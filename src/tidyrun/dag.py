from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from datetime import date, datetime, time
import os
import pickle
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, cast, Literal, Union
from urllib.parse import urlparse

from cloudpathlib import AnyPath, CloudPath
import toml

from tidyrun.job import Job, validate_callable_bindings
from tidyrun.keys import Key, encode_key
from tidyrun.plan import (
    PlanPaths,
    decode_manifest_key,
    failed_path,
    job_definition_file,
    job_output_base,
    job_output_exists,
    normalize_job_id,
    running_path,
    to_path,
    load_callable,
    load_job_definition,
    load_job_inputs,
)

Node = Union[Job, "DAG"]
ProgressCallback = Callable[[str], None]


class DAGExecutionError(Exception):
    """Raised when a DAG job fails during execution.

    Attributes
    ----------
    failed_job_id:
        The job_id of the job that failed.
    cause:
        The original exception raised by the job.
    completed_jobs:
        Set of job_ids that completed successfully before the failure.
    cancelled_jobs:
        Set of job_ids that were pending when the failure occurred and
        were not executed.
    plan_dir:
        Path to the materialized plan directory, if known.
    outputs_path:
        Path where job outputs (and .failed sentinels) are written, if known.
    """

    def __init__(
        self,
        failed_job_id: str,
        cause: BaseException,
        completed_jobs: set[str],
        cancelled_jobs: set[str],
        *,
        plan_dir: Path | None = None,
        outputs_path: Path | None = None,
    ) -> None:
        self.failed_job_id = failed_job_id
        self.cause = cause
        self.completed_jobs = frozenset(completed_jobs)
        self.cancelled_jobs = frozenset(cancelled_jobs)
        self.plan_dir = plan_dir
        self.outputs_path = outputs_path
        super().__init__(str(self))

    def _job_traceback(self) -> str | None:
        """Read the traceback from the .failed sentinel written by the job process."""
        if self.outputs_path is None:
            return None
        sentinel = failed_path(self.outputs_path, self.failed_job_id)
        if not sentinel.is_file():
            return None
        try:
            data = cast(
                dict[str, Any],
                toml.loads(sentinel.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            )
            tb = data.get("traceback")
            return str(tb) if isinstance(tb, str) else None
        except Exception:
            return None

    def rerun_snippet(self) -> str | None:
        """Return a Python snippet that re-runs just the failed job, or None."""
        if self.plan_dir is None:
            return None
        from tidyrun.plan import rerun_snippet as _rerun_snippet

        try:
            return _rerun_snippet(self.plan_dir, self.failed_job_id)
        except Exception:
            return None

    def __str__(self) -> str:
        lines = [f"DAG job {self.failed_job_id!r} failed: {self.cause}"]
        if self.completed_jobs:
            lines.append(f"  Completed jobs: {sorted(self.completed_jobs)}")
        if self.cancelled_jobs:
            lines.append(f"  Cancelled jobs: {sorted(self.cancelled_jobs)}")
        tb = self._job_traceback()
        if tb:
            lines.append("")
            lines.append("Job traceback:")
            lines.append(tb.rstrip())
        snippet = self.rerun_snippet()
        if snippet:
            lines.append("")
            lines.append("To re-run this job interactively:")
            lines.append(snippet)
        return "\n".join(lines)


def is_s3_location(location: Any) -> bool:
    if not isinstance(location, str):
        return False
    parsed = urlparse(location)
    return parsed.scheme == "s3" and bool(parsed.netloc)


def upload_local_tree_to_s3(local_root: Path, s3_location: str) -> None:
    destination = AnyPath(s3_location)
    if not isinstance(destination, CloudPath):
        raise ValueError(f"Expected cloud destination, got: {s3_location!r}")

    leaf_name = _s3_leaf_name(s3_location)

    for source_file in local_root.rglob("*"):
        if not source_file.is_file():
            continue

        relative = source_file.relative_to(local_root)
        relative_parts = relative.parts
        if relative_parts and relative_parts[0] == leaf_name:
            relative = Path(*relative_parts[1:])
        target = destination / relative.as_posix()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_file.read_bytes())


def _s3_leaf_name(location: str) -> str:
    parsed = urlparse(location)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError(f"Invalid S3 location: {location!r}")
    return key.rsplit("/", 1)[-1]


def _default_progress_callback(message: str) -> None:
    print(message)


class _ProgressReporter:
    def __init__(
        self,
        enabled: bool,
        callback: ProgressCallback | None,
        phase: str,
        total: int,
    ) -> None:
        self.enabled = enabled
        self.callback = callback
        self.phase = phase
        self.total = total
        self.done = 0
        self.inline = callback is None
        self._last_render_length = 0

    def _emit(self, message: str) -> None:
        if self.inline:
            padding = max(0, self._last_render_length - len(message))
            print(f"\r{message}{' ' * padding}", end="", flush=True)
            self._last_render_length = len(message)
            return

        callback = self.callback or _default_progress_callback
        callback(message)

    def _finish_inline(self) -> None:
        if self.inline:
            print()
            self._last_render_length = 0

    def _bar(self) -> str:
        width = 24
        if self.total <= 0:
            return "#" * width
        filled = min(width, int((self.done / self.total) * width))
        return ("#" * filled) + ("-" * (width - filled))

    def info(self, message: str) -> None:
        if not self.enabled:
            return
        if self.inline:
            if message.startswith("starting"):
                self._emit(
                    f"[{self.phase}] [{self._bar()}] {self.done}/{self.total} starting"
                )
                return
            if message == "done":
                self._emit(
                    f"[{self.phase}] [{self._bar()}] {self.done}/{self.total} done"
                )
                self._finish_inline()
                return

        self._emit(f"[{self.phase}] {message}")

    def step(self, job_id: str, *, skipped: bool = False) -> None:
        if not self.enabled:
            return
        self.done += 1
        status = "skipped" if skipped else "completed"
        if self.inline:
            self._emit(
                f"[{self.phase}] [{self._bar()}] {self.done}/{self.total} {status}: {job_id}"
            )
            return

        self._emit(f"[{self.phase}] [{self.done}/{self.total}] {status}: {job_id}")


def _count_unique_jobs(
    node: Node,
    seen: set[int],
    _keep_alive: list[Any] | None = None,
) -> int:
    """Count unique Job leaves under *node*.

    *_keep_alive* accumulates all ephemeral sub-nodes created by
    ``ParametrizedJob.__getitem__`` so that Python cannot reuse their object
    IDs while we are still traversing the tree, which would produce false
    cache hits in *seen*.
    """
    if _keep_alive is None:
        _keep_alive = []

    node_id = id(node)
    if node_id in seen:
        return 0
    seen.add(node_id)

    if isinstance(node, Job):
        return 1

    if isinstance(node, ParametrizedJob):
        # Create all children first and stash them in _keep_alive so they
        # remain alive (and keep their IDs stable) for the entire tree walk.
        children = [node[key] for key in node]
        _keep_alive.extend(children)
        return sum(_count_unique_jobs(child, seen, _keep_alive) for child in children)

    return sum(
        _count_unique_jobs(subnode, seen, _keep_alive) for subnode in node.values()
    )


def _job_id_from_path(path: tuple[Key, ...]) -> str:
    if not path:
        raise ValueError("Cannot derive job_id from empty path")
    return "/".join(_encode_key_checked(key) for key in path)


def _job_id_from_path_hint(path_hint: tuple[Any, ...]) -> str | None:
    """Best-effort conversion from internal path hints to a job_id.

    `path_hint` is used in several compilation contexts and can include
    non-user path fragments (for example owner job ids and argument markers).
    In those cases we return ``None`` rather than raising.
    """
    key_path: tuple[Key, ...] = tuple(
        value
        for value in path_hint
        if isinstance(
            value,
            (str, int, float, bool, date, datetime, time),
        )
    )
    try:
        return _job_id_from_path(key_path)
    except ValueError:
        return None


def _encode_key_checked(key: Key) -> str:
    """Encode a key and reject hidden-path components.

    Hidden path components (starting with ".") are disallowed to avoid
    accidental hidden files/directories in materialized DAG plans.
    """
    encoded = encode_key(key)
    if encoded.startswith("."):
        raise ValueError(
            "DAG key encodes to a hidden path component and is not allowed: "
            f"{encoded!r}"
        )
    return encoded


def _callable_import_spec(func: Any) -> tuple[str, str] | None:
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    if module == "__main__" or "<locals>" in qualname:
        return None
    return module, qualname


def _run_compiled_job(plan_paths: PlanPaths, job_id: str) -> None:
    import datetime
    import traceback

    from tidyrun.serialization.api import serialize

    plan_dir = plan_paths.definitions.parent
    definition = load_job_definition(plan_dir, job_id)
    inputs = load_job_inputs(definition, plan_dir)
    func = load_callable(definition)

    running = running_path(plan_paths.outputs, job_id)
    running.parent.mkdir(parents=True, exist_ok=True)
    running.touch()
    try:
        outputs = func(**inputs)
        output_base = job_output_base(plan_paths.outputs, job_id)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        serialize(outputs, output_base)
    except Exception as exc:
        failed_path(plan_paths.outputs, job_id).write_text(
            toml.dumps(  # pyright: ignore[reportUnknownMemberType]
                {
                    "job_id": job_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        raise
    finally:
        if running.exists():
            running.unlink()


def run_materialized_job(dag_path: Any, job_id: str) -> None:
    """Run one job from a materialized DAG plan."""
    _run_compiled_job(PlanPaths.from_root(dag_path), job_id)


def batch_entrypoint() -> None:
    """Container entry point for AWS Batch jobs.

    Reads the plan directory and job identity from environment variables, then
    runs the job exactly as the local subprocess executor would.

    For regular (non-array) jobs the required variables are:

    - ``TIDYRUN_PLAN_DIR`` — S3 URI or path of the materialised plan directory.
    - ``TIDYRUN_JOB_ID`` — the job id to execute.

    For array jobs AWS Batch sets ``AWS_BATCH_JOB_ARRAY_INDEX`` automatically.
    In that case the job id is resolved from ``TIDYRUN_JOB_IDS_JSON`` (a JSON
    array of all job ids in the array), indexed by ``AWS_BATCH_JOB_ARRAY_INDEX``.
    ``TIDYRUN_JOB_ID`` is ignored for array children.

    This function is registered as the ``tidyrun-batch-entrypoint`` console
    script and should be the ``CMD`` of your Batch container image.
    """
    import json
    import sys

    plan_dir = os.environ.get("TIDYRUN_PLAN_DIR")
    if not plan_dir:
        print("TIDYRUN_PLAN_DIR is not set", file=sys.stderr)
        sys.exit(1)

    array_index_str = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if array_index_str is not None:
        job_ids_json = os.environ.get("TIDYRUN_JOB_IDS_JSON")
        if not job_ids_json:
            print(
                "AWS_BATCH_JOB_ARRAY_INDEX is set but TIDYRUN_JOB_IDS_JSON is not",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            job_ids: list[str] = json.loads(job_ids_json)
            index = int(array_index_str)
            job_id = job_ids[index]
        except (json.JSONDecodeError, ValueError, IndexError) as exc:
            print(f"Cannot resolve job_id from array index: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        job_id = os.environ.get("TIDYRUN_JOB_ID")
        if not job_id:
            print("TIDYRUN_JOB_ID is not set", file=sys.stderr)
            sys.exit(1)

    run_materialized_job(plan_dir, job_id)


def _run_compiled_job_entrypoint() -> None:  # pyright: ignore[reportUnusedFunction]
    # Accepts either: <runner_string> <job_id>  (new and compat)
    if len(sys.argv) != 3:
        raise ValueError("Expected arguments: <runner_string> <job_id>")
    plan_paths = PlanPaths.from_runner_string(sys.argv[1])
    _run_compiled_job(plan_paths, sys.argv[2])


def _run_job_in_thread(runner_string: str, job_id: str) -> None:
    """Execute a job directly in the current thread (for ThreadPoolExecutor)."""
    _run_compiled_job(PlanPaths.from_runner_string(runner_string), job_id)


def _run_job_in_process(runner_string: str, job_id: str) -> None:
    """Execute a job in a separate process (for ProcessPoolExecutor)."""
    _run_compiled_job(PlanPaths.from_runner_string(runner_string), job_id)


def _run_job_in_subprocess(runner_string: str, job_id: str) -> None:
    """Execute a job in an isolated subprocess."""
    current_pythonpath = os.pathsep.join(path for path in sys.path if path)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    combined_pythonpath = (
        current_pythonpath
        if not existing_pythonpath
        else f"{current_pythonpath}{os.pathsep}{existing_pythonpath}"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = combined_pythonpath
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tidyrun.dag import _run_compiled_job_entrypoint; "
                "_run_compiled_job_entrypoint()"
            ),
            runner_string,
            job_id,
        ],
        check=True,
        env=env,
    )


def _build_top_level_ref(node: Node, prefix: str) -> dict[str, Any]:
    """Reconstruct the job-ref dict for *node* using *prefix* as its job_id stem.

    This produces the same structure as the ``top_level`` section of the old
    ``plan.tidyrun`` manifest, derived purely from the DAG node tree so that no
    separate metadata file is needed.
    """
    if isinstance(node, Job):
        return {"kind": "job", "job_id": prefix}
    if isinstance(node, ParametrizedJob):
        entries: dict[str, Any] = {}
        for key in node:
            encoded = _encode_key_checked(key)
            entries[encoded] = _build_top_level_ref(node[key], f"{prefix}/{encoded}")
        return {"kind": "group", "entries": entries}
    # nested DAG
    entries = {}
    for key, subnode in node.items():
        encoded = _encode_key_checked(key)
        entries[encoded] = _build_top_level_ref(subnode, f"{prefix}/{encoded}")
    return {"kind": "group", "entries": entries}


def _build_aggregator_deps(
    node: Node, prefix: str, agg_deps: dict[str, list[str]]
) -> str:
    """Walk the DAG tree and register every group node in *agg_deps*.

    Returns the effective job_id for *node*: the leaf job_id for a Job, or
    *prefix* (which becomes the aggregator id) for any group node.
    Populates *agg_deps* mapping each group id to its direct children ids.
    """
    if isinstance(node, Job):
        return prefix

    child_ids: list[str] = []
    for key, subnode in node.items():
        encoded = _encode_key_checked(key)
        child_id = _build_aggregator_deps(subnode, f"{prefix}/{encoded}", agg_deps)
        child_ids.append(child_id)

    agg_deps[prefix] = child_ids
    return prefix


def _run_aggregator_inline(
    agg_id: str, child_ids: list[str], outputs_path: Path
) -> None:
    """Write the dict-folder .tidyrun metadata for a DAG group node.

    Reads the checksum from each child's existing .tidyrun file, combines
    them with checksum_for_named_children, and writes the result at the
    path that corresponds to agg_id inside outputs_path.
    """
    from tidyrun.serialization.metadata import (
        checksum_for_named_children,
        read_metadata,
        write_metadata,
    )

    children: list[tuple[str, Any]] = []
    for child_id in child_ids:
        encoded_name = child_id.rsplit("/", 1)[-1]
        meta = read_metadata(job_output_base(outputs_path, child_id))
        children.append((encoded_name, meta["checksum"]))

    combined = checksum_for_named_children(children)
    write_metadata(
        job_output_base(outputs_path, agg_id),
        encoding="dict-folder",
        suffix="",
        checksum=combined,
    )


def _literal_path_str(abs_path: Path, plan_paths: PlanPaths) -> str:
    """Return the path to store in a definition file for a literal input.

    Relative (to the plan root) when the inputs directory is the standard
    sibling; absolute otherwise (separate PlanPaths).
    """
    plan_root = plan_paths.definitions.parent
    try:
        return str(abs_path.relative_to(plan_root).as_posix())
    except ValueError:
        return str(abs_path)


def _substitute_dependency_placeholders(
    job_id: str,
    parameter_values: Mapping[str, Any],
) -> str:
    """Substitute ``{param}`` tokens in a dependency job_id string."""
    resolved = job_id
    for name, value in parameter_values.items():
        resolved = resolved.replace(f"{{{name}}}", encode_key(value))
    return normalize_job_id(resolved)


def _patch_parameter_values(
    definitions_dir: Path,
    parameter_value_lists: dict[tuple[str, str], list[Any]],
    written_definitions: set[Path],
) -> None:
    """Write accumulated parameter values into their definition TOML files."""
    patched: set[Path] = set()
    for (array_group, arg_name), values in parameter_value_lists.items():
        def_file = job_definition_file(definitions_dir, array_group)
        if def_file not in written_definitions:
            continue
        if def_file in patched:
            definition = cast(
                dict[str, Any],
                toml.loads(def_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
            )
        else:
            definition = cast(
                dict[str, Any],
                toml.loads(def_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
            )
        args = cast(dict[str, Any], definition.setdefault("args", {}))
        args[arg_name] = {"kind": "parameter", "values": values}
        def_file.write_text(
            toml.dumps(definition),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            encoding="utf-8",
        )
        patched.add(def_file)


class DAG(Mapping[Key, Node]):
    """A mapping from keys to deferred computations.

    DAG is the write-time dual of LazyDict: it maps :data:`Key` values to
    nodes (:class:`~tidyrun.Job`, :class:`~tidyrun.ParametrizedJob`, or nested
    :class:`DAG` instances). Evaluating a DAG to disk produces the same on-disk
    layout as
    :func:`~tidyrun.serialize`, so that :func:`~tidyrun.deserialize` returns a
    :class:`~tidyrun.LazyDict` with the same key tree.

    DAG execution is materialized-first: compile a plan on disk, then execute
    each job in a dedicated Python subprocess with arguments loaded through
    :func:`~tidyrun.deserialize`.

    Example::

        import tempfile, pathlib
        from tidyrun.job import Job
        from tidyrun.dag import DAG

        def square(x: int) -> int:
            return x * x

        dag = DAG()
        dag["a"] = Job(func=square, kwargs={"x": 3})

        with tempfile.TemporaryDirectory() as tmp:
            result = dag.evaluate(pathlib.Path(tmp) / "outputs")
            assert result["a"] == 9
    """

    def __init__(self, nodes: Mapping[Key, Node] | None = None) -> None:
        self._nodes: dict[Key, Node] = dict(nodes) if nodes is not None else {}

    def __getitem__(self, key: Key) -> Node:
        return self._nodes[key]

    def __setitem__(self, key: Key, value: Node) -> None:
        self._nodes[key] = value

    def __iter__(self) -> Iterator[Key]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def materialize(
        self,
        dag_path: Any,
        *,
        prefix: str | None = None,
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Path | CloudPath:
        """Write job definitions and literal inputs for process execution.

        Parameters
        ----------
        dag_path :
            Destination for the materialized plan. May be a plain path
            (definitions, inputs, and outputs are created as subdirectories)
            or a :class:`PlanPaths` object that places each component at an
            independent location. Accepts ``s3://`` URIs when the optional
            ``boto3`` dependency is installed.
        prefix :
            Optional string prepended to all job IDs in this plan.
        progress :
            When ``True``, emit progress logs during compilation.
        progress_callback :
            Optional callback that receives each progress message string.

        Returns
        -------
        Path
            The plan root directory (``dag_path`` as a ``Path``, or the
            first component's parent for a ``PlanPaths``).
        """
        from tidyrun.serialization.api import serialize

        seen_nodes: set[int] = set()
        total_jobs = sum(
            _count_unique_jobs(node, seen_nodes) for node in self._nodes.values()
        )
        reporter = _ProgressReporter(
            enabled=progress,
            callback=progress_callback,
            phase="materialize",
            total=total_jobs,
        )
        reporter.info(f"starting ({total_jobs} jobs)")

        # Resolve PlanPaths — support both a plain path and an explicit PlanPaths.
        if isinstance(dag_path, PlanPaths):
            plan_paths = dag_path
            s3_dag_path: str | None = None
            tempdir: TemporaryDirectory[str] | None = None
            plan_dir = plan_paths.definitions.parent
        else:
            s3_dag_path = dag_path if is_s3_location(dag_path) else None
            tempdir = None
            temp_root: Path | None = None
            if s3_dag_path is not None:
                tempdir = TemporaryDirectory()
                temp_root = Path(tempdir.name)
                plan_dir = temp_root / _s3_leaf_name(s3_dag_path)
            else:
                plan_dir = to_path(dag_path)
            plan_paths = PlanPaths.from_root(plan_dir)

        plan_paths.definitions.mkdir(parents=True, exist_ok=True)
        plan_paths.inputs.mkdir(parents=True, exist_ok=True)
        plan_paths.outputs.mkdir(parents=True, exist_ok=True)

        node_to_ref: dict[int, tuple[Node, dict[str, Any]]] = {}
        preferred_job_ids: dict[int, str] = {}
        shared_literal_paths: dict[tuple[str, str], str] = {}
        # Maps (array_group, arg_name) -> list of raw parameter values (one per instance)
        parameter_value_lists: dict[tuple[str, str], list[Any]] = {}
        # Maps array_group -> ordered parameter names
        array_group_parameter_names: dict[str, list[str]] = {}
        collected_job_ids_cache: dict[int, frozenset[str]] = {}
        written_definitions: set[Path] = set()

        prefix_tuple: tuple[Any, ...] = (prefix,) if prefix else ()
        # Maps id(pjob) -> canonical DAG path for top-level ParametrizedJob nodes.
        pjob_paths: dict[int, str] = {}
        # Maps job_id -> compiled ref for deduplication across different Job objects
        # that represent the same logical job (e.g. two calls to pjob[key]).
        compiled_job_ids: dict[str, dict[str, Any]] = {}
        dag_member_path_tuples: dict[int, tuple[Any, ...]] = {}
        for key, node in self._nodes.items():
            path_tuple: tuple[Any, ...] = (*prefix_tuple, key)  # type: ignore[arg-type]
            dag_member_path_tuples[id(node)] = path_tuple
            if isinstance(node, Job):
                preferred_job_ids[id(node)] = _job_id_from_path(path_tuple)
            elif isinstance(node, ParametrizedJob):
                pjob_paths[id(node)] = _job_id_from_path(path_tuple)

        def _collect_job_ids(ref: Mapping[str, Any]) -> frozenset[str]:
            cached = collected_job_ids_cache.get(id(ref))
            if cached is not None:
                return cached

            kind = ref.get("kind")
            if kind == "job":
                job_id = ref.get("job_id")
                if not isinstance(job_id, str):
                    raise ValueError(f"Invalid job reference: {ref!r}")
                resolved = frozenset({job_id})
                collected_job_ids_cache[id(ref)] = resolved
                return resolved

            if kind == "group":
                raw_entries = ref.get("entries")
                if not isinstance(raw_entries, dict):
                    raise ValueError(f"Invalid group reference: {ref!r}")
                entries = cast(Mapping[str, Any], raw_entries)
                collected: set[str] = set()
                for entry in entries.values():
                    collected.update(_collect_job_ids(cast(Mapping[str, Any], entry)))
                resolved = frozenset(collected)
                collected_job_ids_cache[id(ref)] = resolved
                return resolved

            raise ValueError(f"Unknown reference kind: {kind!r}")

        def _write_dep_symlink(
            owner_job_id: str, arg_name: str, dep_output_id: str
        ) -> None:
            """Record dependency: symlink on local FS, sidecar on non-local (S3)."""
            input_dir = plan_paths.inputs / owner_job_id
            input_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(plan_paths.inputs, Path) and isinstance(
                plan_paths.outputs, Path
            ):
                symlink_path = plan_paths.inputs / owner_job_id / arg_name
                target = plan_paths.outputs / dep_output_id
                relative_target = Path(os.path.relpath(target, symlink_path.parent))
                if symlink_path.is_symlink():
                    symlink_path.unlink()
                if not symlink_path.exists():
                    symlink_path.symlink_to(relative_target)
            else:
                sidecar_path = plan_paths.inputs / owner_job_id / f"{arg_name}.tidyrun"
                sidecar_path.write_text(dep_output_id, encoding="utf-8")

        def _compile_operand(
            value: Any,
            owner_job_id: str,
            arg_name: str,
            array_group: str | None,
            group_parameter_names: tuple[str, ...] | None,
        ) -> dict[str, Any]:
            if isinstance(value, (Job, DAG)):
                if id(value) not in dag_member_path_tuples:
                    raise ValueError(
                        f"Argument {arg_name!r} of job {owner_job_id!r} depends on a "
                        "Job or DAG that is not a member of this DAG. "
                        "Register it as a DAG member before using it as a dependency."
                    )
                member_path_tuple = dag_member_path_tuples[id(value)]
                ref = _compile_node(
                    value,
                    member_path_tuple,
                    array_group=None,
                    group_parameter_names=None,
                )

                if ref.get("kind") == "job":
                    dep_output_id = ref.get("job_id")
                    if isinstance(dep_output_id, str):
                        _write_dep_symlink(owner_job_id, arg_name, dep_output_id)
                elif isinstance(value, ParametrizedJob):
                    group_root = _node_dag_path(value)
                    if group_root is not None:
                        _write_dep_symlink(owner_job_id, arg_name, group_root)

                return {"kind": "dependency", "ref": ref}

            if array_group is not None and group_parameter_names is not None:
                shared_key = (array_group, arg_name)

                if arg_name in group_parameter_names:
                    # Parameter arg: accumulate raw value, will be inlined in definition.
                    values_list = parameter_value_lists.setdefault(shared_key, [])
                    values_list.append(value)
                    return {"kind": "parameter"}

                # Shared non-parameter literal (same value for all instances).
                shared_path = shared_literal_paths.get(shared_key)
                if shared_path is not None:
                    return {"kind": "literal", "path": shared_path}

                inputs_base = plan_paths.inputs / Path(array_group) / arg_name
                inputs_base.parent.mkdir(parents=True, exist_ok=True)
                serialize(value, inputs_base)
                inputs_base_str = _literal_path_str(inputs_base, plan_paths)
                shared_literal_paths[shared_key] = inputs_base_str
                return {"kind": "literal", "path": inputs_base_str}

            input_base = plan_paths.inputs / Path(owner_job_id) / arg_name
            input_base.parent.mkdir(parents=True, exist_ok=True)
            serialize(value, input_base)
            return {
                "kind": "literal",
                "path": _literal_path_str(input_base, plan_paths),
            }

        def _node_dag_path(node: Any) -> str | None:
            """Walk the _pjob_parent chain to find the canonical DAG job_id.

            When ParametrizedJob.__getitem__ creates a child it tags it with
            ``_pjob_parent`` (the ParametrizedJob) and ``_pjob_key`` (the lookup
            key).  Top-level ParametrizedJob nodes are registered in ``pjob_paths``
            keyed by id().  Following the chain back to one of those entries yields
            the canonical path (e.g. "produce/1" or "a/m1/train").
            """
            direct = pjob_paths.get(id(node))
            if direct is not None:
                return direct
            parent = getattr(node, "_pjob_parent", None)
            pjob_key = getattr(node, "_pjob_key", None)
            if parent is None or pjob_key is None:
                return None
            parent_path = _node_dag_path(parent)
            if parent_path is None:
                return None
            try:
                return f"{parent_path}/{_encode_key_checked(pjob_key)}"
            except ValueError:
                return None

        def _compile_node(
            node: Node,
            path_hint: tuple[Any, ...],
            array_group: str | None,
            group_parameter_names: tuple[str, ...] | None,
        ) -> dict[str, Any]:
            node_id = id(node)
            existing = node_to_ref.get(node_id)
            if existing is not None and existing[0] is node:
                return existing[1]

            if isinstance(node, Job):
                preferred = preferred_job_ids.get(node_id)
                if preferred is None:
                    preferred = _node_dag_path(node)
                # If the same logical job was already compiled via a different
                # object (another call to pjob[key] returns a fresh instance),
                # reuse the existing ref rather than writing a duplicate definition.
                if preferred is not None:
                    existing_ref = compiled_job_ids.get(preferred)
                    if existing_ref is not None:
                        node_to_ref[node_id] = (node, existing_ref)
                        return existing_ref
                if preferred is None:
                    preferred = _job_id_from_path_hint(path_hint)
                if preferred is None:
                    raise ValueError(
                        "Could not determine a job_id for this node. "
                        "Ensure every Job is reachable from the DAG's top-level nodes."
                    )

                job_id = preferred
                ref: dict[str, Any] = {"kind": "job", "job_id": job_id}
                node_to_ref[node_id] = (node, ref)
                compiled_job_ids[job_id] = ref

                import_spec = _callable_import_spec(node.func)

                args_spec: dict[str, Any] = {}
                dependencies: set[str] = set()
                for arg_name, arg_value in node.kwargs.items():
                    spec = _compile_operand(
                        arg_value,
                        job_id,
                        arg_name,
                        array_group,
                        group_parameter_names,
                    )
                    args_spec[arg_name] = spec
                    if spec["kind"] == "dependency":
                        dependencies.update(
                            _collect_job_ids(cast(Mapping[str, Any], spec["ref"]))
                        )

                definition_group = array_group if array_group is not None else job_id
                definition_file = job_definition_file(
                    plan_paths.definitions, definition_group
                )

                if definition_file not in written_definitions:
                    definition: dict[str, Any] = {
                        "kind": "job_definition",
                        "schema_version": 1,
                        "dependencies": sorted(dependencies),
                        "args": args_spec,
                    }
                    if array_group is not None:
                        # array_group is derived from file path at load time; not stored.
                        param_names = list(
                            array_group_parameter_names.get(array_group, [])
                        )
                        definition["parameter_names"] = param_names
                    if import_spec is not None:
                        definition["callable_module"] = import_spec[0]
                        definition["callable_qualname"] = import_spec[1]
                    else:
                        try:
                            import cloudpickle as _cpickle  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

                            _pickle_bytes = _cpickle.dumps(node.func)  # pyright: ignore[reportUnknownMemberType]
                        except ImportError:
                            _pickle_bytes = pickle.dumps(node.func)
                        definition["callable_data"] = {
                            "encoding": "pickle-base64",
                            "data": base64.b64encode(_pickle_bytes).decode("ascii"),
                        }
                    definition_file.parent.mkdir(parents=True, exist_ok=True)
                    definition_file.write_text(
                        toml.dumps(definition),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                        encoding="utf-8",
                    )
                    written_definitions.add(definition_file)

                reporter.step(job_id)
                return ref

            if isinstance(node, ParametrizedJob):
                entries: dict[str, Any] = {}
                # If this ParametrizedJob is registered at the DAG top level,
                # use its canonical path rather than the path_hint.
                canonical_path = pjob_paths.get(node_id)
                effective_array_group = (
                    array_group
                    if array_group is not None
                    else canonical_path
                    if canonical_path is not None
                    else _job_id_from_path_hint(path_hint)
                )
                effective_group_parameter_names = (
                    group_parameter_names
                    if group_parameter_names is not None
                    else tuple(node.parameter_names)
                )
                if effective_array_group is not None:
                    array_group_parameter_names[effective_array_group] = list(
                        effective_group_parameter_names
                    )

                for key in node:
                    entries[_encode_key_checked(key)] = _compile_node(
                        node[key],
                        (*path_hint, key),
                        effective_array_group,
                        effective_group_parameter_names,
                    )

                ref = {"kind": "group", "entries": entries}
                node_to_ref[node_id] = (node, ref)
                return ref

            entries = {
                _encode_key_checked(key): _compile_node(
                    subnode,
                    (*path_hint, key),
                    array_group,
                    group_parameter_names,
                )
                for key, subnode in node.items()
            }
            ref = {"kind": "group", "entries": entries}
            node_to_ref[node_id] = (node, ref)
            return ref

        for key, node in self._nodes.items():
            _compile_node(
                node,
                (*prefix_tuple, key),
                array_group=None,
                group_parameter_names=None,
            )

        # Patch parameter arg specs into written definitions with accumulated values.
        _patch_parameter_values(
            plan_paths.definitions,
            parameter_value_lists,
            written_definitions,
        )

        if s3_dag_path is not None:
            assert tempdir is not None
            upload_local_tree_to_s3(plan_dir.parent, s3_dag_path)
            tempdir.cleanup()
            reporter.info("done")
            return AnyPath(s3_dag_path)

        if tempdir is not None:
            tempdir.cleanup()

        reporter.info("done")
        return plan_dir

    def execute_materialized(
        self,
        dag_path: Any,
        executor: Executor | None = None,
        max_workers: int | None = None,
        job_resources: Mapping[Key, Mapping[str, str | int]] | None = None,
        execution_mode: Literal["subprocess", "thread", "process"] = "subprocess",
        skip_completed: bool = False,
        skip_running: bool = False,
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Execute a previously materialized plan with dependency ordering.

        Parameters
        ----------
        dag_path:
            Path to the materialized DAG directory. Outputs are always written
            to ``dag_path/outputs``.
        executor:
            Optional custom :class:`~concurrent.futures.Executor`.
        max_workers:
            Number of workers for parallel execution. Creates a
            ThreadPoolExecutor if execution_mode is "thread", or
            ProcessPoolExecutor if execution_mode is "process".
        job_resources:
            Optional per-node submission options.
        execution_mode:
            How to execute jobs: "subprocess" (default, isolated Python processes),
            "thread" (shared memory in threads), or "process" (ProcessPoolExecutor).
        skip_completed:
            When ``True``, skip any job whose output already exists on disk.
            This enables resuming a partially-completed DAG after a failure
            without re-running jobs that already succeeded.
        progress:
            When ``True``, emit progress logs while executing jobs.
        progress_callback:
            Optional callback used for progress messages.
        """
        from tidyrun.serialization.api import deserialize

        if executor is not None and max_workers is not None:
            raise ValueError("Pass either executor or max_workers, not both")

        plan_dir = to_path(dag_path)
        plan_paths = PlanPaths.from_root(plan_dir)
        plan_paths.outputs.mkdir(parents=True, exist_ok=True)
        runner_string = plan_paths.to_runner_string()

        # Discover jobs by scanning definitions/.
        definitions_dir = plan_paths.definitions
        if not definitions_dir.is_dir():
            raise ValueError(
                f"No materialized plan found at {plan_dir}. Run materialize() first."
            )

        dependencies: dict[str, set[str]] = {}
        array_group_by_job_id: dict[str, str] = {}
        array_groups: dict[str, set[str]] = {}

        # Top-level is always reconstructed from self._nodes (no file needed).
        top_level: dict[str, Any] = {
            _encode_key_checked(key): _build_top_level_ref(
                node, _encode_key_checked(key)
            )
            for key, node in self._nodes.items()
        }

        for def_file in sorted(definitions_dir.rglob("*.tidyrun")):
            definition = cast(
                dict[str, Any],
                toml.loads(def_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
            )
            parameter_names = list(definition.get("parameter_names", []))
            dep_list = [
                normalize_job_id(d)
                for d in cast(list[str], definition.get("dependencies", []))
            ]
            if parameter_names:
                # Derive array_group from file path (not stored in file).
                array_group = str(
                    def_file.relative_to(definitions_dir).with_suffix("").as_posix()
                )
                args = cast(dict[str, Any], definition.get("args", {}))
                per_param_values: list[list[Any]] = []
                for pname in parameter_names:
                    arg_spec = cast(Mapping[str, Any], args.get(pname, {}))
                    per_param_values.append(list(arg_spec.get("values", [])))
                if per_param_values:
                    n_instances = len(per_param_values[0])
                    for i in range(n_instances):
                        parameter_value_by_name = {
                            parameter_names[p]: per_param_values[p][i]
                            for p in range(len(parameter_names))
                        }
                        job_id = (
                            array_group
                            + "/"
                            + "/".join(
                                encode_key(per_param_values[p][i])
                                for p in range(len(parameter_names))
                            )
                        )
                        dependencies[job_id] = {
                            _substitute_dependency_placeholders(
                                dep,
                                parameter_value_by_name,
                            )
                            for dep in dep_list
                        }
                        array_group_by_job_id[job_id] = array_group
                        array_groups.setdefault(array_group, set()).add(job_id)
                else:
                    dependencies[array_group] = set(dep_list)
                    array_group_by_job_id[array_group] = array_group
                    array_groups.setdefault(array_group, set()).add(array_group)
            else:
                rel = def_file.relative_to(definitions_dir).with_suffix("")
                job_id = normalize_job_id(rel.as_posix())
                dependencies[job_id] = set(dep_list)

        if max_workers is not None:
            if execution_mode == "process":
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    return self.execute_materialized(
                        plan_dir,
                        executor=pool,
                        job_resources=job_resources,
                        execution_mode=execution_mode,
                        skip_completed=skip_completed,
                        skip_running=skip_running,
                        progress=progress,
                        progress_callback=progress_callback,
                    )
            else:  # "subprocess" or "thread"
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    return self.execute_materialized(
                        plan_dir,
                        executor=pool,
                        job_resources=job_resources,
                        execution_mode=execution_mode,
                        skip_completed=skip_completed,
                        skip_running=skip_running,
                        progress=progress,
                        progress_callback=progress_callback,
                    )

        # Build synthetic aggregator jobs for every group node in the DAG tree.
        # These run inline (no subprocess) after their children complete and write
        # the dict-folder .tidyrun metadata for intermediate and root output folders.
        aggregator_deps: dict[str, list[str]] = {}
        root_children: list[str] = []
        for key, node in self._nodes.items():
            encoded = _encode_key_checked(key)
            child_id = _build_aggregator_deps(node, encoded, aggregator_deps)
            root_children.append(child_id)

        aggregator_job_ids: set[str] = set(aggregator_deps)
        for agg_id, child_ids in aggregator_deps.items():
            dependencies[agg_id] = set(child_ids)

        resources_by_key: Mapping[Key, Mapping[str, str | int]] = (
            {} if job_resources is None else job_resources
        )
        unknown_keys = [key for key in resources_by_key if key not in self._nodes]
        if unknown_keys:
            raise ValueError(f"job_resources contains unknown DAG keys: {unknown_keys}")

        top_level_job_ids: dict[Key, str] = {}
        for key, raw_ref in top_level.items():
            decoded = decode_manifest_key(key)
            ref = cast(Mapping[str, Any], raw_ref)
            if isinstance(ref, dict) and ref.get("kind") == "job":
                job_id = ref.get("job_id")
                if isinstance(job_id, str):
                    top_level_job_ids[decoded] = normalize_job_id(job_id)

        resources_by_job_id = {
            job_id: dict(resources_by_key[key])
            for key, job_id in top_level_job_ids.items()
            if key in resources_by_key
        }

        real_job_count = len(dependencies) - len(aggregator_job_ids)
        reporter = _ProgressReporter(
            enabled=progress,
            callback=progress_callback,
            phase="execute",
            total=real_job_count,
        )
        reporter.info(f"starting ({real_job_count} jobs)")

        # Guard against accidentally mixing old and new results when reusing a
        # partially executed plan without resume semantics.
        if not skip_completed:
            existing_outputs = sorted(
                job_id
                for job_id in dependencies
                if job_id not in aggregator_job_ids
                and job_output_exists(plan_paths.outputs, job_id)
            )
            if existing_outputs:
                raise ValueError(
                    "Materialized plan already has existing job outputs. "
                    "Use skip_completed=True to resume, or clear outputs before "
                    "re-running. Existing job ids: "
                    f"{existing_outputs}"
                )

        dependents: dict[str, set[str]] = {job_id: set() for job_id in dependencies}
        for job_id, deps in dependencies.items():
            for dep in deps:
                dependents.setdefault(dep, set()).add(job_id)

        ready = sorted(job_id for job_id, deps in dependencies.items() if not deps)
        submitted: set[str] = set()
        completed: set[str] = set()

        if execution_mode == "thread":
            job_runner = _run_job_in_thread
        elif execution_mode == "process":
            job_runner = _run_job_in_process
        else:  # "subprocess"
            job_runner = _run_job_in_subprocess

        def _should_skip_em(job_id: str) -> bool:
            if skip_completed and job_output_exists(plan_paths.outputs, job_id):
                return True
            if skip_running and running_path(plan_paths.outputs, job_id).exists():
                return True
            return False

        if executor is None:
            while ready:
                job_id = ready.pop(0)
                if _should_skip_em(job_id):
                    completed.add(job_id)
                    if job_id not in aggregator_job_ids:
                        reporter.step(job_id, skipped=True)
                elif job_id in aggregator_job_ids:
                    _run_aggregator_inline(
                        job_id, aggregator_deps[job_id], plan_paths.outputs
                    )
                    completed.add(job_id)
                else:
                    try:
                        job_runner(runner_string, job_id)
                    except Exception as exc:
                        remaining = set(ready)
                        remaining.update(
                            jid
                            for jid in dependencies
                            if jid not in completed and jid != job_id
                        )
                        raise DAGExecutionError(
                            failed_job_id=job_id,
                            cause=exc,
                            completed_jobs=completed,
                            cancelled_jobs=remaining - completed,
                            plan_dir=plan_dir,
                            outputs_path=plan_paths.outputs,
                        ) from exc
                    completed.add(job_id)
                    reporter.step(job_id)
                for dependent in dependents.get(job_id, set()):
                    dependencies[dependent].discard(job_id)
                    if not dependencies[dependent]:
                        ready.append(dependent)
                ready.sort()

            if len(completed) != len(dependencies):
                raise ValueError("Cycle detected in materialized job dependencies")
        else:
            futures: dict[Future[Any], set[str]] = {}
            submit_with_options = getattr(executor, "submit_with_options", None)
            submit_array_with_options = getattr(
                executor, "submit_array_with_options", None
            )

            if resources_by_job_id and submit_with_options is None:
                raise ValueError(
                    "job_resources requires an executor that supports "
                    "submit_with_options"
                )

            def _mark_completed(job_id: str, *, skipped: bool = False) -> None:
                if job_id in completed:
                    return
                completed.add(job_id)
                if job_id not in aggregator_job_ids:
                    reporter.step(job_id, skipped=skipped)
                for dependent in dependents.get(job_id, set()):
                    dependencies[dependent].discard(job_id)
                    if not dependencies[dependent]:
                        ready.append(dependent)

            def _common_options_for_jobs(
                job_ids: list[str],
            ) -> dict[str, str | int] | None:
                if not job_ids:
                    return {}
                options_list = [
                    dict(resources_by_job_id.get(job_id, {})) for job_id in job_ids
                ]
                first = options_list[0]
                for options in options_list[1:]:
                    if options != first:
                        return None
                return first

            def _submit_ready() -> None:
                while ready:
                    job_id = ready.pop(0)
                    if job_id in submitted:
                        continue
                    submitted.add(job_id)
                    if _should_skip_em(job_id):
                        _mark_completed(job_id, skipped=True)
                        continue

                    if job_id in aggregator_job_ids:
                        _run_aggregator_inline(
                            job_id, aggregator_deps[job_id], plan_paths.outputs
                        )
                        _mark_completed(job_id)
                        continue

                    array_group = array_group_by_job_id.get(job_id)
                    if (
                        array_group is not None
                        and submit_array_with_options is not None
                        and array_group in array_groups
                    ):
                        ready_set = set(ready)
                        batch = sorted(
                            jid
                            for jid in array_groups[array_group]
                            if jid == job_id or jid in ready_set
                        )
                        common_options = _common_options_for_jobs(batch)
                        if len(batch) > 1 and common_options is not None:
                            for jid in batch:
                                submitted.add(jid)
                            ready[:] = [jid for jid in ready if jid not in set(batch)]

                            to_run: list[str] = []
                            for jid in batch:
                                if _should_skip_em(jid):
                                    _mark_completed(jid, skipped=True)
                                else:
                                    to_run.append(jid)

                            if not to_run:
                                continue

                            array_options = dict(common_options)
                            array_options.setdefault("job_name", array_group)
                            submission = submit_array_with_options(
                                job_runner,
                                runner_string,
                                to_run,
                                sbatch_options=array_options,
                            )
                            array_future = cast(
                                Future[Any],
                                getattr(submission, "future", submission),
                            )
                            submitted_job_ids = cast(
                                tuple[str, ...],
                                getattr(submission, "job_ids", tuple(to_run)),
                            )
                            futures[array_future] = set(submitted_job_ids)
                            continue

                    if (
                        job_id in resources_by_job_id
                        and submit_with_options is not None
                    ):
                        future = submit_with_options(
                            job_runner,
                            runner_string,
                            job_id,
                            sbatch_options=dict(resources_by_job_id[job_id]),
                        )
                    else:
                        future = executor.submit(job_runner, runner_string, job_id)
                    futures[future] = {job_id}

            _submit_ready()
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                failed_job_id: str | None = None
                failed_exc: BaseException | None = None
                for future in done:
                    finished_job_ids = futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        failed_job_id = sorted(finished_job_ids)[0]
                        failed_exc = exc
                        break
                    for job_id in sorted(finished_job_ids):
                        _mark_completed(job_id)
                    ready.sort()
                if failed_job_id is not None:
                    cancelled: set[str] = set()
                    for f, job_ids in list(futures.items()):
                        f.cancel()
                        cancelled.update(job_ids)
                    cancelled.update(ready)
                    assert failed_exc is not None
                    raise DAGExecutionError(
                        failed_job_id=failed_job_id,
                        cause=failed_exc,
                        completed_jobs=completed,
                        cancelled_jobs=cancelled,
                        plan_dir=plan_dir,
                        outputs_path=plan_paths.outputs,
                    )
                _submit_ready()

            if len(completed) != len(dependencies):
                raise ValueError("Cycle detected in materialized job dependencies")

        # Write the root dict-folder .tidyrun for the outputs directory so the
        # on-disk layout is identical to serialize(dict, outputs_path).
        from tidyrun.serialization.metadata import (
            checksum_for_named_children,
            metadata_exists,
            read_metadata,
            write_metadata,
        )

        if not (skip_completed and metadata_exists(plan_paths.outputs)):
            root_items: list[tuple[str, Any]] = []
            for child_id in root_children:
                encoded_name = child_id.rsplit("/", 1)[-1]
                meta = read_metadata(job_output_base(plan_paths.outputs, child_id))
                root_items.append((encoded_name, meta["checksum"]))
            root_checksum = checksum_for_named_children(root_items)
            write_metadata(
                plan_paths.outputs,
                encoding="dict-folder",
                suffix="",
                checksum=root_checksum,
            )

        reporter.info("done")
        return deserialize(plan_paths.outputs)

    def evaluate_in_subprocesses(
        self,
        dag_path: Any,
        executor: Executor | None = None,
        max_workers: int | None = None,
        job_resources: Mapping[Key, Mapping[str, str | int]] | None = None,
        execution_mode: Literal["subprocess", "thread", "process"] = "subprocess",
        skip_completed: bool = False,
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Materialize and execute jobs.

        Parameters
        ----------
        dag_path:
            Location for the materialized plan. Outputs are written to
            ``dag_path/outputs``.
        executor:
            Optional custom executor.
        max_workers:
            Number of workers for parallel execution.
        job_resources:
            Optional per-node submission options.
        execution_mode:
            How to execute jobs: "subprocess" (default, isolated Python processes),
            "thread" (shared memory in threads), or "process" (ProcessPoolExecutor).
        skip_completed:
            When ``True``, skip jobs whose outputs already exist in the
            materialized plan.
        progress:
            When ``True``, emit progress logs for materialization and execution.
        progress_callback:
            Optional callback used for progress messages.
        """
        plan_dir = self.materialize(
            dag_path,
            progress=progress,
            progress_callback=progress_callback,
        )
        return self.execute_materialized(
            plan_dir,
            executor=executor,
            max_workers=max_workers,
            job_resources=job_resources,
            execution_mode=execution_mode,
            skip_completed=skip_completed,
            progress=progress,
            progress_callback=progress_callback,
        )

    def evaluate(
        self,
        dag_path: Any,
        executor: Executor | None = None,
        max_workers: int | None = None,
        job_resources: Mapping[Key, Mapping[str, str | int]] | None = None,
        execution_mode: Literal["subprocess", "thread", "process"] = "subprocess",
        skip_completed: bool = False,
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Evaluate this DAG to disk.

        Parameters
        ----------
        dag_path:
            Directory for the materialized plan. Outputs are written to
            ``dag_path/outputs``.
        executor:
            Optional :class:`~concurrent.futures.Executor` for parallel
            job launches.
        max_workers:
            Number of local workers for parallel evaluation. When set,
            creates a :class:`~concurrent.futures.ThreadPoolExecutor` (for
            "thread" or "subprocess" modes) or
            :class:`~concurrent.futures.ProcessPoolExecutor` (for "process" mode).
            Cannot be combined with `executor`.
        job_resources:
            Optional per-node submission options keyed by DAG key. This is
            primarily useful with executors that expose
            ``submit_with_options(..., sbatch_options=...)``, such as
            ``SlurmExecutor``.
        execution_mode:
            How to execute jobs:

            - ``"subprocess"`` (default): Each job runs in an isolated Python
              subprocess with full process separation. Safest for reproducibility
              but has subprocess spawn overhead.
            - ``"thread"``: Jobs run in threads within the same Python process
              with shared memory. Fast for test DAGs with small jobs but may
              encounter GIL contention.
            - ``"process"``: Jobs run in separate processes via
              :class:`~concurrent.futures.ProcessPoolExecutor`. Similar to
              subprocess but with potentially faster worker pool management.
        skip_completed:
            When ``True``, skip jobs whose outputs already exist in the
            materialized plan.
        progress:
            When ``True``, emit progress logs for materialization and execution.
        progress_callback:
            Optional callback used for progress messages.

        Returns
        -------
        LazyDict
            The deserialized :class:`~tidyrun.LazyDict` at ``dag_path/outputs``
            after all nodes have been written.
        """
        return self.evaluate_in_subprocesses(
            dag_path,
            executor=executor,
            max_workers=max_workers,
            job_resources=job_resources,
            execution_mode=execution_mode,
            skip_completed=skip_completed,
            progress=progress,
            progress_callback=progress_callback,
        )

    def clear_outputs(
        self,
        dag_path: Any,
        job_ids: list[str] | None = None,
    ) -> None:
        """Delete serialized outputs for jobs in a materialized plan.

        Use this to discard stale or incorrect outputs before resubmitting a
        DAG.  When *job_ids* is ``None`` the entire outputs directory is
        removed.  Otherwise only the specified jobs' output files are deleted.

        Parameters
        ----------
        dag_path:
            Path to the materialized DAG directory.
        job_ids:
            Optional list of job IDs whose outputs should be cleared.
            When ``None``, all outputs are removed.
        """
        import shutil

        from tidyrun.serialization.metadata import metadata_path, read_metadata

        plan_dir = to_path(dag_path)
        outputs_dir = PlanPaths.from_root(plan_dir).outputs

        if job_ids is None:
            if outputs_dir.exists():
                shutil.rmtree(outputs_dir)
            return

        for job_id in job_ids:
            base = job_output_base(outputs_dir, job_id)
            meta = metadata_path(base)
            if not meta.is_file():
                continue
            try:
                md = read_metadata(base)
                suffix = md.get("suffix", "")
            except Exception:
                suffix = ""
            if suffix:
                payload = Path(str(base) + suffix)
            else:
                payload = base
            if payload.exists():
                if payload.is_dir():
                    shutil.rmtree(payload)
                else:
                    payload.unlink()
            meta.unlink()


class ParametrizedJob(DAG):
    """A deferred computation indexed by parameter keys.

    Parameters are declared through ``parameter_names`` and populated through
    ``parameter_values``. Accessing a key fixes the first parameter and returns
    either a :class:`Job` (when one parameter remains) or another
    :class:`ParametrizedJob` (when more parameters remain).

    Being a subclass of :class:`DAG`, a ``ParametrizedJob`` inherits all
    execution methods (``materialize``, ``execute_materialized``,
    ``evaluate_in_subprocesses``, ``evaluate``, ``clear_outputs``) with
    identical semantics: the top-level keys are the first-level parameter
    values and no extra wrapping level is added.
    """

    func: Callable[..., Any]
    parameter_names: tuple[str, ...]
    parameter_values: tuple[tuple[Key, ...], ...]
    kwargs: Mapping[str, Any]

    def __init__(
        self,
        func: Callable[..., Any],
        parameter_names: list[str] | tuple[str, ...],
        parameter_values: list[tuple[Key, ...]] | tuple[tuple[Key, ...], ...],
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.func = func
        self.parameter_names = tuple(parameter_names)
        self.parameter_values = tuple(tuple(v) for v in parameter_values)
        self.kwargs = {} if kwargs is None else kwargs
        self._validate()

    @property  # type: ignore[override]
    def _nodes(self) -> dict[Key, Node]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return {k: self[k] for k in self}

    def __setitem__(self, key: Key, value: Node) -> None:
        raise TypeError(f"{type(self).__name__!r} does not support item assignment")

    def __getitem__(self, key: Key) -> Job | ParametrizedJob:
        # When a string that matches a parameter name is used as the key (and
        # is not itself one of the actual parameter values), return a sentinel
        # Job that the compiler recognises as a "same-parameter selector" and
        # translates into a per-instance dep template like "produce/{x}".
        if (
            isinstance(key, str)
            and key in self.parameter_names
            and all(values[0] != key for values in self.parameter_values)
        ):
            sentinel: Job = object.__new__(Job)
            sentinel.__dict__["func"] = self.func
            sentinel.__dict__["kwargs"] = self.kwargs
            sentinel._pjob_parent = self  # type: ignore[attr-defined]
            sentinel._pjob_key = key  # type: ignore[attr-defined]
            return sentinel

        matching = [values for values in self.parameter_values if values[0] == key]
        if not matching:
            raise KeyError(key)

        parameter_name = self.parameter_names[0]
        bound_kwargs = dict(self.kwargs)
        bound_kwargs[parameter_name] = key

        if len(self.parameter_names) == 1:
            result: Job | ParametrizedJob = Job(func=self.func, kwargs=bound_kwargs)
        else:
            remaining_names = self.parameter_names[1:]
            remaining_values = [values[1:] for values in matching]
            result = ParametrizedJob(
                func=self.func,
                parameter_names=remaining_names,
                parameter_values=remaining_values,
                kwargs=bound_kwargs,
            )

        # Tag the result so the compiler can resolve its canonical job_id from the
        # DAG path of this ParametrizedJob, without relying on Python object identity.
        result._pjob_parent = self  # type: ignore[attr-defined]
        result._pjob_key = key  # type: ignore[attr-defined]
        return result

    def __iter__(self) -> Iterator[Key]:
        seen: set[Key] = set()
        for values in self.parameter_values:
            first = values[0]
            if first in seen:
                continue
            seen.add(first)
            yield first

    def __len__(self) -> int:
        return len(set(values[0] for values in self.parameter_values))

    def _validate(self) -> None:
        if not self.parameter_names:
            raise ValueError("parameter_names must not be empty")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique")
        expected_arity = len(self.parameter_names)
        seen: set[tuple[Key, ...]] = set()
        for values in self.parameter_values:
            if len(values) != expected_arity:
                raise ValueError(
                    f"Each parameter tuple must have length {expected_arity}"
                )
            for key in values:
                encode_key(key)
            if values in seen:
                raise ValueError("parameter_values must not contain duplicates")
            seen.add(values)
        validate_callable_bindings(
            func=self.func,
            kwargs=self.kwargs,
            parameter_names=self.parameter_names,
        )


def execute_plan(
    plan_dir: Any,
    *,
    skip_completed: bool = False,
    skip_running: bool = False,
    max_workers: int | None = None,
    execution_mode: Literal["subprocess", "thread", "process"] = "subprocess",
    progress: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Execute all jobs found by scanning *plan_dir*/definitions/.

    This is the standalone counterpart to :meth:`DAG.execute_materialized` for
    the decentralised case where multiple DAGs have been independently
    materialised into the same plan directory using the ``prefix=`` parameter.
    Results are written to *plan_dir*/outputs/; use
    :func:`~tidyrun.deserialize` to read individual job outputs.

    Parameters
    ----------
    plan_dir:
        Root of the materialised plan (must contain a ``definitions/`` subdir).
    skip_completed:
        Skip jobs whose output already exists.
    skip_running:
        Skip jobs whose ``.running`` sentinel exists (already in-flight from
        another process).
    max_workers:
        Number of workers for parallel execution.
    execution_mode:
        ``"subprocess"`` (default), ``"thread"``, or ``"process"``.
    progress:
        Emit progress messages.
    progress_callback:
        Optional progress callback; defaults to :func:`print`.
    """
    from tidyrun.plan import enumerate_job_ids_from_definitions

    plan_path = to_path(plan_dir)
    plan_paths = PlanPaths.from_root(plan_path)
    definitions_dir = plan_paths.definitions

    if not definitions_dir.is_dir():
        raise ValueError(
            f"No materialized plan found at {plan_path}. Run materialize() first."
        )

    job_map = enumerate_job_ids_from_definitions(definitions_dir)
    if not job_map:
        return

    # Build dependency graph by reading each definition.
    dependencies: dict[str, set[str]] = {}
    array_group_by_job_id: dict[str, str] = {}
    array_groups: dict[str, set[str]] = {}

    for def_file in sorted(definitions_dir.rglob("*.tidyrun")):
        definition = cast(
            dict[str, Any],
            toml.loads(def_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
        )
        parameter_names = list(definition.get("parameter_names", []))
        dep_list = [
            normalize_job_id(d)
            for d in cast(list[str], definition.get("dependencies", []))
        ]
        if parameter_names:
            array_group = str(
                def_file.relative_to(definitions_dir).with_suffix("").as_posix()
            )
            args = cast(dict[str, Any], definition.get("args", {}))
            per_param_values: list[list[Any]] = []
            for pname in parameter_names:
                arg_spec = cast(Mapping[str, Any], args.get(pname, {}))
                per_param_values.append(list(arg_spec.get("values", [])))
            if per_param_values:
                n_instances = len(per_param_values[0])
                for i in range(n_instances):
                    parameter_value_by_name = {
                        parameter_names[p]: per_param_values[p][i]
                        for p in range(len(parameter_names))
                    }
                    job_id = (
                        array_group
                        + "/"
                        + "/".join(
                            encode_key(per_param_values[p][i])
                            for p in range(len(parameter_names))
                        )
                    )
                    dependencies[job_id] = {
                        _substitute_dependency_placeholders(
                            dep, parameter_value_by_name
                        )
                        for dep in dep_list
                    }
                    array_group_by_job_id[job_id] = array_group
                    array_groups.setdefault(array_group, set()).add(job_id)
            else:
                dependencies[array_group] = set(dep_list)
                array_group_by_job_id[array_group] = array_group
                array_groups.setdefault(array_group, set()).add(array_group)
        else:
            rel = def_file.relative_to(definitions_dir).with_suffix("")
            job_id = normalize_job_id(rel.as_posix())
            dependencies[job_id] = set(dep_list)

    runner_string = plan_paths.to_runner_string()
    if execution_mode == "thread":
        job_runner = _run_job_in_thread
    elif execution_mode == "process":
        job_runner = _run_job_in_process
    else:
        job_runner = _run_job_in_subprocess

    reporter = _ProgressReporter(
        enabled=progress,
        callback=progress_callback,
        phase="execute",
        total=len(dependencies),
    )
    reporter.info(f"starting ({len(dependencies)} jobs)")

    dependents: dict[str, set[str]] = {job_id: set() for job_id in dependencies}
    for job_id, deps in dependencies.items():
        for dep in deps:
            dependents.setdefault(dep, set()).add(job_id)

    ready = sorted(job_id for job_id, deps in dependencies.items() if not deps)
    completed: set[str] = set()

    def _should_skip(job_id: str) -> bool:
        if skip_completed and job_output_exists(plan_paths.outputs, job_id):
            return True
        if skip_running and running_path(plan_paths.outputs, job_id).exists():
            return True
        return False

    if max_workers is None:
        while ready:
            job_id = ready.pop(0)
            if _should_skip(job_id):
                completed.add(job_id)
                reporter.step(job_id, skipped=True)
            else:
                try:
                    job_runner(runner_string, job_id)
                except Exception as exc:
                    remaining = set(ready)
                    remaining.update(
                        jid
                        for jid in dependencies
                        if jid not in completed and jid != job_id
                    )
                    raise DAGExecutionError(
                        failed_job_id=job_id,
                        cause=exc,
                        completed_jobs=completed,
                        cancelled_jobs=remaining - completed,
                        plan_dir=plan_path,
                        outputs_path=plan_paths.outputs,
                    ) from exc
                completed.add(job_id)
                reporter.step(job_id)
            for dependent in dependents.get(job_id, set()):
                dependencies[dependent].discard(job_id)
                if not dependencies[dependent]:
                    ready.append(dependent)
            ready.sort()
    else:
        if execution_mode == "process":
            pool_cls = ProcessPoolExecutor
        else:
            pool_cls = ThreadPoolExecutor
        with pool_cls(max_workers=max_workers) as pool:
            futures: dict[Future[Any], str] = {}
            submitted: set[str] = set()

            def _submit_ready_ep() -> None:
                while ready:
                    jid = ready.pop(0)
                    if jid in submitted:
                        continue
                    submitted.add(jid)
                    if _should_skip(jid):
                        completed.add(jid)
                        reporter.step(jid, skipped=True)
                        for dep in dependents.get(jid, set()):
                            dependencies[dep].discard(jid)
                            if not dependencies[dep]:
                                ready.append(dep)
                        continue
                    futures[pool.submit(job_runner, runner_string, jid)] = jid

            _submit_ready_ep()
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    jid = futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        for f in futures:
                            f.cancel()
                        raise DAGExecutionError(
                            failed_job_id=jid,
                            cause=exc,
                            completed_jobs=completed,
                            cancelled_jobs=set(futures.values()) | set(ready),
                            plan_dir=plan_path,
                            outputs_path=plan_paths.outputs,
                        ) from exc
                    completed.add(jid)
                    reporter.step(jid)
                    for dep in dependents.get(jid, set()):
                        dependencies[dep].discard(jid)
                        if not dependencies[dep]:
                            ready.append(dep)
                _submit_ready_ep()

    if len(completed) != len(job_map):
        raise ValueError("Cycle detected in materialized job dependencies")
    reporter.info("done")
