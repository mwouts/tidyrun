from __future__ import annotations

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
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast, Literal, Union

import toml

from tidyrun.job import Job, ParametrizedJob
from tidyrun.keys import Key, decode_key, encode_key

Node = Union[Job, ParametrizedJob, "DAG"]
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
    """

    def __init__(
        self,
        failed_job_id: str,
        cause: BaseException,
        completed_jobs: set[str],
        cancelled_jobs: set[str],
    ) -> None:
        self.failed_job_id = failed_job_id
        self.cause = cause
        self.completed_jobs = frozenset(completed_jobs)
        self.cancelled_jobs = frozenset(cancelled_jobs)
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [f"DAG job {self.failed_job_id!r} failed: {self.cause}"]
        if self.completed_jobs:
            lines.append(f"  Completed jobs: {sorted(self.completed_jobs)}")
        if self.cancelled_jobs:
            lines.append(f"  Cancelled jobs: {sorted(self.cancelled_jobs)}")
        return "\n".join(lines)


def _to_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


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


def _count_unique_jobs(node: Node, seen: set[int]) -> int:
    node_id = id(node)
    if node_id in seen:
        return 0
    seen.add(node_id)

    if isinstance(node, Job):
        return 1

    if isinstance(node, ParametrizedJob):
        return sum(_count_unique_jobs(node[key], seen) for key in node)

    return sum(_count_unique_jobs(subnode, seen) for subnode in node.values())


def _resolve_plan_and_output_paths(
    target: Any | None,
    dag_path: Any | None,
    output_path: Any | None,
) -> tuple[Path, Path]:
    """Resolve concrete plan/output paths from optional evaluate inputs."""
    if target is None:
        if dag_path is None or output_path is None:
            raise ValueError(
                "Pass target, or pass both dag_path and output_path when target is None"
            )
        return _to_path(dag_path), _to_path(output_path)

    target_path = _to_path(target)
    resolved_plan = target_path / "plan" if dag_path is None else _to_path(dag_path)
    resolved_output = (
        target_path / "outputs" if output_path is None else _to_path(output_path)
    )
    return resolved_plan, resolved_output


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


def _job_output_base(plan_dir: Path, job_id: str) -> Path:
    return plan_dir / "outputs" / Path(job_id)


def _job_output_exists(plan_dir: Path, job_id: str) -> bool:
    """Return True if a job's output metadata file already exists on disk."""
    from tidyrun.serialization.metadata import metadata_exists

    return metadata_exists(_job_output_base(plan_dir, job_id))


def _job_definition_file(plan_dir: Path, job_id: str) -> Path:
    return plan_dir / "definitions" / Path(f"{job_id}.tidyrun")


def _decode_manifest_key(encoded_key: str) -> Key:
    try:
        return decode_key(encoded_key)
    except ValueError:
        # TOML serialization may preserve escaped quotes in keys.
        return decode_key(encoded_key.replace('\\"', '"'))


def _normalize_job_id(job_id: str) -> str:
    return job_id.replace('\\"', '"')


def _callable_import_spec(func: Any) -> tuple[str, str] | None:
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    if module == "__main__" or "<locals>" in qualname:
        return None
    return module, qualname


def _callable_from_import_spec(module: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def load_job_definition(dag_path: Any, job_id: str) -> dict[str, Any]:
    """Load a materialized job definition from disk."""
    plan_dir = _to_path(dag_path)
    definition_file = _job_definition_file(plan_dir, _normalize_job_id(job_id))
    if not definition_file.is_file():
        raise ValueError(f"Missing job definition file: {definition_file}")
    return cast(
        dict[str, Any],
        toml.loads(definition_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )


def load_callable(job_definition: Mapping[str, Any], dag_path: Any) -> Any:
    """Load a callable from job definition metadata."""
    from tidyrun.serialization.api import deserialize

    module = job_definition.get("callable_module")
    qualname = job_definition.get("callable_qualname")
    if isinstance(module, str) and isinstance(qualname, str):
        try:
            return _callable_from_import_spec(module, qualname)
        except Exception:
            # Fall back to serialized callable payload when import path is invalid.
            pass

    callable_path = job_definition.get("callable_path")
    if not isinstance(callable_path, str):
        raise ValueError("Invalid callable metadata in job definition")
    return deserialize(_to_path(dag_path) / Path(callable_path))


def load_job_inputs(job_definition: Mapping[str, Any], dag_path: Any) -> dict[str, Any]:
    """Load all job inputs by deserializing materialized argument specs."""
    kwargs_table = job_definition.get("args")
    if not isinstance(kwargs_table, dict):
        raise ValueError("Invalid args section in job definition")
    typed_kwargs_table = cast(dict[str, Any], kwargs_table)
    plan_dir = _to_path(dag_path)
    resolved_inputs: dict[str, Any] = {}
    for name, raw_spec in typed_kwargs_table.items():
        spec = cast(Mapping[str, Any], raw_spec)
        resolved_inputs[name] = _resolve_arg(plan_dir, spec)
    return resolved_inputs


def _resolve_ref_from_outputs(plan_dir: Path, ref: Mapping[str, Any]) -> Any:
    from tidyrun.serialization.api import deserialize

    kind = ref.get("kind")
    if kind == "job":
        job_id = ref.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError(f"Invalid job reference: {ref!r}")
        return deserialize(_job_output_base(plan_dir, _normalize_job_id(job_id)))

    if kind == "group":
        raw_entries = ref.get("entries")
        if not isinstance(raw_entries, dict):
            raise ValueError(f"Invalid group reference: {ref!r}")
        entries = cast(Mapping[str, Any], raw_entries)
        return {
            _decode_manifest_key(encoded_key): _resolve_ref_from_outputs(
                plan_dir, cast(Mapping[str, Any], entry_ref)
            )
            for encoded_key, entry_ref in entries.items()
        }

    raise ValueError(f"Unknown reference kind: {kind!r}")


def _resolve_arg(plan_dir: Path, spec: Mapping[str, Any]) -> Any:
    from tidyrun.serialization.api import deserialize

    kind = spec.get("kind")
    if kind == "literal":
        relative_path = spec.get("path")
        if not isinstance(relative_path, str):
            raise ValueError(f"Invalid literal arg spec: {spec!r}")
        value = deserialize(plan_dir / Path(relative_path))
        literal_job_id = spec.get("job_id")
        if literal_job_id is None:
            return value
        if not isinstance(literal_job_id, str):
            raise ValueError(f"Invalid literal job_id in arg spec: {spec!r}")
        if isinstance(value, Mapping):
            if literal_job_id not in value:
                raise ValueError(
                    f"Missing grouped literal value for job_id {literal_job_id!r}"
                )
            return cast(Any, value[literal_job_id])
        if isinstance(value, list):
            typed_value = cast(list[object], value)
            for item in typed_value:
                if isinstance(item, tuple):
                    pair = cast(tuple[object, ...], item)
                    if len(pair) != 2:
                        continue
                    key, item_value = pair
                elif isinstance(item, list):
                    pair = cast(list[object], item)
                    if len(pair) != 2:
                        continue
                    key, item_value = pair
                else:
                    continue

                if isinstance(key, str) and key == literal_job_id:
                    return item_value
            raise ValueError(
                f"Missing grouped literal value for job_id {literal_job_id!r}"
            )
        raise ValueError(
            "Grouped literal expects a mapping or list payload keyed by job_id"
        )

    if kind == "dependency":
        ref = spec.get("ref")
        if not isinstance(ref, dict):
            raise ValueError(f"Invalid dependency arg spec: {spec!r}")
        return _resolve_ref_from_outputs(plan_dir, cast(Mapping[str, Any], ref))

    raise ValueError(f"Unknown arg kind: {kind!r}")


def _run_compiled_job(plan_dir: Path, job_id: str) -> None:
    from tidyrun.serialization.api import serialize

    definition = load_job_definition(plan_dir, job_id)
    func = load_callable(definition, plan_dir)
    kwargs = load_job_inputs(definition, plan_dir)

    result = func(**kwargs)
    serialize(result, _job_output_base(plan_dir, job_id))


def run_materialized_job(dag_path: Any, job_id: str) -> None:
    """Run one job from a materialized DAG plan."""
    _run_compiled_job(_to_path(dag_path), job_id)


def _run_compiled_job_entrypoint() -> None:  # pyright: ignore[reportUnusedFunction]
    if len(sys.argv) != 3:
        raise ValueError("Expected arguments: <plan_dir> <job_id>")
    run_materialized_job(sys.argv[1], sys.argv[2])


def _run_job_in_thread(plan_dir: Path | str, job_id: str) -> None:
    """Execute a job directly in the current thread (for ThreadPoolExecutor)."""
    _run_compiled_job(_to_path(plan_dir), job_id)


def _run_job_in_process(plan_dir: Path | str, job_id: str) -> None:
    """Execute a job in a separate process (for ProcessPoolExecutor)."""
    _run_compiled_job(_to_path(plan_dir), job_id)


def _run_job_in_subprocess(plan_dir: Path | str, job_id: str) -> None:
    """Execute a job in an isolated subprocess."""
    plan_dir_path = _to_path(plan_dir)
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
            str(plan_dir_path),
            job_id,
        ],
        check=True,
        env=env,
    )


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
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Write job definitions and literal inputs for process execution.

        This creates a compilable execution plan under *dag_path* and returns
        the created directory path.
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

        plan_dir = _to_path(dag_path)
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "definitions").mkdir(parents=True, exist_ok=True)
        (plan_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (plan_dir / "callables").mkdir(parents=True, exist_ok=True)
        (plan_dir / "outputs").mkdir(parents=True, exist_ok=True)

        jobs: dict[str, dict[str, Any]] = {}
        top_level_refs: dict[str, dict[str, Any]] = {}
        node_to_ref: dict[int, tuple[Node, dict[str, Any]]] = {}
        preferred_job_ids: dict[int, str] = {}
        shared_literal_paths: dict[tuple[str, str], str] = {}
        parameter_literal_maps: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        written_callables: set[Path] = set()
        synthetic_counter = 0

        for key, node in self._nodes.items():
            if isinstance(node, Job):
                preferred_job_ids[id(node)] = _job_id_from_path((key,))

        def _collect_job_ids(ref: Mapping[str, Any]) -> set[str]:
            kind = ref.get("kind")
            if kind == "job":
                job_id = ref.get("job_id")
                if not isinstance(job_id, str):
                    raise ValueError(f"Invalid job reference: {ref!r}")
                return {job_id}

            if kind == "group":
                raw_entries = ref.get("entries")
                if not isinstance(raw_entries, dict):
                    raise ValueError(f"Invalid group reference: {ref!r}")
                entries = cast(Mapping[str, Any], raw_entries)
                collected: set[str] = set()
                for entry in entries.values():
                    collected.update(_collect_job_ids(cast(Mapping[str, Any], entry)))
                return collected

            raise ValueError(f"Unknown reference kind: {kind!r}")

        def _compile_operand(
            value: Any,
            owner_job_id: str,
            arg_name: str,
            array_group: str | None,
            group_parameter_names: frozenset[str] | None,
        ) -> dict[str, Any]:
            if isinstance(value, (Job, ParametrizedJob, DAG)):
                return {
                    "kind": "dependency",
                    "ref": _compile_node(
                        value,
                        (owner_job_id, "arg", arg_name),
                        array_group=None,
                        group_parameter_names=None,
                    ),
                }

            if array_group is not None and group_parameter_names is not None:
                shared_key = (array_group, arg_name)

                if arg_name in group_parameter_names:
                    literal_path = shared_literal_paths.get(shared_key)
                    if literal_path is None:
                        shared_base = plan_dir / "inputs" / Path(array_group) / arg_name
                        literal_path = str(shared_base.relative_to(plan_dir).as_posix())
                        shared_literal_paths[shared_key] = literal_path
                    values = parameter_literal_maps.setdefault(shared_key, [])
                    values.append((owner_job_id, value))
                    return {
                        "kind": "literal",
                        "path": literal_path,
                        "job_id": owner_job_id,
                    }

                shared_path = shared_literal_paths.get(shared_key)
                if shared_path is not None:
                    return {
                        "kind": "literal",
                        "path": shared_path,
                    }

                shared_base = plan_dir / "inputs" / Path(array_group) / arg_name
                shared_base.parent.mkdir(parents=True, exist_ok=True)
                serialize(value, shared_base)
                literal_path = str(shared_base.relative_to(plan_dir).as_posix())
                shared_literal_paths[shared_key] = literal_path
                return {
                    "kind": "literal",
                    "path": literal_path,
                }

            input_base = plan_dir / "inputs" / Path(owner_job_id) / arg_name
            input_base.parent.mkdir(parents=True, exist_ok=True)
            serialize(value, input_base)
            literal_path = str(input_base.relative_to(plan_dir).as_posix())

            return {
                "kind": "literal",
                "path": literal_path,
            }

        def _compile_node(
            node: Node,
            path_hint: tuple[Any, ...],
            array_group: str | None,
            group_parameter_names: frozenset[str] | None,
        ) -> dict[str, Any]:
            nonlocal synthetic_counter

            node_id = id(node)
            existing = node_to_ref.get(node_id)
            if existing is not None and existing[0] is node:
                ref = existing[1]
                if array_group is not None and ref.get("kind") == "job":
                    existing_job_id = ref.get("job_id")
                    if isinstance(existing_job_id, str):
                        payload = jobs.get(existing_job_id)
                        if payload is not None:
                            payload.setdefault("array_group", array_group)
                return existing[1]

            if isinstance(node, Job):
                preferred = preferred_job_ids.get(node_id)
                if preferred is None:
                    preferred = _job_id_from_path_hint(path_hint)
                if preferred is None:
                    synthetic_counter += 1
                    preferred = f"__job_{synthetic_counter}"

                job_id = preferred
                ref: dict[str, Any] = {"kind": "job", "job_id": job_id}
                node_to_ref[node_id] = (node, ref)

                callable_base = (
                    plan_dir / "callables" / Path(array_group) / "callable"
                    if array_group is not None
                    else plan_dir / "callables" / Path(job_id) / "callable"
                )
                if callable_base not in written_callables:
                    callable_base.parent.mkdir(parents=True, exist_ok=True)
                    serialize(node.func, callable_base)
                    written_callables.add(callable_base)

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

                definition = {
                    "kind": "job_definition",
                    "schema_version": 1,
                    "job_id": job_id,
                    "callable_path": str(
                        callable_base.relative_to(plan_dir).as_posix()
                    ),
                    "dependencies": sorted(dependencies),
                    "args": args_spec,
                }
                if import_spec is not None:
                    definition["callable_module"] = import_spec[0]
                    definition["callable_qualname"] = import_spec[1]
                definition_file = _job_definition_file(plan_dir, job_id)
                definition_file.parent.mkdir(parents=True, exist_ok=True)
                definition_file.write_text(
                    toml.dumps(definition),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    encoding="utf-8",
                )

                jobs[job_id] = {"dependencies": sorted(dependencies)}
                if array_group is not None:
                    jobs[job_id]["array_group"] = array_group
                reporter.step(job_id)
                return ref

            if isinstance(node, ParametrizedJob):
                entries: dict[str, Any] = {}
                effective_array_group = (
                    array_group
                    if array_group is not None
                    else _job_id_from_path_hint(path_hint)
                )
                effective_group_parameter_names = (
                    group_parameter_names
                    if group_parameter_names is not None
                    else frozenset(node.parameter_names)
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
            top_level_refs[_encode_key_checked(key)] = _compile_node(
                node,
                (key,),
                array_group=None,
                group_parameter_names=None,
            )

        for (array_group, arg_name), values in parameter_literal_maps.items():
            shared_base = plan_dir / "inputs" / Path(array_group) / arg_name
            shared_base.parent.mkdir(parents=True, exist_ok=True)
            serialize(values, shared_base)

        manifest = {
            "kind": "dag_plan",
            "schema_version": 1,
            "jobs": jobs,
            "top_level": top_level_refs,
        }
        (plan_dir / "plan.tidyrun").write_text(
            toml.dumps(manifest),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            encoding="utf-8",
        )
        reporter.info("done")
        return plan_dir

    def execute_materialized(
        self,
        target: Any | None = None,
        dag_path: Any | None = None,
        output_path: Any | None = None,
        executor: Executor | None = None,
        max_workers: int | None = None,
        job_resources: Mapping[Key, Mapping[str, str | int]] | None = None,
        execution_mode: Literal["subprocess", "thread", "process"] = "subprocess",
        skip_completed: bool = False,
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Execute a previously materialized plan with dependency ordering.

        Parameters
        ----------
        target:
            Optional target run directory. By default, plan and outputs are
            resolved as ``<target>/plan`` and ``<target>/outputs``.
            May be omitted when both ``dag_path`` and ``output_path`` are
            provided explicitly.
        dag_path:
            Optional path to the materialized DAG directory.
        output_path:
            Optional explicit output path. When omitted, defaults to
            ``<target>/outputs``.
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
        from tidyrun.serialization.api import deserialize, serialize

        if executor is not None and max_workers is not None:
            raise ValueError("Pass either executor or max_workers, not both")

        plan_dir, resolved_output = _resolve_plan_and_output_paths(
            target=target,
            dag_path=dag_path,
            output_path=output_path,
        )
        manifest_file = plan_dir / "plan.tidyrun"
        if not manifest_file.is_file():
            raise ValueError(f"Missing materialized plan file: {manifest_file}")

        manifest = cast(
            dict[str, Any],
            toml.loads(manifest_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
        )
        jobs_table = cast(dict[str, Any], manifest.get("jobs", {}))
        top_level = cast(dict[str, Any], manifest.get("top_level", {}))

        if max_workers is not None:
            if execution_mode == "process":
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    return self.execute_materialized(
                        target=None,
                        dag_path=plan_dir,
                        output_path=resolved_output,
                        executor=pool,
                        job_resources=job_resources,
                        execution_mode=execution_mode,
                        skip_completed=skip_completed,
                        progress=progress,
                        progress_callback=progress_callback,
                    )
            else:  # "subprocess" or "thread"
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    return self.execute_materialized(
                        target=None,
                        dag_path=plan_dir,
                        output_path=resolved_output,
                        executor=pool,
                        job_resources=job_resources,
                        execution_mode=execution_mode,
                        skip_completed=skip_completed,
                        progress=progress,
                        progress_callback=progress_callback,
                    )

        resources_by_key: Mapping[Key, Mapping[str, str | int]] = (
            {} if job_resources is None else job_resources
        )
        unknown_keys = [key for key in resources_by_key if key not in self._nodes]
        if unknown_keys:
            raise ValueError(f"job_resources contains unknown DAG keys: {unknown_keys}")

        top_level_job_ids: dict[Key, str] = {}
        for key, raw_ref in top_level.items():
            decoded = _decode_manifest_key(key)
            ref = cast(Mapping[str, Any], raw_ref)
            if isinstance(ref, dict) and ref.get("kind") == "job":
                job_id = ref.get("job_id")
                if isinstance(job_id, str):
                    top_level_job_ids[decoded] = _normalize_job_id(job_id)

        resources_by_job_id = {
            job_id: dict(resources_by_key[key])
            for key, job_id in top_level_job_ids.items()
            if key in resources_by_key
        }

        dependencies: dict[str, set[str]] = {
            _normalize_job_id(job_id): {
                _normalize_job_id(dep)
                for dep in cast(list[str], payload.get("dependencies", []))
            }
            for job_id, payload in jobs_table.items()
        }
        reporter = _ProgressReporter(
            enabled=progress,
            callback=progress_callback,
            phase="execute",
            total=len(dependencies),
        )
        reporter.info(f"starting ({len(dependencies)} jobs)")

        array_group_by_job_id: dict[str, str] = {}
        array_groups: dict[str, set[str]] = {}
        for raw_job_id, raw_payload in jobs_table.items():
            if not isinstance(raw_payload, dict):
                continue
            payload = cast(Mapping[str, Any], raw_payload)
            normalized_job_id = _normalize_job_id(raw_job_id)
            array_group = payload.get("array_group")
            if isinstance(array_group, str) and array_group:
                array_group_by_job_id[normalized_job_id] = array_group
                array_groups.setdefault(array_group, set()).add(normalized_job_id)

        # Guard against accidentally mixing old and new results when reusing a
        # partially executed plan without resume semantics.
        if not skip_completed:
            existing_outputs = sorted(
                job_id
                for job_id in dependencies
                if _job_output_exists(plan_dir, job_id)
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

        # Select the appropriate runner based on execution_mode
        plan_dir_str = str(plan_dir)
        if execution_mode == "thread":
            job_runner = _run_job_in_thread
        elif execution_mode == "process":
            job_runner = _run_job_in_process
        else:  # "subprocess"
            job_runner = _run_job_in_subprocess

        if executor is None:
            while ready:
                job_id = ready.pop(0)
                if skip_completed and _job_output_exists(plan_dir, job_id):
                    completed.add(job_id)
                    reporter.step(job_id, skipped=True)
                else:
                    try:
                        job_runner(plan_dir_str, job_id)
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
                    if skip_completed and _job_output_exists(plan_dir, job_id):
                        _mark_completed(job_id, skipped=True)
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
                                if skip_completed and _job_output_exists(plan_dir, jid):
                                    _mark_completed(jid, skipped=True)
                                else:
                                    to_run.append(jid)

                            if not to_run:
                                continue

                            array_options = dict(common_options)
                            array_options.setdefault("job_name", array_group)
                            submission = submit_array_with_options(
                                job_runner,
                                plan_dir_str,
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
                            plan_dir_str,
                            job_id,
                            sbatch_options=dict(resources_by_job_id[job_id]),
                        )
                    else:
                        future = executor.submit(job_runner, plan_dir_str, job_id)
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
                    # Fail fast: cancel all still-pending futures and stop scheduling.
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
                    )
                _submit_ready()

            if len(completed) != len(dependencies):
                raise ValueError("Cycle detected in materialized job dependencies")

        resolved: dict[Key, Any] = {}
        for encoded_key, ref in top_level.items():
            resolved[_decode_manifest_key(encoded_key)] = _resolve_ref_from_outputs(
                plan_dir, cast(Mapping[str, Any], ref)
            )

        serialize(resolved, resolved_output)
        reporter.info("done")
        return deserialize(resolved_output)

    def evaluate_in_subprocesses(
        self,
        target: Any | None = None,
        dag_path: Any | None = None,
        output_path: Any | None = None,
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
        target:
            Optional target run directory. By default, plans and outputs are
            written to ``<target>/plan`` and ``<target>/outputs``.
            May be omitted when both ``dag_path`` and ``output_path`` are
            provided explicitly.
        dag_path:
            Optional location for materialized plan.
        output_path:
            Optional explicit location for final output serialization. When
            omitted, defaults to ``<target>/outputs``.
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
        resolved_plan, resolved_output = _resolve_plan_and_output_paths(
            target=target,
            dag_path=dag_path,
            output_path=output_path,
        )
        plan_dir = self.materialize(
            resolved_plan,
            progress=progress,
            progress_callback=progress_callback,
        )
        return self.execute_materialized(
            target=None,
            dag_path=plan_dir,
            output_path=resolved_output,
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
        target: Any | None = None,
        dag_path: Any | None = None,
        output_path: Any | None = None,
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
        target:
            Optional target run directory on disk. By default, the
            materialized plan is written to ``<target>/plan`` and final
            outputs to ``<target>/outputs``. May be omitted when both
            ``dag_path`` and ``output_path`` are provided explicitly.
        dag_path:
            Optional location for the materialized execution plan.
        output_path:
            Optional explicit location for final output serialization.
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
            The deserialized :class:`~tidyrun.LazyDict` at the resolved output
            path after all nodes have been written.
        """
        return self.evaluate_in_subprocesses(
            target=target,
            dag_path=dag_path,
            output_path=output_path,
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
        DAG.  When *job_ids* is ``None`` the entire ``outputs/`` directory is
        removed.  Otherwise only the specified jobs' output files are deleted.

        Parameters
        ----------
        dag_path:
            Path to the materialized DAG directory (as passed to
            :meth:`execute_materialized`).
        job_ids:
            Optional list of job IDs whose outputs should be cleared.
            When ``None``, all outputs are removed.
        """
        import shutil

        from tidyrun.serialization.metadata import metadata_path, read_metadata

        plan_dir = _to_path(dag_path)
        outputs_dir = plan_dir / "outputs"

        if job_ids is None:
            if outputs_dir.exists():
                shutil.rmtree(outputs_dir)
            return

        for job_id in job_ids:
            base = _job_output_base(plan_dir, job_id)
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
