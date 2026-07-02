"""The DAG node model and its compilation to materialized plans.

This module defines the deferred-computation containers (:class:`DAG` and
:class:`ParametrizedJob`) and the plan compiler that writes them to disk.
Running a materialized plan lives in :mod:`tidyrun.execute`; reading one back
lives in :mod:`tidyrun.plan`.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Executor
from functools import partial
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast, Union
from urllib.parse import urlparse

from cloudpathlib import AnyPath, CloudPath
import toml

# Re-exported for backward compatibility: these names historically lived in
# tidyrun.dag and are part of its de-facto public surface.
from tidyrun.execute import (
    DAGExecutionError as DAGExecutionError,
    ExecutionMode,
    batch_entrypoint as batch_entrypoint,
    execute_graph,
    execute_plan as execute_plan,
    run_materialized_job as run_materialized_job,
    write_group_metadata,
    write_root_metadata,
)
from tidyrun.job import Job, validate_callable_bindings
from tidyrun.keys import Key, encode_key
from tidyrun.plan import (
    PlanPaths,
    job_definition_file,
    job_output_base as job_output_base,
    job_output_exists as job_output_exists,
    load_callable as load_callable,
    load_job_definition as load_job_definition,
    load_job_inputs as load_job_inputs,
    read_plan_graph,
    to_path,
)
from tidyrun.progress import ProgressCallback, ProgressReporter as _ProgressReporter

Node = Union[Job, "DAG"]


# ---------------------------------------------------------------------------
# S3 upload helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Node-tree helpers
# ---------------------------------------------------------------------------


def _count_unique_jobs(node: Node, seen: set[int]) -> int:
    """Count the runnable jobs under *node*, deduplicated by object identity."""
    node_id = id(node)
    if node_id in seen:
        return 0
    seen.add(node_id)

    if isinstance(node, ParametrizedJob):
        # One job per parameter-value tuple; tuples are validated unique.
        return len(node.parameter_values)
    if isinstance(node, Job):
        return 1
    return sum(_count_unique_jobs(subnode, seen) for subnode in node.values())


def _job_id_from_path(path: tuple[Key, ...]) -> str:
    if not path:
        raise ValueError("Cannot derive job_id from empty path")
    return "/".join(_encode_key_checked(key) for key in path)


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


def _pickled_callable_data(func: Callable[..., Any]) -> dict[str, str]:
    try:
        import cloudpickle  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

        pickle_bytes: bytes = cloudpickle.dumps(func)  # pyright: ignore[reportUnknownMemberType]
    except ImportError:
        pickle_bytes = pickle.dumps(func)
    return {
        "encoding": "pickle-base64",
        "data": base64.b64encode(pickle_bytes).decode("ascii"),
    }


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


# ---------------------------------------------------------------------------
# Plan compilation
# ---------------------------------------------------------------------------


class _PlanCompiler:
    """Compile a DAG node tree into definition and input files on disk.

    Every :class:`Job` leaf becomes a TOML definition file plus serialized
    literal inputs; every :class:`ParametrizedJob` becomes a single "array
    group" definition shared by all its instances.  Dependencies between jobs
    are recorded as symlinks (or sidecar files on non-local filesystems) from
    the dependent job's inputs to the dependency's output location.
    """

    def __init__(self, plan_paths: PlanPaths, reporter: _ProgressReporter) -> None:
        self._plan_paths = plan_paths
        self._reporter = reporter
        # id(node) -> (node, ref). Keeping the node in the value pins ephemeral
        # ParametrizedJob children in memory so their ids cannot be reused.
        self._refs: dict[int, tuple[Node, dict[str, Any]]] = {}
        # id(top-level node) -> canonical key path, for dependency resolution.
        self._member_paths: dict[int, tuple[Key, ...]] = {}
        # (array_group, arg_name) -> stored path of a shared literal input.
        self._shared_literal_paths: dict[tuple[str, str], str] = {}
        # (array_group, arg_name) -> parameter values, one per group instance.
        self._parameter_values: dict[tuple[str, str], list[Any]] = {}
        # array_group -> ordered parameter names.
        self._group_parameter_names: dict[str, list[str]] = {}
        # id(ref) -> job ids reachable from that ref (dependency collection).
        self._ref_job_ids: dict[int, frozenset[str]] = {}
        self._written_definitions: set[Path] = set()

    def compile(self, nodes: Mapping[Key, Node], prefix: str | None) -> None:
        self._plan_paths.definitions.mkdir(parents=True, exist_ok=True)
        self._plan_paths.inputs.mkdir(parents=True, exist_ok=True)
        self._plan_paths.outputs.mkdir(parents=True, exist_ok=True)

        prefix_tuple: tuple[Key, ...] = (prefix,) if prefix else ()
        # Register all members first so forward dependencies resolve no matter
        # the insertion order of the DAG.
        for key, node in nodes.items():
            self._member_paths[id(node)] = (*prefix_tuple, key)
        for key, node in nodes.items():
            self._compile_node(node, (*prefix_tuple, key), None, None)
        self._patch_parameter_values()

    # -- node compilation ----------------------------------------------------

    def _compile_node(
        self,
        node: Node,
        path: tuple[Key, ...],
        array_group: str | None,
        group_parameter_names: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        existing = self._refs.get(id(node))
        if existing is not None and existing[0] is node:
            return existing[1]

        if isinstance(node, ParametrizedJob):
            return self._compile_parametrized_job(
                node, path, array_group, group_parameter_names
            )
        if isinstance(node, Job):
            return self._compile_job(node, path, array_group, group_parameter_names)

        # Nested DAG: a plain group of sub-nodes.
        entries = {
            _encode_key_checked(key): self._compile_node(
                subnode, (*path, key), array_group, group_parameter_names
            )
            for key, subnode in node.items()
        }
        ref: dict[str, Any] = {"kind": "group", "entries": entries}
        self._refs[id(node)] = (node, ref)
        return ref

    def _compile_parametrized_job(
        self,
        node: ParametrizedJob,
        path: tuple[Key, ...],
        array_group: str | None,
        group_parameter_names: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        # The outermost ParametrizedJob defines the array group; nested levels
        # (remaining parameters) reuse it.
        if array_group is None:
            array_group = _job_id_from_path(path)
            group_parameter_names = tuple(node.parameter_names)
        assert group_parameter_names is not None
        self._group_parameter_names[array_group] = list(group_parameter_names)

        entries: dict[str, Any] = {}
        for key in node:
            entries[_encode_key_checked(key)] = self._compile_node(
                node[key], (*path, key), array_group, group_parameter_names
            )
        ref: dict[str, Any] = {"kind": "group", "entries": entries}
        self._refs[id(node)] = (node, ref)
        return ref

    def _compile_job(
        self,
        node: Job,
        path: tuple[Key, ...],
        array_group: str | None,
        group_parameter_names: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        job_id = _job_id_from_path(path)
        ref: dict[str, Any] = {"kind": "job", "job_id": job_id}
        self._refs[id(node)] = (node, ref)

        args_spec: dict[str, Any] = {}
        dependencies: set[str] = set()
        for arg_name, arg_value in node.kwargs.items():
            spec = self._compile_operand(
                arg_value, job_id, arg_name, array_group, group_parameter_names
            )
            args_spec[arg_name] = spec
            if spec["kind"] == "dependency":
                dependencies.update(
                    self._collect_job_ids(cast(Mapping[str, Any], spec["ref"]))
                )

        self._write_definition(job_id, node.func, args_spec, dependencies, array_group)
        self._reporter.step(job_id)
        return ref

    # -- operand compilation ---------------------------------------------------

    def _compile_operand(
        self,
        value: Any,
        owner_job_id: str,
        arg_name: str,
        array_group: str | None,
        group_parameter_names: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        if isinstance(value, (Job, DAG)):
            member_path = self._member_paths.get(id(value))
            if member_path is None:
                raise ValueError(
                    f"Argument {arg_name!r} of job {owner_job_id!r} depends on a "
                    "Job or DAG that is not a member of this DAG. "
                    "Register it as a DAG member before using it as a dependency."
                )
            ref = self._compile_node(value, member_path, None, None)

            if ref.get("kind") == "job":
                self._write_dep_link(owner_job_id, arg_name, cast(str, ref["job_id"]))
            elif isinstance(value, ParametrizedJob):
                # Whole-group dependency: link to the group's output folder.
                self._write_dep_link(
                    owner_job_id, arg_name, _job_id_from_path(member_path)
                )

            return {"kind": "dependency", "ref": ref}

        if array_group is not None and group_parameter_names is not None:
            shared_key = (array_group, arg_name)

            if arg_name in group_parameter_names:
                # Parameter arg: accumulate the raw value; the full list is
                # patched into the definition file at the end of compilation.
                self._parameter_values.setdefault(shared_key, []).append(value)
                return {"kind": "parameter"}

            # Shared non-parameter literal (same value for all instances).
            shared_path = self._shared_literal_paths.get(shared_key)
            if shared_path is None:
                shared_path = self._serialize_literal(
                    Path(array_group), arg_name, value
                )
                self._shared_literal_paths[shared_key] = shared_path
            return {"kind": "literal", "path": shared_path}

        return {
            "kind": "literal",
            "path": self._serialize_literal(Path(owner_job_id), arg_name, value),
        }

    def _serialize_literal(self, owner_dir: Path, arg_name: str, value: Any) -> str:
        from tidyrun.serialization.api import serialize

        input_base = self._plan_paths.inputs / owner_dir / arg_name
        input_base.parent.mkdir(parents=True, exist_ok=True)
        serialize(value, input_base)
        return _literal_path_str(input_base, self._plan_paths)

    def _write_dep_link(
        self, owner_job_id: str, arg_name: str, dep_output_id: str
    ) -> None:
        """Record a dependency: symlink on local FS, sidecar on non-local (S3)."""
        input_dir = self._plan_paths.inputs / owner_job_id
        input_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = input_dir / arg_name
        target = self._plan_paths.outputs / dep_output_id
        try:
            relative_target = Path(os.path.relpath(target, symlink_path.parent))
            if symlink_path.is_symlink():
                symlink_path.unlink()
            if not symlink_path.exists():
                symlink_path.symlink_to(relative_target)
        except OSError:
            # Filesystems without symlink support fall back to a sidecar file.
            sidecar_path = input_dir / f"{arg_name}.tidyrun"
            sidecar_path.write_text(dep_output_id, encoding="utf-8")

    def _collect_job_ids(self, ref: Mapping[str, Any]) -> frozenset[str]:
        cached = self._ref_job_ids.get(id(ref))
        if cached is not None:
            return cached

        kind = ref.get("kind")
        if kind == "job":
            job_id = ref.get("job_id")
            if not isinstance(job_id, str):
                raise ValueError(f"Invalid job reference: {ref!r}")
            resolved = frozenset({job_id})
        elif kind == "group":
            raw_entries = ref.get("entries")
            if not isinstance(raw_entries, dict):
                raise ValueError(f"Invalid group reference: {ref!r}")
            entries = cast(Mapping[str, Any], raw_entries)
            resolved = frozenset(
                job_id
                for entry in entries.values()
                for job_id in self._collect_job_ids(cast(Mapping[str, Any], entry))
            )
        else:
            raise ValueError(f"Unknown reference kind: {kind!r}")

        self._ref_job_ids[id(ref)] = resolved
        return resolved

    # -- definition files ------------------------------------------------------

    def _write_definition(
        self,
        job_id: str,
        func: Callable[..., Any],
        args_spec: dict[str, Any],
        dependencies: set[str],
        array_group: str | None,
    ) -> None:
        definition_group = array_group if array_group is not None else job_id
        definition_file = job_definition_file(
            self._plan_paths.definitions, definition_group
        )
        if definition_file in self._written_definitions:
            return

        definition: dict[str, Any] = {
            "kind": "job_definition",
            "schema_version": 1,
            "dependencies": sorted(dependencies),
            "args": args_spec,
        }
        if array_group is not None:
            # array_group itself is derived from the file path at load time.
            definition["parameter_names"] = self._group_parameter_names.get(
                array_group, []
            )
        import_spec = _callable_import_spec(func)
        if import_spec is not None:
            definition["callable_module"] = import_spec[0]
            definition["callable_qualname"] = import_spec[1]
        else:
            definition["callable_data"] = _pickled_callable_data(func)

        definition_file.parent.mkdir(parents=True, exist_ok=True)
        definition_file.write_text(
            toml.dumps(definition),  # pyright: ignore[reportUnknownMemberType]
            encoding="utf-8",
        )
        self._written_definitions.add(definition_file)

    def _patch_parameter_values(self) -> None:
        """Write the accumulated parameter values into their definition files."""
        for (array_group, arg_name), values in self._parameter_values.items():
            def_file = job_definition_file(self._plan_paths.definitions, array_group)
            if def_file not in self._written_definitions:
                continue
            definition = cast(
                dict[str, Any],
                toml.loads(def_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
            )
            args = cast(dict[str, Any], definition.setdefault("args", {}))
            args[arg_name] = {"kind": "parameter", "values": values}
            def_file.write_text(
                toml.dumps(definition),  # pyright: ignore[reportUnknownMemberType]
                encoding="utf-8",
            )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------


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

        if isinstance(dag_path, PlanPaths):
            _PlanCompiler(dag_path, reporter).compile(self._nodes, prefix)
            reporter.info("done")
            return dag_path.definitions.parent

        if is_s3_location(dag_path):
            # Compile locally, then upload the whole plan tree.
            with TemporaryDirectory() as temp_root:
                plan_dir = Path(temp_root) / _s3_leaf_name(dag_path)
                _PlanCompiler(PlanPaths.from_root(plan_dir), reporter).compile(
                    self._nodes, prefix
                )
                upload_local_tree_to_s3(plan_dir.parent, dag_path)
            reporter.info("done")
            return AnyPath(dag_path)

        plan_dir = to_path(dag_path)
        _PlanCompiler(PlanPaths.from_root(plan_dir), reporter).compile(
            self._nodes, prefix
        )
        reporter.info("done")
        return plan_dir

    def execute_materialized(
        self,
        dag_path: Any,
        executor: Executor | None = None,
        max_workers: int | None = None,
        job_resources: Mapping[Key, Mapping[str, str | int]] | None = None,
        execution_mode: ExecutionMode = "subprocess",
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
        skip_running:
            When ``True``, skip jobs whose ``.running`` sentinel exists.
        progress:
            When ``True``, emit progress logs while executing jobs.
        progress_callback:
            Optional callback used for progress messages.
        """
        from tidyrun.serialization.api import deserialize
        from tidyrun.serialization.metadata import metadata_exists

        if executor is not None and max_workers is not None:
            raise ValueError("Pass either executor or max_workers, not both")

        plan_dir = to_path(dag_path)
        plan_paths = PlanPaths.from_root(plan_dir)
        plan_paths.outputs.mkdir(parents=True, exist_ok=True)

        if not plan_paths.definitions.is_dir():
            raise ValueError(
                f"No materialized plan found at {plan_dir}. Run materialize() first."
            )
        graph = read_plan_graph(plan_paths.definitions)

        # Synthetic aggregator jobs for every group node in the DAG tree. They
        # run inline (no subprocess) after their children complete and write
        # the dict-folder .tidyrun metadata for intermediate output folders.
        aggregator_deps: dict[str, list[str]] = {}
        root_children = [
            _build_aggregator_deps(node, _encode_key_checked(key), aggregator_deps)
            for key, node in self._nodes.items()
        ]
        dependencies = dict(graph.dependencies)
        for agg_id, child_ids in aggregator_deps.items():
            dependencies[agg_id] = set(child_ids)
        inline_runners: dict[str, Callable[[], None]] = {
            agg_id: partial(write_group_metadata, agg_id, child_ids, plan_paths.outputs)
            for agg_id, child_ids in aggregator_deps.items()
        }

        resources_by_key: Mapping[Key, Mapping[str, str | int]] = (
            {} if job_resources is None else job_resources
        )
        unknown_keys = [key for key in resources_by_key if key not in self._nodes]
        if unknown_keys:
            raise ValueError(f"job_resources contains unknown DAG keys: {unknown_keys}")
        # Per-job submission options currently apply to top-level Job nodes only.
        resources_by_job_id = {
            _encode_key_checked(key): dict(options)
            for key, options in resources_by_key.items()
            if isinstance(self._nodes[key], Job)
        }

        reporter = _ProgressReporter(
            enabled=progress,
            callback=progress_callback,
            phase="execute",
            total=len(graph.dependencies),
        )
        reporter.info(f"starting ({len(graph.dependencies)} jobs)")

        # Guard against accidentally mixing old and new results when reusing a
        # partially executed plan without resume semantics.
        if not skip_completed:
            existing_outputs = sorted(
                job_id
                for job_id in graph.dependencies
                if job_output_exists(plan_paths.outputs, job_id)
            )
            if existing_outputs:
                raise ValueError(
                    "Materialized plan already has existing job outputs. "
                    "Use skip_completed=True to resume, or clear outputs before "
                    "re-running. Existing job ids: "
                    f"{existing_outputs}"
                )

        execute_graph(
            dependencies,
            plan_paths,
            plan_dir,
            executor=executor,
            max_workers=max_workers,
            execution_mode=execution_mode,
            skip_completed=skip_completed,
            skip_running=skip_running,
            reporter=reporter,
            inline_runners=inline_runners,
            array_groups=graph.array_groups,
            array_group_by_job_id=graph.array_group_by_job_id,
            resources_by_job_id=resources_by_job_id,
        )

        # Write the root dict-folder .tidyrun for the outputs directory so the
        # on-disk layout is identical to serialize(dict, outputs_path).
        if not (skip_completed and metadata_exists(plan_paths.outputs)):
            write_root_metadata(plan_paths.outputs, root_children)

        reporter.info("done")
        return deserialize(plan_paths.outputs)

    def evaluate(
        self,
        dag_path: Any,
        executor: Executor | None = None,
        max_workers: int | None = None,
        job_resources: Mapping[Key, Mapping[str, str | int]] | None = None,
        execution_mode: ExecutionMode = "subprocess",
        skip_completed: bool = False,
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Evaluate this DAG to disk (materialize, then execute).

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

    def evaluate_in_subprocesses(
        self,
        dag_path: Any,
        executor: Executor | None = None,
        max_workers: int | None = None,
        job_resources: Mapping[Key, Mapping[str, str | int]] | None = None,
        execution_mode: ExecutionMode = "subprocess",
        skip_completed: bool = False,
        progress: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        """Alias of :meth:`evaluate`, kept for backward compatibility."""
        return self.evaluate(
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
                suffix = read_metadata(base).get("suffix", "")
            except Exception:
                # A corrupt metadata file should not prevent clearing the job;
                # fall back to deleting the suffix-less payload path.
                suffix = ""
            payload = Path(str(base) + suffix) if suffix else base
            if payload.exists():
                if payload.is_dir():
                    shutil.rmtree(payload)
                else:
                    payload.unlink()
            meta.unlink()


# ---------------------------------------------------------------------------
# ParametrizedJob
# ---------------------------------------------------------------------------


class ParametrizedJob(DAG):
    """A deferred computation indexed by parameter keys.

    Parameters are declared through ``parameter_names`` and populated through
    ``parameter_values``. Accessing a key fixes the first parameter and returns
    either a :class:`Job` (when one parameter remains) or another
    :class:`ParametrizedJob` (when more parameters remain).

    Being a subclass of :class:`DAG`, a ``ParametrizedJob`` inherits all
    execution methods (``materialize``, ``execute_materialized``, ``evaluate``,
    ``clear_outputs``) with identical semantics: the top-level keys are the
    first-level parameter values and no extra wrapping level is added.
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
        matching = [values for values in self.parameter_values if values[0] == key]
        if not matching:
            raise KeyError(key)

        parameter_name = self.parameter_names[0]
        bound_kwargs = dict(self.kwargs)
        bound_kwargs[parameter_name] = key

        if len(self.parameter_names) == 1:
            return Job(func=self.func, kwargs=bound_kwargs)
        return ParametrizedJob(
            func=self.func,
            parameter_names=self.parameter_names[1:],
            parameter_values=[values[1:] for values in matching],
            kwargs=bound_kwargs,
        )

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
