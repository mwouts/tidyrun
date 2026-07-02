"""Plan-directory reading, job loading, and state inspection utilities."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import pickle
from pathlib import Path
from typing import Any, cast, Literal

import toml

from tidyrun.keys import decode_key, encode_key


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def to_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def normalize_job_id(job_id: str) -> str:
    return job_id.replace('\\"', '"')


def job_definition_file(definitions_path: Path, job_id: str) -> Path:
    return definitions_path / Path(f"{job_id}.tidyrun")


def job_output_base(outputs_path: Path, job_id: str) -> Path:
    return outputs_path / Path(job_id)


def job_output_exists(outputs_path: Path, job_id: str) -> bool:
    from tidyrun.serialization.metadata import metadata_exists

    return metadata_exists(job_output_base(outputs_path, job_id))


def running_path(outputs_path: Path, job_id: str) -> Path:
    return outputs_path / Path(job_id + ".running")


def failed_path(outputs_path: Path, job_id: str) -> Path:
    return outputs_path / Path(job_id + ".failed")


# ---------------------------------------------------------------------------
# PlanPaths
# ---------------------------------------------------------------------------


@dataclass
class PlanPaths:
    """Locations for the three components of a materialized DAG plan.

    By default all three live under a single root directory; use
    :meth:`from_root` for that case.  Providing separate paths lets you
    keep large literal inputs on a different filesystem than the definition
    files, or write outputs to a scratch volume.
    """

    definitions: Path
    inputs: Path
    outputs: Path

    def __post_init__(self) -> None:
        self.definitions = to_path(self.definitions)
        self.inputs = to_path(self.inputs)
        self.outputs = to_path(self.outputs)

    @classmethod
    def from_root(cls, root: Any) -> "PlanPaths":
        """Create a PlanPaths with definitions/, inputs/, outputs/ under *root*."""
        root_path = to_path(root)
        return cls(
            definitions=root_path / "definitions",
            inputs=root_path / "inputs",
            outputs=root_path / "outputs",
        )

    def to_runner_string(self) -> str:
        """Encode as a single string for passing to job runner functions."""
        sep = ":::"
        defs = str(self.definitions)
        inps = str(self.inputs)
        outs = str(self.outputs)
        for part in (defs, inps, outs):
            if sep in part:
                raise ValueError(
                    f"Plan path {part!r} contains reserved separator {sep!r}"
                )
        return f"{defs}{sep}{inps}{sep}{outs}"

    @classmethod
    def from_runner_string(cls, s: str) -> "PlanPaths":
        """Decode a string produced by :meth:`to_runner_string`."""
        sep = ":::"
        if sep in s:
            parts = s.split(sep, 2)
            return cls(Path(parts[0]), Path(parts[1]), Path(parts[2]))
        return cls.from_root(s)


# ---------------------------------------------------------------------------
# Definition-file discovery
# ---------------------------------------------------------------------------


def _find_definition_file(definitions_dir: Path, job_id: str) -> Path | None:
    """Find the definition file for a job, trying direct file then shorter prefixes.

    As a final fallback, checks for the ROOT_ARRAY_GROUP definition file used when
    a :class:`~tidyrun.ParametrizedJob` is compiled standalone via its own
    :meth:`~tidyrun.ParametrizedJob.materialize`.
    """
    candidate = job_definition_file(definitions_dir, job_id)
    if candidate.is_file():
        return candidate
    parts = job_id.split("/")
    for n in range(len(parts) - 1, 0, -1):
        group_candidate = job_definition_file(definitions_dir, "/".join(parts[:n]))
        if group_candidate.is_file():
            return group_candidate
    return None


# ---------------------------------------------------------------------------
# Callable helpers
# ---------------------------------------------------------------------------


def _callable_from_import_spec(module: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


# ---------------------------------------------------------------------------
# Public plan-reading API
# ---------------------------------------------------------------------------


def load_job_definition(dag_path: Any, job_id: str) -> dict[str, Any]:
    """Load a materialized job definition from disk."""
    plan_dir = to_path(dag_path)
    normalized_job_id = normalize_job_id(job_id)

    definitions_dir = plan_dir / "definitions"
    definition_file = _find_definition_file(definitions_dir, normalized_job_id)

    if definition_file is None:
        raise ValueError(f"Missing job definition file for job {normalized_job_id!r}")

    definition = cast(
        dict[str, Any],
        toml.loads(definition_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )

    # Derive array_group from the definition file path (not stored in the file).
    if "parameter_names" in definition:
        rel = definition_file.relative_to(definitions_dir).with_suffix("")
        definition["array_group"] = rel.as_posix()

    definition["job_id"] = normalized_job_id
    definition["_requested_job_id"] = normalized_job_id
    return definition


def load_callable(job_definition: Mapping[str, Any]) -> Any:
    """Load the callable described by a job definition."""
    module = job_definition.get("callable_module")
    qualname = job_definition.get("callable_qualname")
    if isinstance(module, str) and isinstance(qualname, str):
        try:
            return _callable_from_import_spec(module, qualname)
        except Exception:
            pass

    callable_data = job_definition.get("callable_data")
    if isinstance(callable_data, dict):
        callable_data_dict = cast(dict[str, Any], callable_data)
        encoding = callable_data_dict.get("encoding")
        data = callable_data_dict.get("data")
        if encoding == "pickle-base64" and isinstance(data, str):
            raw = base64.b64decode(data)
            try:
                import cloudpickle as _cpickle  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

                return _cpickle.loads(raw)
            except ImportError:
                return pickle.loads(raw)

    raise ValueError(
        "Cannot load callable: no valid callable metadata in job definition"
    )


def _resolve_parameter_value(
    arg_name: str,
    job_id: str,
    definition: Mapping[str, Any],
) -> Any:
    """Decode a parameter value from the job_id path segments."""
    array_group = definition.get("array_group", "")
    parameter_names = list(definition.get("parameter_names", []))
    prefix = f"{array_group}/" if array_group else ""
    if not job_id.startswith(prefix):
        raise ValueError(
            f"job_id {job_id!r} does not start with array_group prefix {prefix!r}"
        )
    suffix = job_id[len(prefix) :]
    key_segments = suffix.split("/")
    if arg_name not in parameter_names:
        raise ValueError(
            f"Parameter {arg_name!r} not in parameter_names {parameter_names!r}"
        )
    param_idx = parameter_names.index(arg_name)
    if param_idx >= len(key_segments):
        raise ValueError(
            f"job_id {job_id!r} has too few path segments for parameter index {param_idx}"
        )
    return decode_key(key_segments[param_idx])


def _resolve_arg(
    plan_dir: Path,
    spec: Mapping[str, Any],
    *,
    arg_name: str | None = None,
    requested_job_id: str | None = None,
) -> Any:
    from tidyrun.serialization.api import deserialize

    kind = spec.get("kind")
    if kind == "literal":
        raw_path = spec.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"Invalid literal arg spec: {spec!r}")
        p = Path(raw_path)
        return deserialize(p if p.is_absolute() else plan_dir / p)

    if kind == "dependency":
        if arg_name is None or requested_job_id is None:
            raise ValueError(
                "arg_name and requested_job_id are required to resolve a dependency arg"
            )
        dep_path = plan_dir / "inputs" / requested_job_id / arg_name
        if dep_path.is_symlink():
            return deserialize(dep_path.resolve())
        # Non-local (S3) or missing symlink: use sidecar written by the compiler
        sidecar = plan_dir / "inputs" / requested_job_id / f"{arg_name}.tidyrun"
        dep_output_id = sidecar.read_text(encoding="utf-8").strip()
        return deserialize(plan_dir / "outputs" / dep_output_id)

    raise ValueError(f"Unknown arg kind: {kind!r}")


def load_job_inputs(
    job_definition: Mapping[str, Any],
    dag_path: Any,
) -> dict[str, Any]:
    """Load all job inputs by deserializing materialized argument specs.

    Parameters
    ----------
    job_definition:
        Loaded job definition (from :func:`load_job_definition`).
    dag_path:
        Root of the materialised plan (contains ``definitions/``).
    """
    kwargs_table = job_definition.get("args")
    if not isinstance(kwargs_table, dict):
        raise ValueError("Invalid args section in job definition")
    typed_kwargs_table = cast(dict[str, Any], kwargs_table)
    plan_dir = to_path(dag_path)
    requested_job_id = job_definition.get("_requested_job_id")
    if requested_job_id is not None and not isinstance(requested_job_id, str):
        raise ValueError("Invalid _requested_job_id metadata in job definition")

    resolved_inputs: dict[str, Any] = {}
    for name, raw_spec in typed_kwargs_table.items():
        spec = cast(Mapping[str, Any], raw_spec)
        if spec.get("kind") == "parameter":
            if requested_job_id is None:
                raise ValueError(
                    f"Cannot resolve parameter arg {name!r}: missing job_id"
                )
            resolved_inputs[name] = _resolve_parameter_value(
                name, requested_job_id, job_definition
            )
        else:
            resolved_inputs[name] = _resolve_arg(
                plan_dir,
                spec,
                arg_name=name,
                requested_job_id=requested_job_id,
            )
    return resolved_inputs


def load_inputs_and_callable(dag_path: Any, job_id: str) -> tuple[Any, dict[str, Any]]:
    """Return (callable, inputs) for a materialised job, ready to call."""
    definition = load_job_definition(dag_path, job_id)
    return load_callable(definition), load_job_inputs(definition, dag_path)


def rerun_snippet(dag_path: Any, job_id: str) -> str:
    """Return a Python snippet that loads and re-runs a single job."""
    plan_str = str(to_path(dag_path))
    return "\n".join(
        [
            "from pathlib import Path",
            "from tidyrun import load_inputs_and_callable",
            "",
            f"func, inputs = load_inputs_and_callable(Path({plan_str!r}), {job_id!r})",
            "outputs = func(**inputs)",
        ]
    )


@dataclass
class PlanGraph:
    """Dependency graph of a materialized plan, derived from ``definitions/``.

    One entry per runnable job.  A parametrized definition file (an "array
    group") expands to one job per parameter-value tuple; those jobs are also
    indexed by their group so executors can submit them as job arrays.
    """

    #: job_id -> ids of the jobs it depends on
    dependencies: dict[str, set[str]]
    #: job_id -> array group id, for jobs that belong to a parametrized group
    array_group_by_job_id: dict[str, str]
    #: array group id -> all job_ids in the group
    array_groups: dict[str, set[str]]


def read_plan_graph(definitions_dir: Path) -> PlanGraph:
    """Scan ``definitions/`` and build the job dependency graph."""
    graph = PlanGraph(dependencies={}, array_group_by_job_id={}, array_groups={})
    if not definitions_dir.is_dir():
        return graph
    for def_file in sorted(definitions_dir.rglob("*.tidyrun")):
        definition = cast(
            dict[str, Any],
            toml.loads(def_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
        )
        deps = {
            normalize_job_id(dep)
            for dep in cast(list[str], definition.get("dependencies", []))
        }
        # The job (or array group) id is the definition file path itself.
        group_id = def_file.relative_to(definitions_dir).with_suffix("").as_posix()
        parameter_names = cast(list[str], definition.get("parameter_names", []))
        if not parameter_names:
            graph.dependencies[normalize_job_id(group_id)] = deps
            continue

        args = cast(dict[str, Any], definition.get("args", {}))
        per_param_values = [
            list(cast(Mapping[str, Any], args.get(name, {})).get("values", []))
            for name in parameter_names
        ]
        for row in zip(*per_param_values):
            job_id = "/".join([group_id, *(encode_key(value) for value in row)])
            graph.dependencies[job_id] = set(deps)
            graph.array_group_by_job_id[job_id] = group_id
            graph.array_groups.setdefault(group_id, set()).add(job_id)
    return graph


def enumerate_job_ids_from_definitions(
    definitions_dir: Path,
) -> dict[str, str | None]:
    """Scan definitions/ and return {job_id: array_group or None}."""
    graph = read_plan_graph(definitions_dir)
    return {
        job_id: graph.array_group_by_job_id.get(job_id) for job_id in graph.dependencies
    }


def get_job_states(
    dag_path: Any,
    output_path: Any | None = None,
) -> dict[str, Literal["pending", "running", "failed", "succeeded"]]:
    """Return the execution state of every job in a materialized plan.

    States (in priority order):
    - ``"succeeded"`` — output metadata file exists.
    - ``"failed"`` — ``.failed`` sentinel file exists.
    - ``"running"`` — ``.running`` sentinel file exists.
    - ``"pending"`` — none of the above.

    Parameters
    ----------
    dag_path:
        Root of the materialised plan (contains ``definitions/``).
    output_path:
        Location where job outputs are written.  Defaults to
        ``dag_path/outputs`` for backward compatibility, but should match the
        value passed to :meth:`~tidyrun.DAG.execute_materialized`.
    """
    plan_dir = to_path(dag_path)
    outputs_path = (
        to_path(output_path) if output_path is not None else plan_dir / "outputs"
    )
    definitions_dir = plan_dir / "definitions"

    job_ids = enumerate_job_ids_from_definitions(definitions_dir)
    states: dict[str, Literal["pending", "running", "failed", "succeeded"]] = {}
    for job_id in job_ids:
        if job_output_exists(outputs_path, job_id):
            states[job_id] = "succeeded"
        elif failed_path(outputs_path, job_id).exists():
            states[job_id] = "failed"
        elif running_path(outputs_path, job_id).exists():
            states[job_id] = "running"
        else:
            states[job_id] = "pending"
    return states
