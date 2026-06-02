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

from tidyrun.keys import decode_key, encode_key, Key


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def to_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def normalize_job_id(job_id: str) -> str:
    return job_id.replace('\\"', '"')


def decode_manifest_key(encoded_key: str) -> Key:
    try:
        return decode_key(encoded_key)
    except ValueError:
        return decode_key(encoded_key.replace('\\"', '"'))


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

    if definition_file is None or not definition_file.is_file():
        raise ValueError(f"Missing job definition file for job {normalized_job_id!r}")

    definition = cast(
        dict[str, Any],
        toml.loads(definition_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )

    # Derive array_group from the definition file path (not stored in the file).
    if "parameter_names" in definition:
        rel = definition_file.relative_to(definitions_dir).with_suffix("")
        definition["array_group"] = rel.as_posix()

    def _hydrate_requested_job_id(value: Any) -> Any:
        if isinstance(value, dict):
            value_dict = cast(dict[str, Any], value)
            hydrated: dict[str, Any] = {
                key: _hydrate_requested_job_id(item) for key, item in value_dict.items()
            }
            if hydrated.get("job_id_from_request") is True:
                hydrated.pop("job_id_from_request", None)
                hydrated["job_id"] = normalized_job_id
            return hydrated
        if isinstance(value, list):
            return [_hydrate_requested_job_id(item) for item in cast(list[Any], value)]
        return value

    args_payload = definition.get("args")
    if isinstance(args_payload, dict):
        definition["args"] = _hydrate_requested_job_id(args_payload)

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


def resolve_ref_from_outputs(outputs_path: Path, ref: Mapping[str, Any]) -> Any:
    from tidyrun.serialization.api import deserialize

    kind = ref.get("kind")
    if kind == "job":
        job_id = ref.get("job_id")
        if not isinstance(job_id, str):
            raise ValueError(f"Invalid job reference: {ref!r}")
        return deserialize(job_output_base(outputs_path, normalize_job_id(job_id)))

    if kind == "group":
        raw_entries = ref.get("entries")
        if not isinstance(raw_entries, dict):
            raise ValueError(f"Invalid group reference: {ref!r}")
        entries = cast(Mapping[str, Any], raw_entries)
        return {
            decode_manifest_key(encoded_key): resolve_ref_from_outputs(
                outputs_path, cast(Mapping[str, Any], entry_ref)
            )
            for encoded_key, entry_ref in entries.items()
        }

    raise ValueError(f"Unknown reference kind: {kind!r}")


def _resolve_arg(
    plan_dir: Path,
    spec: Mapping[str, Any],
    *,
    requested_job_id: str | None = None,
    outputs_path: Path | None = None,
) -> Any:
    from tidyrun.serialization.api import deserialize

    kind = spec.get("kind")
    if kind == "literal":
        raw_path = spec.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"Invalid literal arg spec: {spec!r}")
        p = Path(raw_path)
        full_path = p if p.is_absolute() else plan_dir / p
        value = deserialize(full_path)
        literal_job_id = spec.get("job_id")
        if spec.get("job_id_from_request") is True:
            if requested_job_id is None:
                raise ValueError(
                    "Missing requested job id for grouped literal selector"
                )
            literal_job_id = requested_job_id
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
        dep_outputs = outputs_path if outputs_path is not None else plan_dir / "outputs"
        return resolve_ref_from_outputs(dep_outputs, cast(Mapping[str, Any], ref))

    raise ValueError(f"Unknown arg kind: {kind!r}")


def load_job_inputs(
    job_definition: Mapping[str, Any],
    dag_path: Any,
    outputs_path: Any | None = None,
) -> dict[str, Any]:
    """Load all job inputs by deserializing materialized argument specs.

    Parameters
    ----------
    job_definition:
        Loaded job definition (from :func:`load_job_definition`).
    dag_path:
        Root of the materialised plan (contains ``definitions/``).
    outputs_path:
        Directory where job outputs are stored.  Defaults to
        ``dag_path/outputs``; pass the same value that was supplied to
        :meth:`~tidyrun.DAG.execute_materialized` when a custom path was used.
    """
    kwargs_table = job_definition.get("args")
    if not isinstance(kwargs_table, dict):
        raise ValueError("Invalid args section in job definition")
    typed_kwargs_table = cast(dict[str, Any], kwargs_table)
    plan_dir = to_path(dag_path)
    resolved_out = (
        to_path(outputs_path) if outputs_path is not None else plan_dir / "outputs"
    )
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
                requested_job_id=requested_job_id,
                outputs_path=resolved_out,
            )
    return resolved_inputs


def rerun_snippet(dag_path: Any, job_id: str) -> str:
    """Return a Python snippet that loads and re-runs a single job.

    The snippet is valid for both plain jobs and parametrised job instances.
    For a parametrised instance (e.g. ``job_id="sum_7/3"``), ``load_job_inputs``
    resolves the parameter values from the job_id path segments automatically.

    To re-run the job and also save the output (as the DAG executor does), append::

        from tidyrun import serialize
        serialize(outputs, plan_dir / "outputs" / job_id)
    """
    plan_str = str(to_path(dag_path))
    return "\n".join(
        [
            "from pathlib import Path",
            "from tidyrun import load_callable, load_job_definition, load_job_inputs",
            "",
            f"plan_dir = Path({plan_str!r})",
            f"job_id = {job_id!r}",
            "definition = load_job_definition(plan_dir, job_id)",
            "inputs = load_job_inputs(definition, plan_dir)  # resolves parameters for parametrised jobs",
            "func = load_callable(definition)",
            "outputs = func(**inputs)",
        ]
    )


def enumerate_job_ids_from_definitions(
    definitions_dir: Path,
) -> dict[str, str | None]:
    """Scan definitions/ and return {job_id: array_group or None}."""
    result: dict[str, str | None] = {}
    if not definitions_dir.is_dir():
        return result
    for def_file in sorted(definitions_dir.rglob("*.tidyrun")):
        definition = cast(
            dict[str, Any],
            toml.loads(def_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
        )
        parameter_names = list(definition.get("parameter_names", []))
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
                    encoded_values = "/".join(
                        encode_key(per_param_values[p][i])
                        for p in range(len(parameter_names))
                    )
                    job_id = array_group + "/" + encoded_values
                    result[job_id] = array_group
            else:
                result[array_group] = array_group
        else:
            rel = def_file.relative_to(definitions_dir).with_suffix("")
            job_id = normalize_job_id(rel.as_posix())
            result[job_id] = None
    return result


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
