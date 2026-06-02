from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

# Operand is any value accepted as a Job argument:
# a plain Python value, a LazyDict, a Job, or a DAG.
Operand = Any


def validate_callable_bindings(
    *,
    func: Callable[..., Any],
    kwargs: Mapping[str, Operand],
    parameter_names: tuple[str, ...],
) -> None:
    """Validate that provided names match callable requirements exactly."""
    provided_names = set(kwargs).union(parameter_names)

    overlap = set(kwargs).intersection(parameter_names)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"kwargs already contains parametrized names: {names}")

    signature = inspect.signature(func)
    parameters = signature.parameters

    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )

    accepted_names = {
        name
        for name, param in parameters.items()
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    required_names = {
        name
        for name, param in parameters.items()
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.default is inspect.Parameter.empty
    }

    positional_only_required = [
        name
        for name, param in parameters.items()
        if param.kind == inspect.Parameter.POSITIONAL_ONLY
        and param.default is inspect.Parameter.empty
    ]
    if positional_only_required:
        names = ", ".join(positional_only_required)
        raise ValueError(
            "Callables with required positional-only parameters are not supported: "
            f"{names}"
        )

    missing = sorted(required_names - provided_names)
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required callable arguments: {names}")

    if not has_var_keyword:
        unknown = sorted(provided_names - accepted_names)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"Unknown callable arguments: {names}")


@dataclass
class Job:
    """A deferred computation: a callable with named arguments.

    Each argument value is an Operand, which may be a plain Python value,
    a LazyDict (existing on-disk outputs), another Job, or a DAG.
    Arguments are resolved recursively before the function is called.

    Example::

        def add(x: int, y: int) -> int:
            return x + y

        job = Job(func=add, kwargs={"x": 1, "y": 2})
    """

    func: Callable[..., Any]
    kwargs: Mapping[str, Operand]

    def __post_init__(self) -> None:
        validate_callable_bindings(
            func=self.func,
            kwargs=self.kwargs,
            parameter_names=(),
        )

    def rerun_snippet(self, *, dag_path: str | Path, job_id: str) -> str:
        """Return a Python snippet that reruns this job from a materialized plan."""
        from tidyrun.plan import rerun_snippet as _rerun_snippet

        return _rerun_snippet(dag_path, job_id)
