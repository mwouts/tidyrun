from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Union

from tidyrun.keys import Key, encode_key

# Operand is any value accepted as a Job argument:
# a plain Python value, a LazyDict, a Job, or a DAG.
Operand = Any


def _validate_callable_bindings(
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
        _validate_callable_bindings(
            func=self.func,
            kwargs=self.kwargs,
            parameter_names=(),
        )

    def rerun_snippet(self, *, dag_path: str | Path, job_id: str) -> str:
        """Return a Python snippet that reruns this job from a materialized plan."""
        from tidyrun.plan import rerun_snippet as _rerun_snippet

        return _rerun_snippet(dag_path, job_id)


@dataclass
class ParametrizedJob(Mapping[Key, Union[Job, "ParametrizedJob"]]):
    """A deferred computation indexed by parameter keys.

    Parameters are declared through `parameter_names` and populated through
    `parameter_values`. Accessing a key fixes the first parameter and returns
    either a `Job` (when one parameter remains) or another `ParametrizedJob`
    (when more parameters remain).
    """

    func: Callable[..., Any]
    parameter_names: tuple[str, ...]
    parameter_values: tuple[tuple[Key, ...], ...]
    kwargs: Mapping[str, Operand]

    def __init__(
        self,
        func: Callable[..., Any],
        parameter_names: list[str] | tuple[str, ...],
        parameter_values: list[tuple[Key, ...]] | tuple[tuple[Key, ...], ...],
        kwargs: Mapping[str, Operand] | None = None,
    ) -> None:
        self.func = func
        self.parameter_names = tuple(parameter_names)
        self.parameter_values = tuple(tuple(v) for v in parameter_values)
        self.kwargs = {} if kwargs is None else kwargs
        self._validate()

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

            # Reuse key validation logic (type + encoding constraints).
            for key in values:
                encode_key(key)

            if values in seen:
                raise ValueError("parameter_values must not contain duplicates")
            seen.add(values)

        _validate_callable_bindings(
            func=self.func,
            kwargs=self.kwargs,
            parameter_names=self.parameter_names,
        )

    def __getitem__(self, key: Key) -> Job | ParametrizedJob:
        matching = [values for values in self.parameter_values if values[0] == key]
        if not matching:
            raise KeyError(key)

        parameter_name = self.parameter_names[0]
        bound_kwargs = dict(self.kwargs)
        bound_kwargs[parameter_name] = key

        if len(self.parameter_names) == 1:
            return Job(func=self.func, kwargs=bound_kwargs)

        remaining_names = self.parameter_names[1:]
        remaining_values = [values[1:] for values in matching]
        return ParametrizedJob(
            func=self.func,
            parameter_names=remaining_names,
            parameter_values=remaining_values,
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
