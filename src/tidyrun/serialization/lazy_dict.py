from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable, cast

from tidyrun.keys import Key


class LazyDict(Mapping[Key, Any]):
    """Dictionary-like object that loads each child value on first access."""

    def __init__(self, base_dir: Path, entries: dict[Key, str]) -> None:
        self._base_dir = base_dir
        self._entries = entries
        self._keepalive: object | None = None

    def __getitem__(self, key: Key) -> Any:
        if key not in self._entries:
            raise KeyError(key)

        from .api import deserialize

        encoded_name = self._entries[key]
        return deserialize(self._base_dir / encoded_name)

    def __iter__(self) -> Iterator[Key]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[Key, Any]:
        result: dict[Key, Any] = {}
        for key in self:
            value = self[key]
            if isinstance(value, LazyDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def concat(
        self,
        names: list[str] | None = None,
        transform: Callable[[Any], Any] | None = None,
        select: Callable[[tuple[Key, ...], Any], bool] | None = None,
    ) -> Any:
        """Concatenate leaf pandas objects from a nested LazyDict.

        Parameters mirror the intended high-level behavior:
        - names: names for the concatenation key levels
        - transform: optional function applied to each leaf before concat
        - select: optional predicate `(path, value) -> bool` to filter leaves
        """
        import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

        pd = cast(Any, pd)

        def _identity(value: Any) -> Any:
            return value

        def _select_all(_path: tuple[Key, ...], _value: Any) -> bool:
            return True

        selected_transform: Callable[[Any], Any] = (
            transform if transform is not None else _identity
        )
        selected_filter: Callable[[tuple[Key, ...], Any], bool] = (
            select if select is not None else _select_all
        )

        keys: list[tuple[Key, ...]] = []
        values: list[Any] = []

        def _collect(node: LazyDict, prefix: tuple[Key, ...]) -> None:
            for key in node:
                value = node[key]
                current_path = prefix + (key,)
                if isinstance(value, LazyDict):
                    _collect(value, current_path)
                    continue

                if not selected_filter(current_path, value):
                    continue

                transformed = selected_transform(value)
                if isinstance(transformed, pd.Series):
                    frame: Any = transformed.to_frame()
                elif isinstance(transformed, pd.DataFrame):
                    frame = transformed
                else:
                    raise TypeError(
                        "concat expects leaves (or transformed leaves) to be "
                        "pandas DataFrame or Series"
                    )

                keys.append(current_path)
                values.append(frame)

        _collect(self, ())

        if not values:
            raise ValueError("No values selected for concatenation")

        return pd.concat(values, keys=keys, names=names)

    def __repr__(self) -> str:
        return f"LazyDict(keys={list(self._entries)!r})"
