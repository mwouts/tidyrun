from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, cast

from tidyrun.keys import Key


def _entry_is_lazy_dict(base_dir: Path, encoded_name: str) -> bool:
    """Return True if entry at *base_dir/encoded_name* deserialises to a LazyDict.

    Peeks at the metadata sidecar (if present) instead of loading the value,
    so this is cheap regardless of payload size.
    """
    from .metadata import metadata_exists, read_metadata

    entry_path = base_dir / encoded_name
    if metadata_exists(entry_path):
        try:
            md = read_metadata(entry_path)
            return md.get("encoding") == "dict-folder"
        except Exception:
            return False
    # No metadata sidecar: legacy or unknown — treat directories as LazyDicts
    return entry_path.is_dir()


class LazyDict(Mapping[Key, Any]):
    """Dictionary-like object that loads each child value on first access."""

    def __init__(
        self,
        base_dir: Path,
        entries: dict[Key, str],
        ensure_local_path: Callable[[Path], None] | None = None,
    ) -> None:
        self._base_dir = base_dir
        self._entries = entries
        self._keepalive: object | None = None
        self._ensure_local_path = ensure_local_path

    def _ensure_available(self, path: Path) -> None:
        if self._ensure_local_path is not None:
            self._ensure_local_path(path)

    def __getitem__(self, key: Key) -> Any:
        if key not in self._entries:
            raise KeyError(key)

        from .api import deserialize

        encoded_name = self._entries[key]
        entry_path = self._base_dir / encoded_name
        self._ensure_available(entry_path)
        value = deserialize(entry_path)
        if isinstance(value, LazyDict):
            value._ensure_local_path = self._ensure_local_path
        return value

    def __iter__(self) -> Iterator[Key]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def _ipython_key_completions_(self) -> list[str]:
        """Return string keys for bracket-completion in IPython/Jupyter."""
        return [key for key in self._entries if isinstance(key, str)]

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
        names: list[str | None] | None = None,
        transform: Callable[[Any], Any] | None = None,
        select: Callable[[tuple[Key, ...]], bool] | None = None,
        max_workers: int | None = None,
    ) -> Any:
        """Concatenate leaf values from a nested LazyDict.

        Parameters:
        - names: names for the concatenation key levels. Use None to drop a level
          and concatenate values without that index level.
        - transform: optional function applied to each selected leaf value.
          The transformed value may be a pandas DataFrame, pandas Series,
          or a scalar value.
        - select: optional predicate called as ``select(path)`` where ``path``
          is a tuple of keys to a node (for example, ``("run_001", "metrics")``).
          Selection is evaluated before loading child values, so paths that are
          filtered out are never deserialized.
        - max_workers: when set, leaf values are loaded in parallel using a
          :class:`~concurrent.futures.ThreadPoolExecutor` with this many worker
          threads.  Useful when leaves are large files (e.g. Parquet) and I/O
          dominates.  When ``None`` (default), loading is sequential.

        Returns a pandas DataFrame from ``pd.concat`` with a MultiIndex built
        from selected leaf paths. Scalar transformed values are wrapped as a
        single-row Series before concatenation.

        Raises ValueError if a LazyDict is encountered but names is not deep
        enough to reach all leaf values.
        """
        import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

        pd = cast(Any, pd)

        def _identity(value: Any) -> Any:
            return value

        def _select_all(_path: tuple[Key, ...]) -> bool:
            return True

        selected_transform: Callable[[Any], Any] = (
            transform if transform is not None else _identity
        )
        selected_filter: Callable[[tuple[Key, ...]], bool] = (
            select if select is not None else _select_all
        )

        # Phase 1 (serial): walk the tree to discover leaf (path, node, key)
        # without loading the leaf values.  Intermediate LazyDict nodes are
        # detected cheaply via metadata peek; only directories that resolve to
        # another LazyDict are recursed into.
        leaf_refs: list[tuple[tuple[Key, ...], LazyDict, Key]] = []

        def _collect_refs(node: LazyDict, prefix: tuple[Key, ...]) -> None:
            for key in node:
                current_path = prefix + (key,)
                if not selected_filter(current_path):
                    continue

                encoded_name = node._entries[key]
                node._ensure_available(node._base_dir / encoded_name)
                if _entry_is_lazy_dict(node._base_dir, encoded_name):
                    if names is not None and len(current_path) >= len(names):
                        raise ValueError(
                            f"Encountered LazyDict at depth {len(current_path)}, "
                            f"but names only has {len(names)} levels. "
                            f"Provide more levels in names to reach leaf values."
                        )
                    child = node[
                        key
                    ]  # cheap: returns a LazyDict without loading children
                    _collect_refs(child, current_path)
                else:
                    leaf_refs.append((current_path, node, key))

        _collect_refs(self, ())

        if not leaf_refs:
            raise ValueError("No values selected for concatenation")

        # Phase 2: load leaf values — parallel when max_workers is set.
        def _load(ref: tuple[tuple[Key, ...], LazyDict, Key]) -> Any:
            _, node, key = ref
            return node[key]

        if max_workers is not None and len(leaf_refs) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                loaded_values = list(executor.map(_load, leaf_refs))
        else:
            loaded_values = [_load(ref) for ref in leaf_refs]

        # Phase 3: apply transform and build frames.
        keys: list[tuple[Key, ...]] = []
        values: list[Any] = []
        for (current_path, _, _key), value in zip(leaf_refs, loaded_values):
            transformed = selected_transform(value)
            if isinstance(transformed, pd.Series):
                frame: Any = transformed.to_frame()
            elif isinstance(transformed, pd.DataFrame):
                frame = transformed
            else:
                frame = pd.Series([transformed], name="value").to_frame()
            keys.append(current_path)
            values.append(frame)

        # Handle None values in names by dropping those levels
        if names is not None and any(n is None for n in names):
            # Create filtered keys and names, keeping only non-None levels
            keep_indices = [i for i, n in enumerate(names) if n is not None]
            filtered_names = [names[i] for i in keep_indices]

            if not filtered_names:
                # All levels are dropped, just concatenate without keys
                return pd.concat(values)
            else:
                # Some levels remain
                filtered_keys = [tuple(k[i] for i in keep_indices) for k in keys]
                return pd.concat(values, keys=filtered_keys, names=filtered_names)
        else:
            return pd.concat(values, keys=keys, names=names)

    def __repr__(self) -> str:
        return f"LazyDict(keys={list(self._entries)!r})"
