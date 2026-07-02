from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, cast

from cloudpathlib import CloudPath
from tidyrun.constants import TIDYRUN_METADATA_EXTENSION

from tidyrun.keys import Key, encode_key, decode_key
from .metadata import metadata_exists, read_metadata
from .types import ChecksumInfo
from .types import TidyRunDeserializationError


def _decoded_name_from_payload_name(name: str) -> str | None:
    if name.endswith(TIDYRUN_METADATA_EXTENSION):
        return None

    candidate_suffixes = (".parquet", ".h5", ".json", ".pickle")
    for suffix in candidate_suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]

    return name


def _entry_is_lazy_dict(serialized_path: Path | CloudPath, encoded_name: str) -> bool:
    child_path = serialized_path / encoded_name
    if metadata_exists(child_path):
        metadata = read_metadata(child_path)
        return metadata["encoding"] == "dict-folder"
    return child_path.is_dir()


class LazyDict(Mapping[Key, Any]):
    """Dictionary-like object that loads each child value on first access."""

    def __init__(
        self,
        serialized_path: Path | CloudPath,
        checksum: ChecksumInfo | None = None,
    ) -> None:
        self.__serialized_path__ = serialized_path
        self.__checksum__ = checksum

    """Marker that this LazyDict can be serialized as a symlink reference."""

    def __fspath__(self) -> str:
        """Return the path this LazyDict was loaded from.

        This enables symlink serialization via os.fspath() protocol.
        """
        return str(self.__serialized_path__)

    def _encoded_entries(self) -> list[str]:
        if not self.__serialized_path__.is_dir():
            raise TidyRunDeserializationError(
                f"Expected directory, got: {self.__serialized_path__}"
            )

        serialized_path = cast(Any, self.__serialized_path__)
        entries: list[str] = []
        metadata_named: set[str] = set()

        metadata_files = sorted(
            serialized_path.glob(f"*{TIDYRUN_METADATA_EXTENSION}"), key=lambda p: p.name
        )
        for metadata_file in metadata_files:
            encoded_name = metadata_file.name[: -len(TIDYRUN_METADATA_EXTENSION)]
            entries.append(encoded_name)
            metadata_named.add(encoded_name)

        if metadata_files:
            # When metadata files are present also include bare subdirectories
            # not already covered by a metadata sidecar.  This handles output
            # trees where parametrised-job group directories sit alongside
            # individual job outputs that do have metadata sidecars.
            for entry in sorted(serialized_path.iterdir(), key=lambda p: p.name):
                if entry.is_dir() and entry.name not in metadata_named:
                    entries.append(entry.name)
        else:
            # No metadata — fall back to scanning all payload candidates.
            for entry in sorted(serialized_path.iterdir(), key=lambda p: p.name):
                encoded_name = _decoded_name_from_payload_name(entry.name)
                if encoded_name is None:
                    continue
                entries.append(encoded_name)

        return entries

    def __getitem__(self, key: Key) -> Any:
        from .api import deserialize

        if isinstance(key, str) and ("/" in key or "\\" in key):
            # This is a convenience shortcut that allows users to
            # load nested data by passing a path-like key
            name = key
        else:
            name = encode_key(key)

        key_dir = self.__serialized_path__ / name
        return deserialize(key_dir)

    def __iter__(self) -> Iterator[Key]:
        for encoded_name in self._encoded_entries():
            yield decode_key(encoded_name)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _ipython_key_completions_(self) -> list[str]:
        """Return string keys for bracket-completion in IPython/Jupyter."""
        return [key for key in self if isinstance(key, str)]

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

        Parameters
        ----------
        names :
            Names for the MultiIndex levels built from nested keys. Pass
            ``None`` for a level to drop it from the index and concatenate
            without that level.
        transform :
            Optional function applied to each selected leaf value before
            concatenation. The result may be a ``pd.DataFrame``,
            ``pd.Series``, or a scalar (which is wrapped as a one-row
            ``"value"`` column).
        select :
            Optional predicate called as ``select(path)`` where ``path`` is a
            tuple of keys leading to a node (e.g. ``("run_001", "metrics")``).
            Evaluated before loading child values, so filtered paths are never
            deserialized.
        max_workers :
            When set, leaf values are loaded in parallel using a
            :class:`~concurrent.futures.ThreadPoolExecutor` with this many
            worker threads. Useful when leaves are large files (e.g. Parquet)
            and I/O dominates. When ``None`` (default), loading is sequential.

        Returns
        -------
        pd.DataFrame
            Result of ``pd.concat`` with a MultiIndex built from the selected
            leaf paths.

        Raises
        ------
        ValueError
            When a ``LazyDict`` node is encountered but ``names`` does not have
            enough levels to reach leaf values, or when no values are selected.
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

                encoded_name = encode_key(key)
                if _entry_is_lazy_dict(node.__serialized_path__, encoded_name):
                    if names is not None and len(current_path) >= len(names):
                        if transform is not None:
                            leaf_refs.append((current_path, node, key))
                            continue
                        raise ValueError(
                            f"Encountered LazyDict at depth {len(current_path)}, "
                            f"but names only has {len(names)} levels. "
                            f"Provide more levels in names to reach leaf values."
                        )
                    child = node[key]
                    _collect_refs(child, current_path)
                else:
                    leaf_refs.append((current_path, node, key))

        _collect_refs(self, ())

        if not leaf_refs:
            raise ValueError("No values selected for concatenation")

        # Phase 2: load leaf values — parallel when max_workers is set.
        def _load(ref: tuple[tuple[Key, ...], LazyDict, Key]) -> Any:
            _, node, key = ref
            value = node[key]
            transformed = selected_transform(value)
            if isinstance(transformed, LazyDict):
                raise ValueError(
                    f"Transform returned LazyDict for path {ref[0]}, but "
                    f"concat expects leaf values. Adjust transform or provide "
                    f"more levels in names to reach leaf values."
                )
            if isinstance(transformed, (pd.Series, pd.DataFrame)):
                return transformed
            return pd.Series([transformed], name="value").to_frame()

        if max_workers is not None and len(leaf_refs) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                frames = list(executor.map(_load, leaf_refs))
        else:
            frames = [_load(ref) for ref in leaf_refs]

        # Phase 3: apply transform and build frames.
        keys: list[tuple[Key, ...]] = []
        values: list[Any] = []
        for (current_path, _, _key), frame in zip(leaf_refs, frames):
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
        return f"LazyDict(keys={list(self)!r})"
