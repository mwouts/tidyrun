from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, cast

import toml

from tidyrun.constants import (
    TIDYRUN_DEFAULT_HASH_ALGORITHM,
    TIDYRUN_METADATA_EXTENSION,
    TIDYRUN_METADATA_VERSION,
)

from .types import (
    DEFAULT_HDF5_EXTENSION,
    DEFAULT_JSON_EXTENSION,
    DEFAULT_PARQUET_EXTENSION,
    DEFAULT_PICKLE_EXTENSION,
    TidyRunDeserializationError,
    TidyRunSerializationError,
)


def metadata_path(base_path: Path) -> Path:
    return Path(f"{base_path}{TIDYRUN_METADATA_EXTENSION}")


def metadata_exists(base_path: Path) -> bool:
    return metadata_path(base_path).is_file()


def checksum_for_path(
    path: Path, algorithm: str = TIDYRUN_DEFAULT_HASH_ALGORITHM
) -> str:
    """Return a deterministic checksum for a file or directory tree.

    For files, this hashes file bytes directly.
    For directories, this hashes each file's relative path and content digest,
    iterating in sorted order for reproducibility.
    """

    digest = hashlib.new(algorithm)
    if path.is_file():
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if path.is_dir():
        for file_path in sorted((p for p in path.rglob("*") if p.is_file())):
            relative = file_path.relative_to(path).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            with file_path.open("rb") as file_handle:
                for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return digest.hexdigest()

    raise TidyRunSerializationError(f"Cannot checksum missing path: {path}")


def suffix_for_encoder(encoder_name: str) -> str:
    if encoder_name == "dict-folder":
        return ""
    if encoder_name == "dataframe-parquet":
        return DEFAULT_PARQUET_EXTENSION
    if encoder_name == "series-parquet":
        return DEFAULT_PARQUET_EXTENSION
    if encoder_name == "pandas-hdf5":
        return DEFAULT_HDF5_EXTENSION
    if encoder_name == "fallback-json":
        return DEFAULT_JSON_EXTENSION
    if encoder_name == "fallback-pickle":
        return DEFAULT_PICKLE_EXTENSION
    raise TidyRunSerializationError(f"Unknown encoder name: {encoder_name!r}")


def write_metadata(
    base_path: Path,
    *,
    encoding: str,
    suffix: str,
    metadata_extra: Mapping[str, Any] | None = None,
) -> None:
    metadata_file = metadata_path(base_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "version": TIDYRUN_METADATA_VERSION,
        "encoding": encoding,
        "suffix": suffix,
    }
    if metadata_extra is not None:
        overlap = set(metadata).intersection(metadata_extra)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise TidyRunSerializationError(
                f"metadata_extra contains reserved keys: {names}"
            )
        metadata.update(dict(metadata_extra))

    metadata_file.write_text(
        toml.dumps(metadata),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        encoding="utf-8",
    )


def read_metadata(base_path: Path) -> dict[str, str]:
    metadata_file = metadata_path(base_path)
    if not metadata_file.is_file():
        raise TidyRunDeserializationError(f"Missing metadata file: {metadata_file}")

    data = cast(
        dict[str, Any],
        toml.loads(metadata_file.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )
    version = data.get("version")
    encoding = data.get("encoding")
    suffix = data.get("suffix")
    if not isinstance(version, int):
        raise TidyRunDeserializationError(
            f"Invalid metadata version in file: {metadata_file}"
        )

    if version != TIDYRUN_METADATA_VERSION:
        raise TidyRunDeserializationError(
            f"Unsupported metadata version {version!r} in file: {metadata_file}"
        )

    if not isinstance(encoding, str) or not isinstance(suffix, str):
        raise TidyRunDeserializationError(f"Invalid metadata in file: {metadata_file}")

    return {"encoding": encoding, "suffix": suffix}
