from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import toml

from tidyrun.constants import TIDYRUN_METADATA_EXTENSION, TIDYRUN_METADATA_VERSION

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


def write_metadata(base_path: Path, *, encoding: str, suffix: str) -> None:
    metadata_file = metadata_path(base_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, str | int] = {
        "version": TIDYRUN_METADATA_VERSION,
        "encoding": encoding,
        "suffix": suffix,
    }
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
