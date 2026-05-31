from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import toml
from cloudpathlib import CloudPath

from tidyrun.constants import (
    TIDYRUN_DEFAULT_HASH_ALGORITHM,
    TIDYRUN_METADATA_EXTENSION,
    TIDYRUN_METADATA_VERSION,
)

from .types import (
    ChecksumInfo,
    DEFAULT_HDF5_EXTENSION,
    DEFAULT_JSON_EXTENSION,
    DEFAULT_PARQUET_EXTENSION,
    DEFAULT_PICKLE_EXTENSION,
    TidyRunDeserializationError,
    TidyRunSerializationError,
)
from .paths import with_suffix


def metadata_path(base_path: Path | CloudPath) -> Path | CloudPath:
    return with_suffix(base_path, TIDYRUN_METADATA_EXTENSION)


def metadata_exists(base_path: Path | CloudPath) -> bool:
    return metadata_path(base_path).exists()


def suffix_for_encoder(encoder_name: str) -> str:
    if encoder_name == "symlink":
        return ""
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
    base_path: Path | CloudPath,
    *,
    encoding: str,
    suffix: str,
    checksum: ChecksumInfo,
    symlink_target: str | None = None,
) -> None:
    metadata_file = metadata_path(base_path)
    metadata: dict[str, Any] = {
        "version": TIDYRUN_METADATA_VERSION,
        "encoding": encoding,
        "suffix": suffix,
        "checksum": {
            "algorithm": checksum.algorithm,
            "digest": checksum.digest,
        },
    }

    if symlink_target is not None:
        metadata["symlink"] = {"target": symlink_target}

    metadata_file.write_text(
        toml.dumps(metadata),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        encoding="utf-8",
    )


def read_metadata(base_path: Path | CloudPath) -> dict[str, Any]:
    metadata_file = metadata_path(base_path)
    if not metadata_file.exists():
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

    result: dict[str, Any] = {"encoding": encoding, "suffix": suffix}

    # Extract checksum if present
    checksum_data = data.get("checksum")
    if isinstance(checksum_data, dict):
        checksum_data = cast(dict[str, Any], checksum_data)
        algorithm = checksum_data.get("algorithm")
        digest = checksum_data.get("digest")
        if isinstance(algorithm, str) and isinstance(digest, str):
            result["checksum"] = ChecksumInfo(
                algorithm=algorithm,
                digest=digest,
            )

    # Add symlink target if present
    symlink_data = data.get("symlink")
    if isinstance(symlink_data, dict):
        symlink_data = cast(dict[str, Any], symlink_data)
        target = symlink_data.get("target")
        if isinstance(target, str):
            result["symlink_target"] = target

    return result


def checksum_from_bytes(
    payload: bytes, algorithm: str = TIDYRUN_DEFAULT_HASH_ALGORITHM
) -> ChecksumInfo:
    digest = hashlib.new(algorithm)
    digest.update(payload)
    return ChecksumInfo(algorithm=algorithm, digest=digest.hexdigest())


def checksum_for_path(
    path: Path | CloudPath, algorithm: str = TIDYRUN_DEFAULT_HASH_ALGORITHM
) -> ChecksumInfo:
    digest = hashlib.new(algorithm)

    assert path.is_file()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return ChecksumInfo(algorithm=algorithm, digest=digest.hexdigest())


def checksum_for_named_children(
    children: list[tuple[str, ChecksumInfo]],
    algorithm: str = TIDYRUN_DEFAULT_HASH_ALGORITHM,
) -> ChecksumInfo:
    digest = hashlib.new(algorithm)
    for name, checksum in sorted(children, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(checksum.algorithm.encode("utf-8"))
        digest.update(b":")
        digest.update(checksum.digest.encode("utf-8"))
        digest.update(b"\0")
    return ChecksumInfo(algorithm=algorithm, digest=digest.hexdigest())
