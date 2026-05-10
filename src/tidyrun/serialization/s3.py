from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import urlparse

from .types import EncoderSpec, Location


def is_s3_location(location: Location) -> bool:
    return isinstance(location, str) and location.startswith("s3://")


def _require_boto3() -> Any:
    try:
        import boto3  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as exc:
        raise NotImplementedError(
            "S3 support requires the optional 'boto3' dependency. "
            "Install tidyrun[s3] to enable it."
        ) from exc

    return cast(Any, boto3)


def _parse_s3_location(location: str) -> tuple[str, str]:
    parsed = urlparse(location)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 location: {location!r}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 location must include a key: {location!r}")

    return bucket, key


def _prefix_from_key(key: str) -> str:
    return key.rsplit("/", 1)[0] if "/" in key else ""


def _leaf_name_from_key(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def _join_remote_key(prefix: str, relative_path: str) -> str:
    if not prefix:
        return relative_path
    return f"{prefix}/{relative_path}"


def upload_local_tree_to_s3(local_root: Path, location: str) -> None:
    boto3 = _require_boto3()
    bucket, key = _parse_s3_location(location)
    prefix = _prefix_from_key(key)
    s3 = boto3.client("s3")

    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(local_root).as_posix()
        remote_key = _join_remote_key(prefix, relative_path)
        s3.upload_file(str(path), bucket, remote_key)


def download_s3_tree_to_local_root(location: str, local_root: Path) -> None:
    boto3 = _require_boto3()
    bucket, key = _parse_s3_location(location)
    prefix = _prefix_from_key(key)
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for item in page.get("Contents", []):
            remote_key = item["Key"]
            if (
                remote_key != key
                and not remote_key.startswith(f"{key}/")
                and not remote_key.startswith(f"{key}.")
            ):
                continue

            relative_path = remote_key[len(prefix) + 1 :] if prefix else remote_key
            destination = local_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, remote_key, str(destination))


def _location_as_str(location: Location) -> str:
    if not isinstance(location, str):
        raise TypeError("S3 locations must be provided as string URIs")
    return location


def serialize_to_s3(
    value: object,
    location: Location,
    encoders: tuple[EncoderSpec, ...],
) -> None:
    from .api import serialize

    location_str = _location_as_str(location)
    _, key = _parse_s3_location(location_str)
    with TemporaryDirectory() as tempdir:
        local_root = Path(tempdir)
        local_target = local_root / _leaf_name_from_key(key)
        serialize(value, local_target, encoders=encoders)
        upload_local_tree_to_s3(local_root, location_str)


def deserialize_from_s3(
    location: Location,
    encoders: tuple[EncoderSpec, ...],
) -> object:
    from .api import deserialize

    location_str = _location_as_str(location)
    _, key = _parse_s3_location(location_str)
    tempdir = TemporaryDirectory()
    local_root = Path(tempdir.name)
    download_s3_tree_to_local_root(location_str, local_root)
    result = deserialize(local_root / _leaf_name_from_key(key), encoders=encoders)
    if hasattr(result, "_keepalive"):
        setattr(result, "_keepalive", tempdir)
    else:
        tempdir.cleanup()
    return result
