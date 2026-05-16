from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import urlparse

from tidyrun.constants import TIDYRUN_METADATA_EXTENSION
from tidyrun.keys import Key, decode_key

from .lazy_dict import LazyDict
from .metadata import read_metadata, suffix_for_encoder
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


def _try_download_object(s3: Any, bucket: str, key: str, destination: Path) -> bool:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(destination))
        return True
    except Exception:
        return False


def _build_entries_from_prefix(
    s3: Any,
    bucket: str,
    key: str,
    encoders: tuple[EncoderSpec, ...],
) -> dict[Key, str]:
    prefix = f"{key}/"
    entries: dict[Key, str] = {}
    suffixes = {
        suffix
        for suffix in (suffix_for_encoder(encoder.name) for encoder in encoders)
        if suffix
    }

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            child_prefix = cast(str, common_prefix.get("Prefix", ""))
            if not child_prefix.startswith(prefix):
                continue
            encoded_name = child_prefix[len(prefix) :].rstrip("/")
            if not encoded_name:
                continue
            try:
                key_obj = decode_key(encoded_name)
            except ValueError:
                continue
            entries.setdefault(key_obj, encoded_name)

        for item in page.get("Contents", []):
            object_key = cast(str, item.get("Key", ""))
            if not object_key.startswith(prefix):
                continue
            remainder = object_key[len(prefix) :]
            if not remainder or "/" in remainder:
                continue

            encoded_name: str | None
            if remainder.endswith(TIDYRUN_METADATA_EXTENSION):
                encoded_name = remainder[: -len(TIDYRUN_METADATA_EXTENSION)]
            else:
                encoded_name = None
                for suffix in sorted(suffixes, key=len, reverse=True):
                    if remainder.endswith(suffix):
                        encoded_name = remainder[: -len(suffix)]
                        break

            if not encoded_name:
                continue

            try:
                key_obj = decode_key(encoded_name)
            except ValueError:
                continue
            entries.setdefault(key_obj, encoded_name)

    return entries


def _ensure_local_base_from_s3(
    s3: Any,
    bucket: str,
    remote_base_key: str,
    local_root: Path,
    prefix: str,
    encoders: tuple[EncoderSpec, ...],
) -> None:
    subtree_prefix = f"{remote_base_key}/"
    paginator = s3.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=subtree_prefix):
        for item in page.get("Contents", []):
            object_key = cast(str, item.get("Key", ""))
            if not object_key.startswith(subtree_prefix):
                continue
            found = True
            rel = object_key[len(prefix) + 1 :] if prefix else object_key
            destination = local_root / rel
            _try_download_object(s3, bucket, object_key, destination)
    if found:
        return

    metadata_key = f"{remote_base_key}{TIDYRUN_METADATA_EXTENSION}"
    metadata_rel = metadata_key[len(prefix) + 1 :] if prefix else metadata_key
    metadata_destination = local_root / metadata_rel
    if _try_download_object(s3, bucket, metadata_key, metadata_destination):
        local_base_rel = (
            remote_base_key[len(prefix) + 1 :] if prefix else remote_base_key
        )
        local_base = local_root / local_base_rel
        metadata = read_metadata(local_base)
        suffix = metadata.get("suffix", "")
        if suffix:
            payload_key = f"{remote_base_key}{suffix}"
            payload_rel = payload_key[len(prefix) + 1 :] if prefix else payload_key
            payload_destination = local_root / payload_rel
            _try_download_object(s3, bucket, payload_key, payload_destination)
        return

    object_rel = remote_base_key[len(prefix) + 1 :] if prefix else remote_base_key
    object_destination = local_root / object_rel
    if _try_download_object(s3, bucket, remote_base_key, object_destination):
        return

    for encoder in encoders:
        suffix_candidate = suffix_for_encoder(encoder.name)
        if not suffix_candidate:
            continue
        payload_key = f"{remote_base_key}{suffix_candidate}"
        payload_rel = payload_key[len(prefix) + 1 :] if prefix else payload_key
        payload_destination = local_root / payload_rel
        if _try_download_object(s3, bucket, payload_key, payload_destination):
            return


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
    bucket, key = _parse_s3_location(location_str)
    prefix = _prefix_from_key(key)
    leaf = _leaf_name_from_key(key)
    boto3 = _require_boto3()
    s3 = boto3.client("s3")

    tempdir = TemporaryDirectory()
    local_root = Path(tempdir.name)

    local_base = local_root / leaf

    remote_metadata_key = f"{key}{TIDYRUN_METADATA_EXTENSION}"
    local_metadata = local_root / f"{leaf}{TIDYRUN_METADATA_EXTENSION}"
    metadata_downloaded = _try_download_object(
        s3, bucket, remote_metadata_key, local_metadata
    )

    if metadata_downloaded:
        metadata = read_metadata(local_base)
        encoding = metadata.get("encoding")
        suffix = metadata.get("suffix")

        if encoding == "dict-folder" and suffix == "":
            local_base.mkdir(parents=True, exist_ok=True)

            def _ensure_local_path(path: Path) -> None:
                try:
                    relative = path.relative_to(local_root).as_posix()
                except ValueError:
                    return

                remote_base_key = _join_remote_key(prefix, relative)
                _ensure_local_base_from_s3(
                    s3=s3,
                    bucket=bucket,
                    remote_base_key=remote_base_key,
                    local_root=local_root,
                    prefix=prefix,
                    encoders=encoders,
                )

            entries = _build_entries_from_prefix(s3, bucket, key, encoders)
            result: object = LazyDict(
                local_base,
                entries,
                ensure_local_path=_ensure_local_path,
            )
        else:
            if suffix:
                remote_payload_key = f"{key}{suffix}"
                local_payload = local_root / f"{leaf}{suffix}"
                _try_download_object(s3, bucket, remote_payload_key, local_payload)
            result = deserialize(local_base, encoders=encoders)
    else:
        local_base.mkdir(parents=True, exist_ok=True)
        entries = _build_entries_from_prefix(s3, bucket, key, encoders)
        if entries:

            def _ensure_local_path(path: Path) -> None:
                try:
                    relative = path.relative_to(local_root).as_posix()
                except ValueError:
                    return

                remote_base_key = _join_remote_key(prefix, relative)
                _ensure_local_base_from_s3(
                    s3=s3,
                    bucket=bucket,
                    remote_base_key=remote_base_key,
                    local_root=local_root,
                    prefix=prefix,
                    encoders=encoders,
                )

            result = LazyDict(
                local_base,
                entries,
                ensure_local_path=_ensure_local_path,
            )
        else:
            for encoder in encoders:
                suffix = suffix_for_encoder(encoder.name)
                if not suffix:
                    continue
                remote_payload_key = f"{key}{suffix}"
                local_payload = local_root / f"{leaf}{suffix}"
                if _try_download_object(s3, bucket, remote_payload_key, local_payload):
                    break
            result = deserialize(local_base, encoders=encoders)

    if hasattr(result, "_keepalive"):
        setattr(result, "_keepalive", tempdir)
    else:
        tempdir.cleanup()
    return result
