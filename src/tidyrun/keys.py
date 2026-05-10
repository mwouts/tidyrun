"""
Job names, and storage keys, are constraint to live within
simple Python types, like strings (*), integers, floats, and date/times.

These keys are encoded as strings when used as part of a path, and decoded back
into their original values using toml.

(*) Strings must be valid path parts and cannot contain path separators.
"""

from datetime import date, datetime, time
from typing import Any, Type, Union, cast

import toml
from .constants import TIDYRUN_METADATA_EXTENSION

Key = Union[str, int, float, bool, date, datetime, time]
_KEY_NAME = "key"


class TidyRunKeyEncodingError(ValueError):
    """Raised when a key value cannot be encoded as a valid key name."""


class TidyRunKeyDecodingError(ValueError):
    """Raised when a key name cannot be decoded back into a key value."""


def _is_supported_key(value: object) -> bool:
    return isinstance(value, (str, int, float, bool, date, datetime, time))


def _validate_name(name: str, *, error_type: Type[ValueError]) -> None:
    if not name:
        raise error_type("Key names cannot be empty")
    if "/" in name or "\\" in name:
        raise error_type("Key names cannot contain path separators")
    if name.startswith("."):
        raise error_type("Key names cannot start with reserved prefix '.'")
    if name.endswith(TIDYRUN_METADATA_EXTENSION):
        raise error_type(
            f"Key names cannot end with reserved suffix '{TIDYRUN_METADATA_EXTENSION}'"
        )


def encode_key(key: Key) -> str:
    if not _is_supported_key(key):
        raise TidyRunKeyEncodingError(f"Unsupported key type: {type(key).__name__}")

    try:
        toml_module = cast(Any, toml)
        toml_doc = cast(str, toml_module.dumps({_KEY_NAME: key}))
    except (TypeError, ValueError) as exc:
        raise TidyRunKeyEncodingError(
            f"Unsupported key type: {type(key).__name__}"
        ) from exc

    prefix = f"{_KEY_NAME} = "
    assert toml_doc.startswith(prefix)
    assert toml_doc.endswith("\n")
    name = toml_doc[len(prefix) : -1]

    _validate_name(name, error_type=TidyRunKeyEncodingError)
    return name


def decode_key(name: str) -> Key:
    _validate_name(name, error_type=TidyRunKeyDecodingError)

    try:
        toml_module = cast(Any, toml)
        value = cast(dict[str, Any], toml_module.loads(f"{_KEY_NAME} = {name}\n"))[
            _KEY_NAME
        ]
    except toml.TomlDecodeError as exc:
        raise TidyRunKeyDecodingError(f"Invalid encoded key: {name!r}") from exc

    if not _is_supported_key(value):
        raise TidyRunKeyDecodingError(
            f"Decoded value has unsupported type: {type(value).__name__}"
        )

    return value
