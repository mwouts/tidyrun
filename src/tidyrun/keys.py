"""
Job names, and storage keys, are constraint to live within
simple Python types, like strings (*), integers, floats, and date/times.

These keys are encoded as strings when used as part of a path, and decoded back
into their original values using toml.

(*) Strings must be valid path parts and cannot contain path separators.
"""

from datetime import date, datetime, time
from typing import Any, Type, cast

import toml
from .constants import TIDYRUN_METADATA_EXTENSION

Key = str | int | float | bool | date | datetime | time
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
    """Encode a Python key value to a path-safe string using TOML.

    Supported types are ``str``, ``int``, ``float``, ``bool``, ``date``,
    ``datetime``, and ``time``. Plain strings that round-trip unambiguously
    through TOML are left unquoted; strings that would otherwise be
    interpreted as another type (e.g. ``"true"``, ``"42"``) are
    TOML-quoted.

    Parameters
    ----------
    key :
        The key value to encode.

    Returns
    -------
    str
        A non-empty string suitable for use as a filesystem path component.

    Raises
    ------
    TidyRunKeyEncodingError
        When the key type is not supported or the resulting name violates
        path constraints (empty, contains ``/`` or ``\\``, starts with
        ``.``, or ends with ``.tidyrun``).

    Examples
    --------
    >>> encode_key(42)
    '42'
    >>> encode_key("hello")
    'hello'
    >>> encode_key("true")
    '"true"'
    >>> encode_key(True)
    'true'
    """
    if not _is_supported_key(key):
        raise TidyRunKeyEncodingError(f"Unsupported key type: {type(key).__name__}")

    if isinstance(key, str):
        _validate_name(key, error_type=TidyRunKeyEncodingError)

        # Keep plain strings unquoted unless parsing them as TOML would
        # coerce them to a different type (e.g. int/bool/date).
        try:
            toml_module = cast(Any, toml)
            parsed = cast(dict[str, Any], toml_module.loads(f"{_KEY_NAME} = {key}\n"))[
                _KEY_NAME
            ]
        except toml.TomlDecodeError:
            return key

        if parsed == key and isinstance(parsed, str):
            return key

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
    """Decode a stored key name back to its original Python type.

    Reverses the encoding produced by :func:`encode_key`.

    Parameters
    ----------
    name :
        The encoded key name (a filesystem path component).

    Returns
    -------
    Key
        The original Python value (``str``, ``int``, ``float``, ``bool``,
        ``date``, ``datetime``, or ``time``).

    Raises
    ------
    TidyRunKeyDecodingError
        When the name is empty, contains path separators, or cannot be
        decoded to a supported key type.

    Examples
    --------
    >>> decode_key("hello")
    'hello'
    >>> decode_key('"true"')
    'true'
    >>> decode_key("42")
    42
    """
    _validate_name(name, error_type=TidyRunKeyDecodingError)

    try:
        toml_module = cast(Any, toml)
        value = cast(dict[str, Any], toml_module.loads(f"{_KEY_NAME} = {name}\n"))[
            _KEY_NAME
        ]
    except toml.TomlDecodeError as exc:
        if name.startswith('"') or name.startswith("'"):
            raise TidyRunKeyDecodingError(f"Invalid encoded key: {name!r}") from exc
        return name

    if not _is_supported_key(value):
        raise TidyRunKeyDecodingError(
            f"Decoded value has unsupported type: {type(value).__name__}"
        )

    return value
