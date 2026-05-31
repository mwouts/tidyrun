from __future__ import annotations

from dataclasses import dataclass
from cloudpathlib import CloudPath
from pathlib import Path
from typing import Any, Callable

Predicate = Callable[[Any], bool]


@dataclass(frozen=True)
class ChecksumInfo:
    algorithm: str
    digest: str


Serializer = Callable[[Any, Path | CloudPath], ChecksumInfo]
Deserializer = Callable[[Path | CloudPath, ChecksumInfo | None], Any]

DEFAULT_JSON_EXTENSION = ".json"
DEFAULT_PARQUET_EXTENSION = ".parquet"
DEFAULT_HDF5_EXTENSION = ".h5"
DEFAULT_PICKLE_EXTENSION = ".pickle"


class TidyRunSerializationError(TypeError):
    """Raised when a value cannot be serialized with the available encoders."""


class TidyRunDeserializationError(ValueError):
    """Raised when a location cannot be deserialized."""


class GoToNextEncoderException(Exception):
    """Raised by an encoder to indicate the next encoder should be attempted."""


@dataclass(frozen=True)
class EncoderSpec:
    """A pluggable encoder/decoder pair with a selection predicate."""

    name: str
    predicate: Predicate
    serializer: Serializer
    deserializer: Deserializer
