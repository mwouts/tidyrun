# Type stubs for tidyrun.serialization

from .api import (
    default_encoders,
    deserialize,
    select_encoder,
    serialize,
)
from .encoders import (
    can_encode_series_with_parquet,
    can_encode_with_hdf5,
    can_encode_with_parquet,
    is_dataframe,
    is_json_serializable,
    is_mapping,
    is_parquet_available,
)
from .lazy_dict import LazyDict
from .types import (
    DEFAULT_HDF5_EXTENSION,
    DEFAULT_JSON_EXTENSION,
    DEFAULT_PARQUET_EXTENSION,
    DEFAULT_PICKLE_EXTENSION,
    Deserializer,
    EncoderSpec,
    GoToNextEncoderException,
    Location,
    Predicate,
    Serializer,
    TidyRunDeserializationError,
    TidyRunSerializationError,
    TIDYRUN_METADATA_VERSION,
)

__all__ = [
    "DEFAULT_HDF5_EXTENSION",
    "DEFAULT_JSON_EXTENSION",
    "DEFAULT_PARQUET_EXTENSION",
    "DEFAULT_PICKLE_EXTENSION",
    "Deserializer",
    "EncoderSpec",
    "GoToNextEncoderException",
    "LazyDict",
    "Location",
    "Predicate",
    "Serializer",
    "TidyRunDeserializationError",
    "TidyRunSerializationError",
    "TIDYRUN_METADATA_VERSION",
    "can_encode_series_with_parquet",
    "can_encode_with_hdf5",
    "can_encode_with_parquet",
    "default_encoders",
    "deserialize",
    "is_dataframe",
    "is_json_serializable",
    "is_mapping",
    "is_parquet_available",
    "select_encoder",
    "serialize",
]
