from __future__ import annotations

from typing import Any, Iterable

from tidyrun.constants import TIDYRUN_DEFAULT_HASH_ALGORITHM

from .encoders import (
    can_encode_with_hdf5,
    can_encode_with_parquet,
    can_encode_series_with_parquet,
    decode_dataframe_from_parquet,
    decode_dict_from_folder,
    decode_other_from_json,
    decode_pandas_from_hdf5,
    decode_series_from_parquet,
    decode_value_from_pickle,
    encode_dataframe_as_parquet,
    encode_dict_as_folder,
    encode_other_as_json,
    encode_pandas_as_hdf5,
    encode_series_as_parquet,
    encode_value_as_pickle,
    is_json_serializable,
    is_mapping,
)
from .metadata import (
    checksum_for_path,
    metadata_exists,
    read_metadata,
    suffix_for_encoder,
    write_metadata,
)
from .paths import to_local_path, with_suffix
from .s3 import deserialize_from_s3, is_s3_location, serialize_to_s3
from .types import (
    EncoderSpec,
    GoToNextEncoderException,
    Location,
    TidyRunDeserializationError,
    TidyRunSerializationError,
)


def default_encoders() -> tuple[EncoderSpec, ...]:
    """Return default encoder order."""
    return (
        EncoderSpec(
            name="dict-folder",
            predicate=is_mapping,
            serializer=encode_dict_as_folder,
            deserializer=decode_dict_from_folder,
        ),
        EncoderSpec(
            name="dataframe-parquet",
            predicate=can_encode_with_parquet,
            serializer=encode_dataframe_as_parquet,
            deserializer=decode_dataframe_from_parquet,
        ),
        EncoderSpec(
            name="series-parquet",
            predicate=can_encode_series_with_parquet,
            serializer=encode_series_as_parquet,
            deserializer=decode_series_from_parquet,
        ),
        EncoderSpec(
            name="pandas-hdf5",
            predicate=can_encode_with_hdf5,
            serializer=encode_pandas_as_hdf5,
            deserializer=decode_pandas_from_hdf5,
        ),
        EncoderSpec(
            name="fallback-json",
            predicate=is_json_serializable,
            serializer=encode_other_as_json,
            deserializer=decode_other_from_json,
        ),
        EncoderSpec(
            name="fallback-pickle",
            predicate=lambda _value: True,
            serializer=encode_value_as_pickle,
            deserializer=decode_value_from_pickle,
        ),
    )


def encoder_by_name(encoders: Iterable[EncoderSpec]) -> dict[str, EncoderSpec]:
    return {encoder.name: encoder for encoder in encoders}


def _candidate_payload_path(base_path: Any, encoder_name: str) -> Any:
    suffix = suffix_for_encoder(encoder_name)
    return base_path if suffix == "" else with_suffix(base_path, suffix)


def _candidate_exists(payload_path: Any, encoder_name: str) -> bool:
    if suffix_for_encoder(encoder_name) == "":
        return payload_path.is_dir()
    return payload_path.is_file()


def _deserialize_without_metadata(
    base_path: Any, encoders: Iterable[EncoderSpec]
) -> Any:
    last_error: Exception | None = None

    for encoder in encoders:
        payload_path = _candidate_payload_path(base_path, encoder.name)
        if not _candidate_exists(payload_path, encoder.name):
            continue

        try:
            return encoder.deserializer(payload_path)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise TidyRunDeserializationError(
            f"Could not deserialize {base_path} without metadata sidecar"
        ) from last_error

    raise TidyRunDeserializationError(
        f"Missing metadata file and no known payload candidates found for: {base_path}"
    )


def select_encoder(value: Any, encoders: Iterable[EncoderSpec]) -> EncoderSpec:
    """Pick the first encoder whose predicate returns True."""
    for encoder in encoders:
        if encoder.predicate(value):
            return encoder
    raise TidyRunSerializationError(
        f"No encoder found for value of type {type(value).__name__!r}"
    )


def serialize(
    value: Any, target: Location, encoders: Iterable[EncoderSpec] | None = None
) -> None:
    """Serialize a value with the configured encoder sequence."""
    if is_s3_location(target):
        encoder_list = tuple(default_encoders() if encoders is None else encoders)
        serialize_to_s3(value, target, encoder_list)
        return

    base_path = to_local_path(target)

    encoder_list = tuple(default_encoders() if encoders is None else encoders)
    selected_encoder: EncoderSpec | None = None
    for encoder in encoder_list:
        if not encoder.predicate(value):
            continue

        try:
            encoder.serializer(value, target)
        except GoToNextEncoderException:
            continue

        selected_encoder = encoder
        break

    if selected_encoder is None:
        raise TidyRunSerializationError(
            f"No encoder found for value of type {type(value).__name__!r}"
        )

    payload_path = (
        base_path
        if suffix_for_encoder(selected_encoder.name) == ""
        else with_suffix(base_path, suffix_for_encoder(selected_encoder.name))
    )
    output_digest = checksum_for_path(
        payload_path,
        algorithm=TIDYRUN_DEFAULT_HASH_ALGORITHM,
    )

    write_metadata(
        base_path,
        encoding=selected_encoder.name,
        suffix=suffix_for_encoder(selected_encoder.name),
        metadata_extra={
            "checksums": {
                "output": {
                    "algorithm": TIDYRUN_DEFAULT_HASH_ALGORITHM,
                    "digest": output_digest,
                }
            }
        },
    )


def deserialize(source: Location, encoders: Iterable[EncoderSpec] | None = None) -> Any:
    """Deserialize a value from an extension-free location using metadata."""
    if is_s3_location(source):
        encoder_list = tuple(default_encoders() if encoders is None else encoders)
        return deserialize_from_s3(source, encoder_list)

    base_path = to_local_path(source)

    encoder_list = tuple(default_encoders() if encoders is None else encoders)
    if not metadata_exists(base_path):
        return _deserialize_without_metadata(base_path, encoder_list)

    metadata = read_metadata(base_path)
    encoder_name = metadata["encoding"]
    suffix = metadata["suffix"]

    encoder_map = encoder_by_name(encoder_list)
    encoder = encoder_map.get(encoder_name)
    if encoder is None:
        raise TidyRunDeserializationError(
            f"Unknown encoder in metadata: {encoder_name!r}"
        )

    payload_path = base_path if suffix == "" else with_suffix(base_path, suffix)
    return encoder.deserializer(payload_path)
