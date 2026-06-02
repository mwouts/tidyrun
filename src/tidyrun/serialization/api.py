from pathlib import Path
from typing import Any, Iterable, cast

from cloudpathlib import AnyPath, CloudPath


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
    decode_value_from_symlink,
    encode_dataframe_as_parquet,
    encode_dict_as_folder,
    encode_other_as_json,
    encode_pandas_as_hdf5,
    encode_series_as_parquet,
    encode_value_as_pickle,
    encode_value_as_symlink,
    has_serialized_path,
    is_json_serializable,
    is_mapping,
)
from .metadata import (
    metadata_exists,
    read_metadata,
    suffix_for_encoder,
    write_metadata,
)
from .paths import with_suffix
from .types import (
    ChecksumInfo,
    EncoderSpec,
    GoToNextEncoderException,
    TidyRunDeserializationError,
    TidyRunSerializationError,
)


def default_encoders() -> tuple[EncoderSpec, ...]:
    """Return default encoder order."""
    return (
        EncoderSpec(
            name="symlink",
            predicate=has_serialized_path,
            serializer=encode_value_as_symlink,
            deserializer=decode_value_from_symlink,
        ),
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
            return encoder.deserializer(base_path, None)
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
    value: Any,
    target: str | Path | CloudPath,
    encoders: Iterable[EncoderSpec] | None = None,
) -> ChecksumInfo:
    """Serialize a Python value to disk using the configured encoder pipeline.

    Parameters
    ----------
    value :
        The value to serialize (dict, DataFrame, Series, scalar, etc.).
    target :
        Where to write the output. Extension-free; the encoder appends the
        appropriate suffix. Accepts local paths or ``s3://`` URIs (requires
        the optional ``boto3`` dependency installed via ``pip install tidyrun[s3]``).
    encoders :
        Custom encoder pipeline. Defaults to :func:`default_encoders`.

    Returns
    -------
    ChecksumInfo
        Checksum (``algorithm``, ``digest``) for the serialized payload.

    Raises
    ------
    TidyRunSerializationError
        When no encoder matches the value type.
    NotImplementedError
        When an S3 target is requested without the optional dependency.
    """
    path = AnyPath(target)
    encoder_list = tuple(default_encoders() if encoders is None else encoders)
    selected_encoder: EncoderSpec | None = None
    checksum: ChecksumInfo | None = None
    for encoder in encoder_list:
        if not encoder.predicate(value):
            continue

        try:
            checksum = encoder.serializer(value, path)
        except GoToNextEncoderException:
            continue

        selected_encoder = encoder
        break

    if selected_encoder is None:
        raise TidyRunSerializationError(
            f"No encoder found for value of type {type(value).__name__!r}"
        )

    assert checksum is not None, "Encoder did not return checksum"

    symlink_target: str | None = None
    if selected_encoder.name == "symlink":
        import os

        symlink_target = cast(str, os.fspath(value))

    write_metadata(
        path,
        encoding=selected_encoder.name,
        suffix=suffix_for_encoder(selected_encoder.name),
        checksum=checksum,
        symlink_target=symlink_target,
    )
    return checksum


def deserialize(
    source: str | Path | CloudPath, encoders: Iterable[EncoderSpec] | None = None
) -> Any:
    """Deserialize a value from disk using metadata to determine the format.

    Directories encoded as ``dict-folder`` are returned as :class:`LazyDict`
    objects whose values are loaded on first access.

    Parameters
    ----------
    source :
        Location to read from. Accepts local paths or ``s3://`` URIs (requires
        the optional ``boto3`` dependency installed via ``pip install tidyrun[s3]``).
    encoders :
        Custom encoder pipeline. Defaults to :func:`default_encoders`.

    Returns
    -------
    Any
        ``LazyDict`` for dict-folder outputs; ``pd.DataFrame`` / ``pd.Series``
        for tabular outputs; the original Python object for scalar or pickle
        outputs.

    Raises
    ------
    TidyRunDeserializationError
        When metadata is missing, the encoder name is unknown, or the payload
        cannot be read.
    """
    path = AnyPath(source)
    encoder_list = tuple(default_encoders() if encoders is None else encoders)
    if not metadata_exists(path):
        return _deserialize_without_metadata(path, encoder_list)

    metadata = read_metadata(path)
    encoder_name = metadata["encoding"]
    encoder_map = encoder_by_name(encoder_list)
    encoder = encoder_map.get(encoder_name)
    if encoder is None:
        raise TidyRunDeserializationError(
            f"Unknown encoder in metadata: {encoder_name!r}"
        )

    checksum = metadata.get("checksum")
    assert checksum is None or isinstance(checksum, ChecksumInfo)

    # Symlink metadata can point to another serialized location.
    if encoder_name == "symlink":
        target = metadata.get("symlink_target")
        if isinstance(target, str):
            if "://" in target:
                target_path = AnyPath(target)
            else:
                target_path = AnyPath(path.parent / Path(target))
            return deserialize(target_path, encoders=encoder_list)

    return encoder.deserializer(path, checksum)
