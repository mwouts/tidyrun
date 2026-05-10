from __future__ import annotations

import importlib.util
import json
from typing import Any

from tidyrun.constants import TIDYRUN_METADATA_EXTENSION
from tidyrun.keys import Key, decode_key, encode_key

from .lazy_dict import LazyDict
from .metadata import suffix_for_encoder
from .paths import to_local_path, with_suffix
from .types import (
    DEFAULT_HDF5_EXTENSION,
    DEFAULT_JSON_EXTENSION,
    DEFAULT_PARQUET_EXTENSION,
    DEFAULT_PICKLE_EXTENSION,
    GoToNextEncoderException,
    Location,
    TidyRunDeserializationError,
    TidyRunSerializationError,
)


def _decoded_name_from_payload_name(name: str) -> str | None:
    if name.endswith(TIDYRUN_METADATA_EXTENSION):
        return None

    candidate_suffixes = (
        suffix_for_encoder("dataframe-parquet"),
        suffix_for_encoder("pandas-hdf5"),
        suffix_for_encoder("fallback-json"),
        suffix_for_encoder("fallback-pickle"),
    )
    for suffix in candidate_suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]

    return name


def is_mapping(value: Any) -> bool:
    """Return whether a value should be encoded as a folder."""
    return isinstance(value, dict)


def is_dataframe(value: Any) -> bool:
    """Return whether a value is a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        return False

    return isinstance(value, pd.DataFrame)


def is_pandas_series(value: Any) -> bool:
    """Return whether a value is a pandas Series."""
    try:
        import pandas as pd
    except ImportError:
        return False

    return isinstance(value, pd.Series)


def is_parquet_available() -> bool:
    """Return whether a parquet backend is available to pandas."""
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def can_encode_with_parquet(value: Any) -> bool:
    """Only use parquet encoder when dataframe support and engine are available."""
    return is_dataframe(value) and is_parquet_available()


def can_encode_series_with_parquet(value: Any) -> bool:
    """Use parquet for pandas Series when a parquet engine is available."""
    return is_pandas_series(value) and is_parquet_available()


def can_encode_with_hdf5(value: Any) -> bool:
    """Use HDF5 for pandas DataFrame/Series.

    This encoder sits after parquet encoders in the default order, so it acts
    as a fallback when parquet cannot be used or explicitly asks to skip.
    """
    return is_dataframe(value) or is_pandas_series(value)


def encode_dict_as_folder(value: dict[Any, Any], target_dir: Location) -> None:
    """Encode a nested dictionary as a folder tree."""
    from .api import serialize

    base_dir = to_local_path(target_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for key, item in value.items():
        name = encode_key(key)
        serialize(item, base_dir / name)


def decode_dict_from_folder(source_dir: Location) -> Any:
    """Decode a folder tree into a lazy dictionary-like object."""
    base_dir = to_local_path(source_dir)
    if not base_dir.is_dir():
        raise TidyRunDeserializationError(f"Expected directory, got: {base_dir}")

    entries: dict[Key, str] = {}
    metadata_files = sorted(
        base_dir.glob(f"*{TIDYRUN_METADATA_EXTENSION}"), key=lambda p: p.name
    )
    if metadata_files:
        for metadata_file in metadata_files:
            encoded_name = metadata_file.name[: -len(TIDYRUN_METADATA_EXTENSION)]
            key = decode_key(encoded_name)
            entries[key] = encoded_name
    else:
        for entry in sorted(base_dir.iterdir(), key=lambda p: p.name):
            encoded_name = _decoded_name_from_payload_name(entry.name)
            if encoded_name is None or encoded_name in entries:
                continue

            try:
                key = decode_key(encoded_name)
            except ValueError:
                continue

            entries[key] = encoded_name

    return LazyDict(base_dir, entries)


def encode_dataframe_as_parquet(value: Any, target_file: Location) -> None:
    """Encode a dataframe to a `.parquet` file."""
    if not is_dataframe(value):
        raise TidyRunSerializationError("Value is not a supported dataframe")

    file_path = with_suffix(to_local_path(target_file), DEFAULT_PARQUET_EXTENSION)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        value.to_parquet(file_path)
    except (ImportError, TypeError, ValueError) as exc:
        raise GoToNextEncoderException(
            "Parquet encoder could not serialize this dataframe"
        ) from exc


def decode_dataframe_from_parquet(source_file: Location) -> Any:
    """Decode a dataframe from a `.parquet` file."""
    import pandas as pd

    file_path = with_suffix(to_local_path(source_file), DEFAULT_PARQUET_EXTENSION)
    return pd.read_parquet(file_path)


def encode_series_as_parquet(value: Any, target_file: Location) -> None:
    """Encode a pandas Series to a `.parquet` file."""
    if not is_pandas_series(value):
        raise TidyRunSerializationError("Value is not a pandas Series")

    file_path = with_suffix(to_local_path(target_file), DEFAULT_PARQUET_EXTENSION)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    series_name = value.name if value.name is not None else "__tidyrun_series__"
    try:
        value.to_frame(name=series_name).to_parquet(file_path)
    except (ImportError, TypeError, ValueError) as exc:
        raise GoToNextEncoderException(
            "Parquet encoder could not serialize this series"
        ) from exc


def decode_series_from_parquet(source_file: Location) -> Any:
    """Decode a pandas Series from a `.parquet` file."""
    import pandas as pd

    file_path = with_suffix(to_local_path(source_file), DEFAULT_PARQUET_EXTENSION)
    df = pd.read_parquet(file_path)
    if len(df.columns) != 1:
        raise TidyRunDeserializationError(
            "Invalid parquet payload for pandas Series: expected exactly one column"
        )

    series = df.iloc[:, 0]
    if series.name == "__tidyrun_series__":
        series.name = None
    return series


def encode_pandas_as_hdf5(value: Any, target_file: Location) -> None:
    """Encode a pandas DataFrame or Series to HDF5 under key `data`."""
    if not (is_dataframe(value) or is_pandas_series(value)):
        raise TidyRunSerializationError("Value is not a supported pandas object")

    file_path = with_suffix(to_local_path(target_file), DEFAULT_HDF5_EXTENSION)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    value.to_hdf(file_path, key="data")


def decode_pandas_from_hdf5(source_file: Location) -> Any:
    """Decode a pandas DataFrame or Series from HDF5 key `data`."""
    import pandas as pd

    file_path = with_suffix(to_local_path(source_file), DEFAULT_HDF5_EXTENSION)
    return pd.read_hdf(file_path, key="data")


def encode_other_as_json(value: Any, target_file: Location) -> None:
    """Fallback encoder for non-dict, non-dataframe values using JSON."""
    file_path = with_suffix(to_local_path(target_file), DEFAULT_JSON_EXTENSION)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(value, f)


def decode_other_from_json(source_file: Location) -> Any:
    """Decode values previously written with JSON fallback."""
    file_path = with_suffix(to_local_path(source_file), DEFAULT_JSON_EXTENSION)

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_json_serializable(value: Any) -> bool:
    """Return whether a value can be round-tripped through JSON."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def encode_value_as_pickle(value: Any, target_file: Location) -> None:
    """Last-resort encoder: serialize any Python object with pickle."""
    import pickle

    file_path = with_suffix(to_local_path(target_file), DEFAULT_PICKLE_EXTENSION)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as f:
        pickle.dump(value, f)


def decode_value_from_pickle(source_file: Location) -> Any:
    """Decode a value previously written with pickle."""
    import pickle

    file_path = with_suffix(to_local_path(source_file), DEFAULT_PICKLE_EXTENSION)

    with file_path.open("rb") as f:
        return pickle.load(f)
