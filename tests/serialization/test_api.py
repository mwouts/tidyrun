from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import toml

from tidyrun.constants import TIDYRUN_METADATA_EXTENSION, TIDYRUN_METADATA_VERSION
from tidyrun.serialization import (
    GoToNextEncoderException,
    LazyDict,
    TidyRunDeserializationError,
    TidyRunSerializationError,
    deserialize,
    serialize,
)


class _Picklable:
    """Module-level class needed so pickle can locate it by qualified name."""

    def __init__(self, value: int) -> None:
        self.value = value


def test_serialize_deserialize_scalar_json(tmp_path: Path) -> None:
    target = tmp_path / "scalar"
    serialize({"a": 1}, target)

    root_metadata = tmp_path / f"scalar{TIDYRUN_METADATA_EXTENSION}"
    assert root_metadata.is_file()
    root_meta_data = toml.loads(root_metadata.read_text(encoding="utf-8"))
    assert root_meta_data["version"] == TIDYRUN_METADATA_VERSION

    assert (target / f'"a"{TIDYRUN_METADATA_EXTENSION}').is_file()
    assert (target / '"a".json').is_file()
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)
    assert dict(loaded) == {"a": 1}


def test_serialize_deserialize_nested_dict(tmp_path: Path) -> None:
    value = {
        "a": {
            1: "x",
            date(2026, 5, 10): True,
        },
        "b": 3.14,
    }

    target = tmp_path / "nested"
    serialize(value, target)

    assert (tmp_path / f"nested{TIDYRUN_METADATA_EXTENSION}").is_file()

    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)
    assert loaded.to_dict() == value


def test_deserialize_reloads_lazy_dict_children_on_each_access(tmp_path: Path) -> None:
    target = tmp_path / "nested"
    serialize({"a": {"value": 1}}, target)

    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    first = loaded["a"]
    second = loaded["a"]

    assert isinstance(first, LazyDict)
    assert isinstance(second, LazyDict)
    assert first is not second


def test_deserialize_without_metadata_sidecars_tries_known_candidates(
    tmp_path: Path,
) -> None:
    value = {
        "a": {
            1: "x",
            date(2026, 5, 10): True,
        },
        "b": 3.14,
    }

    target = tmp_path / "nested"
    serialize(value, target)

    for metadata_file in tmp_path.rglob(f"*{TIDYRUN_METADATA_EXTENSION}"):
        metadata_file.unlink()

    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)
    assert loaded.to_dict() == value


def test_serialize_deserialize_s3_round_trip() -> None:
    pytest.importorskip("boto3")
    pytest.importorskip("moto")

    import boto3  # pyright: ignore[reportMissingImports]
    from moto import mock_aws  # pyright: ignore[reportMissingImports]

    value = {
        "a": {
            1: "x",
            date(2026, 5, 10): True,
        },
        "b": 3.14,
    }

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "tidyrun-test-bucket"
        client.create_bucket(Bucket=bucket_name)

        target = f"s3://{bucket_name}/results/run_001"
        serialize(value, target)

        response = client.list_objects_v2(Bucket=bucket_name, Prefix="results/run_001")
        assert response.get("Contents")

        loaded = deserialize(target)
        assert isinstance(loaded, LazyDict)
        assert loaded.to_dict() == value


def test_serialize_non_json_value_uses_pickle(tmp_path: Path) -> None:
    obj = _Picklable(99)
    target = tmp_path / "custom"
    serialize(obj, target)

    assert (tmp_path / f"custom{TIDYRUN_METADATA_EXTENSION}").is_file()
    assert (tmp_path / "custom.pickle").is_file()
    loaded = deserialize(tmp_path / "custom")
    assert loaded.value == 99


def test_select_encoder_raises_serialization_error_for_unserializable(
    tmp_path: Path,
) -> None:
    with pytest.raises(TidyRunSerializationError):
        serialize(object(), tmp_path / "x", encoders=[])


def test_serialize_deserialize_dataframe_parquet(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="no parquet engine available")

    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    target = tmp_path / "frame"
    serialize(df, target)

    assert (tmp_path / f"frame{TIDYRUN_METADATA_EXTENSION}").is_file()
    assert (tmp_path / "frame.parquet").is_file()

    loaded = deserialize(tmp_path / "frame")
    pd.testing.assert_frame_equal(loaded, df)


def test_serialize_deserialize_pandas_series_prefers_parquet(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    series = pd.Series([10, 20, 30], name="data")
    target = tmp_path / "series"
    serialize(series, target)

    assert (tmp_path / f"series{TIDYRUN_METADATA_EXTENSION}").is_file()
    assert (tmp_path / "series.parquet").is_file()

    loaded = deserialize(tmp_path / "series")
    pd.testing.assert_series_equal(loaded, series)


def test_series_falls_back_to_hdf5_when_parquet_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables", reason="PyTables is required for HDF5 support")

    monkeypatch.setattr(
        "tidyrun.serialization.encoders.is_parquet_available", lambda: False
    )

    series = pd.Series([10, 20, 30], name="data")
    target = tmp_path / "series_hdf"
    serialize(series, target)

    metadata = toml.loads(
        (tmp_path / f"series_hdf{TIDYRUN_METADATA_EXTENSION}").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["encoding"] == "pandas-hdf5"
    assert (tmp_path / "series_hdf.h5").is_file()

    loaded = deserialize(tmp_path / "series_hdf")
    pd.testing.assert_series_equal(loaded, series)


def test_dataframe_falls_back_to_hdf5_when_parquet_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables", reason="PyTables is required for HDF5 support")

    monkeypatch.setattr(
        "tidyrun.serialization.encoders.is_parquet_available", lambda: False
    )

    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    target = tmp_path / "frame_hdf"
    serialize(df, target)

    assert (tmp_path / "frame_hdf.h5").is_file()
    loaded = deserialize(tmp_path / "frame_hdf")
    pd.testing.assert_frame_equal(loaded, df)


def test_dataframe_falls_back_to_hdf5_when_parquet_serialize_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables", reason="PyTables is required for HDF5 support")

    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"], "v": [10, 20]}).set_index(
        ["x", "y"]
    )

    def _raise_parquet_failure(_value: object, _target: object) -> None:
        raise GoToNextEncoderException(
            "Parquet serializer cannot handle this dataframe"
        )

    monkeypatch.setattr(
        "tidyrun.serialization.api.encode_dataframe_as_parquet", _raise_parquet_failure
    )

    target = tmp_path / "frame_parquet_fallback"
    serialize(df, target)

    metadata = toml.loads(
        (tmp_path / f"frame_parquet_fallback{TIDYRUN_METADATA_EXTENSION}").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["encoding"] == "pandas-hdf5"
    assert (tmp_path / "frame_parquet_fallback.h5").is_file()

    loaded = deserialize(tmp_path / "frame_parquet_fallback")
    pd.testing.assert_frame_equal(loaded, df)


def test_serialize_accepts_target_with_extension(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    serialize({"a": 1}, target)

    assert (tmp_path / f"x.json{TIDYRUN_METADATA_EXTENSION}").is_file()
    assert (target / '"a".json').is_file()


def test_deserialize_rejects_source_with_extension(tmp_path: Path) -> None:
    with pytest.raises(TidyRunDeserializationError):
        deserialize(tmp_path / "x.json")
