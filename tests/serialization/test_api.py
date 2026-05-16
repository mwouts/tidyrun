from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import toml

from tidyrun.constants import (
    TIDYRUN_DEFAULT_HASH_ALGORITHM,
    TIDYRUN_METADATA_EXTENSION,
    TIDYRUN_METADATA_VERSION,
)
from tidyrun.serialization.metadata import checksum_for_path
from tidyrun.serialization import (
    GoToNextEncoderException,
    LazyDict,
    TidyRunDeserializationError,
    TidyRunSerializationError,
    deserialize,
    serialize,
)
from tidyrun.serialization.s3 import upload_local_tree_to_s3


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
    checksum = root_meta_data["checksums"]["output"]
    assert checksum["algorithm"] == TIDYRUN_DEFAULT_HASH_ALGORITHM
    expected_digest = checksum_for_path(
        target, algorithm=TIDYRUN_DEFAULT_HASH_ALGORITHM
    )
    assert checksum["digest"] == expected_digest

    assert (target / f"a{TIDYRUN_METADATA_EXTENSION}").is_file()
    assert (target / "a.json").is_file()
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


def test_dag_materialize_uploads_plan_tree_to_s3() -> None:
    pytest.importorskip("boto3")
    pytest.importorskip("moto")

    import boto3  # pyright: ignore[reportMissingImports]
    from moto import mock_aws  # pyright: ignore[reportMissingImports]

    from tidyrun.dag import DAG
    from tidyrun.job import Job

    def _return_number() -> int:
        return 7

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "tidyrun-test-bucket"
        client.create_bucket(Bucket=bucket_name)

        dag = DAG({"a": Job(func=_return_number, kwargs={})})
        target = f"s3://{bucket_name}/plans/run_001"

        dag.materialize(target)

        response = client.list_objects_v2(Bucket=bucket_name, Prefix="plans/run_001")
        keys = {item["Key"] for item in response.get("Contents", [])}

        assert "plans/run_001/plan.tidyrun" in keys
        assert "plans/run_001/definitions/a.tidyrun" in keys
        assert any(key.startswith("plans/run_001/callables/a/") for key in keys)


def test_upload_local_tree_to_s3_uses_multiple_workers_for_many_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploaded: list[tuple[str, str, str]] = []
    created_executors: list["_FakeThreadPoolExecutor"] = []

    class _FakeFuture:
        def __init__(self, fn: Any, *args: Any, **kwargs: Any) -> None:
            self._fn = fn
            self._args = args
            self._kwargs = kwargs

        def result(self) -> Any:
            return self._fn(*self._args, **self._kwargs)

    class _FakeThreadPoolExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers
            self.submissions = 0

        def __enter__(self) -> "_FakeThreadPoolExecutor":
            created_executors.append(self)
            return self

        def __exit__(
            self,
            _exc_type: Any,
            _exc: Any,
            _tb: Any,
        ) -> None:
            return None

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> _FakeFuture:
            self.submissions += 1
            return _FakeFuture(fn, *args, **kwargs)

    class _FakeS3Client:
        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            uploaded.append((filename, bucket, key))

    class _FakeBoto3:
        def client(self, service_name: str) -> _FakeS3Client:
            assert service_name == "s3"
            return _FakeS3Client()

    for index in range(4):
        (tmp_path / f"f{index}.txt").write_text(f"payload-{index}", encoding="utf-8")

    monkeypatch.setattr("tidyrun.serialization.s3._require_boto3", lambda: _FakeBoto3())
    monkeypatch.setattr(
        "tidyrun.serialization.s3.ThreadPoolExecutor", _FakeThreadPoolExecutor
    )

    upload_local_tree_to_s3(tmp_path, "s3://bucket/results/run")

    assert len(uploaded) == 4
    assert len(created_executors) == 1
    assert created_executors[0].max_workers > 1
    assert created_executors[0].submissions == 4


def test_deserialize_s3_without_metadata_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("boto3")
    pytest.importorskip("moto")

    import boto3  # pyright: ignore[reportMissingImports]
    from moto import mock_aws  # pyright: ignore[reportMissingImports]

    value = {
        "a": {"x": 1},
        "b": {"y": 2},
    }

    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "tidyrun-test-bucket"
        client.create_bucket(Bucket=bucket_name)

        target = f"s3://{bucket_name}/results/run_lazy"
        serialize(value, target)

        response = client.list_objects_v2(Bucket=bucket_name, Prefix="results/run_lazy")
        for item in response.get("Contents", []):
            key = item["Key"]
            if key.endswith(TIDYRUN_METADATA_EXTENSION):
                client.delete_object(Bucket=bucket_name, Key=key)

        download_calls = 0
        original_download = client.download_file

        def _counting_download(bucket: str, key: str, filename: str) -> None:
            nonlocal download_calls
            download_calls += 1
            original_download(bucket, key, filename)

        monkeypatch.setattr(client, "download_file", _counting_download)
        monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: client)

        loaded = deserialize(target)
        assert isinstance(loaded, LazyDict)
        # Top-level deserialize should not download every payload eagerly.
        assert download_calls <= 1

        _ = loaded["a"]
        assert download_calls > 0


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
    assert (target / "a.json").is_file()


def test_deserialize_rejects_source_with_extension(tmp_path: Path) -> None:
    with pytest.raises(TidyRunDeserializationError):
        deserialize(tmp_path / "x.json")


def test_deserialize_ignores_optional_metadata_sections(tmp_path: Path) -> None:
    target = tmp_path / "with_optional_sections"
    serialize({"a": 1}, target)

    metadata_file = tmp_path / f"with_optional_sections{TIDYRUN_METADATA_EXTENSION}"
    metadata = toml.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["provenance"] = {
        "job_id": "root/a",
        "definition_ref": "definitions/root/a.tidyrun",
    }
    metadata["job_definition"] = {
        "callable_module": "my_pkg.jobs",
        "callable_qualname": "transform",
    }
    metadata_file.write_text(toml.dumps(metadata), encoding="utf-8")

    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)
    assert loaded.to_dict() == {"a": 1}
