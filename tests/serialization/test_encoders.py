from __future__ import annotations

import pytest

from tidyrun.serialize import can_encode_with_hdf5, is_dataframe, is_json_serializable


def test_is_dataframe_without_pandas_dependency() -> None:
    assert is_dataframe({"not": "a dataframe"}) is False


def test_is_json_serializable() -> None:
    assert is_json_serializable(42) is True
    assert is_json_serializable("hello") is True
    assert is_json_serializable([1, 2]) is True
    assert is_json_serializable(object()) is False


def test_can_encode_with_hdf5_for_dataframe_when_parquet_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")

    monkeypatch.setattr(
        "tidyrun.serialization.encoders.is_parquet_available", lambda: False
    )

    df = pd.DataFrame({"x": [1, 2]})
    assert can_encode_with_hdf5(df) is True
