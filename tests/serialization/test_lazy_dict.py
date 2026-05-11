from __future__ import annotations

from pathlib import Path

import pytest

from tidyrun.serialization import LazyDict, deserialize, serialize


def test_lazy_dict_concat_nested_dataframes(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    value = {
        "A": {
            "x": pd.DataFrame({"v": [1, 2]}),
            "y": pd.DataFrame({"v": [3, 4]}),
        },
        "B": {
            "z": pd.DataFrame({"v": [5, 6]}),
        },
    }

    target = tmp_path / "concat_nested"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    concatenated = loaded.concat(names=["outer", "inner"])

    assert list(concatenated.index.names) == ["outer", "inner", None]
    assert list(concatenated["v"]) == [1, 2, 3, 4, 5, 6]


def test_lazy_dict_concat_transform_and_select(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    value = {
        "A": {
            "x": pd.DataFrame({"v": [1]}),
            "y": pd.DataFrame({"v": [2]}),
        },
        "B": {
            "z": pd.DataFrame({"v": [3]}),
        },
    }

    target = tmp_path / "concat_filter"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    concatenated = loaded.concat(
        names=["outer", "inner"],
        transform=lambda frame: frame.assign(v2=frame["v"] * 10),
        select=lambda path, _value: path[0] == "A",
    )

    assert set(concatenated.index.get_level_values("outer")) == {"A"}
    assert list(concatenated["v"]) == [1, 2]
    assert list(concatenated["v2"]) == [10, 20]
