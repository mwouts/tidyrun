from __future__ import annotations

from pathlib import Path

import pytest

from tidyrun.constants import TIDYRUN_METADATA_EXTENSION
from tidyrun.serialization import LazyDict, deserialize, serialize


def test_lazy_dict_ipython_key_completions_returns_string_keys(tmp_path: Path) -> None:
    target = tmp_path / "ipython_completion"
    serialize({"alpha": 1, "beta": 2}, target)

    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    completions = loaded._ipython_key_completions_()
    assert sorted(completions) == ["alpha", "beta"]


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
        select=lambda path: path[0] == "A",
    )

    assert set(concatenated.index.get_level_values("outer")) == {"A"}
    assert list(concatenated["v"]) == [1, 2]
    assert list(concatenated["v2"]) == [10, 20]


def test_lazy_dict_concat_transform_sum_returns_scalar(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    value = {
        "A": {
            "x": pd.DataFrame({"v": [1, 2]}),
            "y": pd.DataFrame({"v": [3]}),
        },
        "B": {
            "z": pd.DataFrame({"v": [4, 5]}),
        },
    }

    target = tmp_path / "concat_sum"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    concatenated = loaded.concat(
        names=["outer", "inner"],
        transform=lambda frame: frame["v"].sum(),
    )

    assert list(concatenated.index.names) == ["outer", "inner", None]
    assert list(concatenated["value"]) == [3, 3, 9]


def test_lazy_dict_concat_select_callback_receives_path_only(
    tmp_path: Path,
) -> None:
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

    target = tmp_path / "concat_select_args"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    seen: list[tuple[object, ...]] = []

    def _select(path: tuple[object, ...]) -> bool:
        seen.append(path)
        return path[0] == "A"

    concatenated = loaded.concat(names=["outer", "inner"], select=_select)

    assert set(seen) == {("A",), ("A", "x"), ("A", "y"), ("B",)}
    assert set(concatenated.index.get_level_values("outer")) == {"A"}


def test_lazy_dict_concat_does_not_load_unselected_keys(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    value = {
        "keep": pd.DataFrame({"v": [1, 2]}),
        "drop": pd.DataFrame({"v": [99]}),
    }

    target = tmp_path / "concat_no_load_unselected"
    serialize(value, target)

    # Make unselected branch unreadable; concat should still succeed.
    (target / "drop.parquet").unlink()

    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    concatenated = loaded.concat(
        names=["dataset"],
        select=lambda path: path[0] == "keep",
    )

    assert set(concatenated.index.get_level_values("dataset")) == {"keep"}
    assert list(concatenated["v"]) == [1, 2]


def test_lazy_dict_concat_does_not_load_unselected_subfolders_and_guards_depth(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    value = {
        "keep_group": {
            "keep_leaf": pd.DataFrame({"v": [1, 2]}),
            "other_leaf": pd.DataFrame({"v": [3]}),
        },
        "drop_group": {
            "drop_leaf": pd.DataFrame({"v": [99]}),
        },
    }

    target = tmp_path / "concat_no_load_unselected_subfolders"
    serialize(value, target)

    # If drop_group is loaded, concat would try to read this and fail.
    (target / "drop_group" / "drop_leaf.parquet").unlink()

    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    # select is evaluated at multiple depths, so guard path[1] access.
    def _select(path: tuple[object, ...]) -> bool:
        if path[0] != "keep_group":
            return False
        if len(path) == 1:
            return True
        return path[1] == "keep_leaf"

    concatenated = loaded.concat(
        names=["group", "leaf"],
        select=_select,
    )

    assert set(concatenated.index.get_level_values("group")) == {"keep_group"}
    assert set(concatenated.index.get_level_values("leaf")) == {"keep_leaf"}
    assert list(concatenated["v"]) == [1, 2]


def test_deserialize_parent_folder_with_subfolders_containing_parquets(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    # Create a structure with parent folder, subfolders a and b,
    # each containing two parquet files with small dataframes
    value = {
        "a": {
            "file1": pd.DataFrame({"col": [1, 2, 3]}),
            "file2": pd.DataFrame({"col": [4, 5, 6]}),
        },
        "b": {
            "file1": pd.DataFrame({"col": [7, 8, 9]}),
            "file2": pd.DataFrame({"col": [10, 11, 12]}),
        },
    }

    parent_folder = tmp_path / "parent"
    serialize(value, parent_folder)

    # Remove all tidyrun metadata files to test deserialization without metadata
    for metadata_file in tmp_path.rglob(f"*{TIDYRUN_METADATA_EXTENSION}"):
        metadata_file.unlink()

    # Deserialize from the parent folder
    loaded = deserialize(parent_folder)

    # Verify it's a LazyDict and not empty
    assert isinstance(loaded, LazyDict)
    assert len(loaded) > 0
    assert set(loaded.keys()) == {"a", "b"}
    assert isinstance(loaded["a"], LazyDict)
    assert isinstance(loaded["b"], LazyDict)
    assert len(loaded["a"]) == 2
    assert len(loaded["b"]) == 2

    # Test calling concat on it
    concatenated = loaded.concat(names=["subfolder", "file"])

    # Verify the concatenation worked
    assert list(concatenated.index.names) == ["subfolder", "file", None]
    assert list(concatenated["col"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_lazy_dict_concat_drops_none_level_names(tmp_path: Path) -> None:
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

    target = tmp_path / "concat_drop_level"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    # Use None for the second level to drop it
    concatenated = loaded.concat(names=["outer", None])

    # Only outer level should be in the index
    assert list(concatenated.index.names) == ["outer", None]
    # Inner level should be aggregated
    assert list(concatenated["v"]) == [1, 2, 3, 4, 5, 6]


def test_lazy_dict_concat_drops_multiple_none_level_names(tmp_path: Path) -> None:
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

    target = tmp_path / "concat_drop_multiple_levels"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    # Use None for both levels to drop them all
    concatenated = loaded.concat(names=[None, None])

    # All named levels are dropped, index should be a RangeIndex
    assert list(concatenated["v"]) == [1, 2, 3, 4, 5, 6]


def test_lazy_dict_concat_raises_when_lazydicts_remain_with_insufficient_names(
    tmp_path: Path,
) -> None:
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

    target = tmp_path / "concat_insufficient_names"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    # Only provide one name level, but we have two levels of nesting
    with pytest.raises(ValueError, match="Encountered LazyDict at depth 1"):
        loaded.concat(names=["outer"])


def test_lazy_dict_concat_parallel_max_workers(tmp_path: Path) -> None:
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

    target = tmp_path / "concat_parallel"
    serialize(value, target)
    loaded = deserialize(target)
    assert isinstance(loaded, LazyDict)

    sequential = loaded.concat(names=["outer", "inner"])
    parallel = loaded.concat(names=["outer", "inner"], max_workers=4)

    # Results must be identical regardless of loading strategy
    pd.testing.assert_frame_equal(sequential, parallel)


def test_lazy_dict_concat_parallel_select(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="Parquet engine is required for this test")

    value = {
        "A": {"x": pd.DataFrame({"v": [1]}), "y": pd.DataFrame({"v": [2]})},
        "B": {"z": pd.DataFrame({"v": [3]})},
    }

    target = tmp_path / "concat_parallel_select"
    serialize(value, target)
    loaded = deserialize(target)

    result = loaded.concat(
        names=["outer", "inner"],
        select=lambda path: path[0] == "A",
        max_workers=2,
    )

    assert set(result.index.get_level_values("outer")) == {"A"}
    assert list(result["v"]) == [1, 2]
