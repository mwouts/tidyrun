from __future__ import annotations

from concurrent.futures import Executor, Future
from datetime import date
import os
from pathlib import Path
import time
from typing import Any, cast

import pytest
import toml

from tidyrun import execute_plan, get_job_states
from tidyrun.dag import DAG, ParametrizedJob, load_job_definition
from tidyrun.job import Job, validate_callable_bindings
from tidyrun.keys import encode_key
from tidyrun.serialization.metadata import metadata_exists


def _join_with_sep(left: str, right: str, sep: str = "/") -> str:
    return f"{left}{sep}{right}"


def _thread_id_with_delay(delay: float) -> int:
    time.sleep(delay)
    return os.getpid()


class _OptionsExecutor:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.submit_with_options_calls: list[dict[str, str | int]] = []

    def submit(self, fn: object, /, *args: object, **kwargs: object) -> Future[object]:
        self.submit_calls += 1
        future: Future[object] = Future()
        result = fn(*args, **kwargs)  # type: ignore[operator]
        future.set_result(result)
        return future

    def submit_with_options(
        self,
        fn: object,
        /,
        *args: object,
        sbatch_options: dict[str, str | int],
        **kwargs: object,
    ) -> Future[object]:
        self.submit_with_options_calls.append(dict(sbatch_options))
        future: Future[object] = Future()
        result = fn(*args, **kwargs)  # type: ignore[operator]
        future.set_result(result)
        return future


class _ArrayOptionsExecutor(_OptionsExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.submit_array_with_options_calls: list[dict[str, object]] = []

    class _Submission:
        def __init__(self, future: Future[object], job_ids: tuple[str, ...]) -> None:
            self.future = future
            self.job_ids = job_ids

    def submit_array_with_options(
        self,
        fn: object,
        /,
        *args: object,
        sbatch_options: dict[str, str | int],
    ) -> _Submission:
        plan_dir, job_ids = args
        assert isinstance(plan_dir, str)
        assert isinstance(job_ids, list)
        batch = tuple(str(job_id) for job_id in job_ids)
        self.submit_array_with_options_calls.append(
            {
                "job_ids": batch,
                "sbatch_options": dict(sbatch_options),
            }
        )
        future: Future[object] = Future()
        for job_id in batch:
            fn(plan_dir, job_id)  # type: ignore[operator]
        future.set_result(None)
        return _ArrayOptionsExecutor._Submission(future=future, job_ids=batch)


def test_parametrized_job_slice_returns_job() -> None:
    pjob = ParametrizedJob(
        func=_join_with_sep,
        parameter_names=["left", "right"],
        parameter_values=[("a", "x"), ("a", "y"), ("b", "x")],
        kwargs={"sep": "-"},
    )

    left_slice = pjob["a"]
    assert isinstance(left_slice, ParametrizedJob)

    leaf_job = left_slice["x"]
    assert isinstance(leaf_job, Job)
    assert leaf_job.kwargs["left"] == "a"
    assert leaf_job.kwargs["right"] == "x"
    assert leaf_job.kwargs["sep"] == "-"


def test_parametrized_job_validation() -> None:
    with pytest.raises(ValueError, match="parameter_names must be unique"):
        ParametrizedJob(
            func=_join_with_sep,
            parameter_names=["x", "x"],
            parameter_values=[("a", "b")],
        )

    with pytest.raises(ValueError, match="length"):
        ParametrizedJob(
            func=_join_with_sep,
            parameter_names=["x", "y"],
            parameter_values=[("a",)],
        )

    with pytest.raises(ValueError, match="duplicates"):
        ParametrizedJob(
            func=_join_with_sep,
            parameter_names=["x", "y"],
            parameter_values=[("a", "b"), ("a", "b")],
        )


def test_job_validation_rejects_missing_and_unknown_arguments() -> None:
    with pytest.raises(ValueError, match="Missing required callable arguments: right"):
        Job(func=_join_with_sep, kwargs={"left": "a"})

    with pytest.raises(ValueError, match="Unknown callable arguments: extra"):
        Job(func=_join_with_sep, kwargs={"left": "a", "right": "b", "extra": 1})


def test_parametrized_job_validation_rejects_overlap_missing_and_unknown() -> None:
    with pytest.raises(ValueError, match="parametrized names"):
        ParametrizedJob(
            func=_join_with_sep,
            parameter_names=["left"],
            parameter_values=[("a",)],
            kwargs={"left": "a", "right": "b"},
        )

    with pytest.raises(ValueError, match="Missing required callable arguments: right"):
        ParametrizedJob(
            func=_join_with_sep,
            parameter_names=["left"],
            parameter_values=[("a",)],
            kwargs={},
        )

    with pytest.raises(ValueError, match="Unknown callable arguments: extra"):
        ParametrizedJob(
            func=_join_with_sep,
            parameter_names=["left", "right", "extra"],
            parameter_values=[("a", "b", "x")],
            kwargs={"sep": "-"},
        )


def test_dag_evaluate_with_parametrized_job(tmp_path: Path) -> None:
    pjob = ParametrizedJob(
        func=_join_with_sep,
        parameter_names=["left", "right"],
        parameter_values=[("m1", "train"), ("m1", "test"), ("m2", "train")],
        kwargs={"sep": ":"},
    )

    dag = DAG()
    dag["pairs"] = pjob

    result = dag.evaluate(tmp_path / "outputs")
    assert result.to_dict() == {
        "pairs": {
            "m1": {"train": "m1:train", "test": "m1:test"},
            "m2": {"train": "m2:train"},
        }
    }


def test_materialize_plan_layout_for_single_and_parametrized_jobs(
    tmp_path: Path,
) -> None:
    dag = DAG(
        {
            "single": Job(
                func=_join_with_sep,
                kwargs={"left": "solo", "right": "x", "sep": "-"},
            ),
            "grid": ParametrizedJob(
                func=_join_with_sep,
                parameter_names=["left"],
                parameter_values=[("a",), ("b",)],
                kwargs={"right": "x", "sep": "-"},
            ),
        }
    )

    plan_dir = dag.materialize(tmp_path / "plan")

    # Schema v2: definitions/, inputs/, outputs/ under plan_dir; no callables/ dir.
    assert (plan_dir / "definitions").is_dir()
    assert (plan_dir / "inputs").is_dir()
    assert (plan_dir / "outputs").is_dir()
    assert not (plan_dir / "callables").exists()
    assert not (plan_dir / "plan.tidyrun").exists()

    assert (plan_dir / "definitions" / "single.tidyrun").is_file()
    assert (plan_dir / "definitions" / "grid.tidyrun").is_file()
    assert not (plan_dir / "definitions" / "grid" / "a.tidyrun").exists()
    assert not (plan_dir / "definitions" / "grid" / "b.tidyrun").exists()

    # Callable is now embedded in the definition (import spec); no callables/ dir.
    single_def = toml.loads(
        (plan_dir / "definitions" / "single.tidyrun").read_text(encoding="utf-8")
    )
    assert single_def["callable_module"] == _join_with_sep.__module__
    assert single_def["callable_qualname"] == _join_with_sep.__qualname__
    assert "callable_path" not in single_def

    # Non-parameter literal inputs for plain job still go to inputs/.
    assert metadata_exists(plan_dir / "inputs" / "single" / "left")
    assert metadata_exists(plan_dir / "inputs" / "single" / "right")
    assert metadata_exists(plan_dir / "inputs" / "single" / "sep")

    # Parameter 'left' is inline in the definition; non-parameter literals in inputs/.
    grid_def = toml.loads(
        (plan_dir / "definitions" / "grid.tidyrun").read_text(encoding="utf-8")
    )
    assert grid_def["args"]["left"]["kind"] == "parameter"
    assert grid_def["args"]["left"]["values"] == ["a", "b"]
    assert metadata_exists(plan_dir / "inputs" / "grid" / "right")
    assert metadata_exists(plan_dir / "inputs" / "grid" / "sep")
    assert not metadata_exists(plan_dir / "inputs" / "grid" / "left")

    assert not metadata_exists(plan_dir / "inputs" / "grid" / "a" / "left")
    assert not metadata_exists(plan_dir / "inputs" / "grid" / "b" / "left")

    # The top-level DAG key → job-ref mapping is reconstructed from the DAG at
    # execute time (no top_level.tidyrun written). Verify via execute+result.
    result = dag.evaluate(tmp_path / "plan")
    assert result.to_dict() == {"single": "solo-x", "grid": {"a": "a-x", "b": "b-x"}}

    # load_job_definition works for parametrized instances.
    definition_a = load_job_definition(plan_dir, "grid/a")
    definition_b = load_job_definition(plan_dir, "grid/b")
    assert "callable_path" not in definition_a
    assert definition_a["callable_module"] == _join_with_sep.__module__
    # Parameter args resolve from job_id; non-parameter literals still have paths.
    assert definition_a["args"]["left"]["kind"] == "parameter"
    assert definition_b["args"]["left"]["kind"] == "parameter"
    assert definition_a["args"]["right"]["kind"] == "literal"
    assert definition_a["args"]["sep"]["kind"] == "literal"


@pytest.mark.skip(reason="S3 support to be revisited with PlanPaths update")
def test_materialize_supports_s3_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture_upload(local_root: Path, location: str) -> None:
        captured["local_root"] = local_root
        captured["location"] = location
        # Schema v2: top_level.tidyrun replaces plan.tidyrun
        captured["has_top_level"] = (
            local_root / "run-plan" / "definitions" / "top_level.tidyrun"
        ).is_file()

    monkeypatch.setattr("tidyrun.dag.upload_local_tree_to_s3", _capture_upload)

    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})
    location = "s3://unit-test-bucket/plans/run-plan"
    plan_dir = dag.materialize(location)

    assert plan_dir == Path(location)
    assert captured["location"] == location
    assert captured["has_top_level"] is True


def test_dag_evaluate_with_local_threads(tmp_path: Path) -> None:
    dag = DAG()
    for i in range(8):
        dag[str(i)] = Job(func=_thread_id_with_delay, kwargs={"delay": 0.05})

    result = dag.evaluate(tmp_path / "threaded", max_workers=4)
    values = list(result.to_dict().values())

    assert len(set(values)) > 1


def test_dag_evaluate_rejects_executor_and_max_workers(tmp_path: Path) -> None:
    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})

    with pytest.raises(ValueError, match="either executor or max_workers"):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            dag.evaluate(tmp_path / "invalid", executor=pool, max_workers=2)


def test_dag_evaluate_routes_job_resources_to_submit_with_options(
    tmp_path: Path,
) -> None:
    dag = DAG(
        {
            "a": Job(func=lambda: 1, kwargs={}),
            "b": Job(func=lambda: 2, kwargs={}),
        }
    )
    executor = _OptionsExecutor()

    result = dag.evaluate(
        tmp_path / "resource-map",
        executor=cast(Executor, executor),
        job_resources={"b": {"mem": "8G", "time": "00:30:00"}},
    )

    assert result.to_dict() == {"a": 1, "b": 2}
    assert executor.submit_calls == 1
    assert executor.submit_with_options_calls == [{"mem": "8G", "time": "00:30:00"}]


def test_dag_evaluate_routes_parametrized_jobs_to_array_submission(
    tmp_path: Path,
) -> None:
    pjob = ParametrizedJob(
        func=_join_with_sep,
        parameter_names=["left", "right"],
        parameter_values=[("m1", "train"), ("m1", "test"), ("m2", "train")],
        kwargs={"sep": ":"},
    )
    dag = DAG({"pairs": pjob})
    executor = _ArrayOptionsExecutor()

    result = dag.evaluate(tmp_path / "array-exec", executor=cast(Executor, executor))

    assert result.to_dict() == {
        "pairs": {
            "m1": {"train": "m1:train", "test": "m1:test"},
            "m2": {"train": "m2:train"},
        }
    }
    assert executor.submit_calls == 0
    assert len(executor.submit_array_with_options_calls) == 1
    call = executor.submit_array_with_options_calls[0]
    assert call["job_ids"] == (
        "pairs/m1/test",
        "pairs/m1/train",
        "pairs/m2/train",
    )
    assert call["sbatch_options"] == {"job_name": "pairs"}


def test_dag_evaluate_parametrized_date_keys_use_array_submission(
    tmp_path: Path,
) -> None:
    pjob = ParametrizedJob(
        func=_join_with_sep,
        parameter_names=["left", "right"],
        parameter_values=[
            (date(2026, 1, 1), "train"),
            (date(2026, 1, 2), "train"),
        ],
        kwargs={"sep": ":"},
    )
    dag = DAG({"pairs": pjob})
    executor = _ArrayOptionsExecutor()

    result = dag.evaluate(
        tmp_path / "array-exec-dates", executor=cast(Executor, executor)
    )

    assert result.to_dict() == {
        "pairs": {
            date(2026, 1, 1): {"train": "2026-01-01:train"},
            date(2026, 1, 2): {"train": "2026-01-02:train"},
        }
    }
    assert executor.submit_calls == 0
    assert len(executor.submit_array_with_options_calls) == 1
    call = executor.submit_array_with_options_calls[0]
    assert call["job_ids"] == (
        f"pairs/{encode_key(date(2026, 1, 1))}/train",
        f"pairs/{encode_key(date(2026, 1, 2))}/train",
    )
    assert call["sbatch_options"] == {"job_name": "pairs"}


def test_dag_evaluate_rejects_unknown_job_resources_key(tmp_path: Path) -> None:
    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})

    with pytest.raises(ValueError, match="unknown DAG keys"):
        dag.evaluate(tmp_path / "unknown", job_resources={"missing": {"mem": "4G"}})


def test_job_dependency_runs_after_inputs_ready(tmp_path: Path) -> None:
    def produce() -> int:
        return 2

    def consume(x: int) -> int:
        return x + 1

    producer = Job(func=produce, kwargs={})
    consumer = Job(func=consume, kwargs={"x": producer})

    dag = DAG({"producer": producer, "result": consumer})
    plan_dir = dag.materialize(tmp_path / "plan")
    assert isinstance(plan_dir, Path)

    # Dependency arg is exposed as a symlink in inputs/; no sidecar
    sym = plan_dir / "inputs" / "result" / "x"
    assert sym.is_symlink(), f"Dependency symlink missing: {sym}"
    assert not (plan_dir / "inputs" / "result" / "x.tidyrun").exists()

    result = dag.execute_materialized(plan_dir)
    assert result.to_dict()["result"] == 3


def test_shared_job_dependency_is_memoized(tmp_path: Path) -> None:
    def base() -> int:
        return 10

    def add_one(x: int) -> int:
        return x + 1

    shared = Job(func=base, kwargs={})
    dag = DAG(
        {
            "a": shared,
            "b": Job(func=add_one, kwargs={"x": shared}),
        }
    )

    result = dag.evaluate(tmp_path / "memo", max_workers=2)
    assert result.to_dict() == {"a": 10, "b": 11}


def test_evaluate_default_layout_writes_plan_and_outputs_subdirs(
    tmp_path: Path,
) -> None:
    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})
    run_dir = tmp_path / "run"

    result = dag.evaluate(run_dir)

    assert result.to_dict() == {"a": 1}
    # Plan and outputs are co-located under run_dir
    assert (run_dir / "definitions").is_dir()
    assert not (run_dir / "plan.tidyrun").exists()
    assert (run_dir / "outputs").is_dir()


def test_evaluate_uses_dag_path_as_plan_and_outputs_root(tmp_path: Path) -> None:
    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})
    plan_dir = tmp_path / "my-plan"

    result = dag.evaluate(plan_dir)

    assert result.to_dict() == {"a": 1}
    assert (plan_dir / "definitions").is_dir()
    assert (plan_dir / "outputs").is_dir()


def test_dag_rejects_encoded_keys_starting_with_dot(tmp_path: Path) -> None:
    dag = DAG({".hidden": Job(func=lambda: 1, kwargs={})})

    with pytest.raises(ValueError, match="start with reserved prefix"):
        dag.materialize(tmp_path / "plan")


def test_parametrized_job_plan_has_recursive_structure_and_o1_files(
    tmp_path: Path,
) -> None:
    """Test that parametrized jobs generate O(1) files with recursive plan structure.

    This test validates three key requirements:
    1. Recursive structure: Plan data matches the structure of nested sub-DAGs
    2. O(1) complexity: For N parameter combinations, only O(1) input files are created,
       not O(N) files per parameter
    3. No duplication: Parameter values are inlined in the definition (schema v2),
       not stored as (job_id, value) pairs in external files
    """
    # Create a parametrized job with multiple parameter combinations
    n_params = 5
    parameter_values = tuple((f"param_{i}",) for i in range(n_params))

    pjob = ParametrizedJob(
        func=_join_with_sep,
        parameter_names=["left"],
        parameter_values=parameter_values,  # type: ignore[arg-type]
        kwargs={"right": "constant_value", "sep": "-"},
    )

    dag = DAG({"grid": pjob})
    plan_dir = dag.materialize(tmp_path / "plan")

    # 1. O(1) definition files: one shared definition for the whole group.
    definition_files = sorted((plan_dir / "definitions").glob("grid*.tidyrun"))
    assert definition_files == [plan_dir / "definitions" / "grid.tidyrun"], (
        "Expected one shared definition file for the parametrized group"
    )
    assert not (plan_dir / "callables").exists(), "No callables/ dir in schema v2"
    assert not (plan_dir / "plan.tidyrun").exists(), "No plan.tidyrun in schema v2"

    # 2. Parameter values are inlined in the definition; no per-job input files.
    grid_def = toml.loads(
        (plan_dir / "definitions" / "grid.tidyrun").read_text(encoding="utf-8")
    )
    assert "array_group" not in grid_def, (
        "array_group must not be stored in file (derived from path)"
    )
    assert grid_def["parameter_names"] == ["left"]
    left_spec = grid_def["args"]["left"]
    assert left_spec["kind"] == "parameter"
    assert left_spec["values"] == [f"param_{i}" for i in range(n_params)], (
        "Parameter values stored as raw list (no job_id strings)"
    )

    # Constant (non-parameter) args still go to inputs/ as serialized files.
    input_group_dir = plan_dir / "inputs" / "grid"
    assert metadata_exists(input_group_dir / "right")
    assert metadata_exists(input_group_dir / "sep")
    assert not metadata_exists(input_group_dir / "left"), (
        "Parameter arg 'left' must NOT appear in inputs/ (it is inlined)"
    )
    for param_index in range(n_params):
        assert not (input_group_dir / f"param_{param_index}").exists(), (
            "No per-job input directories"
        )

    # 3. load_job_definition resolves the callable and parameter for each instance.
    definition_files_set: set[str] = set()
    for i in range(n_params):
        job_id = f"grid/param_{i}"
        definition = load_job_definition(plan_dir, job_id)
        definition_files_set.add(definition.get("array_group", ""))
        assert definition["args"]["left"]["kind"] == "parameter"
        assert definition["args"]["right"]["kind"] == "literal"
        assert definition["args"]["sep"]["kind"] == "literal"

    assert definition_files_set == {"grid"}, "All instances share one array_group"

    # 4. Execute and verify results are correct for all instances.
    result = dag.evaluate(tmp_path / "plan")
    grid_result = result.to_dict()["grid"]
    for i in range(n_params):
        assert grid_result[f"param_{i}"] == f"param_{i}-constant_value"


def test_nested_dag(tmp_path: Path) -> None:
    """A DAG stored as a node inside another DAG materializes and executes correctly."""
    inner = DAG()
    inner["x"] = Job(func=_join_with_sep, kwargs={"left": "a", "right": "b"})
    inner["y"] = Job(func=_join_with_sep, kwargs={"left": "c", "right": "d"})

    outer = DAG()
    outer["group"] = inner

    result = outer.evaluate(tmp_path)
    assert result.to_dict() == {"group": {"x": "a/b", "y": "c/d"}}


def test_validate_callable_rejects_positional_only_params() -> None:
    def _pos_only(a: int, b: int, /, c: int = 0) -> int:
        return a + b + c

    with pytest.raises(ValueError, match="positional-only"):
        validate_callable_bindings(
            func=_pos_only,
            kwargs={"a": 1, "b": 2},
            parameter_names=(),
        )


def test_execute_plan(tmp_path: Path) -> None:
    dag = DAG()
    dag["a"] = Job(func=_join_with_sep, kwargs={"left": "hello", "right": "world"})
    dag["b"] = Job(func=_join_with_sep, kwargs={"left": "foo", "right": "bar"})

    plan_dir = dag.materialize(tmp_path / "plan")
    execute_plan(plan_dir)

    from tidyrun import deserialize

    result = deserialize(tmp_path / "plan" / "outputs")
    assert result.to_dict() == {"a": "hello/world", "b": "foo/bar"}


def test_parametrized_job_dependency_through_same_parameter_no_extra_job(
    tmp_path: Path,
) -> None:
    """Passing the whole group as dep creates no extra job for the consumer."""

    def produce(x: int) -> int:
        return x * 2

    def consume(x: int, dep: Any) -> int:
        return dep[x] + x

    values = [1, 2, 3]
    pjob_produce = ParametrizedJob(
        func=produce,
        parameter_names=["x"],
        parameter_values=[(v,) for v in values],
    )
    pjob_consume = ParametrizedJob(
        func=consume,
        parameter_names=["x"],
        parameter_values=[(v,) for v in values],
        kwargs={"dep": pjob_produce},
    )
    dag = DAG({"produce": pjob_produce, "consume": pjob_consume})

    plan_dir = dag.materialize(tmp_path / "plan")
    assert isinstance(plan_dir, Path)

    # Each consumer instance has a symlink to the group root (whole produce group)
    for v in values:
        sym = plan_dir / "inputs" / f"consume/{encode_key(v)}" / "dep"
        assert sym.is_symlink(), f"Symlink missing for consume/{v}/dep"
        assert not (sym.parent / "dep.tidyrun").exists(), "No sidecar expected"
        # Symlink target resolves to the produce group root
        assert sym.resolve() == (plan_dir / "outputs" / "produce").resolve()

    result = dag.evaluate(tmp_path / "plan")
    result_dict = result.to_dict()
    for v in values:
        assert result_dict["consume"][v] == v * 2 + v


def test_parametrized_job_dependency_by_parameter_name_selector(
    tmp_path: Path,
) -> None:
    """Using pjob['x'] (parameter selector) is no longer supported; use whole group."""

    def produce(x: int) -> int:
        return x * 2

    def consume(x: int, dep: Any) -> int:
        return dep[x] + x

    values = [1, 2, 3]
    producer = ParametrizedJob(
        func=produce,
        parameter_names=["x"],
        parameter_values=[(v,) for v in values],
    )
    consumer = ParametrizedJob(
        func=consume,
        parameter_names=["x"],
        parameter_values=[(v,) for v in values],
        kwargs={"dep": producer},
    )

    dag = DAG({"produce": producer, "consume": consumer})
    plan_dir = dag.materialize(tmp_path / "plan")
    assert isinstance(plan_dir, Path)

    definition_names = {
        str(f.relative_to(plan_dir / "definitions").with_suffix("").as_posix())
        for f in (plan_dir / "definitions").rglob("*.tidyrun")
    }
    assert definition_names == {"produce", "consume"}

    # Each consumer instance has a symlink to the whole produce group root.
    for v in values:
        sym = plan_dir / "inputs" / f"consume/{encode_key(v)}" / "dep"
        assert sym.is_symlink(), f"Symlink missing for consume/{v}/dep"
        assert not (sym.parent / "dep.tidyrun").exists(), "No sidecar expected"
        assert sym.resolve() == (plan_dir / "outputs" / "produce").resolve()

    result = dag.evaluate(plan_dir)
    result_dict = result.to_dict()
    assert set(result_dict["produce"].keys()) == set(values)
    assert set(result_dict["consume"].keys()) == set(values)
    for v in values:
        assert result_dict["consume"][v] == v * 3


@pytest.mark.parametrize("produce_first", [True, False])
def test_parametrized_job_dep_whole_pjob_as_kwarg(
    tmp_path: Path, produce_first: bool
) -> None:
    """Passing a whole ParametrizedJob as dep kwarg; consumer slices it by parameter."""

    def produce(x: int) -> int:
        return x * 2

    def consume(x: int, dep: Any) -> int:
        return dep[x] + x

    values = [1, 2, 3]
    producer = ParametrizedJob(
        func=produce,
        parameter_names=["x"],
        parameter_values=[(v,) for v in values],
    )
    consumer = ParametrizedJob(
        func=consume,
        parameter_names=["x"],
        parameter_values=[(v,) for v in values],
        kwargs={"dep": producer},
    )

    dag = DAG()
    if produce_first:
        dag["produce"] = producer
        dag["consume"] = consumer
    else:
        dag["consume"] = consumer
        dag["produce"] = producer

    plan_dir = dag.materialize(tmp_path / "plan")
    assert isinstance(plan_dir, Path)

    definition_names = {
        str(f.relative_to(plan_dir / "definitions").with_suffix("").as_posix())
        for f in (plan_dir / "definitions").rglob("*.tidyrun")
    }
    assert definition_names == {"produce", "consume"}

    # Each consumer instance has a symlink to the whole produce group root (no sidecar).
    for v in values:
        sym = plan_dir / "inputs" / f"consume/{encode_key(v)}" / "dep"
        assert sym.is_symlink(), f"Symlink missing for consume/{v}/dep"
        assert not (sym.parent / "dep.tidyrun").exists(), "No sidecar expected"
        assert sym.resolve() == (plan_dir / "outputs" / "produce").resolve()

    result = dag.evaluate(plan_dir)
    result_dict = result.to_dict()
    assert set(result_dict["produce"].keys()) == set(values)
    assert set(result_dict["consume"].keys()) == set(values)
    for v in values:
        assert result_dict["consume"][v] == v * 3


def _produce_ab(a: int, b: int) -> int:
    return a * 10 + b


def _sum_values(dep: Any) -> int:
    return sum(dep[k] for k in dep)


def _add_one(dep: int) -> int:
    return dep + 1


def test_parametrized_job_subset_as_dependency(tmp_path: Path) -> None:
    """A subset of a member ParametrizedJob (pjob[key]) is a valid dependency.

    pjob[key] creates a fresh object, so the compiler resolves it through its
    provenance rather than object identity, and must not write a duplicate
    definition file for the subset.
    """
    pjob = ParametrizedJob(
        func=_produce_ab,
        parameter_names=["a", "b"],
        parameter_values=[(1, 1), (1, 2), (2, 1)],
    )
    dag = DAG({"grid": pjob})
    grid = dag["grid"]
    assert isinstance(grid, ParametrizedJob)
    # Group subset: fixes a=1, still a ParametrizedJob over b.
    dag["agg"] = Job(func=_sum_values, kwargs={"dep": grid[1]})
    # Single-instance subset: fixes both parameters, a plain Job.
    instance = pjob[2]
    assert isinstance(instance, ParametrizedJob)
    dag["one"] = Job(func=_add_one, kwargs={"dep": instance[1]})

    plan_dir = dag.materialize(tmp_path / "plan")
    assert isinstance(plan_dir, Path)

    definition_names = {
        str(f.relative_to(plan_dir / "definitions").with_suffix("").as_posix())
        for f in (plan_dir / "definitions").rglob("*.tidyrun")
    }
    assert definition_names == {"grid", "agg", "one"}

    agg_link = plan_dir / "inputs" / "agg" / "dep"
    assert agg_link.is_symlink()
    assert agg_link.resolve() == (plan_dir / "outputs" / "grid" / "1").resolve()
    one_link = plan_dir / "inputs" / "one" / "dep"
    assert one_link.is_symlink()
    assert one_link.resolve() == (plan_dir / "outputs" / "grid" / "2" / "1").resolve()

    agg_def = load_job_definition(plan_dir, "agg")
    assert set(agg_def["dependencies"]) == {"grid/1/1", "grid/1/2"}
    one_def = load_job_definition(plan_dir, "one")
    assert set(one_def["dependencies"]) == {"grid/2/1"}

    result = dag.evaluate(plan_dir)
    result_dict = result.to_dict()
    assert result_dict["agg"] == 11 + 12
    assert result_dict["one"] == 21 + 1


def test_parametrized_job_subset_dependency_in_nested_dag(tmp_path: Path) -> None:
    """Subsets resolve for parametrized jobs registered inside nested DAGs."""
    pjob = ParametrizedJob(
        func=_produce_ab,
        parameter_names=["a", "b"],
        parameter_values=[(1, 1), (1, 2)],
    )
    dag = DAG({"sub": DAG({"grid": pjob})})
    dag["agg"] = Job(func=_sum_values, kwargs={"dep": pjob[1]})

    result = dag.evaluate(tmp_path / "plan")
    assert result.to_dict()["agg"] == 11 + 12


def test_parametrized_job_subset_of_non_member_raises(tmp_path: Path) -> None:
    """A subset of a ParametrizedJob that is not a DAG member is still an error."""
    orphan = ParametrizedJob(
        func=_produce_ab,
        parameter_names=["a", "b"],
        parameter_values=[(1, 1)],
    )
    dag = DAG({"agg": Job(func=_sum_values, kwargs={"dep": orphan[1]})})
    with pytest.raises(ValueError, match="not a member of this DAG"):
        dag.materialize(tmp_path / "plan")


def test_get_job_states(tmp_path: Path) -> None:
    dag = DAG()
    dag["a"] = Job(func=_join_with_sep, kwargs={"left": "x", "right": "y"})
    dag["b"] = Job(func=_join_with_sep, kwargs={"left": "p", "right": "q"})

    plan_dir = dag.materialize(tmp_path / "plan")

    assert get_job_states(plan_dir) == {"a": "pending", "b": "pending"}

    dag.execute_materialized(plan_dir)

    assert get_job_states(plan_dir) == {"a": "succeeded", "b": "succeeded"}


def test_flat_dag_writes_root_metadata(tmp_path: Path) -> None:
    """After executing a flat DAG, outputs.tidyrun exists with dict-folder encoding."""
    import toml
    from tidyrun.serialization.metadata import (
        checksum_for_named_children,
        read_metadata,
    )

    dag = DAG(
        {
            "a": Job(func=_join_with_sep, kwargs={"left": "hello", "right": "world"}),
            "b": Job(func=_join_with_sep, kwargs={"left": "foo", "right": "bar"}),
        }
    )
    plan_dir = dag.materialize(tmp_path / "plan")
    dag.execute_materialized(plan_dir)

    outputs_path = plan_dir / "outputs"
    root_meta_file = outputs_path.with_suffix(".tidyrun")
    assert root_meta_file.exists(), (
        "outputs.tidyrun should be written after DAG execution"
    )

    root_meta = toml.loads(root_meta_file.read_text())
    assert root_meta["encoding"] == "dict-folder"
    assert root_meta["suffix"] == ""

    # Checksum must match what checksum_for_named_children would produce
    from tidyrun.keys import encode_key

    children = []
    for key in ["a", "b"]:
        encoded = encode_key(key)
        child_meta = read_metadata(outputs_path / encoded)
        children.append((encoded, child_meta["checksum"]))
    expected = checksum_for_named_children(children)

    assert root_meta["checksum"]["algorithm"] == expected.algorithm
    assert root_meta["checksum"]["digest"] == expected.digest


def test_nested_dag_writes_group_and_root_metadata(tmp_path: Path) -> None:
    """Nested DAGs produce .tidyrun files for every group level."""
    import toml
    from tidyrun import deserialize

    inner = DAG(
        {
            "x": Job(func=_join_with_sep, kwargs={"left": "a", "right": "b"}),
            "y": Job(func=_join_with_sep, kwargs={"left": "c", "right": "d"}),
        }
    )
    dag = DAG({"group": inner})
    plan_dir = dag.materialize(tmp_path / "plan")
    dag.execute_materialized(plan_dir)

    outputs_path = plan_dir / "outputs"
    group_meta_file = outputs_path / "group.tidyrun"
    root_meta_file = outputs_path.with_suffix(".tidyrun")

    assert group_meta_file.exists(), "outputs/group.tidyrun should be written"
    assert root_meta_file.exists(), "outputs.tidyrun should be written"

    group_meta = toml.loads(group_meta_file.read_text())
    assert group_meta["encoding"] == "dict-folder"

    root_meta = toml.loads(root_meta_file.read_text())
    assert root_meta["encoding"] == "dict-folder"

    # deserialize should still return the correct LazyDict
    result = deserialize(outputs_path)
    assert result.to_dict() == {"group": {"x": "a/b", "y": "c/d"}}


def test_dag_metadata_consistent_with_serialize(tmp_path: Path) -> None:
    """The checksum in outputs.tidyrun matches serialize(equivalent_dict)."""
    from tidyrun import serialize
    from tidyrun.serialization.metadata import read_metadata

    dag = DAG(
        {
            "a": Job(func=lambda: 42, kwargs={}),
            "b": Job(func=lambda: "hello", kwargs={}),
        }
    )
    plan_dir = dag.materialize(tmp_path / "plan")
    dag.execute_materialized(plan_dir)

    outputs_path = plan_dir / "outputs"

    # Serialize the equivalent dict to a separate location
    equivalent_dict = {"a": 42, "b": "hello"}
    serialize_path = tmp_path / "serialized"
    serialize(equivalent_dict, serialize_path)

    dag_root_meta = read_metadata(outputs_path)
    ser_root_meta = read_metadata(serialize_path)

    assert dag_root_meta["checksum"].digest == ser_root_meta["checksum"].digest


def test_dag_skip_completed_preserves_metadata(tmp_path: Path) -> None:
    """Re-running with skip_completed=True produces consistent metadata."""
    import toml

    dag = DAG(
        {
            "a": Job(func=_join_with_sep, kwargs={"left": "x", "right": "y"}),
        }
    )
    plan_dir = dag.materialize(tmp_path / "plan")
    dag.execute_materialized(plan_dir)

    outputs_path = plan_dir / "outputs"
    root_meta_before = toml.loads(outputs_path.with_suffix(".tidyrun").read_text())

    # Second run with skip_completed — should not error and metadata should be consistent
    dag.execute_materialized(plan_dir, skip_completed=True)

    root_meta_after = toml.loads(outputs_path.with_suffix(".tidyrun").read_text())
    assert (
        root_meta_before["checksum"]["digest"] == root_meta_after["checksum"]["digest"]
    )


# ---------------------------------------------------------------------------
# Plan structure: symlinks, no sidecars, correct dep resolution
# ---------------------------------------------------------------------------


def test_materialized_plan_content(tmp_path: Path) -> None:
    """Comprehensive check of the materialised plan layout after §3/§4 changes."""

    def produce() -> int:
        return 7

    def group_fn(x: int) -> int:
        return x * 10

    def consume(src: int, grp: Any) -> dict[str, Any]:
        return {"src": src, "grp_1": grp[1], "grp_2": grp[2]}

    producer = Job(func=produce, kwargs={})
    group = ParametrizedJob(
        func=group_fn,
        parameter_names=["x"],
        parameter_values=[(1,), (2,)],
    )
    consumer = Job(func=consume, kwargs={"src": producer, "grp": group})

    dag = DAG({"producer": producer, "group": group, "consumer": consumer})
    plan_dir = dag.materialize(tmp_path / "plan")
    assert isinstance(plan_dir, Path)

    # --- definition files exist ---
    assert (plan_dir / "definitions" / "producer.tidyrun").is_file()
    assert (plan_dir / "definitions" / "group.tidyrun").is_file()
    assert (plan_dir / "definitions" / "consumer.tidyrun").is_file()

    # --- consumer/src: symlink only, no sidecar ---
    src_sym = plan_dir / "inputs" / "consumer" / "src"
    assert src_sym.is_symlink(), "inputs/consumer/src must be a symlink"
    assert not (plan_dir / "inputs" / "consumer" / "src.tidyrun").exists(), "no sidecar"
    assert src_sym.resolve() == (plan_dir / "outputs" / "producer").resolve()

    # --- consumer/grp: symlink to group root, no sidecar ---
    grp_sym = plan_dir / "inputs" / "consumer" / "grp"
    assert grp_sym.is_symlink(), "inputs/consumer/grp must be a symlink"
    assert not (plan_dir / "inputs" / "consumer" / "grp.tidyrun").exists(), "no sidecar"
    assert grp_sym.resolve() == (plan_dir / "outputs" / "group").resolve()

    # --- execute and verify consumer slices dep correctly ---
    result = dag.execute_materialized(plan_dir)
    d = result.to_dict()
    assert d["producer"] == 7
    assert d["consumer"]["src"] == 7
    assert d["consumer"]["grp_1"] == 10
    assert d["consumer"]["grp_2"] == 20
