from __future__ import annotations

from concurrent.futures import Executor, Future
from datetime import date
import os
from pathlib import Path
import time
from typing import cast

import pytest
import toml

from tidyrun.dag import DAG
from tidyrun.job import Job, ParametrizedJob
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

    assert (plan_dir / "definitions").is_dir()
    assert (plan_dir / "inputs").is_dir()
    assert (plan_dir / "callables").is_dir()
    assert (plan_dir / "outputs").is_dir()

    assert (plan_dir / "definitions" / "single.tidyrun").is_file()
    assert (plan_dir / "definitions" / "grid" / "a.tidyrun").is_file()
    assert (plan_dir / "definitions" / "grid" / "b.tidyrun").is_file()

    assert metadata_exists(plan_dir / "callables" / "single" / "callable")
    assert metadata_exists(plan_dir / "callables" / "grid" / "callable")
    assert not metadata_exists(plan_dir / "callables" / "grid" / "a" / "callable")
    assert not metadata_exists(plan_dir / "callables" / "grid" / "b" / "callable")

    assert metadata_exists(plan_dir / "inputs" / "single" / "left")
    assert metadata_exists(plan_dir / "inputs" / "single" / "right")
    assert metadata_exists(plan_dir / "inputs" / "single" / "sep")

    assert metadata_exists(plan_dir / "inputs" / "grid" / "left")
    assert metadata_exists(plan_dir / "inputs" / "grid" / "right")
    assert metadata_exists(plan_dir / "inputs" / "grid" / "sep")

    assert not metadata_exists(plan_dir / "inputs" / "grid" / "a" / "left")
    assert not metadata_exists(plan_dir / "inputs" / "grid" / "a" / "right")
    assert not metadata_exists(plan_dir / "inputs" / "grid" / "a" / "sep")
    assert not metadata_exists(plan_dir / "inputs" / "grid" / "b" / "left")
    assert not metadata_exists(plan_dir / "inputs" / "grid" / "b" / "right")
    assert not metadata_exists(plan_dir / "inputs" / "grid" / "b" / "sep")

    manifest = toml.loads((plan_dir / "plan.tidyrun").read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    assert set(jobs) == {"single", "grid/a", "grid/b"}
    assert jobs["single"]["dependencies"] == []
    assert "array_group" not in jobs["single"]
    assert jobs["grid/a"]["array_group"] == "grid"
    assert jobs["grid/b"]["array_group"] == "grid"

    top_level = manifest["top_level"]
    assert top_level["single"] == {"kind": "job", "job_id": "single"}
    assert top_level["grid"]["kind"] == "group"
    assert top_level["grid"]["entries"] == {
        "a": {"kind": "job", "job_id": "grid/a"},
        "b": {"kind": "job", "job_id": "grid/b"},
    }

    definition_a = toml.loads(
        (plan_dir / "definitions" / "grid" / "a.tidyrun").read_text(encoding="utf-8")
    )
    definition_b = toml.loads(
        (plan_dir / "definitions" / "grid" / "b.tidyrun").read_text(encoding="utf-8")
    )
    assert definition_a["callable_path"] == "callables/grid/callable"
    assert definition_b["callable_path"] == "callables/grid/callable"
    assert definition_a["args"]["left"]["path"] == "inputs/grid/left"
    assert definition_b["args"]["left"]["path"] == "inputs/grid/left"
    assert definition_a["args"]["left"]["job_id"] == "grid/a"
    assert definition_b["args"]["left"]["job_id"] == "grid/b"
    assert definition_a["args"]["right"]["path"] == "inputs/grid/right"
    assert definition_a["args"]["sep"]["path"] == "inputs/grid/sep"
    assert definition_b["args"]["right"]["path"] == "inputs/grid/right"
    assert definition_b["args"]["sep"]["path"] == "inputs/grid/sep"


def test_materialize_supports_s3_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture_upload(local_root: Path, location: str) -> None:
        captured["local_root"] = local_root
        captured["location"] = location
        captured["has_manifest"] = (local_root / "run-plan" / "plan.tidyrun").is_file()

    monkeypatch.setattr("tidyrun.dag.upload_local_tree_to_s3", _capture_upload)

    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})
    location = "s3://unit-test-bucket/plans/run-plan"
    plan_dir = dag.materialize(location)

    assert plan_dir == Path(location)
    assert captured["location"] == location
    assert captured["has_manifest"] is True


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

    dag = DAG({"result": consumer})
    result = dag.evaluate(tmp_path / "dependency")

    assert result.to_dict() == {"result": 3}


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
    assert (run_dir / "plan" / "plan.tidyrun").is_file()
    assert (run_dir / "outputs").is_dir()


def test_evaluate_accepts_explicit_plan_and_output_paths(tmp_path: Path) -> None:
    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})
    run_dir = tmp_path / "run"
    explicit_plan = tmp_path / "custom-plan"
    explicit_output = tmp_path / "custom-output"

    result = dag.evaluate(run_dir, dag_path=explicit_plan, output_path=explicit_output)

    assert result.to_dict() == {"a": 1}
    assert (explicit_plan / "plan.tidyrun").is_file()
    assert explicit_output.is_dir()


def test_evaluate_accepts_explicit_paths_without_target(tmp_path: Path) -> None:
    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})
    explicit_plan = tmp_path / "custom-plan"
    explicit_output = tmp_path / "custom-output"

    result = dag.evaluate(dag_path=explicit_plan, output_path=explicit_output)

    assert result.to_dict() == {"a": 1}
    assert (explicit_plan / "plan.tidyrun").is_file()
    assert explicit_output.is_dir()


def test_evaluate_requires_target_or_both_explicit_paths(tmp_path: Path) -> None:
    dag = DAG({"a": Job(func=lambda: 1, kwargs={})})

    with pytest.raises(
        ValueError, match="Pass target, or pass both dag_path and output_path"
    ):
        dag.evaluate()

    with pytest.raises(
        ValueError, match="Pass target, or pass both dag_path and output_path"
    ):
        dag.evaluate(dag_path=tmp_path / "plan-only")

    with pytest.raises(
        ValueError, match="Pass target, or pass both dag_path and output_path"
    ):
        dag.evaluate(output_path=tmp_path / "output-only")


def test_dag_rejects_encoded_keys_starting_with_dot(tmp_path: Path) -> None:
    dag = DAG({".hidden": Job(func=lambda: 1, kwargs={})})

    with pytest.raises(ValueError, match="start with reserved prefix"):
        dag.materialize(tmp_path / "plan")
