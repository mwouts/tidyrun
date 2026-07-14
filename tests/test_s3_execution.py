"""End-to-end execution of plans materialized to S3 (mocked with moto).

Regression tests for s3:// plan locations being interpreted as local paths in
``execute_materialized`` / ``evaluate`` / ``get_job_states`` / ``clear_outputs``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from tidyrun import get_job_states
from tidyrun.dag import DAG, ParametrizedJob
from tidyrun.job import Job
from tidyrun.plan import PlanPaths

pytest.importorskip("boto3")
pytest.importorskip("moto")

BUCKET = "tidyrun-test-bucket"


@pytest.fixture()
def s3_bucket(tmp_path: Path) -> Iterator[Any]:
    """A mocked S3 bucket; also runs the test from a scratch cwd so any
    accidentally created local path (e.g. ``s3:/``) is detected."""
    import boto3  # pyright: ignore[reportMissingImports]
    from moto import mock_aws  # pyright: ignore[reportMissingImports]

    previous_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            yield client
    finally:
        os.chdir(previous_cwd)


def _add(x: int, y: int) -> int:
    return x + y


def _double(dep: int) -> int:
    return dep * 2


def _scale(x: int, factor: int) -> int:
    return x * factor


def _sum_values(dep: Any) -> int:
    return sum(dep[k] for k in dep)


def test_execute_materialized_from_s3_string(s3_bucket: Any, tmp_path: Path) -> None:
    """An s3:// string plan location must not be interpreted as a local path."""
    dag = DAG()
    dag["a"] = Job(func=_add, kwargs={"x": 1, "y": 2})
    dag["b"] = Job(func=_double, kwargs={"dep": dag["a"]})

    target = f"s3://{BUCKET}/plans/run-001"
    dag.materialize(target)

    # Dependency links are uploaded as sidecar files (symlinks cannot exist on S3).
    keys = {
        obj["Key"]
        for obj in s3_bucket.list_objects_v2(Bucket=BUCKET, Prefix="plans/run-001")[
            "Contents"
        ]
    }
    assert "plans/run-001/inputs/b/dep.tidyrun" in keys

    result = dag.execute_materialized(target, execution_mode="thread")
    assert result.to_dict() == {"a": 3, "b": 6}

    # No stray local directory named after the URI scheme.
    assert not (tmp_path / "s3:").exists()


def test_evaluate_parametrized_dag_on_s3(s3_bucket: Any) -> None:
    """evaluate() with an s3:// destination materializes and executes on S3."""
    pjob = ParametrizedJob(
        func=_scale,
        parameter_names=["x"],
        parameter_values=[(v,) for v in (1, 2, 3)],
        kwargs={"factor": 10},
    )
    dag = DAG({"grid": pjob})
    dag["total"] = Job(func=_sum_values, kwargs={"dep": pjob})

    result = dag.evaluate(f"s3://{BUCKET}/plans/run-002", execution_mode="thread")
    result_dict = result.to_dict()
    assert result_dict["grid"] == {1: 10, 2: 20, 3: 30}
    assert result_dict["total"] == 60


def test_job_states_and_clear_outputs_on_s3(s3_bucket: Any) -> None:
    target = f"s3://{BUCKET}/plans/run-003"
    dag = DAG({"a": Job(func=_add, kwargs={"x": 1, "y": 2})})
    dag.materialize(target)

    assert get_job_states(target) == {"a": "pending"}
    dag.execute_materialized(target, execution_mode="thread")
    assert get_job_states(target) == {"a": "succeeded"}

    dag.clear_outputs(target, job_ids=["a"])
    assert get_job_states(target) == {"a": "pending"}


def test_runner_string_round_trip_uses_plain_root() -> None:
    """Standard-layout plans encode as the bare root (a path or s3:// URI)."""
    local = PlanPaths.from_root("/data/plan")
    assert local.to_runner_string() == "/data/plan"
    assert PlanPaths.from_runner_string(local.to_runner_string()) == local

    cloud = PlanPaths.from_root(f"s3://{BUCKET}/plans/run")
    assert cloud.to_runner_string() == f"s3://{BUCKET}/plans/run"
    assert PlanPaths.from_runner_string(cloud.to_runner_string()) == cloud

    custom = PlanPaths(
        definitions=Path("/data/defs"),
        inputs=Path("/scratch/inputs"),
        outputs=Path("/scratch/outputs"),
    )
    assert ":::" in custom.to_runner_string()
    assert PlanPaths.from_runner_string(custom.to_runner_string()) == custom
