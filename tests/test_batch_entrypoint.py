from __future__ import annotations

import json
from pathlib import Path

import pytest

from tidyrun import batch_entrypoint
from tidyrun.dag import DAG, job_output_exists
from tidyrun.job import Job


def _double(x: int) -> int:
    return x * 2


def _join(left: str, right: str) -> str:
    return f"{left}/{right}"


# ---------------------------------------------------------------------------
# batch_entrypoint: single-job path
# ---------------------------------------------------------------------------


def test_batch_entrypoint_runs_single_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_dir = tmp_path / "plan"
    DAG({"result": Job(func=_double, kwargs={"x": 21})}).materialize(plan_dir)

    monkeypatch.setenv("TIDYRUN_PLAN_DIR", str(plan_dir))
    monkeypatch.setenv("TIDYRUN_JOB_ID", "result")
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)

    batch_entrypoint()

    assert job_output_exists(plan_dir / "outputs", "result")


def test_batch_entrypoint_exits_when_plan_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TIDYRUN_PLAN_DIR", raising=False)
    monkeypatch.setenv("TIDYRUN_JOB_ID", "result")

    with pytest.raises(SystemExit) as exc_info:
        batch_entrypoint()
    assert exc_info.value.code != 0


def test_batch_entrypoint_exits_when_job_id_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_dir = tmp_path / "plan"
    DAG({"result": Job(func=_double, kwargs={"x": 1})}).materialize(plan_dir)

    monkeypatch.setenv("TIDYRUN_PLAN_DIR", str(plan_dir))
    monkeypatch.delenv("TIDYRUN_JOB_ID", raising=False)
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        batch_entrypoint()
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# batch_entrypoint: array-job path (AWS_BATCH_JOB_ARRAY_INDEX present)
# ---------------------------------------------------------------------------


def test_batch_entrypoint_runs_array_job_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tidyrun import ParametrizedJob

    plan_dir = tmp_path / "plan"
    DAG(
        {
            "grid": ParametrizedJob(
                func=_join,
                parameter_names=["left"],
                parameter_values=[("a",), ("b",), ("c",)],
                kwargs={"right": "x"},
            )
        }
    ).materialize(plan_dir)

    job_ids = ["grid/a", "grid/b", "grid/c"]

    monkeypatch.setenv("TIDYRUN_PLAN_DIR", str(plan_dir))
    monkeypatch.setenv("TIDYRUN_JOB_IDS_JSON", json.dumps(job_ids))
    monkeypatch.delenv("TIDYRUN_JOB_ID", raising=False)

    # Simulate Batch running child index 1 → should execute "grid/b"
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "1")
    batch_entrypoint()

    assert job_output_exists(plan_dir / "outputs", "grid/b")
    assert not job_output_exists(plan_dir / "outputs", "grid/a")
    assert not job_output_exists(plan_dir / "outputs", "grid/c")


def test_batch_entrypoint_exits_when_job_ids_json_missing_for_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_dir = tmp_path / "plan"
    DAG({"result": Job(func=_double, kwargs={"x": 1})}).materialize(plan_dir)

    monkeypatch.setenv("TIDYRUN_PLAN_DIR", str(plan_dir))
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "0")
    monkeypatch.delenv("TIDYRUN_JOB_IDS_JSON", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        batch_entrypoint()
    assert exc_info.value.code != 0


def test_batch_entrypoint_exits_when_array_index_out_of_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_dir = tmp_path / "plan"
    DAG({"result": Job(func=_double, kwargs={"x": 1})}).materialize(plan_dir)

    monkeypatch.setenv("TIDYRUN_PLAN_DIR", str(plan_dir))
    monkeypatch.setenv("AWS_BATCH_JOB_IDS_JSON", '["result"]')
    monkeypatch.setenv("TIDYRUN_JOB_IDS_JSON", '["result"]')
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "99")

    with pytest.raises(SystemExit) as exc_info:
        batch_entrypoint()
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# AwsBatchExecutor: extra_env is injected into every submission
# ---------------------------------------------------------------------------


def test_aws_batch_executor_extra_env_injected_into_single_job() -> None:
    from itertools import count
    from typing import Any

    from tidyrun import AwsBatchExecutor

    class _FakeClient:
        def __init__(self) -> None:
            self._counter = count(1)
            self._statuses: dict[str, list[str]] = {}
            self.submit_calls: list[dict[str, Any]] = []

        def submit_job(self, **kwargs: Any) -> dict[str, str]:
            job_id = f"job-{next(self._counter)}"
            self.submit_calls.append(dict(kwargs))
            self._statuses[job_id] = ["RUNNING", "SUCCEEDED"]
            return {"jobId": job_id}

        def describe_jobs(self, *, jobs: list[str]) -> dict[str, Any]:
            result = []
            for jid in jobs:
                seq = self._statuses[jid]
                status = seq[0]
                if len(seq) > 1:
                    self._statuses[jid] = seq[1:]
                result.append({"jobId": jid, "status": status})
            return {"jobs": result}

    client = _FakeClient()
    executor = AwsBatchExecutor(
        job_queue="q",
        job_definition="jd:1",
        batch_client=client,
        poll_interval_seconds=0.0,
        extra_env={
            "GIT_REPO_URL": "https://github.com/org/repo.git",
            "GIT_COMMIT": "abc1234",
        },
    )
    future = executor.submit(object(), "s3://bucket/plan", "job/a")
    future.result(timeout=2.0)
    executor.shutdown()

    env_by_name = {
        item["name"]: item["value"]
        for item in client.submit_calls[0]["containerOverrides"]["environment"]
    }
    assert env_by_name["GIT_REPO_URL"] == "https://github.com/org/repo.git"
    assert env_by_name["GIT_COMMIT"] == "abc1234"
    assert env_by_name["TIDYRUN_PLAN_DIR"] == "s3://bucket/plan"
    assert env_by_name["TIDYRUN_JOB_ID"] == "job/a"
