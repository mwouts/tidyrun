from __future__ import annotations

from itertools import count
from types import SimpleNamespace
from typing import Any

import pytest

from tidyrun import AwsBatchExecutor


class _FakeBatchClient:
    def __init__(self) -> None:
        self._counter = count(1)
        self._statuses: dict[str, list[str]] = {}
        self.submit_calls: list[dict[str, Any]] = []
        self.meta = SimpleNamespace(region_name="eu-west-1")

    def submit_job(self, **kwargs: Any) -> dict[str, str]:
        job_id = f"job-{next(self._counter)}"
        self.submit_calls.append(dict(kwargs))

        status_sequence = ["SUBMITTED", "RUNNING", "SUCCEEDED"]
        if kwargs.get("parameters", {}).get("force_fail") == "1":
            status_sequence = ["SUBMITTED", "FAILED"]
        self._statuses[job_id] = status_sequence
        return {"jobId": job_id}

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, list[dict[str, Any]]]:
        result: list[dict[str, Any]] = []
        for job_id in jobs:
            sequence = self._statuses[job_id]
            status = sequence[0]
            if len(sequence) > 1:
                self._statuses[job_id] = sequence[1:]
            payload: dict[str, Any] = {"jobId": job_id, "status": status}
            if status == "FAILED":
                payload["statusReason"] = "forced failure"
                payload["container"] = {
                    "reason": "OutOfMemoryError: Container killed",
                    "exitCode": 137,
                    "logStreamName": "job-def/default/abc123",
                }
            result.append(payload)
        return {"jobs": result}


def test_aws_batch_executor_submit_provides_plan_and_job_id() -> None:
    client = _FakeBatchClient()
    executor = AwsBatchExecutor(
        job_queue="queue",
        job_definition="job-def:1",
        batch_client=client,
        poll_interval_seconds=0.0,
    )

    future = executor.submit(object(), "s3://bucket/plan", "group/a")
    assert future.result(timeout=1.0) is None
    executor.shutdown()

    call = client.submit_calls[0]
    assert call["jobQueue"] == "queue"
    assert call["jobDefinition"] == "job-def:1"
    env = call["containerOverrides"]["environment"]
    env_by_name = {item["name"]: item["value"] for item in env}
    assert env_by_name["TIDYRUN_PLAN_DIR"] == "s3://bucket/plan"
    assert env_by_name["TIDYRUN_JOB_ID"] == "group/a"
    assert call["jobName"] == "group-a"


def test_aws_batch_executor_submit_with_options_maps_parameters() -> None:
    client = _FakeBatchClient()
    executor = AwsBatchExecutor(
        job_queue="queue",
        job_definition="job-def:1",
        batch_client=client,
        poll_interval_seconds=0.0,
    )

    future = executor.submit_with_options(
        object(),
        "s3://bucket/plan",
        "group/b",
        sbatch_options={"vcpus": 4, "queue_hint": "high"},
    )
    assert future.result(timeout=1.0) is None
    executor.shutdown()

    call = client.submit_calls[0]
    parameters = call["parameters"]
    assert parameters["tidyrun_plan_dir"] == "s3://bucket/plan"
    assert parameters["tidyrun_job_id"] == "group/b"
    assert parameters["vcpus"] == "4"
    assert parameters["queue_hint"] == "high"


def test_aws_batch_executor_propagates_batch_failure() -> None:
    client = _FakeBatchClient()
    executor = AwsBatchExecutor(
        job_queue="queue",
        job_definition="job-def:1",
        batch_client=client,
        poll_interval_seconds=0.0,
    )

    future = executor.submit_with_options(
        object(),
        "s3://bucket/plan",
        "group/c",
        sbatch_options={"force_fail": 1},
    )
    with pytest.raises(RuntimeError, match="forced failure"):
        future.result(timeout=1.0)
    executor.shutdown()


def test_aws_batch_failure_message_links_to_cloudwatch_logs() -> None:
    """A failed job's error names the log stream and links to the console."""
    client = _FakeBatchClient()
    executor = AwsBatchExecutor(
        job_queue="queue",
        job_definition="job-def:1",
        batch_client=client,
        poll_interval_seconds=0.0,
    )

    future = executor.submit_with_options(
        object(),
        "s3://bucket/plan",
        "group/c",
        sbatch_options={"force_fail": 1},
    )
    with pytest.raises(RuntimeError) as exc_info:
        future.result(timeout=1.0)
    executor.shutdown()

    message = str(exc_info.value)
    assert "Container reason: OutOfMemoryError: Container killed" in message
    assert "Exit code: 137" in message
    assert "Log stream: /aws/batch/job/job-def/default/abc123" in message
    assert (
        "https://eu-west-1.console.aws.amazon.com/cloudwatch/home?region=eu-west-1"
        "#logsV2:log-groups/log-group/$252Faws$252Fbatch$252Fjob"
        "/log-events/job-def$252Fdefault$252Fabc123"
    ) in message


class _ArrayFailBatchClient:
    """Array parent fails; child 0 failed with a log stream, child 1 succeeded."""

    def __init__(self) -> None:
        self.meta = SimpleNamespace(region_name="us-east-1")
        self._parent_statuses = ["SUBMITTED", "FAILED"]

    def submit_job(self, **kwargs: Any) -> dict[str, str]:
        del kwargs
        return {"jobId": "array-1"}

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, list[dict[str, Any]]]:
        if jobs == ["array-1"]:
            status = (
                self._parent_statuses.pop(0)
                if len(self._parent_statuses) > 1
                else self._parent_statuses[0]
            )
            job: dict[str, Any] = {"jobId": "array-1", "status": status}
            if status == "FAILED":
                job["statusReason"] = "Array child failed"
                job["arrayProperties"] = {"size": 2}
            return {"jobs": [job]}
        assert jobs == ["array-1:0", "array-1:1"]
        return {
            "jobs": [
                {
                    "jobId": "array-1:0",
                    "status": "FAILED",
                    "statusReason": "Essential container exited",
                    "container": {"logStreamName": "job-def/default/child0"},
                },
                {"jobId": "array-1:1", "status": "SUCCEEDED"},
            ]
        }


def test_aws_batch_array_failure_links_to_failed_child_logs() -> None:
    client = _ArrayFailBatchClient()
    executor = AwsBatchExecutor(
        job_queue="queue",
        job_definition="job-def:1",
        batch_client=client,  # pyright: ignore[reportArgumentType]
        poll_interval_seconds=0.0,
    )

    submission = executor.submit_array_with_options(
        object(),
        "s3://bucket/plan",
        ["grid/a", "grid/b"],
        sbatch_options={"job_name": "grid"},
    )
    with pytest.raises(RuntimeError) as exc_info:
        submission.future.result(timeout=1.0)
    executor.shutdown()

    message = str(exc_info.value)
    assert "Failed array child array-1:0: Essential container exited" in message
    assert "Log stream: /aws/batch/job/job-def/default/child0" in message
    assert (
        "https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1"
        "#logsV2:log-groups/log-group/$252Faws$252Fbatch$252Fjob"
        "/log-events/job-def$252Fdefault$252Fchild0"
    ) in message
    assert "array-1:1" not in message


def test_aws_batch_executor_submit_array_with_options() -> None:
    client = _FakeBatchClient()
    executor = AwsBatchExecutor(
        job_queue="queue",
        job_definition="job-def:1",
        batch_client=client,
        poll_interval_seconds=0.0,
    )

    submission = executor.submit_array_with_options(
        object(),
        "s3://bucket/plan",
        ["group/a", "group/b", "group/c"],
        sbatch_options={"job_name": "group", "queue_hint": "high"},
    )
    assert submission.future.result(timeout=1.0) is None
    executor.shutdown()

    call = client.submit_calls[0]
    assert call["arrayProperties"] == {"size": 3}
    assert call["jobName"] == "group"
    env = call["containerOverrides"]["environment"]
    env_by_name = {item["name"]: item["value"] for item in env}
    assert env_by_name["TIDYRUN_PLAN_DIR"] == "s3://bucket/plan"
    assert env_by_name["TIDYRUN_JOB_ID"] == "group/a"
    assert env_by_name["TIDYRUN_JOB_IDS_JSON"] == '["group/a", "group/b", "group/c"]'

    parameters = call["parameters"]
    assert parameters["tidyrun_plan_dir"] == "s3://bucket/plan"
    assert parameters["tidyrun_job_id"] == "group/a"
    assert parameters["tidyrun_job_ids_json"] == '["group/a", "group/b", "group/c"]'
    assert parameters["queue_hint"] == "high"

    assert submission.job_ids == ("group/a", "group/b", "group/c")
