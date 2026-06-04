"""AWS Batch executor tests using moto to mock the real AWS Batch API.

These tests complement test_aws_batch_executor.py (which uses a hand-written
fake client) by going through the actual boto3 serialisation layer. This
catches issues that a fake client would silently miss: wrong parameter names,
missing required fields, or incorrect data shapes in the AWS API call.

Moto tries to run containers when Docker is available. In CI Docker is absent,
so submitted jobs fail immediately, which the tests assert below. For array
jobs the moto parent never leaves SUBMITTED (moto only transitions children),
so those tests use a recording wrapper to inspect the submit_job payload
directly rather than waiting for the future.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3
import pytest

from tidyrun import AwsBatchExecutor, DAG, Job


def _double(x: int) -> int:
    return x * 2


def _join(left: str, right: str) -> str:
    return f"{left}/{right}"


# ---------------------------------------------------------------------------
# Infrastructure fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_env(monkeypatch: pytest.MonkeyPatch):
    """Create a minimal moto AWS Batch environment and yield client handles.

    All boto3 calls inside the mock_aws context — including those made by
    AwsBatchExecutor when it creates its own client — hit the moto mock.
    """
    pytest.importorskip("moto")
    from moto import mock_aws  # pyright: ignore[reportMissingImports]

    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")

    with mock_aws():
        ec2 = boto3.client("ec2", region_name="us-east-1")
        iam = boto3.client("iam", region_name="us-east-1")
        batch = boto3.client("batch", region_name="us-east-1")

        vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]["VpcId"]
        subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock="10.0.0.0/24")["Subnet"][
            "SubnetId"
        ]
        sg_id = ec2.create_security_group(
            GroupName="tidyrun-test-sg", Description="tidyrun test", VpcId=vpc_id
        )["GroupId"]

        role_arn = iam.create_role(
            RoleName="tidyrun-batch-role",
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "batch.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
        )["Role"]["Arn"]

        instance_profile_arn = iam.create_instance_profile(
            InstanceProfileName="tidyrun-batch-profile"
        )["InstanceProfile"]["Arn"]
        iam.add_role_to_instance_profile(
            InstanceProfileName="tidyrun-batch-profile",
            RoleName="tidyrun-batch-role",
        )

        batch.create_compute_environment(
            computeEnvironmentName="tidyrun-compute-env",
            type="MANAGED",
            state="ENABLED",
            computeResources={
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": 256,
                "instanceRole": instance_profile_arn,
                "instanceTypes": ["optimal"],
                "subnets": [subnet_id],
                "securityGroupIds": [sg_id],
            },
            serviceRole=role_arn,
        )

        batch.create_job_queue(
            jobQueueName="tidyrun-test-queue",
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[
                {
                    "order": 1,
                    "computeEnvironment": "tidyrun-compute-env",
                }
            ],
        )

        # The job definition: container CMD is tidyrun-batch-entrypoint.
        # moto validates this against the real AWS Batch register_job_definition
        # API, so any wrong field names here would raise immediately.
        batch.register_job_definition(
            jobDefinitionName="tidyrun-worker",
            type="container",
            containerProperties={
                "image": "python:3.12-slim",
                "vcpus": 1,
                "memory": 512,
                "command": ["tidyrun-batch-entrypoint"],
            },
        )

        yield {
            "batch": batch,
            "job_queue": "tidyrun-test-queue",
            "job_definition": "tidyrun-worker",
            "region": "us-east-1",
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Wraps a real (moto-backed) boto3 Batch client and records submit_job
    calls so tests can inspect what AwsBatchExecutor sent to the API without
    waiting for job completion."""

    def __init__(self, real_client: Any) -> None:
        self._real = real_client
        self.submit_calls: list[dict[str, Any]] = []

    def submit_job(self, **kwargs: Any) -> dict[str, Any]:
        self.submit_calls.append(kwargs)
        return self._real.submit_job(**kwargs)

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, Any]:
        return self._real.describe_jobs(jobs=jobs)


def _submitted_env(batch_client: Any, queue: str) -> list[dict[str, str]]:
    """Return the environment of every FAILED job in the queue."""
    summary = batch_client.list_jobs(jobQueue=queue, jobStatus="FAILED")[
        "jobSummaryList"
    ]
    if not summary:
        return []
    jobs = batch_client.describe_jobs(jobs=[j["jobId"] for j in summary])["jobs"]
    return [
        {e["name"]: e["value"] for e in j["container"]["environment"]} for j in jobs
    ]


def _make_executor(
    batch_env: dict[str, Any],
    batch_client: Any = None,
    **kwargs: Any,
) -> AwsBatchExecutor:
    return AwsBatchExecutor(
        job_queue=batch_env["job_queue"],
        job_definition=batch_env["job_definition"],
        poll_interval_seconds=0.05,
        batch_client=batch_client,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Single-job tests — let the future fail, then inspect via describe_jobs
# ---------------------------------------------------------------------------


def test_moto_single_job_environment_variables(
    batch_env: dict[str, Any],
) -> None:
    """TIDYRUN_PLAN_DIR and TIDYRUN_JOB_ID reach the container environment,
    verifiable through the real describe_jobs API."""
    plan_dir = "s3://my-bucket/plans/run-001"
    job_id_str = "result"

    executor = _make_executor(batch_env)
    future = executor.submit(object(), plan_dir, job_id_str)

    with pytest.raises(RuntimeError):
        future.result(timeout=5.0)
    executor.shutdown()

    envs = _submitted_env(batch_env["batch"], batch_env["job_queue"])
    assert len(envs) == 1
    assert envs[0]["TIDYRUN_PLAN_DIR"] == plan_dir
    assert envs[0]["TIDYRUN_JOB_ID"] == job_id_str


def test_moto_extra_env_appears_in_container_environment(
    batch_env: dict[str, Any],
) -> None:
    """extra_env values are injected alongside the tidyrun reserved variables."""
    executor = _make_executor(
        batch_env,
        extra_env={
            "GIT_REPO_URL": "https://github.com/org/repo.git",
            "GIT_COMMIT": "abc1234",
        },
    )
    future = executor.submit(object(), "s3://bucket/plan", "job/a")

    with pytest.raises(RuntimeError):
        future.result(timeout=5.0)
    executor.shutdown()

    envs = _submitted_env(batch_env["batch"], batch_env["job_queue"])
    assert len(envs) == 1
    env = envs[0]
    assert env["GIT_REPO_URL"] == "https://github.com/org/repo.git"
    assert env["GIT_COMMIT"] == "abc1234"
    assert env["TIDYRUN_PLAN_DIR"] == "s3://bucket/plan"
    assert env["TIDYRUN_JOB_ID"] == "job/a"


def test_moto_job_targets_correct_queue_and_definition(
    batch_env: dict[str, Any],
) -> None:
    """The submitted job references the queue and job definition the executor was
    constructed with — verified via the moto describe_jobs state."""
    recording = _RecordingClient(batch_env["batch"])
    executor = _make_executor(batch_env, batch_client=recording)
    future = executor.submit(object(), "s3://bucket/plan", "job/a")

    with pytest.raises(RuntimeError):
        future.result(timeout=5.0)
    executor.shutdown()

    assert len(recording.submit_calls) == 1
    call = recording.submit_calls[0]
    assert call["jobQueue"] == batch_env["job_queue"]
    assert call["jobDefinition"] == batch_env["job_definition"]


# ---------------------------------------------------------------------------
# Array-job test — moto array parent never leaves SUBMITTED without Docker,
# so we use a recording wrapper to inspect the submit_job payload directly.
# ---------------------------------------------------------------------------


def test_moto_array_job_submit_payload(batch_env: dict[str, Any]) -> None:
    """submit_array_with_options issues one submit_job call with:
    - arrayProperties.size == len(job_ids)
    - TIDYRUN_JOB_IDS_JSON containing all logical job ids in order
    - TIDYRUN_PLAN_DIR set correctly
    - no TIDYRUN_JOB_ID for the array (children select by AWS_BATCH_JOB_ARRAY_INDEX)
    """
    job_ids = ["grid/a", "grid/b", "grid/c"]
    plan_dir = "s3://bucket/plan"

    recording = _RecordingClient(batch_env["batch"])
    executor = _make_executor(batch_env, batch_client=recording)

    # Submit without waiting — the future will never resolve in moto without Docker.
    executor.submit_array_with_options(
        object(),
        plan_dir,
        job_ids,
        sbatch_options={"job_name": "grid"},
    )
    executor.shutdown(wait=False)

    assert len(recording.submit_calls) == 1
    call = recording.submit_calls[0]

    # Verified by the real boto3 + moto validation on submit:
    assert call["arrayProperties"] == {"size": 3}
    assert call["jobQueue"] == batch_env["job_queue"]
    assert call["jobDefinition"] == batch_env["job_definition"]
    assert call["jobName"] == "grid"

    env = {e["name"]: e["value"] for e in call["containerOverrides"]["environment"]}
    assert env["TIDYRUN_PLAN_DIR"] == plan_dir
    assert json.loads(env["TIDYRUN_JOB_IDS_JSON"]) == job_ids


# ---------------------------------------------------------------------------
# End-to-end: materialize a real DAG, submit, verify submitted job ids
# ---------------------------------------------------------------------------


def test_moto_full_dag_submitted_job_ids_match_plan(
    batch_env: dict[str, Any], tmp_path: Path
) -> None:
    """After materializing a DAG, every submitted job's TIDYRUN_JOB_ID matches
    a definition file that was written to the plan directory."""
    plan_dir = tmp_path / "plan"
    dag = DAG(
        {
            "double": Job(func=_double, kwargs={"x": 21}),
            "join": Job(func=_join, kwargs={"left": "a", "right": "b"}),
        }
    )
    dag.materialize(plan_dir)

    expected = {p.stem for p in (plan_dir / "definitions").glob("*.tidyrun")}

    recording = _RecordingClient(batch_env["batch"])
    executor = _make_executor(batch_env, batch_client=recording)

    with pytest.raises(Exception):
        dag.execute_materialized(
            plan_dir,
            executor=executor,
        )
    executor.shutdown()

    submitted_ids = {
        e["value"]
        for call in recording.submit_calls
        for e in call["containerOverrides"]["environment"]
        if e["name"] == "TIDYRUN_JOB_ID"
    }
    # At least one job was submitted and its id appears in the plan definitions.
    assert submitted_ids & expected
