"""The AWS-Batch container flow against a real S3 API (moto server).

Unlike the in-process moto tests, these run ``tidyrun-batch-entrypoint`` in a
fresh subprocess — like a Batch container — with only the environment
variables AWS Batch would set. This pins the full contract: the plan
materialized to S3 must be visible to the batch runner, and the plan location
submitted by the scheduler must be the plain ``s3://`` root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest

pytest.importorskip("boto3")
pytest.importorskip("moto.server")  # needs the moto[server] extra (flask)

from tidyrun import deserialize  # noqa: E402
from tidyrun.dag import DAG, ParametrizedJob  # noqa: E402
from tidyrun.job import Job  # noqa: E402

BUCKET = "tidyrun-batch-test-bucket"
_TESTS_DIR = str(Path(__file__).parent)


@pytest.fixture()
def s3_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A moto S3 *server* (reachable over HTTP from subprocesses)."""
    import boto3  # pyright: ignore[reportMissingImports]
    from cloudpathlib import S3Client
    from moto.server import ThreadedMotoServer  # pyright: ignore[reportMissingImports]

    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
    # The endpoint is local; make sure no proxy configuration intercepts it.
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")

    boto3.client("s3", endpoint_url=endpoint).create_bucket(Bucket=BUCKET)
    # Pin the in-process cloudpathlib default client to this endpoint; other
    # tests may have cached one pointing elsewhere.
    client = S3Client(endpoint_url=endpoint)
    client.set_as_default_client()
    try:
        yield endpoint
    finally:
        S3Client._default_client = None  # pyright: ignore[reportPrivateUsage]
        server.stop()


def _run_container(env_overrides: Mapping[str, str]) -> None:
    """Run the batch entrypoint as AWS Batch would: a fresh process that only
    receives the plan location and job identity via environment variables."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_TESTS_DIR, *[p for p in sys.path if p]])
    env.update(env_overrides)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tidyrun import batch_entrypoint; batch_entrypoint()",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"batch container failed for {env_overrides}:\n{proc.stdout}\n{proc.stderr}"
        )


def _add(x: int, y: int) -> int:
    return x + y


def _double(dep: int) -> int:
    return dep * 2


def _scale(x: int, factor: int) -> int:
    return x * factor


def _total(dep: Any) -> int:
    return sum(dep[k] for k in dep)


def test_batch_entrypoint_reads_plan_from_s3(s3_server: str) -> None:
    """A containerized batch runner must see a plan materialized to S3."""
    dag = DAG()
    dag["a"] = Job(func=_add, kwargs={"x": 1, "y": 2})
    dag["b"] = Job(func=_double, kwargs={"dep": dag["a"]})
    target = f"s3://{BUCKET}/plans/entrypoint-run"
    dag.materialize(target)

    for job_id in ["a", "b"]:  # dependency order
        _run_container({"TIDYRUN_PLAN_DIR": target, "TIDYRUN_JOB_ID": job_id})

    outputs = deserialize(f"{target}/outputs")
    assert outputs["a"] == 3
    assert outputs["b"] == 6


class _BatchLikeExecutor(Executor):
    """Mimics AwsBatchExecutor's submission contract, running each submission
    as a local subprocess with the environment AWS Batch would set."""

    def __init__(self) -> None:
        self.submitted_plan_dirs: list[str] = []

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        del fn, kwargs
        plan_dir, job_id = str(args[0]), str(args[1])
        self.submitted_plan_dirs.append(plan_dir)
        future: Future[Any] = Future()
        try:
            _run_container({"TIDYRUN_PLAN_DIR": plan_dir, "TIDYRUN_JOB_ID": job_id})
            future.set_result(None)
        except Exception as exc:
            future.set_exception(exc)
        return future

    def submit_array_with_options(
        self,
        fn: Any,
        /,
        plan_dir: Any,
        job_ids: list[str] | tuple[str, ...],
        *,
        sbatch_options: Mapping[str, str | int],
    ) -> Any:
        del fn, sbatch_options
        plan_dir_str = str(plan_dir)
        ids = tuple(str(job_id) for job_id in job_ids)
        self.submitted_plan_dirs.append(plan_dir_str)
        future: Future[Any] = Future()
        try:
            for index in range(len(ids)):
                _run_container(
                    {
                        "TIDYRUN_PLAN_DIR": plan_dir_str,
                        "TIDYRUN_JOB_IDS_JSON": json.dumps(list(ids)),
                        "AWS_BATCH_JOB_ARRAY_INDEX": str(index),
                    }
                )
            future.set_result(None)
        except Exception as exc:
            future.set_exception(exc)

        class _Submission:
            pass

        submission = _Submission()
        submission.future = future  # pyright: ignore[reportAttributeAccessIssue]
        submission.job_ids = ids  # pyright: ignore[reportAttributeAccessIssue]
        return submission


def test_execute_materialized_drives_batch_containers_on_s3(s3_server: str) -> None:
    """The full loop: execute_materialized submits the plain s3:// plan root,
    array children and dependents run in containers, results assemble on S3."""
    pjob = ParametrizedJob(
        func=_scale,
        parameter_names=["x"],
        parameter_values=[(v,) for v in (1, 2)],
        kwargs={"factor": 10},
    )
    dag = DAG({"grid": pjob})
    # Sorts before "grid": probes scheduling around the inline aggregator.
    dag["a_total"] = Job(func=_total, kwargs={"dep": pjob})

    target = f"s3://{BUCKET}/plans/full-run"
    dag.materialize(target)

    executor = _BatchLikeExecutor()
    result = dag.execute_materialized(target, executor=executor)

    assert result.to_dict() == {"grid": {1: 10, 2: 20}, "a_total": 30}
    # Containers must receive the plain plan root, not an internal encoding.
    assert set(executor.submitted_plan_dirs) == {target}
