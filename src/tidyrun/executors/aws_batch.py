from __future__ import annotations

from concurrent.futures import Executor, Future
from dataclasses import dataclass
import json
import re
import threading
import time
from typing import Any, Mapping, Optional, Protocol, cast


class _BatchClient(Protocol):
    def submit_job(self, **kwargs: Any) -> dict[str, Any]: ...

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _SubmittedJob:
    future: Future[Any]
    job_id: str


@dataclass(frozen=True)
class _ArraySubmission:
    future: Future[Any]
    job_ids: tuple[str, ...]


class AwsBatchExecutor(Executor):
    """AWS Batch-backed executor compatible with ``concurrent.futures.Executor``.

    This executor is designed for materialized DAG execution where each submitted
    task corresponds to one compiled job identified by ``(plan_dir, job_id)``.

    The container image must call ``tidyrun-batch-entrypoint`` (or
    :func:`tidyrun.batch_entrypoint`) as its ``CMD``. That command reads
    ``TIDYRUN_PLAN_DIR`` and ``TIDYRUN_JOB_ID`` (or
    ``TIDYRUN_JOB_IDS_JSON`` + ``AWS_BATCH_JOB_ARRAY_INDEX`` for array jobs)
    and runs the job.

    Args:
        job_queue: ARN or name of the AWS Batch job queue.
        job_definition: ARN or name (with optional revision) of the Batch job
            definition whose container image calls ``tidyrun-batch-entrypoint``.
        extra_env: Optional mapping of additional environment variables to
            inject into every submitted container. Use this to pass
            project-specific settings that the container needs before running
            the job — for example ``GIT_REPO_URL`` and ``GIT_COMMIT`` when the
            container checks out source code at runtime. These variables are
            added after the tidyrun-reserved ones and will not override them.
        poll_interval_seconds: Seconds between ``describe_jobs`` polls.
        region_name: AWS region; defaults to the boto3 session default.
        batch_client: Optional pre-built boto3 Batch client (useful for testing
            with ``moto`` or a custom endpoint).

    Note:
        Requires ``boto3`` at runtime. Install with ``pip install tidyrun[s3]``.
    """

    _TERMINAL_SUCCESS = {"SUCCEEDED"}
    _TERMINAL_FAILURE = {"FAILED"}

    def __init__(
        self,
        job_queue: str,
        job_definition: str,
        *,
        extra_env: dict[str, str] | None = None,
        poll_interval_seconds: float = 1.0,
        region_name: str | None = None,
        batch_client: _BatchClient | None = None,
    ) -> None:
        self._job_queue = job_queue
        self._job_definition = job_definition
        self._extra_env: dict[str, str] = dict(extra_env) if extra_env else {}
        self._poll_interval_seconds = poll_interval_seconds
        self._shutdown = False
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

        if batch_client is None:
            try:
                import boto3  # pyright: ignore[reportMissingTypeStubs]
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError(
                    "AwsBatchExecutor requires boto3. "
                    "Install with `pip install tidyrun[s3]`."
                ) from exc
            boto3_client_factory = cast(Any, boto3)
            self._batch_client = cast(
                _BatchClient,
                boto3_client_factory.client("batch", region_name=region_name),
            )
        else:
            self._batch_client = batch_client

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        del fn, kwargs
        return self._submit_internal(args, sbatch_options=None)

    def submit_with_options(
        self,
        fn: Any,
        /,
        *args: Any,
        sbatch_options: Mapping[str, str | int],
        **kwargs: Any,
    ) -> Future[Any]:
        del fn, kwargs
        return self._submit_internal(args, sbatch_options=sbatch_options)

    def submit_array_with_options(
        self,
        fn: Any,
        /,
        plan_dir: Any,
        job_ids: list[str] | tuple[str, ...],
        *,
        sbatch_options: Mapping[str, str | int],
    ) -> _ArraySubmission:
        del fn
        if self._shutdown:
            raise RuntimeError("Cannot submit new jobs after shutdown")
        if not job_ids:
            raise ValueError("job_ids must not be empty")

        normalized_job_ids = tuple(str(job_id) for job_id in job_ids)
        for job_id in normalized_job_ids:
            if not job_id:
                raise ValueError("job_ids must not contain empty job ids")

        plan_dir_str = str(plan_dir)
        job_ids_json = json.dumps(list(normalized_job_ids))
        parameters = {
            "tidyrun_plan_dir": plan_dir_str,
            "tidyrun_job_ids_json": job_ids_json,
            "tidyrun_job_id": normalized_job_ids[0],
        }
        parameters.update({k: str(v) for k, v in sbatch_options.items()})
        job_name = str(parameters.pop("job_name", normalized_job_ids[0]))

        array_env: list[dict[str, str]] = [
            {"name": "TIDYRUN_PLAN_DIR", "value": plan_dir_str},
            {"name": "TIDYRUN_JOB_IDS_JSON", "value": job_ids_json},
            {"name": "TIDYRUN_JOB_ID", "value": normalized_job_ids[0]},
        ]
        array_env.extend({"name": k, "value": v} for k, v in self._extra_env.items())

        submit_kwargs: dict[str, Any] = {
            "jobName": self._build_job_name(job_name),
            "jobQueue": self._job_queue,
            "jobDefinition": self._job_definition,
            "arrayProperties": {"size": len(normalized_job_ids)},
            "containerOverrides": {"environment": array_env},
            "parameters": parameters,
        }

        response = self._batch_client.submit_job(**submit_kwargs)
        batch_job_id = response.get("jobId")
        if not isinstance(batch_job_id, str):
            raise RuntimeError(f"Invalid AWS Batch submit response: {response!r}")

        future: Future[Any] = Future()
        worker = threading.Thread(
            target=self._watch_job,
            args=(_SubmittedJob(future=future, job_id=batch_job_id),),
            daemon=True,
        )
        with self._lock:
            self._threads.append(worker)
        worker.start()
        return _ArraySubmission(future=future, job_ids=normalized_job_ids)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        del cancel_futures
        self._shutdown = True
        if wait:
            with self._lock:
                threads = list(self._threads)
            for thread in threads:
                thread.join()

    def _submit_internal(
        self,
        args: tuple[Any, ...],
        *,
        sbatch_options: Mapping[str, str | int] | None,
    ) -> Future[Any]:
        if self._shutdown:
            raise RuntimeError("Cannot submit new jobs after shutdown")
        if len(args) < 2:
            raise ValueError("AwsBatchExecutor.submit expects args: (plan_dir, job_id)")

        plan_dir = str(args[0])
        job_id = str(args[1])

        env: list[dict[str, str]] = [
            {"name": "TIDYRUN_PLAN_DIR", "value": plan_dir},
            {"name": "TIDYRUN_JOB_ID", "value": job_id},
        ]
        env.extend({"name": k, "value": v} for k, v in self._extra_env.items())

        submit_kwargs: dict[str, Any] = {
            "jobName": self._build_job_name(job_id),
            "jobQueue": self._job_queue,
            "jobDefinition": self._job_definition,
            "containerOverrides": {"environment": env},
            "parameters": {
                "tidyrun_plan_dir": plan_dir,
                "tidyrun_job_id": job_id,
            },
        }
        if sbatch_options is not None:
            submit_kwargs["parameters"].update(
                {k: str(v) for k, v in sbatch_options.items()}
            )

        response = self._batch_client.submit_job(**submit_kwargs)
        batch_job_id = response.get("jobId")
        if not isinstance(batch_job_id, str):
            raise RuntimeError(f"Invalid AWS Batch submit response: {response!r}")

        future: Future[Any] = Future()
        worker = threading.Thread(
            target=self._watch_job,
            args=(_SubmittedJob(future=future, job_id=batch_job_id),),
            daemon=True,
        )
        with self._lock:
            self._threads.append(worker)
        worker.start()
        return future

    def _watch_job(self, submitted: _SubmittedJob) -> None:
        future = submitted.future
        while not self._shutdown and not future.done():
            try:
                response = self._batch_client.describe_jobs(jobs=[submitted.job_id])
                jobs = cast(list[dict[str, Any]], response.get("jobs", []))
                if not jobs:
                    raise RuntimeError(f"AWS Batch job not found: {submitted.job_id}")
                job = jobs[0]
                status = cast(Optional[str], job.get("status"))
                if status in self._TERMINAL_SUCCESS:
                    future.set_result(None)
                    return
                if status in self._TERMINAL_FAILURE:
                    reason = job.get("statusReason") or "Unknown AWS Batch failure"
                    future.set_exception(
                        RuntimeError(
                            f"AWS Batch job {submitted.job_id} failed: {reason}"
                        )
                    )
                    return
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                return
            time.sleep(self._poll_interval_seconds)

        if self._shutdown and not future.done():
            future.set_exception(RuntimeError("Executor shut down before completion"))

    @staticmethod
    def _build_job_name(job_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", job_id)
        if not cleaned:
            cleaned = "tidyrun-job"
        return cleaned[:128]
