from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path
import os
import pickle
import subprocess
import sys
import threading
import time
from typing import Any

SbatchOptions = Mapping[str, str | int]

_RUNNER_SCRIPT_NAME = "_tidyrun_slurm_runner.py"
_RUNNER_SCRIPT = """import os
import pickle
import sys
import traceback


def main() -> int:
    task_path = sys.argv[1]
    result_path = sys.argv[2]
    error_path = sys.argv[3]

    try:
        with open(task_path, "rb") as fp:
            payload = pickle.load(fp)

        if isinstance(payload, dict) and payload.get("mode") == "array":
            calls = payload.get("calls")
            if not isinstance(calls, list):
                raise RuntimeError("Invalid array payload: calls must be a list")
            index_text = os.environ.get("SLURM_ARRAY_TASK_ID")
            if index_text is None:
                raise RuntimeError("Missing SLURM_ARRAY_TASK_ID for array task")
            index = int(index_text)
            if index < 0 or index >= len(calls):
                raise RuntimeError(
                    f"Array index out of bounds: {index} not in [0, {len(calls)})"
                )
            # SLURM only expands %a in --output, not in script arguments.
            result_path = result_path.replace("%a", index_text)
            error_path = error_path.replace("%a", index_text)
            call = calls[index]
            if not isinstance(call, tuple) or len(call) != 3:
                raise RuntimeError("Invalid array call payload")
            func, args, kwargs = call
        else:
            func, args, kwargs = payload

        result = func(*args, **kwargs)
        with open(result_path, "wb") as fp:
            pickle.dump(result, fp)
        return 0
    except Exception:
        with open(error_path, "w", encoding="utf-8") as fp:
            fp.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


@dataclass(frozen=True)
class _ArraySubmission:
    future: Future[Any]
    job_ids: tuple[str, ...]


class SlurmExecutor(Executor):
    """A SLURM-backed executor compatible with ``concurrent.futures.Executor``.

    This executor serializes callables and arguments to a shared directory,
    submits them through ``sbatch``, polls job completion via ``squeue``, and
    returns results through ``Future`` objects.

    Notes:
    - Submitted callables and arguments must be pickle-serializable.
    - ``shared_dir`` must be visible from SLURM compute nodes.
        - Resource limits can be passed as dedicated arguments (``time_limit``,
            ``memory``, ``cpus_per_task``, etc.) or through ``sbatch_options``.
    """

    def __init__(
        self,
        shared_dir: str | os.PathLike[str],
        *,
        poll_interval_seconds: float = 1.0,
        partition: str | None = None,
        qos: str | None = None,
        account: str | None = None,
        constraint: str | None = None,
        time_limit: str | None = None,
        memory: str | int | None = None,
        cpus_per_task: int | None = None,
        gres: str | None = None,
        sbatch_options: Mapping[str, str | int] | None = None,
        cleanup_files: bool = True,
    ) -> None:
        self._shared_dir = Path(shared_dir)
        self._shared_dir.mkdir(parents=True, exist_ok=True)
        self._poll_interval_seconds = poll_interval_seconds
        resource_options: dict[str, str | int] = {}
        if partition is not None:
            resource_options["partition"] = partition
        if qos is not None:
            resource_options["qos"] = qos
        if account is not None:
            resource_options["account"] = account
        if constraint is not None:
            resource_options["constraint"] = constraint
        if time_limit is not None:
            resource_options["time"] = time_limit
        if memory is not None:
            resource_options["mem"] = memory
        if cpus_per_task is not None:
            resource_options["cpus_per_task"] = cpus_per_task
        if gres is not None:
            resource_options["gres"] = gres

        # sbatch_options takes precedence for advanced overrides.
        self._sbatch_options = {**resource_options, **dict(sbatch_options or {})}
        self._cleanup_files = cleanup_files
        self._shutdown = False
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._runner_script = self._ensure_runner_script()

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        if self._shutdown:
            raise RuntimeError("Cannot submit new jobs after shutdown")

        future: Future[Any] = Future()
        worker = threading.Thread(
            target=self._run_task,
            args=(future, fn, args, kwargs, None),
            daemon=True,
        )
        with self._lock:
            self._threads.append(worker)
        worker.start()
        return future

    def submit_with_options(
        self,
        fn: Any,
        /,
        *args: Any,
        sbatch_options: SbatchOptions,
        **kwargs: Any,
    ) -> Future[Any]:
        """Submit one task with per-task sbatch option overrides."""
        if self._shutdown:
            raise RuntimeError("Cannot submit new jobs after shutdown")

        future: Future[Any] = Future()
        worker = threading.Thread(
            target=self._run_task,
            args=(future, fn, args, kwargs, dict(sbatch_options)),
            daemon=True,
        )
        with self._lock:
            self._threads.append(worker)
        worker.start()
        return future

    def submit_array_with_options(
        self,
        fn: Any,
        /,
        plan_dir: Any,
        job_ids: list[str] | tuple[str, ...],
        *,
        sbatch_options: SbatchOptions,
    ) -> _ArraySubmission:
        """Submit a group of jobs as one SLURM array and return one future."""
        if self._shutdown:
            raise RuntimeError("Cannot submit new jobs after shutdown")
        if not job_ids:
            raise ValueError("job_ids must not be empty")

        normalized_job_ids = tuple(str(job_id) for job_id in job_ids)
        for job_id in normalized_job_ids:
            if not job_id:
                raise ValueError("job_ids must not contain empty job ids")

        future: Future[Any] = Future()
        worker = threading.Thread(
            target=self._run_array,
            args=(future, fn, str(plan_dir), normalized_job_ids, dict(sbatch_options)),
            daemon=True,
        )
        with self._lock:
            self._threads.append(worker)
        worker.start()
        return _ArraySubmission(future=future, job_ids=normalized_job_ids)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._shutdown = True
        if cancel_futures:
            # Pending cancellation is best-effort; in-flight SLURM jobs are not
            # cancelled here because this executor does not track job IDs per
            # future after submission.
            pass
        if wait:
            with self._lock:
                threads = list(self._threads)
            for thread in threads:
                thread.join()

    def _run_task(
        self,
        future: Future[Any],
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        sbatch_options_override: dict[str, str | int] | None,
    ) -> None:
        if not future.set_running_or_notify_cancel():
            return

        assert len(args) >= 2, (
            "Expected at least 2 positional arguments for job ID inference"
        )
        assert isinstance(args[1], str), (
            "Expected second positional argument to be job ID string"
        )
        job_id = args[1]
        task_path = self._shared_dir / f"{job_id}.task.pickle"
        result_path = self._shared_dir / f"{job_id}.result.pickle"
        error_path = self._shared_dir / f"{job_id}.error.txt"
        stdout_path = self._shared_dir / f"{job_id}.slurm.out"
        task_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with task_path.open("wb") as fp:
                pickle.dump((fn, args, kwargs), fp)

            sbatch_cmd = self._build_sbatch_command(
                task_path=task_path,
                result_path=result_path,
                error_path=error_path,
                stdout_path=stdout_path,
                sbatch_options=sbatch_options_override,
                job_name=self._infer_job_name(
                    args=args, sbatch_options=sbatch_options_override
                ),
            )
            completed = subprocess.run(
                sbatch_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            slurm_job_id = self._parse_job_id(completed.stdout)
            self._wait_for_job_completion(slurm_job_id)

            if result_path.is_file():
                with result_path.open("rb") as fp:
                    result = pickle.load(fp)
                future.set_result(result)
                return

            if error_path.is_file():
                message = error_path.read_text(encoding="utf-8")
                future.set_exception(RuntimeError(message))
                return

            future.set_exception(
                RuntimeError(
                    "SLURM job finished without producing a result or error file"
                )
            )
        except Exception as exc:
            future.set_exception(exc)
        finally:
            if self._cleanup_files:
                for path in [task_path, result_path, error_path, stdout_path]:
                    if path.exists():
                        path.unlink()

    def _run_array(
        self,
        future: Future[Any],
        fn: Any,
        plan_dir: str,
        job_ids: tuple[str, ...],
        sbatch_options_override: dict[str, str | int],
    ) -> None:
        if not future.set_running_or_notify_cancel():
            return

        array_group = job_ids[0].rsplit("/", 1)[0] or job_ids[0]
        task_path = self._shared_dir / f"{array_group}.array.task.pickle"
        result_template = self._shared_dir / f"{array_group}.result.%a.pickle"
        error_template = self._shared_dir / f"{array_group}.error.%a.txt"
        stdout_template = self._shared_dir / f"{array_group}.slurm.%a.out"
        task_path.parent.mkdir(parents=True, exist_ok=True)

        result_paths = [
            self._shared_dir / f"{array_group}.result.{index}.pickle"
            for index, _ in enumerate(job_ids)
        ]
        error_paths = [
            self._shared_dir / f"{array_group}.error.{index}.txt"
            for index, _ in enumerate(job_ids)
        ]
        stdout_paths = [
            self._shared_dir / f"{array_group}.slurm.{index}.out"
            for index, _ in enumerate(job_ids)
        ]

        try:
            calls: list[tuple[Any, tuple[str, str], dict[str, Any]]] = [
                (fn, (plan_dir, job_id), {}) for job_id in job_ids
            ]
            with task_path.open("wb") as fp:
                pickle.dump({"mode": "array", "calls": calls}, fp)

            job_name_value = sbatch_options_override.pop("job_name", job_ids[0])

            sbatch_cmd = self._build_sbatch_command(
                task_path=task_path,
                result_path=result_template,
                error_path=error_template,
                stdout_path=stdout_template,
                sbatch_options=sbatch_options_override,
                job_name=str(job_name_value),
                array=f"0-{len(job_ids) - 1}",
            )
            completed = subprocess.run(
                sbatch_cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = self._parse_job_id(completed.stdout)
            self._wait_for_job_completion(job_id)

            failures: list[str] = []
            for logical_job_id, result_path, error_path in zip(
                job_ids, result_paths, error_paths
            ):
                if error_path.is_file():
                    message = error_path.read_text(encoding="utf-8")
                    failures.append(f"{logical_job_id}: {message}")
                    continue
                if not result_path.is_file():
                    failures.append(
                        f"{logical_job_id}: missing result and error file after array run"
                    )

            if failures:
                future.set_exception(RuntimeError("\n\n".join(failures)))
            else:
                future.set_result(None)
        except Exception as exc:
            future.set_exception(exc)
        finally:
            if self._cleanup_files:
                for path in [
                    task_path,
                    *result_paths,
                    *error_paths,
                    *stdout_paths,
                ]:
                    if path.exists():
                        path.unlink()

    def _build_sbatch_command(
        self,
        *,
        task_path: Path,
        result_path: Path,
        error_path: Path,
        stdout_path: Path,
        sbatch_options: SbatchOptions | None = None,
        job_name: str | None = None,
        array: str | None = None,
    ) -> list[str]:
        cmd = ["sbatch", "--parsable", "--output", str(stdout_path)]
        merged_options = {**self._sbatch_options, **dict(sbatch_options or {})}
        if job_name:
            merged_options.setdefault("job_name", job_name)
        if array:
            merged_options.setdefault("array", array)
        for key, value in merged_options.items():
            option = key if key.startswith("-") else f"--{key.replace('_', '-')}"
            cmd.extend([option, str(value)])

        cmd.extend(
            [
                str(self._runner_script),
                str(task_path),
                str(result_path),
                str(error_path),
            ]
        )
        return cmd

    @staticmethod
    def _infer_job_name(
        *,
        args: tuple[Any, ...],
        sbatch_options: Mapping[str, str | int] | None,
    ) -> str | None:
        if sbatch_options is not None:
            for key in ("job_name", "job-name", "--job-name"):
                value = sbatch_options.get(key)
                if value is not None:
                    return str(value)
        if len(args) >= 2 and isinstance(args[1], str) and args[1]:
            return args[1]
        return None

    def _wait_for_job_completion(self, job_id: str) -> None:
        while True:
            # squeue exits with 0 and empty output when job is no longer queued.
            check = subprocess.run(
                ["squeue", "-h", "-j", job_id],
                check=False,
                capture_output=True,
                text=True,
            )
            if check.stdout.strip() == "":
                return
            if self._poll_interval_seconds > 0:
                time.sleep(self._poll_interval_seconds)

    def _parse_job_id(self, stdout: str) -> str:
        token = stdout.strip().splitlines()[0]
        # --parsable may return forms like "12345" or "12345;cluster".
        return token.split(";", maxsplit=1)[0]

    def _ensure_runner_script(self) -> Path:
        path = self._shared_dir / _RUNNER_SCRIPT_NAME
        script = f"#!{sys.executable}\n{_RUNNER_SCRIPT}"
        if not path.exists() or path.read_text(encoding="utf-8") != script:
            path.write_text(script, encoding="utf-8")
            current = path.stat().st_mode
            path.chmod(current | 0o111)
        return path
