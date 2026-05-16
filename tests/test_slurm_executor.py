from __future__ import annotations

from concurrent.futures import Executor
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from tidyrun.slurm_executor import SlurmExecutor


def _double(x: int) -> int:
    return x * 2


def _noop_job(_plan_dir: str, _job_id: str) -> None:
    return None


def test_slurm_executor_submit_and_collect_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    squeue_calls = 0
    state: dict[str, object] = {}

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text

        if cmd[0] == "sbatch":
            task_path = Path(cmd[-3])
            result_path = Path(cmd[-2])
            with task_path.open("rb") as fp:
                fn, args, kwargs = pickle.load(fp)
            state["result"] = fn(*args, **kwargs)
            state["result_path"] = result_path
            return subprocess.CompletedProcess(cmd, 0, stdout="12345\n", stderr="")

        if cmd[0] == "squeue":
            nonlocal squeue_calls
            calls = squeue_calls
            squeue_calls += 1
            if calls == 0:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="12345 running\n", stderr=""
                )

            result_path = state["result_path"]
            assert isinstance(result_path, Path)
            with result_path.open("wb") as fp:
                pickle.dump(state["result"], fp)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("tidyrun.slurm_executor.subprocess.run", fake_run)

    executor: Executor = SlurmExecutor(tmp_path, poll_interval_seconds=0.0)
    future = executor.submit(_double, 21)
    assert future.result(timeout=1.0) == 42
    executor.shutdown()


def test_slurm_executor_propagates_remote_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text

        if cmd[0] == "sbatch":
            error_path = Path(cmd[-1])
            error_path.write_text("remote traceback", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="888\n", stderr="")

        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("tidyrun.slurm_executor.subprocess.run", fake_run)

    executor = SlurmExecutor(tmp_path, poll_interval_seconds=0.0, cleanup_files=False)
    future = executor.submit(_double, 2)
    with pytest.raises(RuntimeError, match="remote traceback"):
        future.result(timeout=1.0)
    executor.shutdown()


def test_slurm_executor_builds_sbatch_options(tmp_path: Path) -> None:
    executor = SlurmExecutor(
        tmp_path,
        sbatch_options={"partition": "debug", "--qos": "normal", "cpus_per_task": 2},
    )
    cmd = executor._build_sbatch_command(
        task_path=tmp_path / "task",
        result_path=tmp_path / "result",
        error_path=tmp_path / "error",
        stdout_path=tmp_path / "stdout",
    )

    assert "--partition" in cmd
    assert "debug" in cmd
    assert "--qos" in cmd
    assert "normal" in cmd
    assert "--cpus-per-task" in cmd
    assert "2" in cmd


def test_slurm_executor_builds_resource_limit_options(tmp_path: Path) -> None:
    executor = SlurmExecutor(
        tmp_path,
        partition="debug",
        qos="normal",
        account="science",
        constraint="zen4",
        time_limit="02:00:00",
        memory="16G",
        cpus_per_task=4,
        gres="gpu:1",
    )
    cmd = executor._build_sbatch_command(
        task_path=tmp_path / "task",
        result_path=tmp_path / "result",
        error_path=tmp_path / "error",
        stdout_path=tmp_path / "stdout",
    )

    assert "--partition" in cmd
    assert "debug" in cmd
    assert "--qos" in cmd
    assert "normal" in cmd
    assert "--account" in cmd
    assert "science" in cmd
    assert "--constraint" in cmd
    assert "zen4" in cmd
    assert "--time" in cmd
    assert "02:00:00" in cmd
    assert "--mem" in cmd
    assert "16G" in cmd
    assert "--cpus-per-task" in cmd
    assert "4" in cmd
    assert "--gres" in cmd
    assert "gpu:1" in cmd


def test_slurm_executor_sbatch_options_override_resource_args(tmp_path: Path) -> None:
    executor = SlurmExecutor(
        tmp_path,
        time_limit="02:00:00",
        memory="16G",
        sbatch_options={"time": "00:30:00", "mem": "8G"},
    )
    cmd = executor._build_sbatch_command(
        task_path=tmp_path / "task",
        result_path=tmp_path / "result",
        error_path=tmp_path / "error",
        stdout_path=tmp_path / "stdout",
    )

    assert "--time" in cmd
    assert "00:30:00" in cmd
    assert "02:00:00" not in cmd
    assert "--mem" in cmd
    assert "8G" in cmd
    assert "16G" not in cmd


def test_slurm_executor_submit_with_options_overrides_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted_cmd: list[str] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text

        if cmd[0] == "sbatch":
            submitted_cmd.extend(cmd)
            result_path = Path(cmd[-2])
            with result_path.open("wb") as fp:
                pickle.dump(84, fp)
            return subprocess.CompletedProcess(cmd, 0, stdout="999\n", stderr="")

        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("tidyrun.slurm_executor.subprocess.run", fake_run)

    executor = SlurmExecutor(
        tmp_path,
        time_limit="02:00:00",
        memory="16G",
        poll_interval_seconds=0.0,
    )
    future = executor.submit_with_options(
        _double,
        42,
        sbatch_options={"time": "00:05:00", "mem": "2G"},
    )
    assert future.result(timeout=1.0) == 84
    executor.shutdown()

    assert "--time" in submitted_cmd
    assert "00:05:00" in submitted_cmd
    assert "02:00:00" not in submitted_cmd
    assert "--mem" in submitted_cmd
    assert "2G" in submitted_cmd
    assert "16G" not in submitted_cmd


def test_slurm_runner_script_uses_current_python_executable(tmp_path: Path) -> None:
    executor = SlurmExecutor(tmp_path)
    script = (tmp_path / "_tidyrun_slurm_runner.py").read_text(encoding="utf-8")

    assert script.startswith(f"#!{sys.executable}\n")
    executor.shutdown()


def test_slurm_executor_sets_job_name_from_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted_cmd: list[str] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text

        if cmd[0] == "sbatch":
            submitted_cmd[:] = cmd
            result_path = Path(cmd[-2])
            with result_path.open("wb") as fp:
                pickle.dump(None, fp)
            return subprocess.CompletedProcess(cmd, 0, stdout="2026\n", stderr="")

        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("tidyrun.slurm_executor.subprocess.run", fake_run)

    executor = SlurmExecutor(tmp_path, poll_interval_seconds=0.0)
    future = executor.submit(_noop_job, "plan", "a/b/c")
    future.result(timeout=1.0)
    executor.shutdown()

    assert "--job-name" in submitted_cmd
    assert "a/b/c" in submitted_cmd


def test_slurm_executor_submit_array_with_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted_cmd: list[str] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text

        if cmd[0] == "sbatch":
            submitted_cmd[:] = cmd
            result_template = cmd[-2]
            assert "%a" in result_template
            for i in range(3):
                result_path = Path(result_template.replace("%a", str(i)))
                with result_path.open("wb") as fp:
                    pickle.dump(None, fp)
            return subprocess.CompletedProcess(cmd, 0, stdout="31415\n", stderr="")

        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("tidyrun.slurm_executor.subprocess.run", fake_run)

    executor = SlurmExecutor(tmp_path, poll_interval_seconds=0.0)
    submission = executor.submit_array_with_options(
        _noop_job,
        "plan",
        ["pairs/a", "pairs/b", "pairs/c"],
        sbatch_options={"time": "00:05:00", "job_name": "pairs"},
    )
    submission.future.result(timeout=1.0)
    executor.shutdown()

    assert submission.job_ids == ("pairs/a", "pairs/b", "pairs/c")
    assert "--array" in submitted_cmd
    assert "0-2" in submitted_cmd
    assert "--job-name" in submitted_cmd
    assert "pairs" in submitted_cmd
