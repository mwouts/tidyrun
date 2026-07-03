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
# Version stamping and TIDYRUN_PIP_SPEC bootstrap
# ---------------------------------------------------------------------------


def test_materialize_stamps_plan_with_tidyrun_version(tmp_path: Path) -> None:
    import tidyrun
    from tidyrun.plan import read_plan_info

    plan_dir = tmp_path / "plan"
    DAG({"result": Job(func=_double, kwargs={"x": 1})}).materialize(plan_dir)

    info = read_plan_info(plan_dir)
    assert info["kind"] == "plan_info"
    assert info["tidyrun_version"] == tidyrun.__version__


def test_runner_warns_on_tidyrun_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import toml

    plan_dir = tmp_path / "plan"
    DAG({"result": Job(func=_double, kwargs={"x": 21})}).materialize(plan_dir)

    # Simulate a plan written by another (e.g. dev) version of tidyrun.
    info_file = plan_dir / "plan.toml"
    info = toml.loads(info_file.read_text(encoding="utf-8"))
    info["tidyrun_version"] = "0.0.0.dev0"
    info_file.write_text(toml.dumps(info), encoding="utf-8")

    monkeypatch.setenv("TIDYRUN_PLAN_DIR", str(plan_dir))
    monkeypatch.setenv("TIDYRUN_JOB_ID", "result")
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)
    monkeypatch.delenv("TIDYRUN_PIP_SPEC", raising=False)

    batch_entrypoint()

    assert job_output_exists(plan_dir / "outputs", "result")
    stderr = capsys.readouterr().err
    assert "materialized with tidyrun 0.0.0.dev0" in stderr


def test_batch_entrypoint_installs_tidyrun_pip_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TIDYRUN_PIP_SPEC triggers a pip install followed by a re-exec."""
    import sys

    from tidyrun import execute as execute_module

    pip_spec = "tidyrun[s3] @ git+https://github.com/my-org/tidyrun@my-branch"
    monkeypatch.setenv("TIDYRUN_PIP_SPEC", pip_spec)
    monkeypatch.delenv("_TIDYRUN_BOOTSTRAPPED", raising=False)

    calls: dict[str, object] = {}

    def _fake_run(cmd: list[str], check: bool) -> None:
        calls["pip"] = cmd
        assert check

    class _Reexec(Exception):
        pass

    def _fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
        calls["execve"] = (path, argv, env)
        raise _Reexec

    monkeypatch.setattr(execute_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(execute_module.os, "execve", _fake_execve)

    with pytest.raises(_Reexec):
        batch_entrypoint()

    assert calls["pip"] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        pip_spec,
    ]
    _, _, reexec_env = calls["execve"]  # type: ignore[misc]
    assert reexec_env["_TIDYRUN_BOOTSTRAPPED"] == "1"


def test_batch_entrypoint_skips_pip_spec_after_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-executed entrypoint must not install (or re-exec) again."""
    from tidyrun import execute as execute_module

    plan_dir = tmp_path / "plan"
    DAG({"result": Job(func=_double, kwargs={"x": 21})}).materialize(plan_dir)

    monkeypatch.setenv("TIDYRUN_PIP_SPEC", "tidyrun==0.0.0")
    monkeypatch.setenv("_TIDYRUN_BOOTSTRAPPED", "1")
    monkeypatch.setenv("TIDYRUN_PLAN_DIR", str(plan_dir))
    monkeypatch.setenv("TIDYRUN_JOB_ID", "result")
    monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)

    def _fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("pip must not run after bootstrap")

    monkeypatch.setattr(execute_module.subprocess, "run", _fail_run)

    batch_entrypoint()

    assert job_output_exists(plan_dir / "outputs", "result")


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
