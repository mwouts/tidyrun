from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

from tidyrun import DAG, Job, deserialize


def _const_two() -> int:
    return 2


def _container_runtime() -> str | None:
    for candidate in ("docker", "podman"):
        binary = shutil.which(candidate)
        if binary is None:
            continue
        try:
            subprocess.run(
                [binary, "version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return binary
        except Exception:
            continue
    return None


@pytest.mark.skipif(
    os.environ.get("RUN_CONTAINER_TESTS") != "1",
    reason="Set RUN_CONTAINER_TESTS=1 to run local container integration tests.",
)
def test_local_container_can_run_materialized_job(tmp_path: Path) -> None:
    runtime = _container_runtime()
    if runtime is None:
        pytest.skip("No working docker/podman runtime found")

    dag = DAG({"a": Job(func=_const_two, kwargs={})})
    plan_dir = dag.materialize(tmp_path / "plan")

    image_tag = f"tidyrun-test-runner:{uuid.uuid4().hex[:8]}"
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "\n".join(
            [
                "FROM python:3.11-slim",
                "RUN pip install --no-cache-dir toml cloudpickle",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [runtime, "build", "-t", image_tag, "-f", str(dockerfile), str(tmp_path)],
        check=True,
    )

    repo_root = Path(__file__).resolve().parents[1]
    try:
        subprocess.run(
            [
                runtime,
                "run",
                "--rm",
                "-v",
                f"{tmp_path}:/data",
                "-v",
                f"{repo_root}:/workspace",
                "-w",
                "/workspace",
                image_tag,
                "python",
                "-c",
                (
                    "import sys; "
                    "sys.path.insert(0, '/workspace/src'); "
                    "from tidyrun.dag import run_materialized_job; "
                    "run_materialized_job('/data/plan', 'a')"
                ),
            ],
            check=True,
        )
    finally:
        subprocess.run([runtime, "rmi", "-f", image_tag], check=False)

    assert deserialize(plan_dir / "outputs" / "a") == 2
