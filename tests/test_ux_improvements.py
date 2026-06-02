"""Tests for UX improvements introduced in 0.0.6:

1. ParametrizedJob.materialize / execute_materialized / evaluate
2. DAGExecutionError includes job traceback + rerun snippet
3. _count_unique_jobs progress-bar count is correct (no ID-reuse bug)
4. execute_materialized uses lightweight symlinks (no eager re-serialization)
5. Dependency job IDs no longer fall back to __job_N synthetic counter
6. SlurmExecutor uses job_id-based file names instead of UUID hashes
"""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import pytest

from tidyrun import DAGExecutionError, ParametrizedJob
from tidyrun.dag import DAG, _count_unique_jobs
from tidyrun.job import Job
from tidyrun.plan import failed_path


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _join(left: str, right: str, sep: str = "/") -> str:
    return f"{left}{sep}{right}"


def _const(x: int) -> int:
    return x


def _fail(x: int) -> int:
    raise RuntimeError("deliberate failure")


def _noop_job(_plan_dir: str, _job_id: str) -> None:
    """No-op job runner used in Slurm executor tests."""
    return None


# ---------------------------------------------------------------------------
# Issue #1 — ParametrizedJob.materialize / execute_materialized / evaluate
# ---------------------------------------------------------------------------


def test_parametrized_job_materialize_creates_plan(tmp_path: Path) -> None:
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left"],
        parameter_values=[("a",), ("b",)],
        kwargs={"right": "x", "sep": "-"},
    )
    plan_dir = pjob.materialize(tmp_path / "plan")

    assert (plan_dir / "definitions").is_dir()
    # 1-D pjob expands to individual jobs; no "result" wrapper.
    assert not (plan_dir / "definitions" / "result.tidyrun").exists()
    # Individual job definitions for each parameter value
    assert (plan_dir / "definitions" / "a.tidyrun").is_file()
    assert (plan_dir / "definitions" / "b.tidyrun").is_file()


def test_parametrized_job_evaluate_returns_lazy_dict(tmp_path: Path) -> None:
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left"],
        parameter_values=[("a",), ("b",)],
        kwargs={"right": "x", "sep": "-"},
    )
    result = pjob.evaluate(tmp_path / "run")

    # No extra wrapper level — the result IS the parametrised structure.
    assert result.to_dict() == {"a": "a-x", "b": "b-x"}


def test_parametrized_job_execute_materialized(tmp_path: Path) -> None:
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left", "right"],
        parameter_values=[("m1", "train"), ("m1", "test"), ("m2", "train")],
        kwargs={"sep": ":"},
    )
    plan_dir = pjob.materialize(tmp_path / "plan")
    result = pjob.execute_materialized(
        dag_path=plan_dir,
        output_path=tmp_path / "out",
        execution_mode="thread",
    )

    # No extra wrapper level
    assert result.to_dict() == {
        "m1": {"train": "m1:train", "test": "m1:test"},
        "m2": {"train": "m2:train"},
    }


def test_parametrized_job_evaluate_skip_completed(tmp_path: Path) -> None:
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left"],
        parameter_values=[("a",), ("b",)],
        kwargs={"right": "x", "sep": "-"},
    )
    run_dir = tmp_path / "run"
    pjob.evaluate(run_dir)
    # Second run with skip_completed should succeed
    result = pjob.evaluate(run_dir, skip_completed=True)
    assert result.to_dict() == {"a": "a-x", "b": "b-x"}


# ---------------------------------------------------------------------------
# Issue #2 — DAGExecutionError includes traceback and rerun snippet
# ---------------------------------------------------------------------------


def test_dag_execution_error_includes_job_traceback(tmp_path: Path) -> None:
    dag = DAG(
        {"a": Job(func=_const, kwargs={"x": 1}), "b": Job(func=_fail, kwargs={"x": 1})}
    )
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag.execute_materialized(
            dag_path=plan_dir,
            output_path=tmp_path / "out",
            execution_mode="subprocess",
        )

    err = exc_info.value
    assert err.plan_dir == plan_dir
    assert err.outputs_path is not None

    # The .failed TOML sentinel should have been written
    sentinel = failed_path(err.outputs_path, "b")
    assert sentinel.is_file(), "Expected .failed sentinel to exist"

    # __str__ should include the job traceback
    msg = str(err)
    assert "Job traceback:" in msg
    assert "deliberate failure" in msg


def test_dag_execution_error_includes_rerun_snippet(tmp_path: Path) -> None:
    dag = DAG({"bad": Job(func=_fail, kwargs={"x": 99})})
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag.execute_materialized(
            dag_path=plan_dir,
            output_path=tmp_path / "out",
            execution_mode="subprocess",
        )

    msg = str(exc_info.value)
    assert "To re-run this job interactively:" in msg
    assert "load_job_definition" in msg
    assert "bad" in msg


def test_dag_execution_error_rerun_snippet_method(tmp_path: Path) -> None:
    dag = DAG({"bad": Job(func=_fail, kwargs={"x": 99})})
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag.execute_materialized(
            dag_path=plan_dir,
            output_path=tmp_path / "out",
            execution_mode="subprocess",
        )

    err = exc_info.value
    snippet = err.rerun_snippet()
    assert snippet is not None
    assert "bad" in snippet


def test_dag_execution_error_without_plan_dir_has_no_snippet() -> None:
    err = DAGExecutionError(
        failed_job_id="x",
        cause=RuntimeError("oops"),
        completed_jobs=set(),
        cancelled_jobs=set(),
    )
    assert err.rerun_snippet() is None
    assert "To re-run" not in str(err)


# ---------------------------------------------------------------------------
# Issue #3 — _count_unique_jobs progress-bar count is correct
# ---------------------------------------------------------------------------


def test_count_unique_jobs_multi_level_parametrized() -> None:
    """With 3 unique first-level keys and 100 total combos, count must be 100."""
    n_a = 3
    n_b = 34  # 3*34 = 102, not divisible, test with non-uniform
    parameter_values = [(f"a{i}", f"b{j}") for i in range(n_a) for j in range(n_b)]
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left", "right"],
        parameter_values=parameter_values,  # type: ignore[arg-type]
        kwargs={"sep": ":"},
    )
    dag = DAG({"pairs": pjob})

    seen: set[int] = set()
    count = sum(_count_unique_jobs(node, seen) for node in dag.values())
    assert count == len(parameter_values)


def test_count_unique_jobs_no_id_reuse_with_gc_pressure() -> None:
    """Create enough combinations that Python is likely to reuse object IDs."""
    import gc

    gc.collect()
    parameter_values = [(f"k{i}",) for i in range(200)]
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left"],
        parameter_values=parameter_values,  # type: ignore[arg-type]
        kwargs={"right": "x", "sep": "-"},
    )
    dag = DAG({"jobs": pjob})

    seen: set[int] = set()
    count = sum(_count_unique_jobs(node, seen) for node in dag.values())
    assert count == 200


def test_materialize_progress_total_matches_step_count(tmp_path: Path) -> None:
    """The progress total should equal the number of step() calls during materialize."""
    parameter_values = [(f"k{i}",) for i in range(50)]
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left"],
        parameter_values=parameter_values,  # type: ignore[arg-type]
        kwargs={"right": "x", "sep": "-"},
    )
    dag = DAG({"jobs": pjob})

    steps: list[str] = []
    totals: list[int] = []

    from tidyrun.dag import _ProgressReporter

    original_init = _ProgressReporter.__init__

    def patched_init(self: _ProgressReporter, **kwargs: object) -> None:
        original_init(self, **kwargs)  # type: ignore[call-arg]
        totals.append(self.total)

    original_step = _ProgressReporter.step

    def patched_step(self: _ProgressReporter, job_id: str, **kwargs: object) -> None:
        steps.append(job_id)
        original_step(self, job_id, **kwargs)  # type: ignore[call-arg]

    _ProgressReporter.step = patched_step  # type: ignore[method-assign]
    _ProgressReporter.__init__ = patched_init  # type: ignore[method-assign]
    try:
        dag.materialize(tmp_path / "plan", progress=True)
    finally:
        _ProgressReporter.step = original_step  # type: ignore[method-assign]
        _ProgressReporter.__init__ = original_init  # type: ignore[method-assign]

    assert totals[0] == 50
    assert len(steps) == 50


# ---------------------------------------------------------------------------
# Issue #4 — execute_materialized no longer eagerly loads/re-serializes
# ---------------------------------------------------------------------------


def test_execute_materialized_writes_directly_to_output_path(tmp_path: Path) -> None:
    """Jobs should write outputs directly to output_path — no post-processing step."""
    dag = DAG(
        {
            "a": Job(func=_const, kwargs={"x": 1}),
            "b": Job(func=_const, kwargs={"x": 2}),
        }
    )
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "out"
    dag.materialize(plan_dir)
    result = dag.execute_materialized(
        dag_path=plan_dir,
        output_path=out_dir,
        execution_mode="thread",
    )

    assert result.to_dict() == {"a": 1, "b": 2}

    # Outputs should be written directly under out_dir — no intermediate copy.
    from tidyrun.plan import job_output_exists

    assert job_output_exists(out_dir, "a"), "Job 'a' output should be at out_dir/a"
    assert job_output_exists(out_dir, "b"), "Job 'b' output should be at out_dir/b"


def test_mixed_dag_result_navigable_via_lazy_dict(tmp_path: Path) -> None:
    """A DAG with both single jobs and parametrised jobs must produce a fully
    navigable result: single-job entries (backed by metadata sidecars) and
    parametrised-group entries (bare subdirectories) must both be visible."""
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left"],
        parameter_values=[("a",), ("b",)],  # type: ignore[arg-type]
        kwargs={"right": "x", "sep": "-"},
    )
    dag = DAG(
        {
            "single": Job(func=_const, kwargs={"x": 42}),
            "pairs": pjob,
        }
    )
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "out"
    dag.materialize(plan_dir)
    result = dag.execute_materialized(
        dag_path=plan_dir,
        output_path=out_dir,
        execution_mode="thread",
    )

    assert result["single"] == 42
    assert result.to_dict() == {
        "single": 42,
        "pairs": {"a": "a-x", "b": "b-x"},
    }


def test_parametrized_job_result_tree_is_accessible(tmp_path: Path) -> None:
    """The result tree for a parametrized job should be navigable as a LazyDict."""
    pjob = ParametrizedJob(
        func=_join,
        parameter_names=["left", "right"],
        parameter_values=[("m1", "train"), ("m1", "test"), ("m2", "train")],
        kwargs={"sep": ":"},
    )
    dag = DAG({"pairs": pjob})
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "out"
    dag.materialize(plan_dir)
    result = dag.execute_materialized(
        dag_path=plan_dir,
        output_path=out_dir,
        execution_mode="thread",
    )

    assert result.to_dict() == {
        "pairs": {
            "m1": {"train": "m1:train", "test": "m1:test"},
            "m2": {"train": "m2:train"},
        }
    }


# ---------------------------------------------------------------------------
# Issue #5 — dependency job IDs no longer fall back to __job_N
# ---------------------------------------------------------------------------


def _identity(x: int) -> int:
    return x


def test_shared_dependency_job_id_is_not_synthetic(tmp_path: Path) -> None:
    """A shared dependency Job used by an array job must NOT get a __job_N id."""
    shared = Job(func=_identity, kwargs={"x": 10})

    def _use_dep(dep: int, tag: str) -> str:
        return f"{dep}-{tag}"

    pjob = ParametrizedJob(
        func=_use_dep,
        parameter_names=["tag"],
        parameter_values=[("a",), ("b",), ("c",)],
        kwargs={"dep": shared},
    )
    dag = DAG({"result": pjob})
    plan_dir = dag.materialize(tmp_path / "plan")

    definitions_dir = plan_dir / "definitions"
    # Scan all definition file names
    def_names = [f.stem for f in definitions_dir.rglob("*.tidyrun")]
    # None of them should look like the synthetic __job_N pattern
    for name in def_names:
        assert not name.startswith("__job_"), (
            f"Synthetic job ID found: {name!r}. "
            "Dependency job IDs should be derived from the DAG path."
        )


# ---------------------------------------------------------------------------
# Issue #6 — SlurmExecutor uses job_id-based file names
# ---------------------------------------------------------------------------


def test_slurm_executor_task_files_use_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task/result/error/stdout files should be prefixed with the job id."""
    from tidyrun.executors.slurm import SlurmExecutor

    created_paths: list[str] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "sbatch":
            result_path = Path(cmd[-2])
            created_paths.append(str(result_path))
            with result_path.open("wb") as fp:
                pickle.dump(None, fp)
            return subprocess.CompletedProcess(cmd, 0, stdout="42\n", stderr="")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected: {cmd}")

    monkeypatch.setattr("tidyrun.executors.slurm.subprocess.run", fake_run)

    executor = SlurmExecutor(tmp_path, poll_interval_seconds=0.0)
    future = executor.submit(_noop_job, "plan/dir", "pairs/m1/train")
    future.result(timeout=2.0)
    executor.shutdown()

    assert created_paths, "No result file was created"
    # Path must embed the job id as a sub-path (pairs/m1/train), not flattened with __
    result_path_obj = Path(created_paths[0])
    assert "pairs/m1/train" in created_paths[0], (
        f"Expected job-id path in result path, got: {created_paths[0]!r}"
    )
    # The file must be inside a subdirectory of shared_dir, not directly in it
    assert result_path_obj.parent != tmp_path, (
        f"Result file should be in a subdirectory, not directly in shared_dir: {result_path_obj!r}"
    )
    # No UUID-style 32-hex-char names
    import re

    assert not re.match(r".*[0-9a-f]{32}\.", created_paths[0]), (
        f"Unexpected UUID-style path: {created_paths[0]!r}"
    )


def test_slurm_executor_array_files_use_group_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Array task files should be named after the array group, not a UUID."""
    from tidyrun.executors.slurm import SlurmExecutor

    created_task_paths: list[str] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "sbatch":
            task_path = Path(cmd[-3])
            created_task_paths.append(str(task_path))
            result_template = cmd[-2]
            for i in range(2):
                rp = Path(result_template.replace("%a", str(i)))
                with rp.open("wb") as fp:
                    pickle.dump(None, fp)
            return subprocess.CompletedProcess(cmd, 0, stdout="99\n", stderr="")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected: {cmd}")

    monkeypatch.setattr("tidyrun.executors.slurm.subprocess.run", fake_run)

    executor = SlurmExecutor(tmp_path, poll_interval_seconds=0.0)
    submission = executor.submit_array_with_options(
        _noop_job,
        "plan",
        ["pairs/m1/train", "pairs/m1/test"],
        sbatch_options={"job_name": "pairs"},
    )
    submission.future.result(timeout=2.0)
    executor.shutdown()

    assert created_task_paths, "No task file was created"
    task_path_obj = Path(created_task_paths[0])
    # Path must embed the array group (common prefix pairs/m1) as a sub-path, not a UUID
    assert "pairs/m1" in created_task_paths[0], (
        f"Expected array-group path in task path, got: {created_task_paths[0]!r}"
    )
    # The task file must be inside a subdirectory of shared_dir, not directly in it
    assert task_path_obj.parent != tmp_path, (
        f"Task file should be in a subdirectory, not directly in shared_dir: {task_path_obj!r}"
    )
    import re

    assert not re.match(r".*[0-9a-f]{32}\.", created_task_paths[0]), (
        f"Unexpected UUID-style array task path: {created_task_paths[0]!r}"
    )
