from __future__ import annotations

from pathlib import Path

import pytest

from tidyrun import DAGExecutionError
from tidyrun.dag import DAG, job_output_exists
from tidyrun.job import Job


def _const(x: int) -> int:
    return x


def _add(x: int, y: int) -> int:
    return x + y


def _fail(x: int) -> int:
    raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# DAGExecutionError raised on job failure (serial path)
# ---------------------------------------------------------------------------


def test_dag_execution_error_on_serial_failure(tmp_path: Path) -> None:
    good = Job(func=_const, kwargs={"x": 1})
    bad = Job(func=_fail, kwargs={"x": good})

    dag = DAG({"good": good, "bad": bad})
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag.execute_materialized(plan_dir, execution_mode="thread")

    err = exc_info.value
    assert err.failed_job_id == "bad"
    assert isinstance(err.cause, RuntimeError)
    assert "good" in err.completed_jobs
    assert "bad" not in err.completed_jobs


def test_dag_execution_error_carries_completed_and_cancelled(tmp_path: Path) -> None:
    """With three independent jobs where the middle one fails, the other two
    are either completed or cancelled."""
    a = Job(func=_const, kwargs={"x": 1})
    b = Job(func=_fail, kwargs={"x": 42})  # always fails
    c = Job(func=_const, kwargs={"x": 3})

    dag = DAG({"a": a, "b": b, "c": c})
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag.execute_materialized(plan_dir)

    err = exc_info.value
    assert err.failed_job_id == "b"
    # "a" ran before "b" in alphabetical order; "c" was cancelled
    assert "a" in err.completed_jobs
    assert "c" in err.cancelled_jobs


# ---------------------------------------------------------------------------
# DAGExecutionError raised on job failure (executor / parallel path)
# ---------------------------------------------------------------------------


def test_dag_execution_error_on_parallel_failure(tmp_path: Path) -> None:
    good = Job(func=_const, kwargs={"x": 10})
    bad = Job(func=_fail, kwargs={"x": good})

    dag = DAG({"good": good, "bad": bad})
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag.execute_materialized(plan_dir, max_workers=2, execution_mode="thread")

    err = exc_info.value
    assert err.failed_job_id == "bad"
    assert isinstance(err.cause, RuntimeError)
    assert "good" in err.completed_jobs


def test_dag_execution_error_str_contains_job_id(tmp_path: Path) -> None:
    dag = DAG({"fail": Job(func=_fail, kwargs={"x": 99})})
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag.execute_materialized(plan_dir, execution_mode="thread")

    assert "fail" in str(exc_info.value)
    assert "intentional failure" in str(exc_info.value)


# ---------------------------------------------------------------------------
# skip_completed: already-finished jobs are not re-run
# ---------------------------------------------------------------------------


def test_skip_completed_skips_already_run_jobs(tmp_path: Path) -> None:
    from tidyrun.constants import TIDYRUN_METADATA_EXTENSION
    from tidyrun.dag import job_output_base

    dag = DAG(
        {
            "a": Job(func=_const, kwargs={"x": 1}),
            "b": Job(func=_add, kwargs={"x": 1, "y": 2}),
        }
    )
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)
    dag.execute_materialized(plan_dir, execution_mode="thread")

    # Tamper with "a"'s metadata to confirm it is not overwritten on re-run.
    out_path = plan_dir / "outputs"
    a_meta = Path(str(job_output_base(out_path, "a")) + TIDYRUN_METADATA_EXTENSION)
    sentinel_content = a_meta.read_text() + "\n# sentinel"
    a_meta.write_text(sentinel_content)

    # Second execution with skip_completed=True should NOT re-write "a"'s output
    dag.execute_materialized(plan_dir, execution_mode="thread", skip_completed=True)
    assert a_meta.read_text() == sentinel_content


def test_skip_completed_reruns_only_failed_jobs(tmp_path: Path) -> None:
    """Simulate the iteration workflow: first run fails; fix job; resubmit
    skipping already-completed upstream work."""
    # First run: "a" succeeds, "b" fails
    a = Job(func=_const, kwargs={"x": 1})
    b_fail = Job(func=_fail, kwargs={"x": a})

    dag_first = DAG({"a": a, "b": b_fail})
    plan_dir = tmp_path / "plan"
    dag_first.materialize(plan_dir)

    with pytest.raises(DAGExecutionError) as exc_info:
        dag_first.execute_materialized(plan_dir, execution_mode="thread")
    assert exc_info.value.failed_job_id == "b"

    out_path = plan_dir / "outputs"
    assert job_output_exists(out_path, "a")
    assert not job_output_exists(out_path, "b")

    # Second run: "b" now works; resubmit with skip_completed=True
    b_fixed = Job(func=_add, kwargs={"x": a, "y": 10})
    dag_second = DAG({"a": a, "b": b_fixed})
    # Re-materialize over the same plan_dir so "b" gets a new definition
    dag_second.materialize(plan_dir)

    result = dag_second.execute_materialized(
        plan_dir, execution_mode="thread", skip_completed=True
    )
    assert result.to_dict() == {"a": 1, "b": 11}


def test_execute_materialized_raises_if_outputs_exist_without_skip_completed(
    tmp_path: Path,
) -> None:
    dag = DAG(
        {
            "a": Job(func=_const, kwargs={"x": 1}),
            "b": Job(func=_add, kwargs={"x": 1, "y": 2}),
        }
    )
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    # First execution writes outputs to plan_dir/outputs.
    dag.execute_materialized(plan_dir, execution_mode="thread")

    # Re-running without skip_completed should fail fast.
    with pytest.raises(ValueError, match="skip_completed=True"):
        dag.execute_materialized(plan_dir, execution_mode="thread")


def test_evaluate_skip_completed_resumes_existing_plan(tmp_path: Path) -> None:
    dag = DAG(
        {
            "a": Job(func=_const, kwargs={"x": 1}),
            "b": Job(func=_add, kwargs={"x": 1, "y": 2}),
        }
    )
    run_dir = tmp_path / "run"

    dag.evaluate(run_dir, execution_mode="thread")
    result = dag.evaluate(run_dir, execution_mode="thread", skip_completed=True)

    assert result.to_dict() == {"a": 1, "b": 3}
    assert (run_dir / "definitions").is_dir()
    assert (run_dir / "outputs").is_dir()


# ---------------------------------------------------------------------------
# clear_outputs
# ---------------------------------------------------------------------------


def test_clear_outputs_removes_all_outputs(tmp_path: Path) -> None:
    dag = DAG(
        {
            "a": Job(func=_const, kwargs={"x": 1}),
            "b": Job(func=_const, kwargs={"x": 2}),
        }
    )
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)
    dag.execute_materialized(plan_dir, execution_mode="thread")

    out_dir = plan_dir / "outputs"
    assert job_output_exists(out_dir, "a")
    assert job_output_exists(out_dir, "b")

    dag.clear_outputs(plan_dir)

    assert not job_output_exists(out_dir, "a")
    assert not job_output_exists(out_dir, "b")


def test_clear_outputs_removes_specific_jobs(tmp_path: Path) -> None:
    dag = DAG(
        {
            "a": Job(func=_const, kwargs={"x": 1}),
            "b": Job(func=_const, kwargs={"x": 2}),
        }
    )
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)
    dag.execute_materialized(plan_dir, execution_mode="thread")

    out_dir = plan_dir / "outputs"
    dag.clear_outputs(plan_dir, job_ids=["a"])

    assert not job_output_exists(out_dir, "a")
    assert job_output_exists(out_dir, "b")


def test_clear_outputs_noop_if_no_outputs_dir(tmp_path: Path) -> None:
    dag = DAG({"a": Job(func=_const, kwargs={"x": 1})})
    plan_dir = tmp_path / "plan"
    dag.materialize(plan_dir)

    # No outputs have been written; should not raise
    dag.clear_outputs(plan_dir)
    dag.clear_outputs(plan_dir, job_ids=["a"])
