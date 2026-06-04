from __future__ import annotations

import os
from pathlib import Path

import pytest

from tidyrun import load_callable, load_job_definition, load_job_inputs
from tidyrun import load_inputs_and_callable
from tidyrun.dag import DAG, run_materialized_job
from tidyrun.job import Job


def _const_two() -> int:
    return 2


def _add(x: int, y: int) -> int:
    return x + y


def _current_pid() -> int:
    return os.getpid()


def test_materialize_and_execute_in_threads(tmp_path: Path) -> None:
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})

    plan_dir = dag.materialize(tmp_path / "plan")
    result = dag.execute_materialized(
        plan_dir,
        execution_mode="thread",
        max_workers=2,
    )
    assert result.to_dict() == {"producer": 2, "consumer": 5}


def test_materialize_and_execute_in_subprocesses(tmp_path: Path) -> None:
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})

    plan_dir = tmp_path / "plan"

    materialized = dag.materialize(plan_dir)
    assert materialized == plan_dir
    # Schema v2: no plan.tidyrun; definitions in definitions/ subdir.
    assert not (plan_dir / "plan.tidyrun").exists()
    assert (plan_dir / "definitions" / "producer.tidyrun").is_file()
    assert (plan_dir / "definitions" / "consumer.tidyrun").is_file()
    assert (plan_dir / "inputs" / "consumer" / "y.tidyrun").is_file()

    result = dag.execute_materialized(plan_dir, max_workers=2)
    assert result.to_dict() == {"producer": 2, "consumer": 5}


def test_each_job_runs_in_separate_python_process(tmp_path: Path) -> None:
    dag = DAG(
        {
            "a": Job(func=_current_pid, kwargs={}),
            "b": Job(func=_current_pid, kwargs={}),
        }
    )

    result = dag.evaluate_in_subprocesses(
        tmp_path / "pid_plan",
        max_workers=2,
    )

    parent_pid = os.getpid()
    values = result.to_dict()
    assert values["a"] != parent_pid
    assert values["b"] != parent_pid


def test_job_rerun_snippet_and_public_runner(tmp_path: Path) -> None:
    job = Job(func=_const_two, kwargs={})
    dag = DAG({"producer": job})
    plan_dir = dag.materialize(tmp_path / "plan")

    snippet = job.rerun_snippet(dag_path=plan_dir, job_id="producer")
    assert "load_inputs_and_callable" in snippet
    assert "producer" in snippet

    func, inputs = load_inputs_and_callable(plan_dir, "producer")
    assert func(**inputs) == 2

    # Also verify via the lower-level API
    definition = load_job_definition(plan_dir, "producer")
    callable_obj = load_callable(definition)
    inputs2 = load_job_inputs(definition, plan_dir)
    assert callable_obj(**inputs2) == 2

    run_materialized_job(plan_dir, "producer")
    loaded = DAG({"producer": job}).execute_materialized(
        plan_dir,
        skip_completed=True,
    )
    assert loaded.to_dict() == {"producer": 2}


def test_materialize_progress_callback(tmp_path: Path) -> None:
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})
    messages: list[str] = []

    dag.materialize(
        tmp_path / "plan",
        progress=True,
        progress_callback=messages.append,
    )

    assert any("[materialize] starting" in message for message in messages)
    assert any("[materialize] [1/2] completed:" in message for message in messages)
    assert any("[materialize] [2/2] completed:" in message for message in messages)
    assert any("[materialize] done" in message for message in messages)


def test_execute_progress_callback(tmp_path: Path) -> None:
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})
    messages: list[str] = []

    plan_dir = dag.materialize(tmp_path / "plan")
    result = dag.execute_materialized(
        plan_dir,
        execution_mode="thread",
        progress=True,
        progress_callback=messages.append,
    )

    assert result.to_dict() == {"producer": 2, "consumer": 5}
    assert any("[execute] starting" in message for message in messages)
    assert any("[execute] [1/2] completed:" in message for message in messages)
    assert any("[execute] [2/2] completed:" in message for message in messages)
    assert any("[execute] done" in message for message in messages)


def test_execute_progress_default_single_line_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})

    plan_dir = dag.materialize(tmp_path / "plan")
    dag.execute_materialized(
        plan_dir,
        execution_mode="thread",
        progress=True,
    )

    captured = capsys.readouterr().out
    assert "\r[execute]" in captured
    assert "2/2 done" in captured
