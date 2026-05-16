"""Test DAG execution in different modes: subprocess, thread, and process."""

from __future__ import annotations

import os
import time
from pathlib import Path

from tidyrun.dag import DAG
from tidyrun.job import Job


def _const_two() -> int:
    return 2


def _add(x: int, y: int) -> int:
    return x + y


def _slow_job(delay_seconds: float) -> float:
    """A job that takes time to complete."""
    time.sleep(delay_seconds)
    return delay_seconds


def _current_pid() -> int:
    """Return the current process ID."""
    return os.getpid()


def test_thread_mode_execution(tmp_path: Path) -> None:
    """Test that thread mode executes jobs in the same process."""
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})

    result = dag.evaluate(
        target=tmp_path / "thread_outputs",
        dag_path=tmp_path / "thread_plan",
        execution_mode="thread",
    )

    assert result.to_dict() == {"producer": 2, "consumer": 5}


def test_process_mode_execution(tmp_path: Path) -> None:
    """Test that process mode executes jobs via ProcessPoolExecutor."""
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})

    result = dag.evaluate(
        target=tmp_path / "process_outputs",
        dag_path=tmp_path / "process_plan",
        execution_mode="process",
        max_workers=2,
    )

    assert result.to_dict() == {"producer": 2, "consumer": 5}


def test_subprocess_mode_execution(tmp_path: Path) -> None:
    """Test that subprocess mode executes jobs in separate processes."""
    producer = Job(func=_const_two, kwargs={})
    consumer = Job(func=_add, kwargs={"x": producer, "y": 3})
    dag = DAG({"producer": producer, "consumer": consumer})

    result = dag.evaluate(
        target=tmp_path / "subprocess_outputs",
        dag_path=tmp_path / "subprocess_plan",
        execution_mode="subprocess",
    )

    assert result.to_dict() == {"producer": 2, "consumer": 5}


def test_thread_mode_avoids_process_spawning(tmp_path: Path) -> None:
    """Test that thread mode keeps all jobs in parent process."""
    parent_pid = os.getpid()
    dag = DAG(
        {
            "a": Job(func=_current_pid, kwargs={}),
            "b": Job(func=_current_pid, kwargs={}),
        }
    )

    result = dag.evaluate(
        target=tmp_path / "thread_pid_outputs",
        dag_path=tmp_path / "thread_pid_plan",
        execution_mode="thread",
    )

    values = result.to_dict()
    # In thread mode, jobs run in the parent process
    assert values["a"] == parent_pid
    assert values["b"] == parent_pid


def test_subprocess_mode_spawns_separate_processes(tmp_path: Path) -> None:
    """Test that subprocess mode spawns separate processes for jobs."""
    parent_pid = os.getpid()
    dag = DAG(
        {
            "a": Job(func=_current_pid, kwargs={}),
            "b": Job(func=_current_pid, kwargs={}),
        }
    )

    result = dag.evaluate(
        target=tmp_path / "subprocess_pid_outputs",
        dag_path=tmp_path / "subprocess_pid_plan",
        execution_mode="subprocess",
    )

    values = result.to_dict()
    # In subprocess mode, jobs run in different processes
    assert values["a"] != parent_pid
    assert values["b"] != parent_pid


def test_process_mode_spawns_separate_processes(tmp_path: Path) -> None:
    """Test that process mode spawns separate processes for jobs."""
    parent_pid = os.getpid()
    dag = DAG(
        {
            "a": Job(func=_current_pid, kwargs={}),
            "b": Job(func=_current_pid, kwargs={}),
        }
    )

    result = dag.evaluate(
        target=tmp_path / "process_pid_outputs",
        dag_path=tmp_path / "process_pid_plan",
        execution_mode="process",
        max_workers=2,
    )

    values = result.to_dict()
    # In process mode, jobs run in different processes
    assert values["a"] != parent_pid
    assert values["b"] != parent_pid


def test_thread_mode_is_faster_than_subprocess_for_small_jobs(tmp_path: Path) -> None:
    """Benchmark that thread mode is faster than subprocess for small jobs."""
    # Create multiple small jobs
    jobs = {
        f"job_{i}": Job(func=_slow_job, kwargs={"delay_seconds": 0.01})
        for i in range(5)
    }
    dag = DAG(jobs)

    # Time thread mode
    start_thread = time.time()
    dag.evaluate(
        target=tmp_path / "thread_bench",
        dag_path=tmp_path / "thread_bench_plan",
        execution_mode="thread",
    )
    thread_time = time.time() - start_thread

    # Time subprocess mode
    start_subprocess = time.time()
    dag.evaluate(
        target=tmp_path / "subprocess_bench",
        dag_path=tmp_path / "subprocess_bench_plan",
        execution_mode="subprocess",
    )
    subprocess_time = time.time() - start_subprocess

    # Thread mode should be significantly faster due to no subprocess overhead
    # (subprocess spawning has ~50-100ms overhead per job)
    assert thread_time < subprocess_time, (
        f"Thread mode ({thread_time:.2f}s) should be faster than "
        f"subprocess mode ({subprocess_time:.2f}s) for small jobs"
    )


def test_parallel_execution_with_thread_mode(tmp_path: Path) -> None:
    """Test that thread mode can parallelize job execution with max_workers."""
    jobs = {
        f"job_{i}": Job(func=_slow_job, kwargs={"delay_seconds": 0.05})
        for i in range(4)
    }
    dag = DAG(jobs)

    # Sequential thread execution
    start_seq = time.time()
    dag.evaluate(
        target=tmp_path / "thread_seq",
        dag_path=tmp_path / "thread_seq_plan",
        execution_mode="thread",
    )
    seq_time = time.time() - start_seq

    # Parallel thread execution with max_workers=2
    start_par = time.time()
    dag.evaluate(
        target=tmp_path / "thread_par",
        dag_path=tmp_path / "thread_par_plan",
        execution_mode="thread",
        max_workers=2,
    )
    par_time = time.time() - start_par

    # Parallel should be faster (roughly 2x for 4 jobs with 2 workers)
    # seq_time should be ~0.2s (4 jobs * 0.05s), par_time should be ~0.1s
    assert par_time < seq_time, (
        f"Parallel thread execution ({par_time:.2f}s) should be faster than "
        f"sequential ({seq_time:.2f}s)"
    )
