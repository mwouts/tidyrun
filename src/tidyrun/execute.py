"""Execution of materialized plans: job runners and the dependency scheduler."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast, Literal

from cloudpathlib import CloudPath
import toml

from tidyrun.plan import (
    PlanPaths,
    failed_path,
    job_output_base,
    job_output_exists,
    load_callable,
    load_job_definition,
    load_job_inputs,
    read_plan_graph,
    running_path,
    to_path,
)
from tidyrun.progress import ProgressCallback, ProgressReporter

ExecutionMode = Literal["subprocess", "thread", "process"]

#: A job runner receives (runner_string, job_id) and raises on failure.
JobRunner = Callable[[str, str], None]


class DAGExecutionError(Exception):
    """Raised when a DAG job fails during execution.

    Attributes
    ----------
    failed_job_id:
        The job_id of the job that failed.
    cause:
        The original exception raised by the job.
    completed_jobs:
        Set of job_ids that completed successfully before the failure.
    cancelled_jobs:
        Set of job_ids that were pending when the failure occurred and
        were not executed.
    plan_dir:
        Path to the materialized plan directory, if known.
    outputs_path:
        Path where job outputs (and .failed sentinels) are written, if known.
    """

    def __init__(
        self,
        failed_job_id: str,
        cause: BaseException,
        completed_jobs: set[str],
        cancelled_jobs: set[str],
        *,
        plan_dir: Path | CloudPath | None = None,
        outputs_path: Path | CloudPath | None = None,
    ) -> None:
        self.failed_job_id = failed_job_id
        self.cause = cause
        self.completed_jobs = frozenset(completed_jobs)
        self.cancelled_jobs = frozenset(cancelled_jobs)
        self.plan_dir = plan_dir
        self.outputs_path = outputs_path
        super().__init__(str(self))

    def _job_traceback(self) -> str | None:
        """Read the traceback from the .failed sentinel written by the job process."""
        if self.outputs_path is None:
            return None
        sentinel = failed_path(self.outputs_path, self.failed_job_id)
        if not sentinel.is_file():
            return None
        try:
            data = cast(
                dict[str, Any],
                toml.loads(sentinel.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            )
            tb = data.get("traceback")
            return str(tb) if isinstance(tb, str) else None
        except Exception:
            # The sentinel is best-effort diagnostics; never mask the original
            # job failure with a parse error here.
            return None

    def rerun_snippet(self) -> str | None:
        """Return a Python snippet that re-runs just the failed job, or None."""
        if self.plan_dir is None:
            return None
        from tidyrun.plan import rerun_snippet as _rerun_snippet

        try:
            return _rerun_snippet(self.plan_dir, self.failed_job_id)
        except Exception:
            # Same as above: the snippet is optional help text only.
            return None

    def __str__(self) -> str:
        lines = [f"DAG job {self.failed_job_id!r} failed: {self.cause}"]
        if self.completed_jobs:
            lines.append(f"  Completed jobs: {sorted(self.completed_jobs)}")
        if self.cancelled_jobs:
            lines.append(f"  Cancelled jobs: {sorted(self.cancelled_jobs)}")
        tb = self._job_traceback()
        if tb:
            lines.append("")
            lines.append("Job traceback:")
            lines.append(tb.rstrip())
        snippet = self.rerun_snippet()
        if snippet:
            lines.append("")
            lines.append("To re-run this job interactively:")
            lines.append(snippet)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Job runners
# ---------------------------------------------------------------------------


def _run_compiled_job(plan_paths: PlanPaths, job_id: str) -> None:
    import datetime
    import traceback

    from tidyrun.serialization.api import serialize

    plan_dir = plan_paths.definitions.parent
    definition = load_job_definition(plan_dir, job_id)
    inputs = load_job_inputs(definition, plan_dir)
    func = load_callable(definition)

    running = running_path(plan_paths.outputs, job_id)
    running.parent.mkdir(parents=True, exist_ok=True)
    running.touch()
    try:
        outputs = func(**inputs)
        output_base = job_output_base(plan_paths.outputs, job_id)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        serialize(outputs, output_base)
    except Exception as exc:
        failed_path(plan_paths.outputs, job_id).write_text(
            toml.dumps(  # pyright: ignore[reportUnknownMemberType]
                {
                    "job_id": job_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        raise
    finally:
        if running.exists():
            running.unlink()


def run_materialized_job(dag_path: Any, job_id: str) -> None:
    """Run one job from a materialized DAG plan.

    *dag_path* is the plan root — a local path or an ``s3://`` URI — or a
    runner string with independent component locations (see
    :meth:`~tidyrun.plan.PlanPaths.to_runner_string`).
    """
    _run_compiled_job(PlanPaths.from_runner_string(str(dag_path)), job_id)


def batch_entrypoint() -> None:
    """Container entry point for AWS Batch jobs.

    Reads the plan directory and job identity from environment variables, then
    runs the job exactly as the local subprocess executor would.

    For regular (non-array) jobs the required variables are:

    - ``TIDYRUN_PLAN_DIR`` — S3 URI or path of the materialised plan directory.
    - ``TIDYRUN_JOB_ID`` — the job id to execute.

    For array jobs AWS Batch sets ``AWS_BATCH_JOB_ARRAY_INDEX`` automatically.
    In that case the job id is resolved from ``TIDYRUN_JOB_IDS_JSON`` (a JSON
    array of all job ids in the array), indexed by ``AWS_BATCH_JOB_ARRAY_INDEX``.
    ``TIDYRUN_JOB_ID`` is ignored for array children.

    This function is registered as the ``tidyrun-batch-entrypoint`` console
    script and should be the ``CMD`` of your Batch container image.
    """
    import json

    plan_dir = os.environ.get("TIDYRUN_PLAN_DIR")
    if not plan_dir:
        print("TIDYRUN_PLAN_DIR is not set", file=sys.stderr)
        sys.exit(1)

    array_index_str = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if array_index_str is not None:
        job_ids_json = os.environ.get("TIDYRUN_JOB_IDS_JSON")
        if not job_ids_json:
            print(
                "AWS_BATCH_JOB_ARRAY_INDEX is set but TIDYRUN_JOB_IDS_JSON is not",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            job_ids: list[str] = json.loads(job_ids_json)
            index = int(array_index_str)
            job_id = job_ids[index]
        except (json.JSONDecodeError, ValueError, IndexError) as exc:
            print(f"Cannot resolve job_id from array index: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        job_id = os.environ.get("TIDYRUN_JOB_ID")
        if not job_id:
            print("TIDYRUN_JOB_ID is not set", file=sys.stderr)
            sys.exit(1)

    run_materialized_job(plan_dir, job_id)


def _run_compiled_job_entrypoint() -> None:  # pyright: ignore[reportUnusedFunction]
    # Invoked by _run_job_in_subprocess with: <runner_string> <job_id>
    if len(sys.argv) != 3:
        raise ValueError("Expected arguments: <runner_string> <job_id>")
    plan_paths = PlanPaths.from_runner_string(sys.argv[1])
    _run_compiled_job(plan_paths, sys.argv[2])


def _run_job_inline(runner_string: str, job_id: str) -> None:
    """Execute a job in the current process (thread and process pool modes)."""
    _run_compiled_job(PlanPaths.from_runner_string(runner_string), job_id)


def _run_job_in_subprocess(runner_string: str, job_id: str) -> None:
    """Execute a job in an isolated Python subprocess."""
    current_pythonpath = os.pathsep.join(path for path in sys.path if path)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    combined_pythonpath = (
        current_pythonpath
        if not existing_pythonpath
        else f"{current_pythonpath}{os.pathsep}{existing_pythonpath}"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = combined_pythonpath
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tidyrun.execute import _run_compiled_job_entrypoint; "
                "_run_compiled_job_entrypoint()"
            ),
            runner_string,
            job_id,
        ],
        check=True,
        env=env,
    )


def job_runner_for_mode(execution_mode: ExecutionMode) -> JobRunner:
    """Return the (picklable, module-level) job runner for an execution mode."""
    if execution_mode == "subprocess":
        return _run_job_in_subprocess
    return _run_job_inline


def pool_for_mode(execution_mode: ExecutionMode, max_workers: int) -> Executor:
    """Create the worker pool matching an execution mode.

    ``"thread"`` and ``"subprocess"`` jobs are dispatched from threads;
    ``"process"`` uses a process pool.
    """
    if execution_mode == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    return ThreadPoolExecutor(max_workers=max_workers)


# ---------------------------------------------------------------------------
# Group metadata (dict-folder .tidyrun files for DAG group nodes)
# ---------------------------------------------------------------------------


def _combined_child_checksum(
    outputs_path: Path | CloudPath, child_ids: list[str]
) -> Any:
    from tidyrun.serialization.metadata import (
        checksum_for_named_children,
        read_metadata,
    )

    children: list[tuple[str, Any]] = []
    for child_id in child_ids:
        encoded_name = child_id.rsplit("/", 1)[-1]
        meta = read_metadata(job_output_base(outputs_path, child_id))
        children.append((encoded_name, meta["checksum"]))
    return checksum_for_named_children(children)


def write_group_metadata(
    group_id: str, child_ids: list[str], outputs_path: Path | CloudPath
) -> None:
    """Write the dict-folder .tidyrun metadata for a DAG group node.

    Reads the checksum from each child's existing .tidyrun file, combines
    them, and writes the result at the path that corresponds to *group_id*
    inside *outputs_path*.
    """
    from tidyrun.serialization.metadata import write_metadata

    write_metadata(
        job_output_base(outputs_path, group_id),
        encoding="dict-folder",
        suffix="",
        checksum=_combined_child_checksum(outputs_path, child_ids),
    )


def write_root_metadata(outputs_path: Path | CloudPath, child_ids: list[str]) -> None:
    """Write the root dict-folder .tidyrun so outputs match serialize(dict, ...)."""
    from tidyrun.serialization.metadata import write_metadata

    write_metadata(
        outputs_path,
        encoding="dict-folder",
        suffix="",
        checksum=_combined_child_checksum(outputs_path, child_ids),
    )


# ---------------------------------------------------------------------------
# Dependency-ordered scheduling
# ---------------------------------------------------------------------------


class _GraphScheduler:
    """Run a job dependency graph to completion, serially or on an executor.

    The scheduler owns the topological bookkeeping (ready set, dependents,
    completion, skip and failure handling).  Callers provide:

    - ``job_runner``/``runner_string``: how to run one real job;
    - ``inline_runners``: synthetic jobs (DAG group aggregators) executed
      inline in the scheduling thread once their children complete;
    - optional array-group and per-job-resource information, used when the
      executor supports ``submit_array_with_options``/``submit_with_options``.
    """

    def __init__(
        self,
        dependencies: Mapping[str, set[str]],
        *,
        plan_paths: PlanPaths,
        plan_dir: Path | CloudPath,
        job_runner: JobRunner,
        runner_string: str,
        reporter: ProgressReporter,
        skip_completed: bool = False,
        skip_running: bool = False,
        inline_runners: Mapping[str, Callable[[], None]] | None = None,
        array_groups: Mapping[str, set[str]] | None = None,
        array_group_by_job_id: Mapping[str, str] | None = None,
        resources_by_job_id: Mapping[str, dict[str, str | int]] | None = None,
    ) -> None:
        self._pending = {job_id: set(deps) for job_id, deps in dependencies.items()}
        self._dependents: dict[str, set[str]] = {}
        for job_id, deps in self._pending.items():
            for dep in deps:
                self._dependents.setdefault(dep, set()).add(job_id)

        self._plan_paths = plan_paths
        self._plan_dir = plan_dir
        self._job_runner = job_runner
        self._runner_string = runner_string
        self._reporter = reporter
        self._skip_completed = skip_completed
        self._skip_running = skip_running
        self._inline_runners = dict(inline_runners or {})
        self._array_groups = dict(array_groups or {})
        self._array_group_by_job_id = dict(array_group_by_job_id or {})
        self._resources_by_job_id = dict(resources_by_job_id or {})

        self._ready = sorted(
            job_id for job_id, deps in self._pending.items() if not deps
        )
        self._completed: set[str] = set()

    def run(self, executor: Executor | None) -> None:
        if executor is None:
            self._run_serial()
        else:
            self._run_on_executor(executor)
        if len(self._completed) != len(self._pending):
            raise ValueError("Cycle detected in materialized job dependencies")

    # -- shared helpers -----------------------------------------------------

    def _should_skip(self, job_id: str) -> bool:
        outputs = self._plan_paths.outputs
        if self._skip_completed and job_output_exists(outputs, job_id):
            return True
        if self._skip_running and running_path(outputs, job_id).exists():
            return True
        return False

    def _mark_completed(self, job_id: str, *, skipped: bool = False) -> None:
        if job_id in self._completed:
            return
        self._completed.add(job_id)
        if job_id not in self._inline_runners:
            self._reporter.step(job_id, skipped=skipped)
        for dependent in self._dependents.get(job_id, set()):
            self._pending[dependent].discard(job_id)
            if not self._pending[dependent]:
                self._ready.append(dependent)

    def _execution_error(self, job_id: str, cause: BaseException) -> DAGExecutionError:
        cancelled = {
            other
            for other in self._pending
            if other not in self._completed and other != job_id
        }
        return DAGExecutionError(
            failed_job_id=job_id,
            cause=cause,
            completed_jobs=self._completed,
            cancelled_jobs=cancelled,
            plan_dir=self._plan_dir,
            outputs_path=self._plan_paths.outputs,
        )

    # -- serial execution ---------------------------------------------------

    def _run_serial(self) -> None:
        while self._ready:
            job_id = self._ready.pop(0)
            if self._should_skip(job_id):
                self._mark_completed(job_id, skipped=True)
            elif job_id in self._inline_runners:
                self._inline_runners[job_id]()
                self._mark_completed(job_id)
            else:
                try:
                    self._job_runner(self._runner_string, job_id)
                except Exception as exc:
                    raise self._execution_error(job_id, exc) from exc
                self._mark_completed(job_id)
            self._ready.sort()

    # -- executor-based execution -------------------------------------------

    def _run_on_executor(self, executor: Executor) -> None:
        futures: dict[Future[Any], set[str]] = {}
        submitted: set[str] = set()
        submit_with_options = getattr(executor, "submit_with_options", None)
        submit_array_with_options = getattr(executor, "submit_array_with_options", None)

        if self._resources_by_job_id and submit_with_options is None:
            raise ValueError(
                "job_resources requires an executor that supports submit_with_options"
            )

        def _common_options_for_jobs(
            job_ids: list[str],
        ) -> dict[str, str | int] | None:
            """Return the shared submission options, or None when they differ."""
            options_list = [
                dict(self._resources_by_job_id.get(job_id, {})) for job_id in job_ids
            ]
            first = options_list[0]
            for options in options_list[1:]:
                if options != first:
                    return None
            return first

        def _try_submit_array(job_id: str) -> bool:
            """Submit *job_id* together with its ready array siblings, if possible."""
            array_group = self._array_group_by_job_id.get(job_id)
            if array_group is None or submit_array_with_options is None:
                return False
            ready_set = set(self._ready)
            batch = sorted(
                sibling
                for sibling in self._array_groups.get(array_group, set())
                if sibling == job_id or sibling in ready_set
            )
            if len(batch) <= 1:
                return False
            common_options = _common_options_for_jobs(batch)
            if common_options is None:
                return False

            submitted.update(batch)
            batch_set = set(batch)
            self._ready = [jid for jid in self._ready if jid not in batch_set]

            to_run = [jid for jid in batch if not self._should_skip(jid)]
            to_run_set = set(to_run)
            for jid in batch:
                if jid not in to_run_set:
                    self._mark_completed(jid, skipped=True)
            if not to_run:
                return True

            array_options = dict(common_options)
            array_options.setdefault("job_name", array_group)
            submission = submit_array_with_options(
                self._job_runner,
                self._runner_string,
                to_run,
                sbatch_options=array_options,
            )
            array_future = cast(Future[Any], getattr(submission, "future", submission))
            submitted_job_ids = cast(
                tuple[str, ...], getattr(submission, "job_ids", tuple(to_run))
            )
            futures[array_future] = set(submitted_job_ids)
            return True

        def _submit_ready() -> None:
            self._ready.sort()
            while self._ready:
                job_id = self._ready.pop(0)
                if job_id in submitted:
                    continue
                submitted.add(job_id)
                if self._should_skip(job_id):
                    self._mark_completed(job_id, skipped=True)
                    continue
                if job_id in self._inline_runners:
                    self._inline_runners[job_id]()
                    self._mark_completed(job_id)
                    continue
                if _try_submit_array(job_id):
                    continue
                if job_id in self._resources_by_job_id and submit_with_options:
                    future = submit_with_options(
                        self._job_runner,
                        self._runner_string,
                        job_id,
                        sbatch_options=dict(self._resources_by_job_id[job_id]),
                    )
                else:
                    future = executor.submit(
                        self._job_runner, self._runner_string, job_id
                    )
                futures[cast("Future[Any]", future)] = {job_id}

        _submit_ready()
        while futures:
            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                finished_job_ids = futures.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    for other in futures:
                        other.cancel()
                    raise self._execution_error(
                        sorted(finished_job_ids)[0], exc
                    ) from exc
                for job_id in sorted(finished_job_ids):
                    self._mark_completed(job_id)
            _submit_ready()


def execute_graph(
    dependencies: Mapping[str, set[str]],
    plan_paths: PlanPaths,
    plan_dir: Path | CloudPath,
    *,
    executor: Executor | None = None,
    max_workers: int | None = None,
    execution_mode: ExecutionMode = "subprocess",
    skip_completed: bool = False,
    skip_running: bool = False,
    reporter: ProgressReporter,
    inline_runners: Mapping[str, Callable[[], None]] | None = None,
    array_groups: Mapping[str, set[str]] | None = None,
    array_group_by_job_id: Mapping[str, str] | None = None,
    resources_by_job_id: Mapping[str, dict[str, str | int]] | None = None,
) -> None:
    """Run *dependencies* in topological order.

    When *max_workers* is given a pool matching *execution_mode* is created
    for the duration of the run; when *executor* is given it is used as-is
    (and not shut down); otherwise jobs run serially in this thread.
    """
    scheduler = _GraphScheduler(
        dependencies,
        plan_paths=plan_paths,
        plan_dir=plan_dir,
        job_runner=job_runner_for_mode(execution_mode),
        runner_string=plan_paths.to_runner_string(),
        reporter=reporter,
        skip_completed=skip_completed,
        skip_running=skip_running,
        inline_runners=inline_runners,
        array_groups=array_groups,
        array_group_by_job_id=array_group_by_job_id,
        resources_by_job_id=resources_by_job_id,
    )
    if max_workers is not None:
        with pool_for_mode(execution_mode, max_workers) as pool:
            scheduler.run(pool)
    else:
        scheduler.run(executor)


def execute_plan(
    plan_dir: Any,
    *,
    skip_completed: bool = False,
    skip_running: bool = False,
    max_workers: int | None = None,
    execution_mode: ExecutionMode = "subprocess",
    progress: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Execute all jobs found by scanning *plan_dir*/definitions/.

    This is the standalone counterpart to :meth:`DAG.execute_materialized` for
    the decentralised case where multiple DAGs have been independently
    materialised into the same plan directory using the ``prefix=`` parameter.
    Results are written to *plan_dir*/outputs/; use
    :func:`~tidyrun.deserialize` to read individual job outputs.

    Parameters
    ----------
    plan_dir:
        Root of the materialised plan (must contain a ``definitions/`` subdir).
    skip_completed:
        Skip jobs whose output already exists.
    skip_running:
        Skip jobs whose ``.running`` sentinel exists (already in-flight from
        another process).
    max_workers:
        Number of workers for parallel execution.
    execution_mode:
        ``"subprocess"`` (default), ``"thread"``, or ``"process"``.
    progress:
        Emit progress messages.
    progress_callback:
        Optional progress callback; defaults to :func:`print`.
    """
    plan_path = to_path(plan_dir)
    plan_paths = PlanPaths.from_root(plan_path)

    if not plan_paths.definitions.is_dir():
        raise ValueError(
            f"No materialized plan found at {plan_path}. Run materialize() first."
        )

    graph = read_plan_graph(plan_paths.definitions)
    if not graph.dependencies:
        return

    reporter = ProgressReporter(
        enabled=progress,
        callback=progress_callback,
        phase="execute",
        total=len(graph.dependencies),
    )
    reporter.info(f"starting ({len(graph.dependencies)} jobs)")
    execute_graph(
        graph.dependencies,
        plan_paths,
        plan_path,
        max_workers=max_workers,
        execution_mode=execution_mode,
        skip_completed=skip_completed,
        skip_running=skip_running,
        reporter=reporter,
    )
    reporter.info("done")
