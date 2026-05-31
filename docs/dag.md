# DAG Guide

## Overview

TidyRun provides three core building blocks for deferred computation:

- `Job`: one deferred function call with named kwargs
- `ParametrizedJob`: a parameter grid over keys that slices into nested jobs
- `DAG`: a key-addressable mapping of deferred nodes that can be evaluated to disk

A DAG evaluation writes outputs with the same storage contract used by
`serialize(...)`, so `deserialize(...)` returns a matching `LazyDict` tree.

## Core Concepts

### Job

A `Job` holds a Python callable and named arguments.

```python
from tidyrun import Job


def add(x: int, y: int) -> int:
    return x + y


job = Job(func=add, kwargs={"x": 1, "y": 2})
```

Arguments can be plain values, `LazyDict`, `Job`, `ParametrizedJob`, or `DAG`.

### ParametrizedJob

A `ParametrizedJob` represents a Cartesian-style parameterized computation.
Accessing a key fixes the first parameter:

- returns another `ParametrizedJob` while parameters remain
- returns a concrete `Job` when the last parameter is fixed

```python
from tidyrun import ParametrizedJob


def score(model: str, split: str, prefix: str = "") -> str:
    return f"{prefix}{model}:{split}"


grid = ParametrizedJob(
    func=score,
    parameter_names=["model", "split"],
    parameter_values=[("m1", "train"), ("m1", "test"), ("m2", "train")],
    kwargs={"prefix": "run="},
)

model_slice = grid["m1"]
leaf_job = model_slice["train"]
```

### DAG

A `DAG` maps keys (same key type contract as `keys.py`) to nodes.
Supported node types are:

- `Job`
- `ParametrizedJob`
- nested `DAG`

```python
from tidyrun import DAG, Job


def square(x: int) -> int:
    return x * x


dag = DAG()
dag["a"] = Job(func=square, kwargs={"x": 3})
```

## Evaluation

`DAG.evaluate(...)` is materialize-first:

1. Compile the DAG into a plan directory (by default `<target>/plan`)
2. Execute jobs in dependency order
3. Serialize top-level outputs to the outputs directory (by default `<target>/outputs`)

By default, jobs run in isolated subprocesses.

### Core DAG Lifecycle APIs

These three methods cover the most common lifecycle for local and resumable
execution:

- evaluate: one-call workflow that materializes a plan, executes it, and
    writes top-level outputs to your run outputs directory.
- materialize: compile only (no execution). Use this when you want a stable,
    inspectable on-disk plan before running jobs.
- execute_materialized: run an already materialized plan, optionally with
    skip_completed=True to resume partially completed runs.

Typical pattern:

1. Use evaluate for everyday runs.
2. Use materialize + execute_materialized for debugging, reproducibility,
     or resumable workflows.

Default layout for `dag.evaluate("./exp1")`:

- plan: `./exp1/plan`
- outputs: `./exp1/outputs`

You can also skip `target` entirely when both paths are explicit:

```python
result = dag.evaluate(
    dag_path="./exp1-plan",
    output_path="./exp1-outputs",
)
```

### Sequential Evaluation

```python
result = dag.evaluate("./sequential")
print(result["a"])  # 9
```

### Execution Modes

Select execution behavior with `execution_mode`:

- `"subprocess"` (default): isolated Python subprocess per job
- `"thread"`: run jobs in the current process (lower overhead)
- `"process"`: run jobs using `ProcessPoolExecutor`

As a rule of thumb, start with `"subprocess"` for the safest isolation and
reproducibility. Choose `"thread"` for lightweight local runs where low
overhead matters (for example during rapid iteration or tests). Choose
`"process"` when running many local CPU-bound jobs and you want process-level
parallelism with worker reuse through a process pool.

```python
# Fast local testing (no subprocess spawn per job)
result = dag.evaluate("./thread-mode", execution_mode="thread")

# Process pool execution
result = dag.evaluate(
    "./process-mode",
    execution_mode="process",
    max_workers=4,
)
```

### Local Parallel Evaluation

Use `max_workers` to evaluate independent jobs in parallel.

```python
# Thread pool (thread/subprocess modes)
result = dag.evaluate("./threaded", max_workers=4, execution_mode="thread")

# Process pool (process mode)
result = dag.evaluate("./process-pooled", max_workers=4, execution_mode="process")
```

### Failure Handling and Resume

DAG execution fails fast. If any job fails, scheduling stops and a
`DAGExecutionError` is raised with structured context:

- `failed_job_id`
- `cause`
- `completed_jobs`
- `cancelled_jobs`

```python
from tidyrun import DAGExecutionError

try:
    dag.evaluate("./run")
except DAGExecutionError as exc:
    print("failed:", exc.failed_job_id)
    print("completed:", sorted(exc.completed_jobs))
    print("cancelled:", sorted(exc.cancelled_jobs))
```

To resume after fixing a failing job, run from an existing materialized plan
and set `skip_completed=True` so already-written outputs are reused:

If outputs already exist and `skip_completed=False` (default),
`execute_materialized(...)` now raises an error to prevent accidental mixing
of previous and newly computed results.

```python
plan_dir = dag.materialize("./run/plan")

result = dag.execute_materialized(
    dag_path=plan_dir,
    output_path="./run/outputs",
    skip_completed=True,
)
```

### Progress Logging

Use `progress=True` to emit simple progress logs during plan compilation and
job execution.

```python
result = dag.evaluate(
    "./run",
    progress=True,
)
```

You can also provide a custom callback to collect or redirect progress lines:

```python
messages: list[str] = []
result = dag.evaluate(
    "./run",
    progress=True,
    progress_callback=messages.append,
)
```

If outputs are obsolete or wrong, clear them before resubmitting:

```python
# Remove all outputs
dag.clear_outputs("./run/plan")

# Or remove specific job outputs only
dag.clear_outputs("./run/plan", job_ids=["train/model_a", "metrics/model_a"])
```

### Custom Executor

You can pass your own `concurrent.futures.Executor`.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as pool:
    result = dag.evaluate("./custom", executor=pool, execution_mode="thread")
```

Pass either `executor` or `max_workers`, not both.

### Materialized Plan Helpers

For debugging/reproducibility, you can inspect materialized artifacts directly:

```python
from tidyrun import load_callable, load_job_definition, load_job_inputs

plan_dir = dag.materialize("./experiment/plan")
definition = load_job_definition(plan_dir, "a")
func = load_callable(definition, plan_dir)
kwargs = load_job_inputs(definition, plan_dir)
print(func(**kwargs))
```

### SLURM Executor

`SlurmExecutor` submits each job as an `sbatch` task, polling `squeue` for
completion. The plan directory and `shared_dir` must both be on shared storage
visible from all compute nodes. See the [Executors guide](executors.md#slurm) for setup
instructions, resource configuration, and a full deployment example with git
commit pinning.

```python
from tidyrun import SlurmExecutor

with SlurmExecutor(
    shared_dir="/shared/tidyrun_scratch",
    partition="compute",
    time_limit="01:00:00",
    memory="8G",
) as executor:
    result = dag.execute_materialized(
        dag_path="/shared/plans/run-001",
        output_path="/shared/outputs/run-001",
        executor=executor,
    )
```

### AWS Batch Executor

`AwsBatchExecutor` submits each job as a Batch container task, polling
`describe_jobs` for completion. The plan directory must be an S3 URI, and the
container image must call `tidyrun-batch-entrypoint` as its `CMD`. See the
[Executors guide](executors.md#aws-batch) for the container setup, IAM
requirements, and a full deployment example with git commit pinning.

```python
from tidyrun import AwsBatchExecutor

with AwsBatchExecutor(
    job_queue="my-queue",
    job_definition="my-worker:1",
) as executor:
    result = dag.execute_materialized(
        dag_path="s3://my-bucket/plans/run-001",
        output_path="s3://my-bucket/outputs/run-001",
        executor=executor,
    )
```

## End-to-End Example

```python
from tidyrun import DAG, Job, ParametrizedJob


def metric(model: str, split: str, base: int = 1) -> str:
    return f"{model}:{split}:{base}"


scores = ParametrizedJob(
    func=metric,
    parameter_names=["model", "split"],
    parameter_values=[("m1", "train"), ("m1", "test"), ("m2", "train")],
    kwargs={"base": 10},
)

summary = Job(func=lambda value: f"summary={value}", kwargs={"value": "ok"})


dag = DAG()
dag["scores"] = scores
dag["summary"] = summary

outputs = dag.evaluate("./experiment", max_workers=4)
print(outputs["scores"]["m1"]["train"])  # m1:train:10
print(outputs["summary"])                 # summary=ok
```

## Notes

- Evaluated outputs are serialized using the same metadata sidecar mechanism as
    the serialization API.
- The on-disk plan format is intended for reproducibility: you can re-run one
    job later via `run_materialized_job(plan_dir, job_id)`.
- `job_resources` is keyed by top-level DAG keys and is only applied when the
    executor implements `submit_with_options(...)`.
