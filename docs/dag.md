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

Use `SlurmExecutor` when you want each top-level DAG node to run as a SLURM
batch job.

```python
from tidyrun import DAG, Job, SlurmExecutor


def square(x: int) -> int:
    return x * x


dag = DAG()
dag["a"] = Job(func=square, kwargs={"x": 3})

with SlurmExecutor(
    shared_dir="/shared/tidyrun_jobs",
    partition="debug",
    time_limit="01:00:00",
    memory="8G",
    cpus_per_task=2,
    gres="gpu:1",
) as executor:
    outputs = dag.evaluate("/shared/tidyrun_outputs/run_001", executor=executor)

print(outputs["a"])  # 9
```

Per-node overrides can be provided at evaluation time:

```python
job_resources = {
    "a": {"mem": "32G", "time": "04:00:00"},
    # other keys use executor defaults
}

with SlurmExecutor(shared_dir="/shared/tidyrun_jobs", memory="8G") as executor:
    outputs = dag.evaluate(
        "/shared/tidyrun_outputs/run_001",
        executor=executor,
        job_resources=job_resources,
    )
```

Notes:

- `shared_dir` must be visible from both submission and compute nodes.
- Submitted callables and arguments must be pickle-serializable.
- `SlurmExecutor` uses `sbatch` for submission and `squeue` for completion
  polling.
- Resource settings can be passed directly (`time_limit`, `memory`,
  `cpus_per_task`, etc.).
- You can still use `sbatch_options` for advanced flags; when both are set,
  `sbatch_options` values take precedence.
- `job_resources` applies per-node overrides by DAG key and requires an
  executor that supports `submit_with_options` (such as `SlurmExecutor`).

### AWS Batch Executor

`AwsBatchExecutor` integrates with AWS Batch by submitting each materialized
job as a Batch job and passing:

- `TIDYRUN_PLAN_DIR`
- `TIDYRUN_JOB_ID`

as container environment variables.

For parametrized jobs, executors that implement array submission (including
`AwsBatchExecutor` and `SlurmExecutor`) receive homogeneous ready batches as
one array submission. Job ids are still the normal relative DAG paths
(`group/a`, `scores/m1/train`, etc.) and are passed unchanged in metadata.

```python
from tidyrun import AwsBatchExecutor

with AwsBatchExecutor(
    job_queue="tidyrun-queue",
    job_definition="tidyrun-worker:1",
) as executor:
    outputs = dag.evaluate(
        "./aws",
        executor=executor,
        execution_mode="thread",  # executor handles remote execution
    )
```

Use `job_resources` to pass per-node submit parameters to
`AwsBatchExecutor.submit_with_options(...)`.

`AwsBatchExecutor.submit_array_with_options(...)` submits one AWS Batch array
job and provides both:

- `TIDYRUN_JOB_ID`: first logical job id (for backward compatibility)
- `TIDYRUN_JOB_IDS_JSON`: JSON array of all logical job ids in submission order

along with matching submit parameters:

- `tidyrun_job_id`
- `tidyrun_job_ids_json`

Array workers can select their logical job id using
`AWS_BATCH_JOB_ARRAY_INDEX`.

Submission checklist:

- Install optional dependencies: `pip install tidyrun[s3]`
- Ensure the Batch worker image includes `tidyrun` and your callable modules
- Use a plan path accessible to workers (local shared path or `s3://...`)
- Configure the Batch job definition so the container command reads
    `TIDYRUN_PLAN_DIR` and `TIDYRUN_JOB_ID` and runs one materialized job

Example worker entrypoint logic:

```python
import json
import os
from tidyrun import run_materialized_job

plan_dir = os.environ["TIDYRUN_PLAN_DIR"]
job_ids_json = os.environ.get("TIDYRUN_JOB_IDS_JSON")
if job_ids_json:
    idx = int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])
    job_id = json.loads(job_ids_json)[idx]
else:
    job_id = os.environ["TIDYRUN_JOB_ID"]

run_materialized_job(plan_dir, job_id)
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
