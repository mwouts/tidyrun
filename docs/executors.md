# Executors

TidyRun separates *what to run* (the materialized plan) from *how to run it*
(the executor). Any `concurrent.futures.Executor` can be passed to
`execute_materialized`; TidyRun also ships two remote executors —
`SlurmExecutor` and `AwsBatchExecutor` — that additionally implement
`submit_with_options` and `submit_array_with_options` for per-job resource
control and efficient array-job submission.

## Local execution

The built-in `execution_mode` parameter covers the common local cases without
an external executor:

| Mode | Description |
|---|---|
| `"subprocess"` (default) | One isolated Python subprocess per job. Safest for reproducibility. |
| `"thread"` | Jobs run in threads within the current process. Low overhead; good for tests. |
| `"process"` | Jobs run via `ProcessPoolExecutor`. Process-level parallelism with worker reuse. |

```python
# Fast local test run
result = dag.execute_materialized(plan_dir, execution_mode="thread", max_workers=4)

# Production subprocess run
result = dag.execute_materialized(plan_dir, execution_mode="subprocess", max_workers=8)
```

---

## SLURM

`SlurmExecutor` submits each job as an `sbatch` task, polls `squeue` for
completion, and collects results through a shared filesystem.

### How it works

1. You call `dag.materialize("/shared/plans/run-001")` on the head node.
   TidyRun writes job definitions and serialized inputs to the shared filesystem.
2. You call `dag.execute_materialized(..., executor=SlurmExecutor(...))`.
   For each job TidyRun pickles the runner and job identity into `shared_dir`,
   then calls `sbatch` to launch the compute task.
3. Each compute node executes the `tidyrun` runner script, which unpickles
   the task, loads the job definition from `plan_dir`, deserializes inputs,
   runs the function, and writes outputs back.
4. Back on the head node, `execute_materialized` polls `squeue` until all
   jobs finish, then assembles the final result.

### Setup requirements

| Requirement | Details |
|---|---|
| `shared_dir` | Visible from both head node and all compute nodes (e.g. NFS, Lustre). Used for temporary pickle files and the runner script. |
| `plan_dir` | Also must be on shared storage — compute nodes read job definitions and write outputs there. |
| `sbatch` / `squeue` | Must be on `PATH` on the head node. |
| Python interpreter | `sys.executable` on the head node must be reachable at the same path from compute nodes (typical for NFS-mounted home directories or shared conda environments). |

### Basic usage

```python
from tidyrun import SlurmExecutor

with SlurmExecutor(
    shared_dir="/shared/tidyrun_scratch",
    partition="compute",
    time_limit="01:00:00",
    memory="8G",
    cpus_per_task=2,
) as executor:
    result = dag.execute_materialized(
        dag_path="/shared/plans/run-001",
        output_path="/shared/outputs/run-001",
        executor=executor,
    )
```

Per-node resource overrides can be passed at evaluation time:

```python
with SlurmExecutor(shared_dir="/shared/tidyrun_scratch", memory="8G") as executor:
    result = dag.execute_materialized(
        dag_path="/shared/plans/run-001",
        output_path="/shared/outputs/run-001",
        executor=executor,
        job_resources={
            "a": {"mem": "32G", "time": "04:00:00", "gres": "gpu:1"},
        },
    )
```

### Deploying a local git repository

On HPC clusters all nodes share a filesystem and the same Python environment,
so there is no container image to build. Instead, install your project from
git into the shared Python environment at submission time. The `SlurmExecutor`
runner script uses `sys.executable` — the same interpreter as the head node —
so any package installed before calling `execute_materialized` is immediately
available on compute nodes.

The example below trains a model on several regularisation strengths in
parallel (a `ParametrizedJob` submitted as one SLURM array job), then evaluates
each trained model — a dependent step that only runs after training finishes.

```python
import subprocess
from tidyrun import DAG, ParametrizedJob, SlurmExecutor
from my_research.experiments import train, evaluate

GIT_REPO_URL = "https://github.com/my-org/my-research.git"

# Pin to the current HEAD so every SLURM task runs exactly this code.
GIT_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

# Install this commit into the shared Python environment.
# The SlurmExecutor runner inherits sys.executable, so packages installed here
# are immediately visible on every compute node.
subprocess.run(
    ["pip", "install", "--quiet", f"git+{GIT_REPO_URL}@{GIT_COMMIT}"],
    check=True,
)

# Build the DAG.
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

trained = ParametrizedJob(
    func=train,
    parameter_names=["alpha"],
    parameter_values=[(a,) for a in alphas],
    kwargs={"dataset": "/shared/data/train.parquet"},
)
evaluated = ParametrizedJob(
    func=evaluate,
    parameter_names=["alpha"],
    parameter_values=[(a,) for a in alphas],
    kwargs={
        "model": trained,          # dependency: each alpha waits for its trained model
        "dataset": "/shared/data/test.parquet",
    },
)
dag = DAG({"train": trained, "evaluate": evaluated})

# Materialise to shared storage.
PLAN_DIR = f"/shared/plans/{GIT_COMMIT[:8]}"
dag.materialize(PLAN_DIR)

# Execute on SLURM.
with SlurmExecutor(
    shared_dir=f"/shared/tidyrun/{GIT_COMMIT[:8]}",
    partition="gpu",
    time_limit="02:00:00",
    memory="32G",
    cpus_per_task=8,
    gres="gpu:1",
) as executor:
    results = dag.execute_materialized(
        dag_path=PLAN_DIR,
        output_path=f"/shared/outputs/{GIT_COMMIT[:8]}",
        executor=executor,
    )

for alpha, metrics in results["evaluate"].to_dict().items():
    print(f"alpha={alpha:6.3f}  accuracy={metrics['accuracy']:.4f}")
```

For the `train` step, `SlurmExecutor` submits one SLURM array job with five
tasks (one per alpha value). SLURM sets `SLURM_ARRAY_TASK_ID` to `0`–`4` on
each task; the runner script uses it to pick the right job id. Each task loads
the job definition from `PLAN_DIR` and runs `train(alpha=..., dataset=...)`.
Once all training tasks succeed the five `evaluate` jobs are submitted the same
way.

!!! tip
    Using per-commit install paths (e.g. a conda env named after the commit
    hash) keeps experiments with different code versions reproducible without
    clobbering each other.
    [pixi](https://prefix.dev/docs/pixi/) can manage per-project lockfile
    environments on shared storage.

### `SlurmExecutor` reference

::: tidyrun.SlurmExecutor
    options:
      show_source: false
      members:
        - __init__
        - submit
        - submit_with_options
        - submit_array_with_options
        - shutdown

---

## AWS Batch

`AwsBatchExecutor` submits each job as a Batch container task, polls
`describe_jobs` for completion, and reads/writes all data via S3.

### How it works

1. You call `dag.materialize("s3://bucket/plans/run-001")` on your laptop.
   TidyRun serializes job definitions and inputs to S3.
2. You call `dag.execute_materialized(..., executor=AwsBatchExecutor(...))`.
   For each job TidyRun submits a Batch container task, passing the plan
   location and job id as environment variables.
3. Each container calls `tidyrun-batch-entrypoint`, which reads those
   variables, loads the job definition from S3, deserializes inputs, runs the
   function, and writes outputs back to S3.
4. Back on your laptop, `execute_materialized` polls Batch until all jobs
   finish, then assembles the final result from S3.

When a job fails, the raised error includes the container's exit code and
reason, the CloudWatch log stream name, and a direct link to the log stream
in the AWS console (for array jobs, the links point at the failed children).

### Container entrypoint

The `tidyrun-batch-entrypoint` command (installed with the package) is the only
piece of logic that must run inside the container. It handles two job shapes:

| Shape | Variables read |
|---|---|
| Single job | `TIDYRUN_PLAN_DIR`, `TIDYRUN_JOB_ID` |
| Array job child | `TIDYRUN_PLAN_DIR`, `TIDYRUN_JOB_IDS_JSON`, `AWS_BATCH_JOB_ARRAY_INDEX` (set by Batch) |

For array jobs, Batch sets `AWS_BATCH_JOB_ARRAY_INDEX` to the child's index
automatically. The entrypoint uses it to pick the right job from
`TIDYRUN_JOB_IDS_JSON`. This is how `ParametrizedJob` grids are submitted
efficiently as a single Batch array job.

### Minimal Dockerfile

If your functions live in a published package this is all you need:

```dockerfile
FROM python:3.12-slim
RUN pip install "tidyrun[s3]==<version>" my-package
CMD ["tidyrun-batch-entrypoint"]
```

!!! warning "Keep tidyrun versions aligned"
    The container reads the plan that your submitting machine wrote, so pin
    the **same tidyrun version** in the image as on the submitting machine.
    A container running an older tidyrun may not be able to read the plan at
    all — versions before 0.0.8 fail with
    ``Missing job definition file for job ...`` when ``TIDYRUN_PLAN_DIR`` is
    an ``s3://`` URI, because the runner interpreted the URI as a local path.

    Each plan records the tidyrun version that wrote it (``plan.toml`` at the
    plan root), and the runner logs a warning when its own version differs.

### Using a development version of tidyrun

When the submitting machine runs an unreleased tidyrun (a git checkout or
branch), the container must run that same version. Two ways to do it:

1. **Bake it into the image** — install tidyrun from your git ref instead of
   PyPI:

    ```dockerfile
    RUN pip install "tidyrun[s3] @ git+https://github.com/my-org/tidyrun@my-branch"
    ```

2. **Keep a generic image and bootstrap at runtime** — pass a pip
   requirement through ``TIDYRUN_PIP_SPEC``. The entrypoint installs it and
   re-executes itself before running the job, so every container picks up
   your development version without rebuilding the image:

    ```python
    executor = AwsBatchExecutor(
        job_queue="my-research-queue",
        job_definition="my-research-base:3",
        extra_env={
            "TIDYRUN_PIP_SPEC": "tidyrun[s3] @ git+https://github.com/my-org/tidyrun@my-branch",
        },
    )
    ```

    Any pip requirement works — a git URL with a branch or commit, a version
    pin like ``tidyrun[s3]==0.0.8``, or an S3-downloaded wheel path if your
    image fetches one. Runtime installs add a few seconds per container and
    require network access to the package source; prefer baking the image
    once your version stabilizes.

### Deploying a local git repository

A common research workflow is to run experiments against a specific commit of
an unpublished repository. The pattern below uses a single base image that
clones and installs the repository at a commit hash supplied at submission time.

**Project structure**

```
my_research/
├── Dockerfile
├── entrypoint.sh
├── pyproject.toml
└── my_research/
    ├── __init__.py
    └── experiments.py
```

**`Dockerfile`** — build once, push to ECR. Contains Python, git, and
`tidyrun[s3]`, but *not* the research code, which is cloned at runtime.

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install "tidyrun[s3]==<version>"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

CMD ["/usr/local/bin/entrypoint.sh"]
```

**`entrypoint.sh`**

```bash
#!/bin/bash
set -euo pipefail

: "${GIT_REPO_URL:?GIT_REPO_URL must be set}"
: "${GIT_COMMIT:?GIT_COMMIT must be set}"

git clone --quiet "$GIT_REPO_URL" /workspace
cd /workspace
git checkout --quiet "$GIT_COMMIT"
pip install --quiet -e "."

exec tidyrun-batch-entrypoint
```

!!! tip
    For private repositories pass an SSH key or use an HTTPS token via AWS
    Secrets Manager and inject it as an environment variable. See the
    [AWS Batch secrets documentation](https://docs.aws.amazon.com/batch/latest/userguide/specifying-sensitive-data.html).

**Submission script** — the same train/evaluate DAG as the SLURM example, but
using S3 for the plan directory and `extra_env` to pass the git coordinates to
every container.

```python
import subprocess
from tidyrun import DAG, ParametrizedJob, AwsBatchExecutor
from my_research.experiments import train, evaluate

GIT_REPO_URL = "https://github.com/my-org/my-research.git"

# Pin to the current HEAD so every Batch container runs exactly this code.
GIT_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

# Build the DAG.
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

trained = ParametrizedJob(
    func=train,
    parameter_names=["alpha"],
    parameter_values=[(a,) for a in alphas],
    kwargs={"dataset": "s3://my-bucket/data/train.parquet"},
)
evaluated = ParametrizedJob(
    func=evaluate,
    parameter_names=["alpha"],
    parameter_values=[(a,) for a in alphas],
    kwargs={
        "model": trained,          # dependency: each alpha waits for its trained model
        "dataset": "s3://my-bucket/data/test.parquet",
    },
)
dag = DAG({"train": trained, "evaluate": evaluated})

# Materialise to S3.
PLAN_DIR = f"s3://my-bucket/plans/{GIT_COMMIT[:8]}"
dag.materialize(PLAN_DIR)

# Execute on AWS Batch.
executor = AwsBatchExecutor(
    job_queue="my-research-queue",
    job_definition="my-research-base:3",    # the image built from the Dockerfile above
    extra_env={
        "GIT_REPO_URL": GIT_REPO_URL,
        "GIT_COMMIT": GIT_COMMIT,
    },
)
results = dag.execute_materialized(
    dag_path=PLAN_DIR,  # outputs are written to PLAN_DIR/outputs
    executor=executor,
)

for alpha, metrics in results["evaluate"].to_dict().items():
    print(f"alpha={alpha:6.3f}  accuracy={metrics['accuracy']:.4f}")
```

For the `train` step, `AwsBatchExecutor` submits one Batch array job with five
children. Batch sets `AWS_BATCH_JOB_ARRAY_INDEX` to `0`–`4` on each child;
the entrypoint resolves the correct job id from the index, clones the repo at
`GIT_COMMIT`, installs it, and runs the training function. Once all training
children succeed the five `evaluate` jobs are submitted the same way.

### IAM requirements

The task role attached to your Batch job definition needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ]
    }
  ]
}
```

The machine that calls `dag.materialize()` and `dag.execute_materialized()` also
needs `s3:PutObject` on the plan directory and `batch:SubmitJob`,
`batch:DescribeJobs` on the queue and job definition.

### `AwsBatchExecutor` reference

::: tidyrun.AwsBatchExecutor
    options:
      show_source: false
      members:
        - __init__
        - submit
        - submit_with_options
        - submit_array_with_options
        - shutdown
