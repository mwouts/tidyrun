from importlib.metadata import version

from .dag import DAG, ParametrizedJob
from .execute import (
    DAGExecutionError,
    batch_entrypoint,
    execute_plan,
    run_materialized_job,
)
from .executors import AwsBatchExecutor, SlurmExecutor
from .job import Job
from .plan import (
    PlanPaths,
    get_job_states,
    load_callable,
    load_inputs_and_callable,
    load_job_definition,
    load_job_inputs,
    rerun_snippet,
)
from .serialization import LazyDict, deserialize, serialize

__version__ = version("tidyrun")

__all__ = [
    "__version__",
    "DAG",
    "DAGExecutionError",
    "AwsBatchExecutor",
    "batch_entrypoint",
    "Job",
    "LazyDict",
    "ParametrizedJob",
    "PlanPaths",
    "SlurmExecutor",
    "deserialize",
    "execute_plan",
    "get_job_states",
    "load_callable",
    "load_inputs_and_callable",
    "load_job_definition",
    "load_job_inputs",
    "rerun_snippet",
    "run_materialized_job",
    "serialize",
]
