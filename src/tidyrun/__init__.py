from importlib.metadata import version
from .dag import (
    DAG,
    DAGExecutionError,
    load_callable,
    load_job_definition,
    load_job_inputs,
    run_materialized_job,
)
from .aws_batch_executor import AwsBatchExecutor
from .job import Job, ParametrizedJob
from .serialization import LazyDict, deserialize, serialize
from .slurm_executor import SlurmExecutor

__version__ = version("tidyrun")

__all__ = [
    "__version__",
    "DAG",
    "DAGExecutionError",
    "AwsBatchExecutor",
    "Job",
    "LazyDict",
    "ParametrizedJob",
    "SlurmExecutor",
    "deserialize",
    "load_callable",
    "load_job_definition",
    "load_job_inputs",
    "run_materialized_job",
    "serialize",
]
