TidyRun
=======

A tool to orchestrate the compute and storage of Python DAGs

## Features

### Compute Orchestration

TidyRun provides first-class deferred compute primitives for DAG workflows:

- **Deferred Primitives**: Model work with `Job`, `ParametrizedJob`, and nested `DAG`
- **Dependency-Aware Scheduling**: Evaluate DAGs with topological execution and fail-fast behavior
- **Execution Modes**: Choose `subprocess` (default), `thread`, or `process`
- **Parallel Evaluation**: Run independent nodes with `DAG.evaluate(max_workers=...)`
- **Materialized Plans**: Compile reproducible execution plans before running jobs
- **Resumable Runs**: Re-run materialized plans with `execute_materialized(skip_completed=True)`
- **Pluggable Executors**: Use local executors, `SlurmExecutor`, or `AwsBatchExecutor`

### Serialization and Storage

TidyRun also includes a comprehensive serialization system for storing and retrieving Python objects:

- **Type-Aware Encoding**: Automatically selects folder, Parquet, HDF5, JSON, or pickle based on value type
- **Lazy Evaluation**: Directories deserialize into `LazyDict` objects that load values on-demand
- **Recursive Concatenation**: Aggregate DataFrames across nested structures with `LazyDict.concat()` (optionally parallel with `max_workers`)
- **Metadata Sidecars**: Each output is tracked with `.tidyrun` metadata files for format versioning and checksums
- **Checksum Return Value**: `serialize(...)` returns checksum information (`algorithm`, `digest`) for the serialized payload
- **Extensible Pipeline**: Customize encoders or add support for custom types
- **Intelligent Fallback**: Parquet → HDF5 → JSON → Pickle chain ensures robust serialization

**Quick Example:**

Compute (DAG execution):

```python
from tidyrun import DAG, Job


def square(x: int) -> int:
    return x * x


dag = DAG()
dag["a"] = Job(func=square, kwargs={"x": 3})

# Fast local execution without subprocess spawn overhead
outputs = dag.evaluate("./local_dag", execution_mode="thread", max_workers=4)
print(outputs["a"])  # 9
```

Serialization and lazy loading:

```python
from tidyrun import serialize, deserialize
import pandas as pd

# Save nested data with smart format selection
serialize({
    "metrics": pd.DataFrame({"score": [9]}),
    "config": {"lr": 0.001},
}, "./results/exp_001")

# Load with lazy evaluation
results = deserialize("./results/exp_001")
df = results["metrics"]  # Loads on access

# Aggregate across nested structures
combined = results.concat(names=["run_id"])
```

**Learn More:**
- [Quick Start](docs/quick_start.md) — Local docs workflow and publishing notes
- [DAG Guide](docs/dag.md) — Jobs, parametrized jobs, executors, and evaluation modes
- [Serialization Guide](docs/serialization.md) — Complete API reference, quick reference, and examples
