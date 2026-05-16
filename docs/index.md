# Home

Welcome to **TidyRun** — a tool to orchestrate the compute and storage of Python DAGs.

## What is TidyRun?

TidyRun helps you both run Python DAG computations and store their outputs reliably. You can model deferred work with `Job`, `ParametrizedJob`, and `DAG`, execute locally or on schedulers, and persist results using the same serialization contract used across the project.

## Key Features

### 🧠 DAG Compute Orchestration
Define deferred computation graphs and evaluate them with dependency-aware scheduling:

- **Deferred primitives**: `Job`, `ParametrizedJob`, and nested `DAG`
- **Execution modes**: `subprocess` (default), `thread`, and `process`
- **Parallel execution**: `DAG.evaluate(max_workers=...)` for independent nodes
- **Pluggable executors**: local executors plus `SlurmExecutor` and `AwsBatchExecutor`
- **Robust failure model**: `DAGExecutionError` with failed/completed/cancelled job context
- **Resumable runs**: materialize plans and re-run with `skip_completed=True`

### 🚀 Smart Format Selection
Automatically chooses the best serialization format based on your data type:

- **Nested dicts** → Filesystem hierarchy
- **DataFrames** → Apache Parquet (with intelligent fallback)
- **Series** → Parquet or HDF5
- **Scalars** → JSON
- **Custom objects** → Pickle

### 💾 Metadata Tracking
Every output includes a `.tidyrun` metadata sidecar that tracks:

- Encoding format used
- Version for future compatibility
- Format migration information

### ⚡ Lazy Evaluation
Load only what you need, when you need it:

- Nested directories deserialize as `LazyDict` objects
- Values load on-demand each time they are accessed
- Perfect for large DAG outputs with selective access patterns

### ☁️ Optional S3 Support
Use `s3://...` locations when the optional S3 dependency is installed:

- `pip install tidyrun[s3]`
- Serialization stages through a local temporary directory, then uploads to S3
- Deserialization downloads the S3 object tree and reuses the normal local loader

### 🔄 Recursive Aggregation
Combine data across nested structures with a single call:
```python
results = deserialize("./experiments")
combined = results.concat(names=["run_id"])  # Stack all DataFrames
```

### 🔌 Extensible Pipeline
Add support for custom types by creating custom encoders:
```python
from tidyrun.serialization import EncoderSpec, serialize

my_encoder = EncoderSpec(
    name="my-type",
    predicate=lambda v: isinstance(v, MyClass),
    serializer=encode_func,
    deserializer=decode_func,
)
serialize(value, "./output", encoders=(my_encoder,) + default_encoders())
```

### 🛡️ Intelligent Fallback
If one encoder fails, the next in the chain is automatically tried:

- DataFrame with multi-index → Parquet fails → HDF5 succeeds ✓
- Series without parquet engine → Parquet fails → HDF5 succeeds ✓
- Custom object → Pickle fallback always works ✓

## Quick Start

### DAG Compute

```python
from tidyrun import DAG, Job

# Define and execute deferred compute
def train_step(x: int) -> int:
    return x * x

dag = DAG()
dag["step_1"] = Job(func=train_step, kwargs={"x": 3})
outputs = dag.evaluate("./run_001", execution_mode="thread", max_workers=2)
print(outputs["step_1"])  # 9
```

### Serialization and Lazy Loading

```python
from tidyrun import deserialize, serialize
import pandas as pd

# Save nested data
serialize({
    "config": {"lr": 0.001, "epochs": 100},
    "metrics": pd.DataFrame({"loss": [0.5, 0.3, 0.2]}),
}, "./results/exp_001")

# Load with lazy evaluation
results = deserialize("./results/exp_001")
config = results["config"]      # Loads on access
metrics = results["metrics"]    # DataFrame loaded directly
```

## Documentation

- **[DAG Guide](dag.md)** — Jobs, parametrized jobs, and parallel DAG evaluation
- **[Serialization Guide](serialization.md)** — Complete API reference with examples
- **[Quick Start](quick_start.md)** — Local docs workflow, deployment, and authoring notes

## Installation

```bash
pip install tidyrun
```

Or with optional pandas and parquet support:

```bash
pip install tidyrun[pandas]
```

## Contributing

See [Contributing](contributing.md) for development workflow and project status.

## License

TidyRun is released under the [LICENSE](https://github.com/mwouts/tidyrun/blob/main/LICENSE).

## Support

For issues, questions, or feedback:

- Open an [issue on GitHub](https://github.com/mwouts/tidyrun/issues)
- Check the [documentation](serialization.md)
