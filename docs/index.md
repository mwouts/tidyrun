# Home

Welcome to **TidyRun** — a tool to orchestrate the compute and storage of Python DAGs.

## What is TidyRun?

TidyRun provides a comprehensive framework for managing data serialization in Python DAG workflows. Whether you're building data science pipelines, machine learning experiments, or complex computational workflows, TidyRun handles the storage and retrieval of your results with minimal configuration.

## Key Features

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

```python
from tidyrun import serialize, deserialize
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

- **[Serialization Guide](serialization.md)** — Complete API reference with examples
- **[Design Decisions](design-decisions.md)** — Architecture rationale and trade-offs
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

Contributions are welcome! Please see [Contributing](contributing.md) for guidelines.

## License

TidyRun is released under the [LICENSE](https://github.com/mwouts/tidyrun/blob/main/LICENSE).

## Support

For issues, questions, or feedback:
- Open an [issue on GitHub](https://github.com/mwouts/tidyrun/issues)
- Check the [documentation](serialization.md)
- Review [design decisions](design-decisions.md) for architecture details
