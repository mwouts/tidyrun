TidyRun
=======

A tool to orchestrate the compute and storage of Python DAGs

## Features

### Serialization Framework

TidyRun includes a comprehensive serialization system for storing and retrieving Python objects with smart format selection:

- **Type-Aware Encoding**: Automatically selects the best format (folder, Parquet, HDF5, JSON, or pickle) based on value type
- **Lazy Evaluation**: Directories deserialize into `LazyDict` objects that load values on-demand
- **Recursive Concatenation**: Aggregate DataFrames across nested structures with `LazyDict.concat()`
- **Metadata Sidecars**: Each output is tracked with `.tidyrun` metadata files for format versioning
- **Extensible Pipeline**: Customize encoders or add support for custom types
- **Intelligent Fallback**: Parquet → HDF5 → JSON → Pickle chain ensures robust serialization

**Quick Example:**
```python
from tidyrun import serialize, deserialize

# Save nested data with smart format selection
serialize({
    "metrics": pd.DataFrame(...),
    "config": {"lr": 0.001},
}, "./results/exp_001")

# Load with lazy evaluation
results = deserialize("./results/exp_001")
df = results["metrics"]  # Loads on access, cached

# Aggregate across nested structures
combined = results.concat(names=["run_id"])
```

**Learn More:**
- [Quick Start](docs/quick_start.md) — Local docs workflow and publishing notes
- [Serialization Guide](docs/serialization.md) — Complete API reference, quick reference, and examples
- [Design Decisions](docs/design-decisions.md) — Architecture rationale and trade-offs
