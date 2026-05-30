# TidyRun Serialization Guide

## Overview

This page covers serialization only: storing and retrieving Python objects
(`serialize`, `deserialize`, metadata, encoders, and `LazyDict`).

For deferred-compute APIs (`DAG`, `Job`, `ParametrizedJob`) and DAG execution
patterns, see the dedicated [DAG Guide](dag.md).

The TidyRun serialization framework provides a comprehensive, extensible system
for storing and retrieving Python objects, including nested dictionaries,
DataFrames, Series, and arbitrary Python objects, using a filesystem hierarchy.

### Key Features

- **Type-Aware Encoding**: Automatically selects the best format (folder, parquet, HDF5, JSON, or pickle) based on value type
- **Metadata Sidecars**: Each output is accompanied by a `.tidyrun` metadata file recording encoding format, version, and checksum
- **Lazy Evaluation**: Directories deserialize into `LazyDict` objects that load values on-demand on each access
- **Unified Path Handling**: Paths are normalized via `cloudpathlib.AnyPath` for local and cloud-backed locations
- **Optional S3 Support**: `s3://...` locations work when the `boto3` extra is installed
- **Recursive Concatenation**: `LazyDict.concat()` method provides pandas-style aggregation across nested structures
- **Parallel Leaf Loading**: `LazyDict.concat(max_workers=...)` can load selected leaf values concurrently
- **Fallback Chain**: Intelligent fallback routing (e.g., parquet → HDF5 when parquet encoding fails)
- **Extensible Pipeline**: Users can provide custom encoder sequences and compose encoders
- **Direct Import Path**: Use `tidyrun.serialization` for the serialization API

## Quick Start

### Basic Serialization

```python
from tidyrun import serialize, deserialize
import pandas as pd

# Serialize a simple value
data = {"experiment": "run_1", "results": 42}
serialize(data, "./output/my_result")

# Deserialize back (returns LazyDict for folders)
loaded = deserialize("./output/my_result")
print(loaded["experiment"])  # "run_1" — loaded on access
print(loaded["results"])     # 42
```

### DataFrame Support

```python
import pandas as pd

df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

# Serializes to Parquet (with metadata sidecar)
serialize(df, "./output/dataframe")

# Deserializes directly as DataFrame
loaded_df = deserialize("./output/dataframe")
```

### Nested Data with LazyDict

```python
# Nested dictionary
nested = {
    "run_1": {"metrics": pd.DataFrame(...)},
    "run_2": {"metrics": pd.DataFrame(...)},
}

serialize(nested, "./outputs/runs")

# Deserialize as LazyDict — values load on access
loaded = deserialize("./outputs/runs")

# Lazy access
metrics_1 = loaded["run_1"]["metrics"]  # Loaded only when accessed

# Recursive materialization
full_dict = loaded.to_dict()  # Materializes all nested values
```

### DAG APIs

For deferred-compute APIs (`DAG`, `Job`, `ParametrizedJob`) and local
multithreaded DAG evaluation, see the dedicated DAG page:
[DAG Guide](dag.md).

### Optional S3 Support

TidyRun can read and write `s3://bucket/prefix/...` locations when the optional S3 dependency is installed:

```bash
pip install tidyrun[s3]
```

Under the hood, S3 serialization stages through a local temporary directory and then uploads the generated files. Deserialization downloads the object tree to a temporary local directory and then reuses the normal local loader.

For tests and local development, the S3 round-trip test suite uses the `moto` mock backend.

### Concatenation Across Runs

```python
# Given a nested structure like:
# outputs/
#   run_1/
#     metrics.parquet
#   run_2/
#     metrics.parquet
#   ...

loaded = deserialize("./outputs")

# Concatenate all DataFrames, keyed by run ID
combined = loaded.concat(names=["run_id"])
# Returns: DataFrame with multi-index (run_id) and all metrics stacked

# With transformation (e.g., add timestamp)
combined = loaded.concat(
    names=["run_id"],
    transform=lambda df: df.assign(loaded_at=pd.Timestamp.now())
)

# Transform to scalar values (wrapped as a one-row "value" column)
totals = loaded.concat(
    names=["run_id"],
    transform=lambda df: df["metric"].sum(),
)

# With filtering (load only specific runs)
combined = loaded.concat(
    names=["run_id"],
    select=lambda path: path[0] in ["run_1", "run_2"]
)

# Parallel loading of leaf values (useful when leaf files are large)
combined = loaded.concat(
    names=["run_id"],
    max_workers=8,
)

# select(path) argument:
# - path: tuple of keys to the leaf (e.g. ("run_1", "metrics"))
# Note: select is evaluated at multiple depths while traversing nested folders.
# If you index deeper elements like path[1], guard with len(path) first.
```

## Quick Reference

### One-Minute Overview

TidyRun serialization provides save/load behavior with automatic format selection:

```python
from tidyrun import serialize, deserialize

serialize({"df": my_dataframe, "config": settings}, "./output/result")

result = deserialize("./output/result")
df = result["df"]
```

### What Gets Stored Where

| Type | Format | File | Notes |
|------|--------|------|-------|
| `dict` | Folder tree | `key/name/` | Keys encoded via TOML; values recurse |
| `pd.DataFrame` | Parquet | `*.parquet` | Falls back to HDF5 on failure |
| `pd.Series` | Parquet | `*.parquet` | Falls back to HDF5 on failure |
| Scalar (`int`, `str`, `bool`, `float`, `date`, `datetime`, `time`) | JSON | `*.json` | JSON-serializable scalars |
| Other objects | Pickle | `*.pickle` | Last resort |

Every output gets a `.tidyrun` metadata sidecar recording the selected format and checksum.

### LazyDict in Practice

Directories deserialize as `LazyDict` objects for on-demand loading:

```python
result = deserialize("./large_output")
value = result["key"]
full_dict = result.to_dict()
```

### Encoder Fallback Chain

If one encoder cannot serialize a value, the next compatible encoder is tried:

1. `dict` -> folder tree
2. `DataFrame` -> parquet, then HDF5
3. `Series` -> parquet, then HDF5
4. JSON-serializable values -> JSON
5. Everything else -> pickle

### Common Patterns

Save experiment results:

```python
serialize(
    {
        "config": hyperparams,
        "metrics": pd.DataFrame(training_log),
        "model": trained_model,
    },
    "./experiments/exp_001",
)
```

Load and compare runs:

```python
runs = deserialize("./experiments")
comparison_table = runs.concat(names=["exp_id"])
```

### API Summary

| Function | Purpose |
|----------|---------|
| `serialize(value, target, encoders=None)` | Save a value to disk and return `ChecksumInfo` |
| `deserialize(source, encoders=None)` | Load a value from disk |
| `LazyDict.to_dict()` | Materialize a nested `LazyDict` |
| `LazyDict.concat(names, transform, select, max_workers=None)` | Recursively concatenate DataFrames with optional parallel leaf loading |
| `encode_key(key)` | Encode a Python type to a filename-safe key |
| `decode_key(name)` | Decode a stored key name |

## Architecture

### Module Structure

```
src/tidyrun/serialization/
├── __init__.py          # Public API exports
├── types.py             # Type definitions, exceptions, constants
├── paths.py             # Path/location helpers
├── metadata.py          # Metadata I/O and format mapping
├── encoders.py          # Encoder implementations (dict, parquet, hdf5, etc.)
├── lazy_dict.py         # LazyDict class
└── api.py               # Main serialize/deserialize functions
```

### Encoder Pipeline

The default encoder pipeline tries encoders in this order:

1. **dict-folder**: Maps dictionaries to directory trees (keys become folder names)
2. **dataframe-parquet**: Stores DataFrames as `.parquet` files
3. **series-parquet**: Stores pandas Series as `.parquet` files
4. **pandas-hdf5**: Fallback for DataFrames/Series (HDF5 format with key `"data"`)
5. **fallback-json**: JSON serialization for scalar types (int, float, str, list, dict, etc.)
6. **fallback-pickle**: Last resort for arbitrary Python objects

Encoders are tried in order; the first whose predicate returns `True` is used.

### Fallback Mechanism

When an encoder fails (e.g., parquet cannot serialize a multi-index DataFrame), it raises `GoToNextEncoderException` to signal "skip me and try the next one." This allows:

- DataFrame with multi-index → fails parquet → tries HDF5 ✓
- Series when parquet engine unavailable → fails parquet → tries HDF5 ✓
- Custom object → fails all structured formats → falls back to pickle ✓

### Metadata Sidecars

Every serialized value gets a `.tidyrun` metadata file:

```toml
# output.tidyrun
version = 1
encoding = "dataframe-parquet"
suffix = ".parquet"

[checksum]
algorithm = "sha256"
digest = "..."
```

This metadata:

- Tracks the encoding format used
- Enables schema versioning for future compatibility
- Allows deserialization without requiring file extension guessing

## API Reference

### Core Functions

#### `serialize(value, target, encoders=None)`

Serializes a Python value to disk using the configured encoder pipeline.

**Parameters:**
- `value` (Any): The value to serialize (dict, DataFrame, Series, scalar, etc.)
- `target` (Path | CloudPath): Where to write the output (with or without extension)
- `encoders` (Iterable[EncoderSpec], optional): Custom encoder pipeline; defaults to `default_encoders()`

**Returns:**
- `ChecksumInfo`: checksum (`algorithm`, `digest`) for the serialized payload.

**S3 support:**
- When `target` is an `s3://...` URI, TidyRun stages the output locally and uploads the resulting tree to S3
- This requires the optional `boto3` dependency, available via `pip install tidyrun[s3]`

**Behavior:**
- Selects first encoder whose predicate matches the value
- Calls the encoder's serializer function
- Writes metadata sidecar with encoding info
- Returns payload checksum information to the caller
- If serializer raises `GoToNextEncoderException`, tries next matching encoder

**Raises:**
- `TidyRunSerializationError`: No encoder found for value type
- `NotImplementedError`: S3 support requested without the optional dependency installed
- Various I/O errors (permission denied, disk full, etc.)

**Example:**
```python
serialize({"data": pd.DataFrame(...)}, "./results/exp_1")
# Writes:
#   results/exp_1.tidyrun      (metadata)
#   results/exp_1/data/        (folder for nested dict)
#   results/exp_1/data/data.tidyrun
#   results/exp_1/data/data.parquet
```

#### `deserialize(source, encoders=None)`

Deserializes a value from disk using metadata to determine format.

**Parameters:**
- `source` (Path | CloudPath): Location to read from (with or without extension)
- `encoders` (Iterable[EncoderSpec], optional): Custom encoder pipeline

**S3 support:**
- When `source` is an `s3://...` URI, TidyRun downloads the object tree to a temporary local directory and deserializes from there
- This requires the optional `boto3` dependency, available via `pip install tidyrun[s3]`

**Returns:**
- For dict-folder format: `LazyDict` (not materialized dict)
- For DataFrame: `pd.DataFrame`
- For Series: `pd.Series`
- For scalar/pickle: Original Python object

**Raises:**
- `TidyRunDeserializationError`: Missing metadata, unknown encoder, invalid data
- `NotImplementedError`: Remote locations not yet supported
- Various I/O errors

**Example:**
```python
result = deserialize("./results/exp_1")
# Returns LazyDict with keys from the directory structure
```

### LazyDict

#### `LazyDict.__getitem__(key)`

Lazily loads the value associated with `key`.

```python
loaded = deserialize("./results")
# Value not loaded yet

value = loaded["run_1"]  # Loaded here on first access
value_again = loaded["run_1"]  # Loaded again on access
```

#### `LazyDict.to_dict()`

Recursively materializes all nested LazyDicts into a plain Python dict.

```python
loaded = deserialize("./nested")
full_dict = loaded.to_dict()  # All values now loaded into memory
```

#### `LazyDict.concat(names=None, transform=None, select=None)`

Recursively concatenates DataFrame/Series leaves using `pandas.concat`.

**Parameters:**
- `names` (list[str], optional): Names for multi-index levels created from nested keys
- `transform` (Callable[[Any], Any], optional): Function to apply to each leaf value before concatenation (e.g., add metadata)
- `select` (Callable[[tuple, Any], bool], optional): Predicate to filter which values to include; receives (path_tuple, value)

**Returns:**
- `pd.DataFrame`: Concatenated result with multi-index if nested

**Raises:**
- `ValueError`: No matching values after filtering
- `TypeError`: Selected values are not DataFrame or Series

**Example:**
```python
# Simple concatenation
combined = loaded.concat(names=["experiment_id"])

# With transformation
combined = loaded.concat(
    names=["experiment_id"],
    transform=lambda df: df.assign(dataset="train")
)

# With filtering
combined = loaded.concat(
    names=["experiment_id"],
    select=lambda path: "important" in path
)
```

### Encoders and Predicates

#### `is_mapping(value) -> bool`

Returns `True` if `value` is a dict (will be encoded as folder).

#### `is_dataframe(value) -> bool`

Returns `True` if `value` is a pandas DataFrame (detected at runtime if pandas installed).

#### `is_json_serializable(value) -> bool`

Returns `True` if `value` can round-trip through `json.dumps` and `json.loads`.

#### `can_encode_with_parquet(value) -> bool`

Returns `True` if DataFrame and a parquet engine (pyarrow or fastparquet) is available.

#### `can_encode_with_hdf5(value) -> bool`

Returns `True` if value is a DataFrame or Series (and pandas is installed).

#### `default_encoders() -> tuple[EncoderSpec, ...]`

Returns the default 6-encoder pipeline in priority order.

### Exceptions

#### `TidyRunSerializationError`

Raised when serialization fails (e.g., no encoder matches the value type).

```python
from tidyrun.serialization import TidyRunSerializationError

try:
    serialize(some_unsupported_type(), "./output")
except TidyRunSerializationError as e:
    print(f"Cannot serialize: {e}")
```

#### `TidyRunDeserializationError`

Raised when deserialization fails (e.g., missing metadata, invalid format).

```python
from tidyrun.serialization import TidyRunDeserializationError

try:
    deserialize("./invalid_path")
except TidyRunDeserializationError as e:
    print(f"Cannot deserialize: {e}")
```

#### `GoToNextEncoderException`

Internal exception raised by encoders to signal fallback to next encoder. Users typically don't need to handle this directly.

## Key Encoding

TidyRun keys (used as folder/file names in the hierarchy) are encoded using TOML for type safety.

### `encode_key(key) -> str`

Encodes a simple Python type to a path-safe string using TOML.

**Supported types:** str, int, float, bool, date, datetime, time

String keys are encoded to preserve round-trip type safety. Plain strings are
left unquoted when safe (for example, `"hello" -> "hello"`). Strings that
would otherwise be interpreted as another TOML type (for example `"true"`,
`"42"`, or date-like values) are TOML-quoted to ensure they decode back to
strings.

Encoded keys are used as path parts, so they must satisfy these constraints:

- Non-empty
- No `/` or `\\`
- Must not start with `.`
- Must not end with `.tidyrun`

```python
from datetime import date, datetime
from tidyrun.keys import encode_key, decode_key

encoded = encode_key(42)                    # "42"
encoded = encode_key("hello")               # "hello"
encoded = encode_key("true")                # '"true"'
encoded = encode_key(True)                  # "true"
encoded = encode_key(date(2026, 5, 10))     # "2026-05-10"
encoded = encode_key(datetime(2026, 5, 10, 13, 37, 42))  # "2026-05-10T13:37:42"
```

### `decode_key(name) -> Key`

Decodes a key name back to its original Python type.

```python
original = decode_key("hello")    # "hello"
original = decode_key('"true"')   # "true"
original = decode_key("42")       # 42
```

**Raises:**
- `TidyRunKeyEncodingError`: Unsupported key type
- `TidyRunKeyDecodingError`: Invalid encoded name format

## Customization

### Custom Encoder

To add support for a custom type:

```python
from tidyrun.serialization import EncoderSpec, serialize

def is_my_type(value):
    return isinstance(value, MyType)

def encode_my_type(value, target):
    # Write value to target location
    ...

def decode_my_type(source):
    # Read and return value from source location
    ...

my_encoder = EncoderSpec(
    name="my-custom-type",
    predicate=is_my_type,
    serializer=encode_my_type,
    deserializer=decode_my_type,
)

# Use custom encoder
from tidyrun.serialization import default_encoders

custom_pipeline = (my_encoder,) + default_encoders()
serialize(my_value, "./output", encoders=custom_pipeline)
```

### Custom Encoder Pipeline

To override the default pipeline order:

```python
from tidyrun.serialization import default_encoders, EncoderSpec

# Reorder: put HDF5 before Parquet
encoders = default_encoders()
reordered = (
    encoders[0],  # dict-folder
    encoders[3],  # pandas-hdf5
    encoders[1],  # dataframe-parquet
    *encoders[2:],  # rest
)

serialize(df, "./output", encoders=reordered)
```

## Performance Considerations

### Lazy Loading

`LazyDict` does not load values until accessed:

```python
loaded = deserialize("./large_structure")  # Fast: only reads metadata

result = loaded["expensive_dataframe"]  # Slow: loads large file here
result = loaded["expensive_dataframe"]  # Loaded again on access
```

### Concatenation Memory

`concat()` materializes all selected leaves into memory before concatenating. For very large structures, consider filtering with the `select` parameter:

```python
# Load all 1000 experiments at once (high memory)
combined = loaded.concat()

# Load only 10 specific experiments (low memory)
combined = loaded.concat(
    select=lambda path: path[0] in experiments[:10]
)
```

### Parquet Engine Selection

Parquet serialization uses pyarrow by default; fastparquet is tried if pyarrow is unavailable. HDF5 is tried if parquet encoding fails.

## Limitations and Future Work

### Current Limitations

1. **Remote Storage**: S3 is supported as an optional backend via `tidyrun[s3]`. Other cloud storage providers such as GCS and Azure Blob Storage are not yet supported.
2. **Schema Evolution**: No automatic schema migration for DataFrames. Users must handle schema changes manually (e.g., use `transform` in `concat`).
3. **Parquet Multi-Index**: Multi-index DataFrames cannot be serialized to parquet and fall back to HDF5.

### Planned Features
- **Metadata Sidecars**: Each output is accompanied by a `.tidyrun` metadata file recording encoding format, version, and checksum
- **Unified Path Handling**: Paths are normalized via `cloudpathlib.AnyPath` for local and cloud-backed locations
4. **Custom Metadata**: Allow users to store arbitrary metadata alongside outputs

## Testing

Run the full test suite:

```bash

[checksum]
algorithm = "sha256"
digest = "..."
pixi run pytest tests/serialization tests/test_keys.py
```

Key test modules:

**Returns:**
- `ChecksumInfo`: checksum (`algorithm`, `digest`) for the serialized payload.

- `tests/serialization/test_api.py`: End-to-end serialize/deserialize, metadata, fallback sequencing, S3 round-trip
- `tests/serialization/test_encoders.py`: Encoder predicates and detection logic
- Returns payload checksum information to the caller
- `tests/serialization/test_lazy_dict.py`: LazyDict access patterns and concatenation
- `tests/test_keys.py`: Key encoding and decoding

## Examples

### Example 1: Saving Experiment Results

```python
import pandas as pd
from tidyrun import serialize, deserialize

# After running an experiment
results = {
    "config": {"lr": 0.001, "epochs": 100},
    "metrics": pd.DataFrame({
        "epoch": [1, 2, 3],
        "loss": [0.5, 0.3, 0.2]
    }),
    "model_weights": some_large_array,  # Will pickle
}

serialize(results, "./experiments/exp_001")

# Later, load with lazy access
loaded = deserialize("./experiments/exp_001")
print(loaded["config"])  # Loaded on access
model = loaded["model_weights"]  # Pickled data
```

### Example 2: Comparing Multiple Runs

```python
runs = {
    "run_a": {
        "metrics": pd.DataFrame({"accuracy": [0.8, 0.85, 0.9]}),
    },
    "run_b": {
        "metrics": pd.DataFrame({"accuracy": [0.75, 0.82, 0.88]}),
    },
}

serialize(runs, "./comparison")

# Load and aggregate
loaded = deserialize("./comparison")
combined = loaded.concat(names=["run_id"])
print(combined)
# Output:
#             accuracy
# run_id
# run_a   0        0.80
#         1        0.85
#         2        0.90
# run_b   0        0.75
#         1        0.82
#         2        0.88
```

### Example 3: Filtered Concatenation

```python
# Load results from multiple time periods
results = deserialize("./results_by_month")

# Concatenate only 2026 results, adding month info
combined = results.concat(
    names=["month"],
    select=lambda path: path[0].startswith("2026"),
    transform=lambda df: df.assign(period="2026")
)
```
