# Changelog

All notable changes to TidyRun are documented in this file.

## [0.0.4.dev0] — (unreleased)

### Added

- Enhanced `LazyDict.concat(names=...)` to support `None` values in the names list, allowing selected index levels to be dropped during concatenation.
- Added parallel leaf loading for `LazyDict.concat(max_workers=...)` using a thread pool.
- Added deferred-compute primitives: `Job`, `ParametrizedJob`, and `DAG`.
- Added `DAG.evaluate(max_workers=...)` for local multithreaded execution of independent top-level nodes.
- Added `DAG.evaluate(execution_mode=...)` with `subprocess` (default), `thread`, and `process` execution modes.
- Added `SlurmExecutor` to run DAG nodes through SLURM (`sbatch` + `squeue`) with result/error materialization via shared storage.
- Added `AwsBatchExecutor` for AWS Batch submission via the standard `Executor` interface.
- Added first-class SLURM resource parameters on `SlurmExecutor` (e.g., partition, QoS, account, constraint, time, memory, CPU/GPU requests).
- Added per-node resource overrides via `DAG.evaluate(job_resources=...)` for executors that support `submit_with_options` (including `SlurmExecutor`).
- Added DAG execution mode tests and AWS Batch executor mock tests (no AWS account required).
- Added an opt-in local container integration smoke test (`RUN_CONTAINER_TESTS=1`).
- Added dedicated DAG documentation page (`docs/dag.md`) with API and execution examples.
- Added `DAGExecutionError` to report failed job id, root cause, completed jobs, and cancelled jobs.
- Added `DAG.execute_materialized(skip_completed=True)` for resumable execution that skips jobs whose outputs already exist.
- Added `DAG.clear_outputs(...)` to clear all outputs or selected job outputs from a materialized plan.
- Added `SlurmExecutor.submit_array_with_options(...)` to submit homogeneous ready batches as SLURM array jobs.
- Added `AwsBatchExecutor.submit_array_with_options(...)` to submit homogeneous ready batches as AWS Batch array jobs.
- Added strict callable-signature validation in `Job` and `ParametrizedJob` constructors to reject missing required arguments, unknown arguments, and overlap between `kwargs` and `parameter_names`.
- Added opt-in progress logging for DAG plan compilation and execution via `progress=True` and optional `progress_callback` on `materialize`, `execute_materialized`, and `evaluate` APIs.

### Changed

- Updated `LazyDict.concat()` to raise a `ValueError` when a nested `LazyDict` is encountered but `names` has insufficient levels to reach leaf values, preventing silent failures.
- Updated DAG evaluation to materialize execution plans first, then execute compiled jobs with dependency-aware scheduling.
- Updated DAG evaluation with explicit validation/error messages for invalid executor option combinations and invalid `job_resources` keys.
- Updated DAG execution to fail fast: stop scheduling new jobs when a job fails and surface structured failure context.
- Updated CI workflows to pin Pixi to `v0.68.1`.
- Updated SLURM runner script generation to include a shebang bound to the current Python executable (`sys.executable`) instead of a generic interpreter.
- Updated SLURM submission defaults so job names derive from materialized DAG job ids (slash-concatenated encoded keys).
- Updated AWS Batch submissions to preserve relative materialized job ids in environment/parameters, including array payloads via `TIDYRUN_JOB_IDS_JSON` / `tidyrun_job_ids_json`.
- Updated DAG materialization/execution to tag parametrized jobs with array-group metadata and submit eligible ready jobs through executor array APIs when available.
- Updated parametrized plan compilation to attach array-group metadata inline during node compilation (constant extra work per node, no post-hoc group traversal).
- Updated DAG path-hint to job-id derivation to support all declared `Key` types (`str`, `int`, `float`, `bool`, `date`, `datetime`, `time`) with explicit best-effort handling.

## [0.0.3] — 2026-05-12

### Changed

- Updated key encoding to keep plain strings unquoted when safe, while still quoting string keys that would otherwise be parsed as non-string TOML values.
- Updated key decoding to accept bare string folder names when TOML parsing fails, improving interoperability with manually-created directory trees.
- Updated `LazyDict.concat(select=...)` to use a path-only callback signature (`select(path)`) and to evaluate selection before loading children.
- Updated `LazyDict.concat(transform=...)` to accept scalar transform outputs (wrapped as a single-row pandas object for concatenation).

### Fixed

- Fixed deserialization of nested directories without `.tidyrun` sidecars when subdirectory names are simple strings.
- Updated serialization tests to match the new simple-string on-disk key naming convention.

## [0.0.2] — 2026-05-11

### Changed

- Simplified the public serialization API path by removing the `tidyrun.serialize` compatibility module and standardizing imports on `tidyrun.serialization` (while still exposing `serialize` and `deserialize` from `tidyrun`).
- Updated docs and tests to use the new import paths.

### Fixed

- Fixed wheel packaging for PyPI: the previous published package was effectively empty (metadata only, no `tidyrun` module files).
- Added release/build-time package smoke checks to verify the built artifact exports `tidyrun.__version__` and `tidyrun.deserialize`.
- Fixed packaging/editable install behavior so `tidyrun` is importable in the Pixi dev environment (`src` path is now wired for Hatch dev mode).


## [0.0.1] — 2026-05-10

### Serialization Framework (Initial Release)

**Features:**
- `encode_key()` / `decode_key()` for TOML-based type-safe key serialization
- Pluggable encoder pipeline with 6 default encoders:
    - dict → folder tree
    - DataFrame → Parquet (with HDF5 fallback)
    - Series → Parquet (with HDF5 fallback)
    - Scalar → JSON
    - Any → Pickle
- Metadata sidecars (`.tidyrun` files) with version tracking
- `LazyDict` for lazy on-demand loading without child caching
- `LazyDict.concat(names, transform, select)` for recursive pandas aggregation
- `GoToNextEncoderException` for intelligent encoder fallback
- Support for local filesystem plus optional S3 serialization/deserialization via `tidyrun[s3]`
- Full test suite (44 tests) organized by submodule
- Comprehensive documentation with Material for MkDocs theme
- GitHub Actions workflow for automatic documentation deployment

### Documentation

- Complete API reference guide
- Contributing guidelines
- Live local preview with `mkdocs serve`
- Automated GitHub Pages deployment
- S3 round-trip tests backed by `moto`

### Limitations (Future Work)

- [ ] Additional remote storage backends (GCS, Azure, etc.) via fsspec integration
- [ ] Virtual keys / glob patterns in LazyDict
- [ ] Schema hinting for Parquet files
- [ ] Custom metadata support
- [ ] Nested transform in concat
- [ ] Automatic schema evolution detection

### Known Issues

- Multi-index DataFrames fall back from Parquet to HDF5 (by design; users can work around with `reset_index()`)
- No automatic schema migration for DataFrames across versions (users must handle manually)

---

## Format

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
