# Changelog

All notable changes to TidyRun are documented in this file.

## [0.0.8] — (2026-07-14)

### Added

- `AwsBatchExecutor` failures now include the container's exit code and
  reason, the CloudWatch log stream name, and a direct console link to the
  logs. For failed array jobs the links point at the failed children.
- Development-version support for containerized execution:
  - `materialize` records the writer's tidyrun version in ``plan.toml`` at
    the plan root, and job runners warn when they run under a different
    tidyrun version than the one that wrote the plan.
  - `tidyrun-batch-entrypoint` honors a new ``TIDYRUN_PIP_SPEC`` environment
    variable (pass it via ``AwsBatchExecutor(extra_env=...)``): the requested
    tidyrun distribution — e.g. a git branch — is installed and the
    entrypoint re-executes itself, so a generic container image runs the
    same (development) tidyrun version as the submitting machine.

### Fixed

- `execute_materialized`, `evaluate`, `execute_plan`, `get_job_states`,
  `clear_outputs`, and `run_materialized_job` now accept ``s3://`` locations
  (as strings or `CloudPath` objects). Previously the URI was coerced to a
  local path (e.g. ``s3:/bucket/plan``), so executing a plan materialized to
  S3 failed with "No materialized plan found" — despite the documented AWS
  Batch workflow relying on it.
- Plans materialized to S3 now record dependency inputs as ``.tidyrun``
  sidecar files. The local symlinks previously created during compilation
  were silently dropped by the S3 upload, leaving dependency arguments
  unresolvable.
- `TIDYRUN_PLAN_DIR` submitted by executors is now the plan root itself
  (e.g. ``s3://bucket/plan``) for standard-layout plans instead of an
  internal ``:::``-separated string, and `run_materialized_job` accepts both
  forms.
- The AWS-Batch-style container flow (plan on S3, `tidyrun-batch-entrypoint`
  in a separate process) is now covered by end-to-end tests against a moto S3
  server, including array children and dependencies. The
  "Missing job definition file" error now reports the location that was
  searched, and the docs warn that the container image must run the same
  tidyrun version as the submitting machine (older runners cannot read
  ``s3://`` plans).
- Fixed a 0.0.7 regression where passing a subset of a member
  `ParametrizedJob` (e.g. `pjob[key]`, or `dag["grid"][key]`) as a job
  dependency raised "depends on a Job or DAG that is not a member of this
  DAG". Subsets — whether a smaller `ParametrizedJob` or a single job
  instance — are now resolved to the member's already-compiled jobs, with the
  dependency input linked to the corresponding output subfolder (or single
  job output). Genuinely unregistered jobs still raise the same error.
- Jobs and parametrized jobs registered inside nested DAG members can now be
  referenced as dependencies; previously only top-level members were
  recognized.

## [0.0.7] — (2026-07-02)

### Added

- DAG execution now writes `.tidyrun` metadata for nested output folders, so
  executed DAG outputs deserialize consistently as `LazyDict` values (matching
  `serialize(dict, path)` layout).
- Added `load_inputs_and_callable(dag_path, job_id)` and exported it from the
  top-level package to simplify rerun/debug flows.

### Changed

- Internals were refactored (`dag`, `execute`, `plan`, `progress`) to remove
  duplicated scheduling/plan-reading code and improve maintainability, without
  changing the public API surface of `tidyrun.dag`.
- `execute_materialized`, `evaluate`, and `evaluate_in_subprocesses` now always
  write outputs to `dag_path/outputs/` (no separate `output_path`/`target`).
- Dependency handling is stricter and clearer:
  - `ParametrizedJob` dependencies pass the whole group output (`LazyDict`);
    `pjob["param"]` selector-style access is no longer supported.
  - Dependency jobs/DAGs must belong to the same top-level DAG; implicit
    anonymous dependencies now raise `ValueError`.
  - Unknown `ParametrizedJob.__getitem__` keys consistently raise `KeyError`.
- Dependency inputs are represented as symlinks on local filesystems (with an
  S3 sidecar equivalent), and relinked on re-materialization.
- API docs for core symbols are now generated from docstrings via
  `mkdocstrings`, reducing signature drift.
- `materialize` now returns `Path | CloudPath`, including proper `S3Path`
  behavior for S3 plans.

### Fixed

- Fixed SLURM array jobs incorrectly reporting failure even when outputs were
  written successfully.
- Removed duplicate jobs that could be created when parametrized jobs had
  dependencies.
- Fixed failures when a parametrized job depends on another parametrized job.
- `LazyDict.concat` with `transform` and `names` no longer raises
  "Encountered LazyDict at depth N" when the leaf value is a plain `dict` (or
  any non-`LazyDict` mapping) that `transform` knows how to handle.

## [0.0.6] — (2026-06-02)

### Added

- `ParametrizedJob` is now a subclass of `DAG`, eliminating delegation
  boilerplate and adding the previously missing `evaluate_in_subprocesses`
  and `clear_outputs` methods. All five execution methods (`materialize`,
  `execute_materialized`, `evaluate_in_subprocesses`, `evaluate`,
  `clear_outputs`) are now inherited directly from `DAG`.
- `DAGExecutionError.plan_dir` and `DAGExecutionError.outputs_path`: new attributes
  that carry the materialised plan directory and its outputs path so that callers
  can locate the `.failed` sentinel or construct a rerun snippet.
- `DAGExecutionError.rerun_snippet()`: returns a copy-pasteable Python snippet that
  re-runs just the failed job from the materialised plan.
- `DAGExecutionError.__str__` now appends the job's full traceback (read from the
  `.failed` TOML sentinel) and the rerun snippet, making it straightforward to
  debug a failed SLURM or AWS Batch job.

### Fixed

- Progress bar total was over-reported for multi-level parametrised jobs (e.g.
  `200/3` instead of `200/200`). Root cause: Python reuses object IDs for
  short-lived `ParametrizedJob` sub-nodes created by `__getitem__`, causing false
  hits in the deduplication set used by `_count_unique_jobs`. Fixed by accumulating
  all ephemeral child references in a single list so they remain alive for the
  entire counting pass.
- `execute_materialized` was slow for large parametrised runs because it eagerly
  loaded every job output into memory and re-serialised it to `output_path`.
  Each job now writes its output directly to `output_path/{job_id}` so the
  ``output_path`` directory IS the final result with no post-processing step.
  ``load_job_inputs`` also gained an ``outputs_path`` parameter so dependency
  resolution always finds the right location.
- Dependency job IDs could fall back to synthetic `__job_N` counters when a
  shared `Job` was used as an argument to a parametrised job instance whose
  `job_id` contained path separators (e.g. `"pairs/m1/train"`). Fixed by splitting
  the owner job id on `"/"` before building the path hint in `_compile_operand`.
- `SlurmExecutor` files in `shared_dir` (task pickle, result, error, stdout) now
  use the job id as a prefix (e.g. `pairs__m1__train.task.pickle`) instead of a
  random UUID hex string, making it much easier to correlate log files with jobs.
  For array submissions the prefix is derived from the common group name of the
  submitted job ids.

### Changed

- We have moved the executors module under `executors`

## [0.0.5] — (2026-05-31)

### Added

- `tidyrun-batch-entrypoint` console script: fixes a correctness bug in AWS
  Batch array jobs where all children would run the same job. Single jobs read
  `TIDYRUN_JOB_ID`; array children pick the right id via
  `AWS_BATCH_JOB_ARRAY_INDEX` + `TIDYRUN_JOB_IDS_JSON`.
- `extra_env` on `AwsBatchExecutor`: inject static environment variables (e.g.
  `GIT_REPO_URL`, `GIT_COMMIT`) into every submitted container.
- `execute_plan()`: run all jobs in a plan directory without a `DAG` object,
  for the decentralised case where multiple scripts share one plan dir.
- `DAG.materialize(prefix=...)`: namespace all job ids under a prefix so
  multiple DAGs can write to the same plan directory without conflict.
- Job-state sentinels: `.running` at start, `.failed` on error (TOML with
  traceback). `get_job_states()` returns `"pending"`, `"running"`, `"failed"`,
  or `"succeeded"` for every job in a plan.
- `skip_running` flag on `execute_materialized()` and `execute_plan()`.
- Moto-based AWS Batch integration tests covering the full boto3 serialisation
  path, complementing the existing fake-client unit tests.
- Unified Executors documentation page covering local, SLURM, and AWS Batch
  with a shared git-commit-pinning example.

### Changed

- Reviewed and simplified the DAG plan format and execution path.
- Plan-reading helpers extracted to `plan.py`; executors moved to
  `executors/`; public API unchanged.
- S3 serialisation now handled by `cloudpathlib`.
- `LazyDict` is lazier: keys are not listed until accessed.
- `LazyDict` objects are serialised as symlinks.
- Dropped support for Python 3.10.

## [0.0.4] — (2026-05-16)

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
