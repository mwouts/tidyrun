# Changelog

All notable changes to TidyRun are documented in this file.

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
- Architecture design decisions documentation
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
