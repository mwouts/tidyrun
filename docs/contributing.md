# Contributing

Thank you for your interest in contributing to TidyRun! This guide will help you get started.

## Getting Started

### Set Up Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mwouts/tidyrun.git
   cd tidyrun
   ```

2. **Install with development dependencies:**
   ```bash
   pixi install  # or use conda/pip with the dependencies from pyproject.toml
   ```

3. **Verify installation:**
   ```bash
   pixi run pytest -q tests/
   ```

## Making Changes

### Code Style

TidyRun uses:

- **[Ruff](https://github.com/astral-sh/ruff)** for linting and formatting
- **[Pyright](https://github.com/microsoft/pyright)** for strict type checking

Before committing:
```bash
pixi run ruff check --fix src/ tests/
pixi run pyright src/
```

### Testing

Add tests for any new functionality:

```bash
# Run all tests
pixi run pytest tests/

# Run specific test file
pixi run pytest tests/serialization/test_api.py

# AWS Batch executor unit tests (mocked client, no AWS account required)
pixi run pytest tests/test_aws_batch_executor.py

# Optional S3 round-trip serialization test (requires boto3 + moto)
pixi run pytest tests/serialization/test_api.py -k s3_round_trip

# Optional local container integration smoke test (Docker/Podman)
RUN_CONTAINER_TESTS=1 pixi run pytest tests/test_container_runner.py

# Run with coverage
pixi run pytest --cov=src/tidyrun tests/
```

Test structure mirrors the source code:

- `tests/serialization/` — serialization module tests
- `tests/test_keys.py` — key encoding tests
- `tests/test_version.py` — version tests

Notes:

- `tests/test_aws_batch_executor.py` uses a fake Batch client and does not call AWS.
- `tests/test_container_runner.py` is opt-in and skipped unless `RUN_CONTAINER_TESTS=1`.
- Container tests require a working `docker` or `podman` runtime.

## Documentation

### Building Locally

### Prerequisites

Documentation dependencies are included in `pyproject.toml`. Just install the project:

```bash
pixi install
```

### Live Preview

Serve documentation locally with live reload:

```bash
pixi run mkdocs serve
```

Then open http://localhost:8000

### Build Static Site

Generate the static HTML site:

```bash
pixi run mkdocs build
```

The output is in the `site/` directory.

### Writing Documentation

- Place new documentation in `docs/`
- Update `mkdocs.yml` navigation if adding new pages
- Use code examples from the test suite where possible
- Include both basic usage and advanced patterns

### Documentation Guidelines

- **Code Examples**: Always include runnable examples
- **Design Rationale**: Explain the "why" behind decisions
- **Trade-offs**: Acknowledge limitations and alternatives
- **Cross-links**: Link to related sections and code
- **Type Information**: Include type hints in code examples

## Questions?

- Open an [issue](https://github.com/mwouts/tidyrun/issues) with a `question` label
- Check existing [discussions](https://github.com/mwouts/tidyrun/discussions)

## Code of Conduct

Please be respectful and constructive in all interactions.
