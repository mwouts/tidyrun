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

# Run with coverage
pixi run pytest --cov=src/tidyrun tests/
```

Test structure mirrors the source code:
- `tests/serialization/` — serialization module tests
- `tests/test_keys.py` — key encoding tests
- `tests/test_version.py` — version tests

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

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/description
   ```

2. **Make your changes** and commit with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```

3. **Push to your fork:**
   ```bash
   git push origin feature/description
   ```

4. **Open a Pull Request** with:
   - Clear title and description
   - Reference to any related issues
   - Summary of changes
   - Confirmation that tests pass

5. **Respond to reviews** and make requested changes

## Areas for Contribution

### High Priority

- **Remote Storage**: Add additional remote backends via fsspec integration
- **Documentation**: Expand examples and API documentation
- **Performance**: Optimize LazyDict traversal for very large nested structures

### Medium Priority

- **Virtual Keys**: Add glob-like pattern matching for LazyDict keys
- **Schema Hinting**: Pre-load Parquet metadata to detect schema mismatches
- **Custom Metadata**: Allow users to store arbitrary metadata

### Low Priority

- **Alternative Encoders**: Support additional formats (Arrow, NetCDF, etc.)
- **Async Support**: Add async serialize/deserialize
- **Memory Use**: Reduce memory overhead for large nested structures

## Questions?

- Open an [issue](https://github.com/mwouts/tidyrun/issues) with a `question` label
- Check existing [discussions](https://github.com/mwouts/tidyrun/discussions)
- Review the [design decisions](design-decisions.md) for architecture context

## Code of Conduct

We're committed to fostering a welcoming community. Please be respectful and constructive in all interactions.
