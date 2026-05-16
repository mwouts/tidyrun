# TidyRun Documentation

This directory contains the core documentation for the TidyRun project. The site is built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages via GitHub Actions.

## Documentation Files

- **[index.md](index.md)** — Home page with overview and quick start
- **[serialization.md](serialization.md)** — Comprehensive serialization framework guide
    - Quick start examples
    - Complete API reference
    - Architecture overview
    - Performance considerations
    - Real-world examples
- **[contributing.md](contributing.md)** — Contributing guidelines
    - Development setup
    - Testing procedures
    - Documentation guidelines
- **[dag.md](dag.md)** — DAG compute guide
    - Job, ParametrizedJob, and DAG primitives
    - Execution modes and parallelism
    - Materialized plans and failure handling
- **[quick_start.md](quick_start.md)** — Local docs workflow and publishing notes

## Other Documentation

- **[changelog.md](changelog.md)** — Project changelog included in the documentation site via symlink
- **[serialization.md](serialization.md)** — Serialization guide with API reference and quick reference material

## Changelog

The [Changelog](changelog.md) is symlinked from the root `CHANGELOG.md` file. This ensures:

- **Single source of truth**: The changelog lives in one place
- **Always in sync**: Documentation automatically reflects the latest version history
- **Version tracking**: Full history of features, changes, and known issues
## Building Documentation Locally

### Prerequisites

The documentation dependencies are already included in the `pyproject.toml`. Set up your environment:

```bash
pixi install
```

Or to use the dedicated docs environment:

```bash
pixi shell -e docs
```

### Notebook Workflow

For interactive experimentation with TidyRun in Jupyter:

```bash
# Register a kernel to the development environment
pixi run -e kernel register-kernel

# Launch JupyterLab
pixi run -e notebook jupyter lab
```

This setup uses two Pixi environments:

- `notebook`: hosts the JupyterLab UI
- `kernel`: provides the Python kernel with `ipykernel` and your editable `tidyrun` install

### Live Preview

Serve the documentation locally with live reload:

```bash
pixi run mkdocs serve
```

Then open http://localhost:8000 in your browser. The site will automatically rebuild when you edit files.

### Build Static Site

Generate the static HTML site:

```bash
pixi run mkdocs build
```

Output is in the `site/` directory. Open `site/index.html` to view.

### Preview Deployment

To test the exact deployment as it will appear on GitHub Pages:

```bash
pixi run mkdocs build
pixi run python -m http.server --directory site 8000
```

Visit http://localhost:8000 to preview.

## Deployment

Documentation is automatically deployed to GitHub Pages when:

- Changes are pushed to `main` or `serialize_v01` branch
- Changes affect `docs/`, `mkdocs.yml`, or the workflow itself

The GitHub Actions workflow (`.github/workflows/deploy-docs.yml`) handles:

1. Building the documentation with mkdocs
2. Uploading to GitHub Pages artifact storage
3. Deploying to the live GitHub Pages site

**Live Site:** https://mwouts.github.io/tidyrun/

### Manual Deployment

To deploy manually:

```bash
pixi run mkdocs gh-deploy
```

This builds and pushes the `site/` directory to the `gh-pages` branch.

## Contributing to Documentation

When adding or updating documentation:

- Keep examples practical and runnable
- Include trade-offs and design rationale
- Link to relevant code sections
- Update `mkdocs.yml` navigation when adding pages
- Use Material for MkDocs features (admonitions, code tabs, etc.)
- Test locally with `pixi run mkdocs serve` before committing

### Documentation Style Guide

**Code Examples:**
```python
# Always include complete, runnable examples
from tidyrun import serialize, deserialize

data = {"key": "value"}
serialize(data, "./output")
result = deserialize("./output")
```

**Admonitions (Notes, Warnings, etc.):**
```markdown
!!! note
    This is important information.

!!! warning
    This is a warning.

!!! tip
    This is a helpful tip.
```

**Code Tabs:**
```markdown
=== "Python"
    ```python
    code_example()
    ```

=== "Shell"
    ```bash
    command --flag
    ```
```

## Configuration

The documentation is configured in `mkdocs.yml` at the project root:

- **Theme:** Material for MkDocs with light/dark mode toggle
- **Search:** Full-text search enabled
- **Code highlighting:** Syntax highlighting with copy buttons
- **Navigation:** See `mkdocs.yml` for structure

## Troubleshooting

**Port 8000 already in use:**
```bash
pixi run mkdocs serve -a localhost:8001
```

**Documentation won't build:**
- Check Python version (3.8+)
- Verify the pixi environment is up to date: `pixi install`
- Check for YAML syntax errors in `mkdocs.yml`
- Look for Markdown syntax errors (unclosed code blocks, etc.)

**Changes not appearing:**
- Stop and restart `pixi run mkdocs serve`
- Clear browser cache (Ctrl+Shift+Delete)
- Ensure file is saved before refreshing
