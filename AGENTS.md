# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

This is a Python library called `dsl` (package name: `dsl`, module name: `pdum.dsl`). A Python DSL toolkit

The project uses a modern Python toolchain with UV for dependency management.

## Important Rules

### Version Management
**NEVER modify the version number in any file.** Versions are written by the `release` workflow
via `scripts/_versioning.py` — never by hand, and never by an agent. Do not change:
- `pyproject.toml` version field
- `src/pdum/dsl/__init__.py` `__version__` variable
- Any version references in documentation

Between releases the tree carries an `X.Y.Z+dev` marker (last release + a WIP flag). If you think
a version change is needed, inform the user but do not make the change yourself.

### Release Management
**ABSOLUTELY NEVER TRIGGER THE RELEASE WORKFLOW.** Releasing is a production deployment that
publishes to PyPI (affects real users), creates a permanent public GitHub release, and pushes
commits and tags. It is dispatched **by a human** from the Actions UI.

Do not:
- Run `gh workflow run release.yml` (or dispatch it by any other means) under any circumstances
- Run `./scripts/publish.sh` (the manual PyPI fallback) under any circumstances
- Suggest releasing unless the user explicitly asks about the release process

If the user needs to make a release, explain the process but let them run it themselves.

## Development Commands

### Environment Setup
```bash
# Bootstrap the full toolchain (uv sync, pnpm install, widget build, hooks)
./scripts/setup.sh
```

**Important for Development**:
- Use `uv sync --frozen` to ensure the lockfile is used without modification, maintaining reproducible builds
- Re-run `./scripts/setup.sh` whenever dependencies change

### Testing
```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_example.py

# Run a specific test function
uv run pytest tests/test_example.py::test_version

# Run tests with coverage
uv run pytest --cov=src/pdum/dsl --cov-report=xml --cov-report=term
```

### Code Quality
```bash
# Check code with ruff
uv run ruff check .

# Format code with ruff
uv run ruff format .

# Fix auto-fixable issues
uv run ruff check --fix .
```


### Documentation
```bash
# Serve documentation locally (auto-reloads on changes)
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Test demo notebooks (REQUIRED after any notebook changes)
./scripts/test_notebooks.sh
```

**Important**: After making any changes to notebooks under `docs/` (e.g. the book chapters in `docs/book/*.ipynb`), you MUST run `./scripts/test_notebooks.sh` to verify they execute without errors. Do not consider notebook changes complete until this test passes.

### Publishing
Releasing is **entirely CI** (see [Release Process](#release-process)) — a human dispatches the
`release` workflow. `./scripts/publish.sh` exists only as an out-of-band manual fallback and is
**never** run by an agent.


## Architecture

### Project Structure
- **src/pdum/dsl/**: the redesigned framework (being built; see
  `docs/design/010_proposed-architecture.md` for the design and
  `docs/design/020_implementation-plan.md` for the step sequence — both authoritative)
- **src/pdum/dsl_reference/**: the frozen Milestone-0 proof of concept.
  **Do not extend or refactor it.** Its tests live in `reference/tests/`
  (run on demand: `uv run pytest reference/tests`; not collected by default)
- **reference/**: everything about the frozen asset (README, REVIEW tour, demo)
- **tests/**: the redesign's test suite (pytest default)
- **docs/**: MkDocs site — `desiderata.md` (redesign brief), `m0/` (frozen M0
  docs, historical), `book/` (forthcoming chapter notebooks, one per
  implementation step)
- **docs/design/**: the numbered design canon, part of the mkdocs site
  (`010_proposed-architecture.md` is the master; `020` the plan; `022`/`024`
  the evidence analyses; `03x+` topic notes in book order — see
  `docs/design/README.md`), plus `research/` (frozen corpus)
- **archive/** (repo root): historical motivation material — not design inputs

### Key Constraints
- **Python Version**: Requires Python 3.14+
- **Dependency Management**: Uses UV exclusively; uv.lock is committed
- **Build System**: Uses Hatch/Hatchling for building distributions
- **Documentation Style**: NumPy docstring style (see mkdocs.yml)

### Code Standards
- **Ruff Configuration**:
  - Target: Python 3.14
  - Line length: 120 characters
  - Linting rules: E (pycodestyle errors), F (pyflakes), W (warnings), I (isort)
- **Type Hints**: Use type hints where appropriate
- **Docstrings**: NumPy style, include Parameters, Returns, Raises sections

### Testing Strategy
- Test files must start with `test_` prefix
- Test classes must start with `Test` prefix
- Test functions must start with `test_` prefix
- Tests run with `-s` flag (no capture) by default
- Coverage reporting: use `--cov=src/pdum/dsl --cov-report=xml --cov-report=term`

### Testing Configuration
The pytest configuration is in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-s"
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

Coverage configuration is also in `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/pdum/dsl"]
relative_files = true
omit = [
    "*/tests/*",
    "*/testing.py",
]
```

## CI/CD

### Continuous Integration
The project uses GitHub Actions for CI (`.github/workflows/ci.yml`):
- Runs on every push to main and pull requests
- Installs a C compiler, then asserts the C backend can see it (so `tests/test_backend_c.py`
  cannot silently skip)
- Executes linting with ruff
- Runs unit tests with coverage reporting
- Builds documentation to verify it compiles
- Posts coverage report as a PR comment

This run is **the gate the release requires green** on the commit being released.

### Documentation Deployment
Documentation is deployed to GitHub Pages (`.github/workflows/docs.yml`):
- Triggered when a GitHub release is published (i.e. by the release workflow)
- Can also be triggered manually via workflow_dispatch
- Uses `./scripts/setup.sh` before running `mkdocs build`

### Release Process
Releasing is **entirely CI** — `.github/workflows/release.yml` is the single publish path. There
is no local release script and no tag trigger, which is what makes a local/CI double-publish
impossible. A **human** dispatches it (Actions → release → Run workflow, or
`gh workflow run release.yml -f bump=minor`) with three inputs: `bump` (patch/minor/major),
`skip_ci_check`, and `dry_run`.

The pipeline: **require the commit's `ci.yml` run green** (waits out an in-progress run) →
compute the version → bump + tag + push → build sdist + wheel **from the tag** → publish to PyPI
(token auth via the `PYPI_API_TOKEN` repo secret, `skip-existing`) → create the GitHub Release
(which fires `docs.yml`) → return `main` to the `+dev` marker.

Versioning is **tag-as-truth** (`scripts/_versioning.py`):
- The last release is the highest `vX.Y.Z` git tag. This release is `bump(last_tag, level)`,
  computed **at release time** — the size of a release is chosen against the last real release,
  never guessed in advance. With no tags yet, a first `minor` cuts `0.1.0`.
- Between releases the tree carries `X.Y.Z+dev`. That is a PEP 440 *local* version, which PyPI
  refuses to upload — an accidental-publish guard. The release writes the clean `X.Y.Z` before
  building, so artifacts are never `+dev`.
- Every version-bearing file (root `pyproject.toml`, the `__init__.py` mirror, and any
  `packages/*` workspace member) is written in lockstep; agreement is enforced.
