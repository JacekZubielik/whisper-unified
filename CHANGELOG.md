# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-02-11

### Added

- Modular source structure (`src/whisper_unified/`) replacing monolithic `orchestrator.py`
- `pyproject.toml` with PDM dependency management
- `pydantic-settings` based configuration (`config.py`)
- Full unit test suite with mocked ML dependencies
- CI/CD workflows (lint, test, Docker build validation)
- `.pre-commit-config.yaml` with 8 hooks (ruff, black, isort, mypy, bandit, commitizen)
- `.devcontainer/` for development without GPU
- CLAUDE.md with project documentation for Claude Code
- README.md in English with full API documentation

### Changed

- Split `orchestrator.py` (969 lines) into 6 modules (all under 500 lines)
- Dockerfile CMD changed from `python3 orchestrator.py` to `python -m src.whisper_unified`
- Configuration moved from `os.getenv()` to `pydantic-settings` `Settings` class
- All services accept `Settings` via constructor injection (testable)

### Deprecated

- `orchestrator.py` in root directory (thin wrapper, kept for backwards compatibility)
