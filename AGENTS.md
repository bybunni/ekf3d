# Repository Guidelines

## Project Structure & Module Organization
The package uses a `src` layout:
- `src/ekf3d/ekf_predictor.py`: motion model and EKF prediction step.
- `src/ekf3d/ekf_updater.py`: azimuth/elevation measurement model and EKF update step.
- `src/ekf3d/__init__.py`: public package entry points.
- `README.md`: math, assumptions, and integration notes.
- `pyproject.toml`: package metadata and build backend (`uv_build`).

Keep new runtime code under `src/ekf3d/`. Add tests in a top-level `tests/` directory.

## Build, Test, and Development Commands
- `uv run python -c "import ekf3d; print(ekf3d.hello())"`: quick import/smoke check.
- `uv build`: build sdist and wheel artifacts.
- `uv run python`: open a local REPL in the project environment.
- `uv run pytest`: run tests (after adding `pytest` and test files).

Prefer `uv` commands so local and CI environments stay consistent.

## Coding Style & Naming Conventions
- Target Python `>=3.12`.
- Follow PEP 8 with 4-space indentation.
- Use type hints consistently (`NDArray[np.float64]` patterns are already in use).
- Classes use `PascalCase` (for example, `EKFPredictor3D`).
- Functions and variables use `snake_case`.
- Constants use `UPPER_SNAKE_CASE`.
- Preserve core conventions: state ordering `[x, vx, y, vy, z, vz]`, measurements `[azimuth, elevation]`, angles in radians.

## Defensive Programming for Public APIs
- Treat all public method inputs as untrusted (especially when called by external simulators/integrations).
- Add strict runtime input validation at public entry points:
  - validate shape/dimension expectations
  - validate numeric type coercion
  - validate finiteness (`np.isfinite`)
  - validate allowed enum/string options
- Fail fast with clear, parameter-specific error messages so integration bugs are easy to diagnose.
- Add regression tests for both valid and invalid inputs whenever public API behavior is changed.

## Testing Guidelines
There is no committed automated test suite yet. New contributions should include tests for:
- Prediction matrix/process-noise behavior for known `dt`.
- Measurement function/Jacobian correctness.
- EKF update consistency (shape checks and numeric sanity).

Use `tests/test_*.py` naming and `numpy.testing`/`np.allclose` for numeric assertions.

## Commit & Pull Request Guidelines
Current history is minimal and uses short subject-only messages (for example, `README`, `initial commit`). Keep commits:
- Small and focused.
- Imperative and concise (about 50-72 chars), with optional body when needed.

For pull requests, include:
- What changed and why.
- Any math/assumption changes (especially frame/angle conventions).
- Reproduction steps or command snippets used to validate behavior.
