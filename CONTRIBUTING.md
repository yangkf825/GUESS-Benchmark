# Contributing to GNN-UQ-Bench

Thank you for your interest in contributing!

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/gnn-uq-bench
cd gnn-uq-bench
pip install -e ".[dev]"
pre-commit install   # optional: auto-runs ruff on commit
```

## Code Standards

- **Linting**: `ruff check src/ tests/`
- **Formatting**: `ruff format src/ tests/`
- **Tests**: `pytest` — all PRs must maintain ≥ 90% coverage on changed files.
- **Type hints**: all public functions must have typed signatures.
- **Docstrings**: Google style, including Parameters / Returns sections.

## Adding a New UQ Method

1. Add the model to `src/gnn_uq_bench/models/<method>.py`.
2. Export it from `src/gnn_uq_bench/models/__init__.py`.
3. Add an experiment script `experiments/run_<method>.py` following
   the pattern of `experiments/run_ungnn.py`.
4. Add unit tests in `tests/test_models.py` and `tests/test_metrics.py`.
5. Update `README.md` — add a row to the algorithm table.

## Adding a New Dataset

1. Add a loader to `src/gnn_uq_bench/datasets/<dataset>.py`.
2. Export from `src/gnn_uq_bench/datasets/__init__.py`.
3. Add tests in `tests/test_datasets.py`.

## Pull Request Checklist

- [ ] Branch name is not `main` or `dev`.
- [ ] Tests pass locally (`pytest`).
- [ ] `ruff check` returns no errors.
- [ ] Docstrings are complete for all new public functions.
- [ ] `README.md` updated if a new algorithm or dataset was added.
