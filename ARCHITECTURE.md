# ARCHITECTURE

## Goal
Document the SegEdge zero-shot segmentation pipeline structure and entrypoints.

## Folder Structure
- `config.py`: Pipeline configuration and data paths.
- `segedge/core/`: Core modules (features, banks, kNN, XGB, CRF, I/O, metrics).
- `segedge/pipeline/`: Orchestration entrypoints and shared helpers.
- `main.py`: CLI wrapper for the full pipeline.
- `split_eval.py`: CLI wrapper for split evaluation.
- `tests/`: Smoke and end-to-end tests.
- `scripts/`: Repo health checks (doctest ratio, file length).

## Design Principles
- **Modularity:** Core logic lives in `segedge/core/`, orchestration in `segedge/pipeline/`.
- **Docstrings + doctests:** Public helpers include docstrings and doctests.

## Workflow
1. Configure paths and hyperparameters in `config.py`.
2. Run `python main.py` for full pipeline or `python split_eval.py` for split evaluation.
