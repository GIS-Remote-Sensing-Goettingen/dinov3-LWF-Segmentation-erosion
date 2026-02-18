# ARCHITECTURE

## Goal
Document the SegEdge zero-shot segmentation pipeline structure and entrypoints.

## Folder Structure
- `config.yml`: Primary pipeline configuration (commented, reader-first).
- `segedge/core/config_loader.py`: Typed YAML loader used by runtime modules.
- `segedge/core/`: Core modules (features, banks, kNN, XGB, CRF, I/O, metrics).
- `segedge/pipeline/`: Orchestration entrypoints and shared helpers.
- `main.py`: CLI wrapper for the full pipeline.
- `tests/`: Smoke and end-to-end tests.
- `scripts/`: Repo health checks (doctest ratio, file length).

## Design Principles
- **Modularity:** Core logic lives in `segedge/core/`, orchestration in `segedge/pipeline/`.
- **Docstrings + doctests:** Public helpers include docstrings and doctests.

## Workflow
1. Configure paths and hyperparameters in `config.yml`.
2. If `io.auto_split.enabled=true`, tiles are discovered from `io.auto_split.tiles_dir`
   and split into source/validation using `io.paths.eval_gt_vectors`; tiles without GT become holdout.
3. Run `python main.py` for the full pipeline.
