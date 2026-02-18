# ARCHITECTURE

## Goal
Document the SegEdge zero-shot segmentation pipeline structure and entrypoints.

## Folder Structure
- `config.yml`: Primary pipeline configuration (commented, reader-first).
- `segedge/core/config_loader.py`: Typed YAML loader used by runtime modules.
- `segedge/core/`: Core modules (features, banks, kNN, XGB, CRF, I/O, metrics).
- `segedge/pipeline/`: Orchestration entrypoints and shared helpers.
- `segedge/pipeline/runtime_utils.py`: Runtime-heavy helpers split out of `run.py` (roads masks, proposal filtering, rolling checkpoint utilities, tile context loading).
- `main.py`: CLI wrapper for the full pipeline.
- `tests/`: Smoke and end-to-end tests.
- `scripts/`: Repo health checks (doctest ratio, file length).

## Design Principles
- **Modularity:** Core logic lives in `segedge/core/`, orchestration in `segedge/pipeline/`.
- **Docstrings + doctests:** Public helpers include docstrings and doctests.

## Workflow
1. Configure paths and hyperparameters in `config.yml`.
2. If `io.auto_split.enabled=true`, tiles are discovered from `io.auto_split.tiles_dir`.
   GT-overlap tiles are used for leave-one-out (LOO) folds (`training.loo`), and tiles
   without GT are treated as inference-only holdout tiles.
3. Training artifacts are built per fold from source tiles:
   kNN banks and XGB data can fuse DINO patch embeddings with optional image patch cues
   (`model.hybrid_features`), with train-fold-only z-score stats for XGB.
4. Run `python main.py` for the full pipeline.
5. During execution, `rolling_best_setting.yml` is updated incrementally so best-known settings survive interruptions.
6. Optional runtime time-budget cutover (`runtime.time_budget`) can stop training
   phases after the configured wall-clock budget and switch directly to holdout
   inference using best-so-far fold settings.
7. In inference, champion masks can spawn `postprocess.novel_proposals` outside the
   incomplete source label raster; connected components are filtered by shape
   heuristics and exported as accepted/rejected proposal layers.
8. Plot outputs include unified phase panels plus diagnostics (core qualitative
   boundary errors, score+threshold histogram inset, disagreement/entropy maps,
   proposal overlays, and XGB DINO-channel importance).
