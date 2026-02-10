# ARCHITECTURE

## Goal
Document the SegEdge zero-shot segmentation pipeline structure and entrypoints.

## Folder Structure
- `config.py`: Pipeline configuration and data paths.
- `segedge/core/`: Core modules (features, banks, kNN, XGB, CRF, I/O, metrics, timing CSV helpers).
- `segedge/pipeline/`: Orchestration entrypoints and shared helpers.
- `main.py`: CLI wrapper for the full pipeline.
- `tests/`: Smoke and end-to-end tests.
- `scripts/`: Repo health checks (doctest ratio, file length).

## Design Principles
- **Modularity:** Core logic lives in `segedge/core/`, orchestration in `segedge/pipeline/`.
- **Docstrings + doctests:** Public helpers include docstrings and doctests.

## Workflow
1. Configure paths and hyperparameters in `config.py`.
2. If `AUTO_SPLIT_TILES=True`, tiles are discovered from `TILES_DIR` and split using
   `AUTO_SPLIT_MODE`:
   - `gt_to_val_cap_holdout`: all GT-overlap tiles become validation, source tiles
     come from `SOURCE_TILES`, and non-GT holdout tiles can be capped.
   - `legacy_gt_source_val_holdout`: GT-overlap tiles are split into source/validation,
     and non-GT tiles become holdout.
3. Run `python main.py` for the full pipeline.
4. Per-tile timing telemetry is appended during source training and inference to:
   - `output/run_*/tile_phase_timing.csv` (detailed phase rows)
   - `output/run_*/timing_opportunity_cost.csv` (aggregated phase opportunity-cost table)
