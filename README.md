# SegEdge

SegEdge is a zero-shot segmentation pipeline for linear woody features (LWF) on large orthophoto tile sets.
It combines DINOv3 features, kNN/XGBoost scoring, and geometric post-processing (CRF, continuity bridging, shadow filtering), then exports plots, union shapefiles, and run telemetry.

## What This Repository Contains

- `config.py`: single source of truth for runtime/tuning/split settings.
- `main.py`: entrypoint (`python main.py`).
- `segedge/pipeline/`: orchestration (`run.py`, `tuning.py`, `tuning_bayes.py`).
- `segedge/core/`: feature extraction, models, post-processing, metrics, telemetry, explainability.
- `tests/`: smoke + unit tests for pipeline contracts and utilities.
- `ARCHITECTURE.md`: code-aligned architecture reference.
- `KB.md`: thesis-grade technical knowledge base.
- `CHANGELOG.md`: versioned change history.

## High-Level Pipeline

1. Resolve source/validation/holdout tiles (manual or auto-split).
2. Build source banks and XGBoost training data from source supervision.
3. Tune on validation tiles:
   - `grid` mode: exhaustive combinations.
   - `bayes` mode: staged Optuna search.
4. Freeze best settings.
5. Run inference on validation and holdout tiles.
6. Export plots, unions, settings, summaries, timing CSVs, and XAI artifacts.

## Main Features

- DINOv3 frozen feature extraction with tile prefetch/cache.
- Dual scoring streams: kNN + XGBoost.
- Adaptive top-p thresholding inside SH buffer.
- Champion selection between kNN and XGB streams.
- CRF refinement, optional skeleton gap bridging, shadow filtering.
- Incremental per-tile runtime telemetry CSVs.
- Bayesian optimization telemetry CSVs (trial time series + parameter importances).
- Tier-1 explainability outputs per tile (JSON + plots).
- Resume support for long holdout runs.

## Quick Start

### 1) Configure paths and run settings

Edit `config.py`:

- data locations (`TILES_DIR`, `SOURCE_LABEL_RASTER`, `EVAL_GT_VECTORS`)
- split strategy (`AUTO_SPLIT_TILES`, `AUTO_SPLIT_MODE`)
- runtime caps (`INFERENCE_TILE_CAP_ENABLED`, `INFERENCE_TILE_CAP`)
- tuning mode (`TUNING_MODE = "bayes"` or `"grid"`)

### 2) Run the pipeline

```bash
python main.py
```

### 3) Run quality checks

```bash
pre-commit run --all-files
pytest --doctest-modules
```

## Split Semantics (Important)

When `AUTO_SPLIT_TILES=True`, the current default mode is `gt_to_val_cap_holdout`:

- Validation: all GT-overlap tiles.
- Source: `SOURCE_TILES` from config.
- Holdout: non-GT tiles (optionally capped).

Alternative mode `legacy_gt_source_val_holdout` splits GT-overlap tiles into source/validation with `VAL_SPLIT_FRACTION`.

## Bayesian Tuning (Default)

Default sampler is TPE (`multivariate=True`, `group=True`), with optional CMA-ES.

Current staged flow:

- Stage 1: raw scoring controls (k, neg_alpha, roads/top-p, XGB selection, silver core dilation).
- Stage 2: CRF + shadow parameters (broad search + seeded refinement study).
- Stage 3: bridge/skeleton continuity parameters with frozen upstream maps.

Built-in run guards:

- Optuna pruning support.
- Early stopping on stagnation via:
  - `BO_EARLY_STOP_PATIENCE` (default `20`)
  - `BO_EARLY_STOP_MIN_DELTA`.

## Runtime Artifacts

Each run writes to `output/run_XXX/`:

- `run.log`
- `inference_best_setting.yml`
- `run_summary.yml`
- `plots/`
- `shapes/unions/.../union.shp`
- `processed_tiles.jsonl`
- `tile_phase_timing.csv`
- `timing_opportunity_cost.csv`
- `bayes_trials_timeseries.csv`
- `bayes_hyperparam_importances.csv`
- `xai/` and `xai_summary.csv`

## Timing and Opportunity-Cost Analysis

Structured timing is written incrementally during execution (no log scraping required):

- `tile_phase_timing.csv`: per tile, per phase.
- `timing_opportunity_cost.csv`: aggregated runtime share/rank by phase.

Controls:

- `TIMING_CSV_ENABLED`
- `TIMING_CSV_FILENAME`
- `TIMING_SUMMARY_CSV_FILENAME`
- `TIMING_CSV_FLUSH_EVERY`

## Explainability Outputs

Tier-1 explainability can run during validation and capped holdout inference:

- per tile JSON summaries
- per tile XAI plots
- run-level summary CSV

Controls:

- `XAI_ENABLED`
- `XAI_SAVE_JSON`
- `XAI_SAVE_PLOTS`
- `XAI_HOLDOUT_CAP_ENABLED`

## Performance Controls for Large Tile Collections

If your tile folder is very large, start with these controls:

- Keep holdout bounded: `INFERENCE_TILE_CAP_ENABLED=True`, set `INFERENCE_TILE_CAP`.
- Limit XAI volume: `XAI_HOLDOUT_CAP_ENABLED=True`, set `XAI_HOLDOUT_CAP`.
- Keep feature cache in memory for active tile: `FEATURE_CACHE_MODE="memory"`.
- Use Bayesian tuning (not large grid products) for expensive searches.
- Use `BO_EARLY_STOP_PATIENCE` to stop stagnant studies sooner.

## Documentation Map

- Architecture: `ARCHITECTURE.md`
- Technical deep reference: `KB.md`
- Style and writing conventions: `STYLE.MD`
- Agent/workflow policy: `AGENTS.md`
- Release history: `CHANGELOG.md`
