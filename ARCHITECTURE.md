# ARCHITECTURE

## Goal
Document the current SegEdge pipeline architecture with code-accurate behavior,
interfaces, and run artifacts.

## System Scope
- Repository purpose: zero-shot segmentation of linear woody features (LWF) using
  DINOv3 feature transfer with kNN/XGBoost scoring and post-processing.
- Primary entrypoint: `main.py` -> `segedge/pipeline/run.py`.
- Primary operator controls: `config.py`.

## Repository Topology
- `config.py`
  - Runtime controls: data paths, split mode, tuning ranges, output telemetry.
- `segedge/pipeline/`
  - `run.py`: orchestration of end-to-end run lifecycle.
  - `tuning.py`: validation tuning for kNN/XGB/CRF/shadow/top-p/roads.
  - `common.py`: split resolution, model init, shared prep logic.
  - `inference_utils.py`: tile context load and score-threshold helpers.
- `segedge/core/`
  - Feature extraction, banks, kNN, XGBoost, CRF, continuity, shadow, metrics.
  - I/O exports (`io_utils.py`).
  - Timing and telemetry helpers (`timing_utils.py`, `timing_csv.py`).
- `tests/`
  - Smoke and unit coverage for pipeline behavior and utility contracts.

## Design Principles
- Separation of concerns: core algorithms in `segedge/core`, orchestration in
  `segedge/pipeline`.
- Config-first behavior: defaults and search spaces are centralized in `config.py`.
- Incremental artifacts: long runs persist outputs progressively (unions, processed
  tiles, timing CSV rows).
- Reproducibility: deterministic split seeds and explicit run directories.

## End-to-End Runtime Flow
1. Create or resume run directory under `OUTPUT_DIR`.
2. Configure logging and initialize run-scoped artifact paths.
3. Resolve source/validation/holdout tile sets.
4. Build source-derived banks and XGB dataset from source tiles.
5. Tune on validation tiles (roads/top-p/kNN/XGB/CRF/shadow).
6. Run fixed-setting inference on validation tiles (metrics + plots).
7. Run fixed-setting inference on holdout tiles (plots + union exports + resume log).
8. Emit per-tile explainability artifacts (XGB+kNN) during validation and capped holdout.
9. Consolidate features (disk cache mode only).
10. Emit run summary YAML and telemetry CSV summaries.

## Split Semantics
If `AUTO_SPLIT_TILES=True`, split behavior depends on `AUTO_SPLIT_MODE`:

- `gt_to_val_cap_holdout`
  - Validation tiles: all GT-overlap tiles.
  - Source tiles: `SOURCE_TILES` from config.
  - Holdout tiles: non-GT tiles, optionally capped by
    `INFERENCE_TILE_CAP_ENABLED` / `INFERENCE_TILE_CAP` / `INFERENCE_TILE_CAP_SEED`.

- `legacy_gt_source_val_holdout`
  - GT-overlap tiles are split into source/validation using
    `VAL_SPLIT_FRACTION` and `SPLIT_SEED`.
  - Non-GT tiles are holdout.

Common prefilter:
- Tile candidates are filtered to those overlapping `SOURCE_LABEL_RASTER` bounds.

## Runtime Telemetry Architecture
Per-tile telemetry is emitted during runtime, not post-hoc parsed from logs.

- Detailed timing CSV:
  - `output/run_*/tile_phase_timing.csv`
  - One row per tile-phase timing key.
  - Core columns include: `stage`, `tile_role`, `tile_path`, `phase_name`,
    `duration_s`, split/source metadata, and status.

- Opportunity-cost summary CSV:
  - `output/run_*/timing_opportunity_cost.csv`
  - Aggregated by scope and phase with totals, central tendency, runtime share,
    phase grouping, and rank.

Telemetry controls:
- `TIMING_CSV_ENABLED`
- `TIMING_CSV_FILENAME`
- `TIMING_SUMMARY_CSV_FILENAME`
- `TIMING_CSV_FLUSH_EVERY`

## Explainability Architecture
Tier-1 explainability is generated online during inference:

- Per-tile outputs:
  - `output/run_*/xai/{validation|holdout}/{image_id}.json`
  - `output/run_*/xai/{validation|holdout}/{image_id}_xai.png`
- Run summary:
  - `output/run_*/xai_summary.csv`
- Holdout volume control:
  - `XAI_HOLDOUT_CAP_ENABLED`, `XAI_HOLDOUT_CAP`, `XAI_HOLDOUT_CAP_SEED`
- Per-tile explainability timings:
  - `xai_prepare_s`, `xai_plot_s`, `xai_write_s`, `xai_total_s`

## Artifact Contracts
Run outputs are rooted at `output/run_XXX/`:
- `run.log`: full text log.
- `best_settings.yml`: generic settings path.
- `inference_best_setting.yml`: frozen inference settings + weighted validation stats.
- `run_summary.yml`: phase metrics/deltas/timing rollups.
- `processed_tiles.jsonl`: resume-safe tile completion log.
- `plots/`: validation and holdout plots.
- `shapes/unions/.../union.shp`: stream/variant union shapefiles + backups.
- `tile_phase_timing.csv`, `timing_opportunity_cost.csv`: runtime telemetry.
- `xai/` and `xai_summary.csv`: per-tile explainability artifacts.

## Key Data/Algorithm Contracts
- Adaptive thresholding: top-p within SH buffer using
  `p = clip(a * buffer_density + b, p_min, p_max)`.
- Champion selection: choose better of `knn_raw` and `xgb_raw` by weighted validation IoU.
- Post-processing chain: champion raw -> CRF -> optional bridge -> shadow.
- Silver core: `kNN ∩ XGB` (optional dilation) exported as auxiliary stream.

## Extension Points
- Split policy: `segedge/pipeline/common.py`.
- Phase timing and telemetry schema: `segedge/core/timing_csv.py`.
- Post-processing behavior: `segedge/core/crf_utils.py`, `segedge/core/continuity.py`,
  `segedge/core/shadow_filter.py`.
- Tuning search spaces: `config.py` and `segedge/pipeline/tuning.py`.
