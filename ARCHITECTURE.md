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
  - `tuning.py`: validation tuning orchestration (grid and Bayesian modes).
  - `tuning_bayes.py`: staged Optuna search for raw, CRF/shadow, and bridge params.
  - `common.py`: split resolution, model init, shared prep logic.
  - `inference_utils.py`: tile context load and score-threshold helpers.
- `segedge/core/`
  - Feature extraction, banks, kNN, XGBoost, CRF, continuity, shadow, metrics.
  - I/O exports (`io_utils.py`).
  - Timing and telemetry helpers (`timing_utils.py`, `timing_csv.py`).
  - Curated training-config logging helper (`run_config_logging.py`).
  - Optuna tuning callbacks (`optuna_stop.py`, `optuna_feedback.py`).
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
   - Optional GT-aware source cache gate:
     `SOURCE_PREFETCH_GT_ONLY=True` caches/prefetches only GT-overlap source tiles.
5. Tune on validation tiles.
   - `TUNING_MODE="grid"`: exhaustive Cartesian search (legacy).
   - `TUNING_MODE="bayes"`: staged Bayesian search:
     - Stage 1: roads/top-p/kNN/XGB + silver-core dilation.
     - Stage 2: CRF + shadow parameters with broad-then-refine flow.
       - Refinement runs in a fresh study and seeds top trials with
         `enqueue_trial(...)`.
     - Stage 3: bridge/skeleton continuity parameters with frozen upstream maps.
     - Sampler: configurable (`BO_SAMPLER`), default TPE.
     - Study lifecycle: fresh-by-default namespacing with
       `BO_FORCE_NEW_STUDY` to avoid accidental resume from old trial history.
- Trial telemetry: each trial logs objective, proxy loss (`1-objective`),
  IoU metrics, best-so-far value, and readable progress separators
  (for example `==== Stage2 Broad Trial 3/40 ====`).
  - Compact timing suffix mode:
    when enabled, each trial log line can append one short timing summary
    instead of emitting per-image component timing logs.
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

Source supervision for bank/XGB training:
- `SOURCE_SUPERVISION_MODE="gt_if_available"` (default): source-tile labels come
  from GT vector rasterization when available, with fallback to
  `SOURCE_LABEL_RASTER`.
- `SOURCE_SUPERVISION_MODE="gt_only"`: fail fast if GT supervision is missing.
- `SOURCE_SUPERVISION_MODE="source_raster"`: preserve legacy weak-label-only behavior.
- Anti-leak guardrails run before tuning:
  - detect source/validation tile identity overlap,
  - detect source/validation spatial overlap above
    `ANTI_LEAK_TILE_OVERLAP_MIN_RATIO`,
  - warn when GT-based source supervision reuses evaluation GT vectors.
  - optionally fail fast via `ANTI_LEAK_FAIL_FAST`.

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
- `DEBUG_TIMING`
- `TIMING_TILE_LOGS` (set `False` to suppress per-tile timing lines and keep
  image-level summaries)
- `BO_VERBOSE_TRIAL_SEPARATORS` (set `True` for highly-readable Bayes trial logs)

## Explainability Architecture
Tier-1 explainability is generated online during inference:

- Per-tile outputs:
  - `output/run_*/xai/{validation|holdout}/{image_id}.json`
  - `output/run_*/xai/{validation|holdout}/{image_id}_xai.png`
- Post-training source explainability:
  - `output/run_*/xai/training/{source_image_id}_xgb_pca_top5.png`
  - PCA components are ranked by XGB gain relevance and overlaid on RGB using a blue/red diverging map.
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
- `bayes_trials_timeseries.csv`: Optuna per-trial time series with params/attrs.
- `bayes_hyperparam_importances.csv`: stage-wise Optuna importance table.

## Key Data/Algorithm Contracts
- Adaptive thresholding: top-p within SH buffer using
  `p = clip(a * buffer_density + b, p_min, p_max)`.
- Robust tuning objective (Bayesian mode):
  `score = w_gt * IoU_GT + w_sh * IoU_SH`, with optional light image perturbations.
- BO perturbation feature cache:
  `_bo` feature prefetches now use disk cache in `FEATURE_DIR` so repeated
  perturbation evaluations can reuse cached tiles rather than recomputing all tiles.
- Resize hot path acceleration:
  kNN and XGB patch-to-pixel interpolation prefer CUDA bilinear resize when
  `USE_GPU_RESIZE=True` (fallback remains CPU resize path).
- XGB runtime diagnostics:
  `xgb_score_image_b` now logs image-level total/predict/resize timings,
  mirroring kNN timing introspection.
- Bayesian stagnation guard:
  Optuna studies can stop early after `BO_EARLY_STOP_PATIENCE` non-improving trials
  (with tolerance `BO_EARLY_STOP_MIN_DELTA`).
- Parameter-space precedence:
  range keys (`BO_*_RANGE`) override legacy list keys (`*_VALUES`).
- Bayesian diagnostics artifact:
  run-level hyperparameter importances JSON (`BO_IMPORTANCE_FILENAME`) and CSV
  (`BO_IMPORTANCE_CSV_FILENAME`), plus trial time series CSV
  (`BO_TRIALS_CSV_FILENAME`) from Optuna storage.
- Bayesian phase timing artifact:
  run-level compact phase timing CSV (`BO_TRIAL_PHASE_TIMING_CSV_FILENAME`)
  with one row per trial (`bayes_trial_phase_timing.csv` by default).
- Bayesian log-noise guard:
  `BO_EMIT_COMPONENT_TIMING_LOGS=False` suppresses repeated kNN/XGB image-level
  timing lines inside Bayes tuning while preserving compact trial summaries.
- Champion selection: choose better of `knn_raw` and `xgb_raw` by weighted validation IoU.
- Post-processing chain: champion raw -> CRF -> optional bridge -> shadow.
- Silver core: `kNN ∩ XGB` (optional dilation) exported as auxiliary stream.

## Extension Points
- Split policy: `segedge/pipeline/common.py`.
- Phase timing and telemetry schema: `segedge/core/timing_csv.py`.
- Post-processing behavior: `segedge/core/crf_utils.py`, `segedge/core/continuity.py`,
  `segedge/core/shadow_filter.py`.
- Tuning search spaces and strategy: `config.py`, `segedge/pipeline/tuning.py`,
  `segedge/pipeline/tuning_bayes.py`.
