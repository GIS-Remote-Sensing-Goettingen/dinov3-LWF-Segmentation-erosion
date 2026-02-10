# Project Knowledge Base (Thesis-Grade Reference)

This document is the authoritative technical reference for the current SegEdge
pipeline. It is written for paper/thesis use, engineering onboarding, and
operations.

It focuses on three guarantees:
1. **Code truth alignment**: statements reflect the current implementation.
2. **Reproducibility readiness**: run contracts and artifact schemas are explicit.
3. **Historical traceability**: major implemented experiments and changes are
   documented by version.

---

## 1) Scope, Audience, and Guarantees

### Audience
- Thesis/paper author (methodology and ablation sections).
- Operator running large tile batches.
- Engineer extending the pipeline.

### Scope
- Main pipeline only (`main.py` -> `segedge/pipeline/run.py`).
- Split policy, tuning, inference, exports, telemetry, and failure handling.

### Document Guarantees
- **Normative**: behavior described as "current" is expected to match code.
- **Historical**: experiment/change history is summarized from `CHANGELOG.md`.
- **Template-based evidence**: result sections use explicit placeholders unless
  values are directly tied to run artifacts.

---

## 2) Problem Formulation

### Task
Given one or more source tiles with raster supervision (`SOURCE_LABEL_RASTER`) and
an unlabeled target domain of tiles, segment linear woody features (LWF).

### Inputs
- RGB orthophoto tiles (`*.tif`).
- Source supervision raster (`SOURCE_LABEL_RASTER`).
- Evaluation vectors (`EVAL_GT_VECTORS`) for validation scoring and split logic.
- Optional roads vector mask (`ROADS_MASK_PATH`) for multiplicative score penalty.

### Outputs
- Per-run plots, union shapefiles, frozen settings YAML, run summary YAML.
- Incremental timing telemetry (`tile_phase_timing.csv`,
  `timing_opportunity_cost.csv`).

### Core Modeling Assumption
Semantic transfer is possible via DINOv3 patch embeddings from source tiles to
new tiles, then calibrated with validation-driven thresholding and
post-processing.

---

## 3) Algorithmic Overview

### Stage Graph
1. Resolve source/validation/holdout tiles.
2. Build source feature banks and XGB dataset.
3. Tune settings on validation set.
4. Infer on validation with frozen settings.
5. Infer on holdout with frozen settings.
6. Export unions, settings, summaries, and telemetry.

### Per-Tile Inference Chain
For each inference tile:
1. Load image and masks (SH labels, optional GT).
2. Build SH buffer mask.
3. Prefetch tile features.
4. Score with kNN and XGB.
5. Apply adaptive top-p threshold in SH buffer.
6. Compute silver core (kNN ∩ XGB, optional dilation).
7. Select champion raw stream.
8. CRF refinement.
9. Optional continuity bridging.
10. Shadow filtering.
11. Plot/export/metrics/timing.

---

## 4) Formal Definitions

Let:
- `S_knn(x)` be kNN score map at pixel `x`.
- `S_xgb(x)` be XGB score map at pixel `x`.
- `B(x)` be SH buffer mask (`True/False`).
- `d = mean(B)` be buffer density.

Adaptive top-p:
- `p(d) = clip(a * d + b, p_min, p_max)`
- Threshold `tau` is chosen so only top `p(d)` fraction inside buffer is kept.

Binary mask from score `S`:
- `M_raw(x) = 1[S(x) >= tau and B(x)]`
- Median-filtered before downstream stages.

Champion choice:
- If `IoU(xgb_raw) >= IoU(knn_raw)`, champion source is `xgb`, else `knn`.

Silver core:
- `M_core = dilate(M_knn_raw ∩ M_xgb_raw, r = SILVER_CORE_DILATE_PX)`.

---

## 5) Data and Coordinate System Handling

### Reprojection Policy
- Raster labels are reprojected to tile grid by `reproject_labels_to_image`.
- GT vectors are rasterized in tile CRS by `rasterize_vector_labels`.
- Missing vector CRS triggers fallback warning behavior.

### Buffer Construction
- `BUFFER_M` converted to pixels using tile resolution.
- `build_sh_buffer_mask` constrains candidate prediction area.

### GT Clipping Option
- `CLIP_GT_TO_BUFFER=True` masks GT outside buffer during evaluation.

### Practical Implication
- Metric comparability depends on consistent buffer policy and CRS alignment.

---

## 6) Split and Resume Semantics

### Auto Split (`AUTO_SPLIT_TILES=True`)
Candidates from `TILES_DIR` + `TILE_GLOB` are filtered by overlap with
`SOURCE_LABEL_RASTER` bounds, then partitioned by GT overlap.

#### Mode A: `gt_to_val_cap_holdout`
- Validation = all GT-overlap tiles.
- Source = `SOURCE_TILES` from config.
- Holdout = non-GT tiles, optionally capped by:
  - `INFERENCE_TILE_CAP_ENABLED`
  - `INFERENCE_TILE_CAP`
  - `INFERENCE_TILE_CAP_SEED`

#### Mode B: `legacy_gt_source_val_holdout`
- GT-overlap tiles split into source/validation via:
  - `VAL_SPLIT_FRACTION`
  - `SPLIT_SEED`
- Non-GT tiles become holdout.

### Resume Behavior
- `RESUME_RUN=True` + `RESUME_RUN_DIR` resumes existing run directory.
- Completed holdout tiles are loaded from `processed_tiles.jsonl`.
- Already done tiles are skipped.

---

## 7) Configuration Truth (Current Defaults)

This section reflects current `config.py` defaults.

### Key Runtime Defaults
- `FEATURE_CACHE_MODE = "memory"`
- `TILE_SIZE = 2048`
- `STRIDE = 512`
- `PATCH_SIZE = 16`
- `BUFFER_M = 5.0`

### Split Defaults
- `AUTO_SPLIT_TILES = True`
- `AUTO_SPLIT_MODE = "gt_to_val_cap_holdout"`
- `INFERENCE_TILE_CAP_ENABLED = True`
- `INFERENCE_TILE_CAP = 50`

### Top-p Defaults
- `TOP_P_A = 0.2`
- `TOP_P_B = 0.04`
- `TOP_P_MIN = 0.02`
- `TOP_P_MAX = 0.12`

Top-p candidate grids:
- `TOP_P_A_VALUES = [0.0, 0.2, 0.4]`
- `TOP_P_B_VALUES = [0.02, 0.04, 0.06]`
- `TOP_P_MIN_VALUES = [0.02, 0.03]`
- `TOP_P_MAX_VALUES = [0.06, 0.08, 0.1, 0.12]`

### CRF/Shadow/Bridge Defaults
- CRF softness/weights configured in `PROB_SOFTNESS_VALUES`, `POS_W_VALUES`,
  `BILATERAL_*`.
- Shadow tuned over `SHADOW_WEIGHT_SETS`, `SHADOW_THRESHOLDS`,
  `SHADOW_PROTECT_SCORES`.
- Bridge enabled (`ENABLE_GAP_BRIDGING=True`) with large-gap settings.

### Timing Telemetry Defaults
- `TIMING_CSV_ENABLED = True`
- `TIMING_CSV_FILENAME = "tile_phase_timing.csv"`
- `TIMING_SUMMARY_CSV_FILENAME = "timing_opportunity_cost.csv"`
- `TIMING_CSV_FLUSH_EVERY = 1`

### Explainability Defaults (Tier 1)
- `XAI_ENABLED = True`
- `XAI_INCLUDE_XGB = True`
- `XAI_INCLUDE_KNN = True`
- `XAI_TOP_FEATURES = 20`
- `XAI_TOP_PATCHES = 50`
- `XAI_HOLDOUT_CAP_ENABLED = True`
- `XAI_HOLDOUT_CAP = 10`
- `XAI_HOLDOUT_CAP_SEED = 42`

---

## 8) Module Map and Responsibilities

### Orchestration
- `segedge/pipeline/run.py`: full run lifecycle, exports, telemetry integration.
- `segedge/pipeline/tuning.py`: validation tuning and selection logic.
- `segedge/pipeline/common.py`: split logic, model init, utility flow.
- `segedge/pipeline/inference_utils.py`: context load and inference helpers.

### Core Algorithms
- `segedge/core/features.py`: tiling, feature extraction, prefetch.
- `segedge/core/banks.py`: positive/negative bank construction.
- `segedge/core/knn.py`: kNN scoring and threshold utilities.
- `segedge/core/xdboost.py`: dataset build, train/search, scoring.
- `segedge/core/crf_utils.py`: dense CRF refinement and search.
- `segedge/core/continuity.py`: skeleton-based gap bridging.
- `segedge/core/shadow_filter.py`: dark-region suppression.
- `segedge/core/metrics_utils.py`: IoU/F1/P/R and batch metrics.

### I/O and Telemetry
- `segedge/core/io_utils.py`: image/vector I/O, shapefile/YAML exports.
- `segedge/core/timing_csv.py`: detailed and summary timing CSV generation.
- `segedge/core/summary_utils.py`: phase/timing aggregation for YAML.

---

## 9) Runtime Artifact Contract

Each run writes to `output/run_XXX/`:

### Control and Logs
- `run.log`
- `processed_tiles.jsonl`

### Settings and Summaries
- `best_settings.yml`
- `inference_best_setting.yml`
- `run_summary.yml`

### Timing Telemetry
- `tile_phase_timing.csv`
- `timing_opportunity_cost.csv`

### Explainability
- `xai/{validation|holdout}/{image_id}.json`
- `xai/{validation|holdout}/{image_id}_xai.png`
- `xai_summary.csv`

### Visual and Geospatial Outputs
- `plots/*.png`
- `shapes/unions/{knn|xgb|champion|silver_core}/{variant}/union.shp`

---

## 10) Timing Telemetry Schema (Detailed CSV)

File: `tile_phase_timing.csv`

Row granularity:
- One row per `(tile, phase_name)` timing event.

Columns:
- `run_dir`
- `timestamp_utc`
- `stage` (`source_training`, `validation_inference`, `holdout_inference`)
- `tile_role` (`source`, `validation`, `holdout`)
- `tile_path`
- `image_id`
- `phase_name`
- `duration_s`
- `gt_available`
- `source_mode`
- `auto_split_mode`
- `resample_factor`
- `tile_size`
- `stride`
- `status`

Typical phase keys now include:
- Source: `load_source_image_s`, `reproject_source_labels_s`,
  `prefetch_source_features_s` (memory mode), `build_banks_s`,
  `build_xgb_dataset_s`, `source_tile_total_s`.
- Inference: `load_context_s`, `roads_mask_s`, `prefetch_features_s`,
  `knn_score_s`, `knn_threshold_s`, `knn_metrics_s`, `knn_s`,
  `xgb_score_s`, `xgb_threshold_s`, `xgb_metrics_s`, `xgb_s`, `silver_core_s`,
  `crf_knn_s`, `crf_xgb_s`, `crf_s`, `bridge_s`, `shadow_s`, `skeleton_s`,
  `plot_s`, `xai_prepare_s`, `xai_plot_s`, `xai_write_s`, `xai_total_s`, `total_s`.

---

## 11) Opportunity-Cost Summary Schema

File: `timing_opportunity_cost.csv`

Aggregation scopes:
- Stage-level scopes (`source_training`, `validation_inference`,
  `holdout_inference`).
- Composite scopes (`all_inference`, `all`).

Columns:
- `scope`
- `phase_name`
- `count`
- `total_s`
- `mean_s`
- `median_s`
- `min_s`
- `max_s`
- `runtime_share_pct`
- `optional_phase`
- `phase_group`
- `opportunity_rank`

Interpretation:
- `runtime_share_pct` quantifies opportunity cost within each scope.
- `opportunity_rank=1` is highest total runtime contributor in scope.

---

## 12) Evaluation and Selection Logic

### Primary Metric
- IoU (with F1/P/R as secondary diagnostics).

### Validation Weighting
- Tile metrics are weighted by GT-positive pixel count.

### Champion Chain Reporting
- Reported phase chain includes:
  - `champion_raw`
  - `champion_crf`
  - optional `champion_bridge`
  - `champion_shadow`

### Why This Matters
- Best raw model and best final post-processed model can differ materially.
- Per-phase deltas in `run_summary.yml` and timing CSVs support tradeoff analysis.

---

## 13) Performance and Scaling Considerations

### Dominant Costs
- Feature prefetch on large tiles.
- Score-map resize/aggregation in kNN path.
- CRF grid search when config ranges are wide.

### Known Runtime Accelerators
- Batched feature extraction (`FEATURE_BATCH_SIZE`).
- CPU fallback behavior for unsupported XGB GPU builds.
- Roads spatial index caching.
- Resume mode and incremental holdout processing.

### Memory Risk Areas
- Feature caches and full-size score/probability arrays.
- Large CRF candidate sets.

---

## 14) Failure Atlas (Symptom -> Cause -> Action)

1. **IoU near zero unexpectedly**
- Cause: CRS mismatch or buffer clipping mismatch.
- Action: inspect reprojection logs and buffer coverage diagnostics.

2. **CRF degrades IoU sharply**
- Cause: overly strong unary/pairwise settings.
- Action: soften CRF grid (higher softness, lower pairwise weights), compare raw vs CRF.

3. **Shadow stage changes nothing**
- Cause: thresholds/weights too permissive, or little dark-noise in candidate mask.
- Action: widen/shift shadow threshold set or temporarily disable for ablation.

4. **Run too slow on many tiles**
- Cause: large holdout count, heavy overlap, broad tuning grids.
- Action: cap holdout, narrow search spaces, adjust stride.

5. **Resume does not skip expected tiles**
- Cause: missing/partial `processed_tiles.jsonl` records.
- Action: verify run dir and processed log integrity.

---

## 15) Reproducibility Protocol

### Minimum run manifest to archive per experiment
- Git revision.
- Full `config.py` snapshot.
- `inference_best_setting.yml`.
- `run_summary.yml`.
- `tile_phase_timing.csv` + `timing_opportunity_cost.csv`.

### Determinism-related knobs
- `SPLIT_SEED`
- `INFERENCE_TILE_CAP_SEED`
- Any random seeds inside training/tuning utilities.

### Suggested run discipline
1. Freeze config.
2. Run a single `run_XXX` experiment.
3. Archive output folder without post-editing.
4. Record experiment metadata in thesis table template.

---

## 16) Historical Evolution and Implemented Experiments

This section summarizes implemented changes by version and the experimental intent.
All entries are implemented history, not proposals.

| Version | What Was Implemented | Primary Intent / Impact |
|---|---|---|
| 0.2.25 | Retuned top-p/CRF/shadow ranges | Reduce top-p cap saturation, soften harmful post-processing defaults |
| 0.2.24 | Incremental per-tile timing CSV + opportunity-cost summary CSV | Structured runtime tradeoff analysis without log parsing |
| 0.2.23 | Auto-split mode (`gt_to_val_cap_holdout`) + holdout cap | Align workflow: GT validation coverage + bounded inference runtime |
| 0.2.22 | Tuning preview plots | Earlier qualitative feedback during tuning |
| 0.2.21 | Debug logs promoted to INFO | Better observability with reduced third-party noise |
| 0.2.20 | Roads geometry clipping/simplification pre-rasterization | Faster roads mask build |
| 0.2.19 | Release discipline + file-length checks | Maintainability and per-request version hygiene |
| 0.2.18 | Run summary YAML with metrics/deltas/timing | Durable experiment summaries |
| 0.2.17 | Always-on reprojection/coverage logging | Diagnose label alignment failures |
| 0.2.16 | Adaptive top-p + silver core outputs | Better threshold adaptability and high-precision supervision stream |
| 0.2.15 | Per-tile roads spatial-index rasterization | Avoid global roads raster OOM |
| 0.2.14 | Unified plot enrichment (labels/similarity/score maps) | Better diagnosis of score and label alignment |
| 0.2.13 | Single inference settings YAML after validation | Stable downstream holdout configuration |
| 0.2.12 | Roads mask path realignment | Fix pathing after data layout changes |
| 0.2.11 | Roads multiplicative penalty tuning | Controlled suppression of road false positives |
| 0.2.10 | Weighted IoU tuning objective + phase markers | Better config selection fidelity |
| 0.2.9 | Continuity bridging + unified phase summaries | Improve topology continuity, compare phase gains |
| 0.2.8 | KB updates for rolling unions/resume/split | Documentation alignment |
| 0.2.7 | Holdout-only rolling unions + resume logging | Incremental inference durability |
| 0.2.6 | Rolling union backups | Crash resilience and recovery |
| 0.2.5 | Source-label-overlap filter before auto-split | Avoid pointless GT checks on out-of-coverage tiles |
| 0.2.4 | Batched feature extraction | Better GPU throughput |
| 0.2.3 | Cached vector intersection GT checks | Faster tile scan for split |
| 0.2.2 | Parallel GT presence checks | Faster split prep on many tiles |
| 0.2.1 | Removed split-eval/per-tile shapefile clutter | Simplified main pipeline outputs |
| 0.2.0 | Phase logging, diskless cache mode, parallel CRF tuning, GT auto split | Core modernization and scaling improvements |
| 0.1.0 | Package refactor + doctests + smoke E2E | Structural baseline and validation guardrails |

### Reading This Timeline
- Later versions are cumulative.
- Not all versions affect model quality; many improve observability, runtime,
  or reproducibility.

---

## 17) Thesis/Paper Reporting Templates (Strict Placeholders)

### 17.1 Main Results Table (Template)
| Experiment ID | Split Mode | Source Tiles | Val Tiles | Holdout Tiles | Raw Champion IoU | Final IoU | Runtime (h) | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| EXP-XXX | [mode] | [n] | [n] | [n] | [ ] | [ ] | [ ] | [ ] |

### 17.2 Phase Ablation Table (Template)
| Experiment ID | Raw IoU | +CRF Delta | +Bridge Delta | +Shadow Delta | Final IoU |
|---|---:|---:|---:|---:|---:|
| EXP-XXX | [ ] | [ ] | [ ] | [ ] | [ ] |

### 17.3 Runtime Opportunity-Cost Table (Template)
| Experiment ID | Scope | Rank 1 Phase | Rank 1 Share % | Rank 2 Phase | Rank 2 Share % |
|---|---|---|---:|---|---:|
| EXP-XXX | holdout_inference | [ ] | [ ] | [ ] | [ ] |

### 17.4 Hyperparameter Record (Template)
| Experiment ID | TOP_P (a,b,min,max) | CRF Grid ID | Shadow Grid ID | Roads Penalty | Bridge Params Snapshot |
|---|---|---|---|---|---|
| EXP-XXX | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## 18) Operator Runbook

1. Confirm paths and split mode in `config.py`.
2. Run `python main.py`.
3. Track progress in `run.log` and `processed_tiles.jsonl`.
4. Inspect:
   - `inference_best_setting.yml`
   - `run_summary.yml`
   - `timing_opportunity_cost.csv`
5. For failures/resume:
   - set `RESUME_RUN=True`
   - set `RESUME_RUN_DIR` to interrupted run folder
   - rerun.

---

## 19) Terminology Glossary

- **Source tile**: Tile used to build banks/XGB dataset.
- **Validation tile**: Tile used for tuning and metric-driven selection.
- **Holdout tile**: Final inference tile, usually unlabeled.
- **Champion**: Better of raw kNN vs raw XGB on validation objective.
- **Silver core**: Conservative overlap mask from kNN and XGB.
- **Top-p thresholding**: Keep top `p` fraction of scores within buffer.
- **Opportunity cost (runtime)**: Phase runtime share within a scope.

---

## 20) Change-Control Notes for This KB

When code changes, update this KB if any of the following change:
- Config defaults or tuning ranges.
- Split behavior or resume semantics.
- Output artifact paths or schemas.
- Telemetry schema and phase naming.
- Champion/post-processing ordering.

For thesis usage, treat this KB as the canonical technical appendix seed.
