# Changelog


## [Unreleased]
- Description: Fix LOO tile eligibility mismatch by requiring effective GT positives after optional SH-buffer clipping.
- file touched: `segedge/pipeline/common.py`, `ARCHITECTURE.md`, `CHANGELOG.md`
- reason: Prevent tiles with vector overlap but zero post-clip GT from entering LOO train/val.
- problems fixed: Auto split now performs a second eligibility pass and moves tiles with zero effective GT pixels to inference tiles, avoiding no-GT validation plots during training.

- Description: Restore variable XGB threshold selection for tuning runs.
- file touched: `config.yml`, `CHANGELOG.md`
- reason: Keep threshold adaptive per fold/run instead of pinning to a single fixed value.
- problems fixed: Sets `search.xgb.fixed_threshold` back to `null`, so XGB threshold is selected from the configured threshold range.

- Description: Add fast-stable LOO tuning defaults with fixed XGB threshold/config, low-GT fold skipping, and outside-SH proposal candidate scope.
- file touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `CHANGELOG.md`, `ARCHITECTURE.md`
- reason: Reduce wasted tuning time on sparse folds, preserve best-so-far under 10h budget, and surface novel linear-feature proposals beyond SH bounds during inference.
- problems fixed: Shrinks kNN/XGB search space, supports 2-tile validation folds, skips low-signal folds, applies budget checks at tuning checkpoints, pins XGB threshold when configured, and enables whole-tile proposal sourcing for accepted/rejected overlays and shapefiles.

- Description: Add time-budget cutover that uses best-so-far settings and transitions to inference after budget expiry.
- file touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `CHANGELOG.md`, `ARCHITECTURE.md`
- reason: Long LOO runs need a deterministic wall-clock stop point with interruption-safe continuation to inference.
- problems fixed: Adds configurable runtime budget (`runtime.time_budget`), resume-aware deadline handling, rolling checkpoint time-budget metadata, and immediate inference cutover using best completed fold artifacts when the budget is hit.

- Description: Reduce LOO tuning runtime with stronger caching defaults and XGB scoring/training efficiency improvements.
- file touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/core/features.py`, `segedge/core/banks.py`, `segedge/core/xdboost.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `CHANGELOG.md`
- reason: Address repeated DINO extraction, CPU fallback overhead, expensive roads rasterization, and oversized training samples.
- problems fixed: Defaulted to disk feature cache + auto-force disk in LOO+augmentation mode, added roads-mask disk caching, switched XGB tile inference to `inplace_predict`, capped positive samples (`max_pos_bank`) for banks/XGB dataset, hardened feature-cache validity via `feature_spec_hash`, and reduced repeated GPU retry behavior.

- Description: Add hybrid DINO+image feature fusion for kNN/XGB, shape-filtered novel proposals, and richer inference diagnostics plots.
- file touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/core/features.py`, `segedge/core/banks.py`, `segedge/core/knn.py`, `segedge/core/xdboost.py`, `segedge/core/plotting.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `ARCHITECTURE.md`, `CHANGELOG.md`
- reason: Improve thin-structure segmentation boundaries and auditability by combining semantic embeddings with local patch cues and object-level heuristics.
- problems fixed: Enables leakage-safe train-fold feature standardization for XGB, preserves kNN cosine geometry, exports feature-spec/model metadata, proposes accepted/rejected novel objects outside incomplete labels, and adds boundary/disagreement/uncertainty/importance visual diagnostics.

- Description: Add interruption-safe rolling best-config checkpoints during LOO tuning and holdout inference.
- file touched: `segedge/pipeline/run.py`, `segedge/pipeline/common.py`, `CHANGELOG.md`
- reason: Persist the current best configuration while long runs are still in progress.
- problems fixed: If a run stops early, `rolling_best_setting.yml` preserves the best-known config and progress counters (folds/holdout tiles).

- Description: Switch training/search to directory-driven leave-one-out (LOO) folds and export fold mean/std summaries in inference settings.
- file touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `segedge/core/xdboost.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Replace manual tile split flow with robust GT-presence discovery plus LOO validation.
- problems fixed: Removes dependency on manual source/val lists, trains final model from all GT tiles after LOO selection, and reports process variability (mean/std) in `inference_best_setting.yml`.

- Description: Add source-tile augmentation (flip + 90-degree rotations), compact repetitive timing logs, and richer best-settings YAML model details.
- file touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/core/timing_utils.py`, `segedge/pipeline/run.py`, `segedge/core/io_utils.py`, `KB.md`, `CHANGELOG.md`
- reason: Improve source diversity during bank/XGB building, reduce log spam from per-tile timers, and export clearer model/tuning context.
- problems fixed: Adds configurable augmentation, collapses noisy repeated timing lines into compact summaries, and records XGB/champion/model metadata in `inference_best_setting.yml`.

- Description: Remove continuity bridging and migrate runtime config from `config.py` to commented `config.yml` with typed loader access.
- file touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `segedge/core/plotting.py`, `segedge/core/continuity.py`, `segedge/core/features.py`, `segedge/core/xdboost.py`, `segedge/core/io_utils.py`, `segedge/core/knn.py`, `segedge/core/banks.py`, `segedge/core/timing_utils.py`, `segedge/pipeline/common.py`, `tests/test_e2e_smoke.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Simplify champion postprocessing and centralize configuration in a structured YAML format.
- problems fixed: Removes over-connecting bridge phase regressions and replaces flat Python constants with a readable, validated runtime config.

- Description: Switch roads masking to per-tile spatial index rasterization.
- file touched: `config.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Avoid OOM from global raster builds while keeping tile-resolution masking.
- problems fixed: Prevents roads raster build OOM and keeps per-tile masking fast.

- Description: Add source label, DINO similarity, and score heatmaps to unified plot.
- file touched: `segedge/core/plotting.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Improve plot diagnostics with label context and model score maps.
- problems fixed: Makes score variations and label alignment visible per tile.

- Description: Write a single inference_best_setting.yml after validation and stop per-tile settings.
- file touched: `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Freeze holdout settings once and include weighted validation metrics.
- problems fixed: Removes redundant per-tile settings and captures run-level metrics.

- Description: Point roads mask path to data/roads.
- file touched: `config.py`, `CHANGELOG.md`
- reason: Align roads penalty with new data layout.
- problems fixed: Prevents missing roads mask after folder move.

- Description: Add tunable roads mask penalty for kNN/XGB score maps.
- file touched: `config.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Downweight predictions on roads with a validated multiplier.
- problems fixed: Allows road suppression without hard masking.

- Description: Switch tuning to weighted-mean IoU and add red phase markers.
- file touched: `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Ensure configs are chosen by GT-weighted validation performance and make phases easy to spot.
- problems fixed: Avoids selecting configs by non-weighted aggregation and improves log readability.

- Description: Add continuity bridging, unified plots, and phase summaries for kNN/XGB/Champion.
- file touched: `config.py`, `segedge/core/continuity.py`, `segedge/core/crf_utils.py`, `segedge/core/plotting.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Improve topological continuity and make phase-by-phase gains visible.
- problems fixed: Reduces broken segments and consolidates diagnostic plots.

- Description: Document rolling unions, resume, and auto split updates in KB.
- file touched: `KB.md`, `CHANGELOG.md`
- reason: Keep docs aligned with current pipeline behavior and outputs.
- problems fixed: Prevents outdated output paths and config guidance.

- Description: Add holdout-only rolling unions for kNN/XGB/Champion streams with resume logging.
- file touched: `config.py`, `segedge/core/io_utils.py`, `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Track per-stream unions during holdout inference and resume safely after interruptions.
- problems fixed: Enables incremental union outputs and skip/restart logic based on processed tiles.

- Description: Write rolling union shapefile updates with periodic backups.
- file touched: `config.py`, `segedge/core/io_utils.py`, `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Keep union shapes updated during inference and protect against partial runs.
- problems fixed: Avoids waiting for the final merge step to see union outputs.

- Description: Filter auto-split tiles to those overlapping SOURCE_LABEL_RASTER.
- file touched: `segedge/pipeline/common.py`, `CHANGELOG.md`
- reason: Skip tiles outside label raster coverage during auto split.
- problems fixed: Prevents unnecessary GT scans for tiles without label coverage.

- Description: Added batched feature extraction for single-scale tiles.
- file touched: `config.py`, `segedge/core/features.py`, `CHANGELOG.md`
- reason: Improve GPU utilization by processing multiple tiles per forward pass.
- problems fixed: Reduces overhead from per-tile model calls during prefetch.

- Description: Use cached vector geometry intersection for GT presence checks.
- file touched: `segedge/pipeline/common.py`, `CHANGELOG.md`
- reason: Avoid per-tile rasterization during auto split.
- problems fixed: Eliminates repeated GT reprojection logs and reduces tile scan overhead.

- Description: Parallelized GT presence checks for auto tile splitting.
- file touched: `config.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Speed up auto split when scanning large tile directories.
- problems fixed: Reduces wall time for GT overlap detection with many tiles.

- Description: Removed split-eval entrypoints and per-tile shapefile exports; union shapefile only.
- file touched: `segedge/pipeline/run.py`, `split_eval.py`, `segedge/pipeline/split_eval.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Focus on main pipeline only and reduce shapefile clutter.
- problems fixed: Avoids per-tile shapefiles when only union outputs are needed.

## [0.2.0]
- Description: Added phase logging, diskless feature-cache mode, parallel CRF tuning in the main pipeline, shadow protect score tuning for split evaluation, and GT-driven auto split of tiles.
- file touched: `config.py`, `segedge/core/banks.py`, `segedge/core/knn.py`, `segedge/core/xdboost.py`, `segedge/pipeline/run.py`, `segedge/pipeline/split_eval.py`, `segedge/pipeline/common.py`, `ARCHITECTURE.md`, `CHANGELOG.md`
- reason: Allow memory-scoped feature reuse, speed up CRF search on multi-core nodes, preserve high-confidence positives under shadows, and automate source/val/holdout selection from tiles.
- problems fixed: Avoids redundant feature extraction while keeping memory bounded per image, reduces CRF tuning time, prevents shadow filtering from removing true positives in split evaluation, and removes manual tile list maintenance.

## [0.1.0]
- Description: Refactored pipeline scripts into a package layout and added doctests + smoke E2E.
- file touched: `segedge/`, `main.py`, `split_eval.py`, `tests/test_e2e_smoke.py`, docs
- reason: Improve structure, readability, and test coverage.
- problems fixed: Clarified module boundaries and added end-to-end validation guardrails.

EXAMPLE
## [0.0.1]
- Description:
- file touched:
- reason:
- problems fixed:
