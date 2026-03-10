# Changelog

## [Unreleased]

### Documentation and repository health
- Description: Move human-maintained repository docs under `docs/`, remove `journal.md`, expand workflow/function documentation, and add a function-length guard that excludes leading docstrings and doctests from its count.
- Files touched: `AGENTS.md`, `docs/README.md`, `docs/ARCHITECTURE.md`, `docs/Implementation.md`, `docs/KB.md`, `docs/CHANGELOG.md`, `.pre-commit-config.yaml`, `scripts/check_function_length.py`, `tests/test_function_length.py`, `segedge/core/config_loader.py`, `segedge/pipeline/tuning.py`, `segedge/pipeline/runtime/holdout_inference.py`, `segedge/pipeline/workflows/loo_training.py`
- Reason: Centralize documentation, make the refactored workflow easier to navigate, and enforce a maintainability limit on function size without penalizing documentation blocks.
- Problems fixed: Repository docs now live in one place, the workflow modules and major orchestration functions are documented in detail, pre-commit now fails when a Python function exceeds 400 counted lines after removing its docstring block from the count, and the existing oversized orchestration functions were split so the new guard passes.

### Refactors and code organization
- Description: Split oversized feature, runtime, and pipeline entrypoint modules into focused packages and workflow-specific modules.
- Files touched: `segedge/core/features.py`, `segedge/core/feature_ops/`, `segedge/pipeline/runtime_utils.py`, `segedge/pipeline/runtime/`, `segedge/pipeline/run.py`, `segedge/pipeline/workflows/`, `tests/test_run_dispatch.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Reduce the maintenance cost of `features.py`, `runtime_utils.py`, and `run.py`, which had grown too large to navigate safely.
- Problems fixed: `run.py` is now a bootstrap/dispatch layer, runtime helpers are grouped by concern, feature operations are split into dedicated modules, and a dispatch test now pins the workflow selection behavior.

### Inference, tuning, and runtime stability
- Description: Add `scripts/analyze_performance_log.py` to summarize `performance.jsonl` by stage, substage, and hottest tile contributors, and ignore generated `performance.jsonl` files in the repo file-length guard.
- Files touched: `scripts/analyze_performance_log.py`, `tests/test_analyze_performance_log.py`, `scripts/check_file_length.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Make the new structured performance log usable for iterative bottleneck analysis without hand-parsing JSONL.
- Problems fixed: The repository now has a repeatable EDA entrypoint that reports average durations per process, per sub-process, per tile, can export CSV summaries for deeper comparison across runs, and pre-commit no longer fails on large generated profiling logs.

- Description: Allow inference-only runs with `io.inference.model_bundle_dir: null` to reuse the newest valid previous `output/run_*/model_bundle`.
- Files touched: `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `tests/test_config_loader_inference_mode.py`, `tests/test_run_dispatch.py`, `docs/KB.md`, `docs/Implementation.md`, `docs/CHANGELOG.md`
- Reason: Remove the need to manually copy the last bundle path into the config for every inference-only run.
- Problems fixed: `io.training=false` no longer fails at config load when `model_bundle_dir` is unset, inference-only bootstrap now resolves the latest valid prior bundle automatically, explicit bundle paths still take precedence, and the fallback failure mode is now explicit when no previous bundle exists.

- Description: Add structured `performance.jsonl` logging for inference internals, cache GT vector geometries by CRS, stream XGB cached features more lazily, and refactor novel-proposal evaluation onto local component crops.
- Files touched: `segedge/core/timing_utils.py`, `segedge/core/io_utils.py`, `segedge/core/feature_ops/cache.py`, `segedge/core/feature_ops/extraction.py`, `segedge/core/xdboost.py`, `segedge/pipeline/inference_flow.py`, `segedge/pipeline/runtime/holdout_inference.py`, `segedge/pipeline/runtime/postprocess.py`, `segedge/pipeline/runtime/tile_context.py`, `tests/test_performance_logging.py`, `docs/ARCHITECTURE.md`, `docs/Implementation.md`, `docs/CHANGELOG.md`
- Reason: Make long inference runs measurable at the function-internal level and reduce the dominant CPU overhead without adding new runtime knobs.
- Problems fixed: Each run now writes machine-readable per-span performance data alongside `run.log`, holdout inference records tile/phase-aware timing summaries, GT vector reprojection work is reused across tiles, XGB-only inference no longer materializes every cached feature array up front, and novel-proposal analysis now evaluates components from local bounding boxes instead of rebuilding full-image masks for every candidate.

- Description: Add a manual `io.inference.score_prior` that boosts XGB scores inside `SOURCE_LABEL_RASTER` pixels during the final holdout/inference phase only.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/runtime/holdout_inference.py`, `segedge/pipeline/inference_flow.py`, `segedge/pipeline/workflows/shared.py`, `docs/Implementation.md`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Allow manual recall-oriented score adjustment inside known source-label regions without affecting validation metrics or threshold tuning.
- Problems fixed: Final holdout inference can now apply a configurable XGB-only in-label score boost in both training and inference-only runs, while validation and fold-tuning inference remain unchanged.

- Description: Add per-tile holdout progress logs in the form `Processing tile <path>, <current> / <total>`.
- Files touched: `segedge/pipeline/inference_flow.py`, `tests/test_inference_flow.py`, `docs/Implementation.md`, `docs/CHANGELOG.md`
- Reason: Make long inference runs easier to track from logs, especially when resuming partially completed jobs.
- Problems fixed: Holdout inference now reports stable run-level progress counts that exclude already processed tiles, while preserving the existing skip logs and append/checkpoint ordering.

- Description: Filter folder/list-based inference tiles by actual `SOURCE_LABEL_RASTER` label presence and keep rolling union outputs updated tile by tile.
- Files touched: `segedge/pipeline/common.py`, `segedge/pipeline/inference_flow.py`, `segedge/pipeline/run.py`, `tests/test_inference_flow.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Avoid spending inference time on tiles that only intersect the label-raster extent while containing no source labels, and preserve usable mask geometry if a long job stops mid-run.
- Problems fixed: Inference tile resolution now skips tiles with no positive source-label pixels in both training+inference and inference-only modes, auto split uses the same stricter eligibility rule, empty filtered holdout sets no longer crash the run, and tests verify union-mask append/checkpoint updates happen after each inferred tile.

- Description: Force serial CRF tuning when CUDA is active, cap CRF workers to candidate count, and retry serially if the CRF process pool crashes.
- Files touched: `config.yml`, `segedge/pipeline/tuning.py`, `docs/CHANGELOG.md`
- Reason: Prevent `BrokenProcessPool` and `Bus error` failures during CRF tuning from unsafe process forking and pointless worker oversubscription.
- Problems fixed: Default config no longer requests 16 CRF workers for a single config; runtime now avoids CRF process pools on CUDA, reduces workers to the available config count, and falls back to serial CRF evaluation if a worker pool still dies.

- Description: Prevent kNN GPU OOM during large-bank scoring by chunking top-k similarity matmuls.
- Files touched: `segedge/core/knn.py`, `docs/CHANGELOG.md`
- Reason: Full `[tile_patches x bank_size]` GPU similarity matrices can exceed memory with multi-million negative banks.
- Problems fixed: Replaces full-matrix kNN similarity matmul with chunked top-k aggregation for both positive and negative banks, avoiding 100+ GiB temporary allocations.

- Description: Switch inference visual outputs to single active model panels, split plots by stage, and upgrade novel proposal acceptance and diagnostics.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `segedge/core/plotting.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Reduce confusion from multi-stream inference plots, allow disabling kNN/CRF cleanly, and make proposal decisions auditable.
- Problems fixed: Adds model enable toggles, writes validation/inference plots into separate folders, annotates proposal overlays with acceptance scores, auto-accepts proposal components inside the SH buffer while evaluating outside components heuristically, and exports per-component proposal records with rejection reasons.

- Description: Fix LOO tile eligibility mismatch by requiring effective GT positives after optional SH-buffer clipping.
- Files touched: `segedge/pipeline/common.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Prevent tiles with vector overlap but zero post-clip GT from entering LOO train/validation.
- Problems fixed: Auto split now performs a second eligibility pass and moves tiles with zero effective GT pixels to inference tiles, avoiding no-GT validation plots during training.

- Description: Restore variable XGB threshold selection for tuning runs.
- Files touched: `config.yml`, `docs/CHANGELOG.md`
- Reason: Keep threshold adaptive per fold/run instead of pinning to a single fixed value.
- Problems fixed: `search.xgb.fixed_threshold` is back to `null`, so XGB threshold is selected from the configured threshold range.

- Description: Add fast-stable LOO tuning defaults with fixed XGB threshold/config, low-GT fold skipping, and outside-SH proposal candidate scope.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Reduce wasted tuning time on sparse folds, preserve best-so-far under the 10-hour budget, and surface novel linear-feature proposals beyond SH bounds during inference.
- Problems fixed: Shrinks kNN/XGB search space, supports two-tile validation folds, skips low-signal folds, applies budget checks at tuning checkpoints, pins XGB threshold when configured, and enables whole-tile proposal sourcing for accepted/rejected overlays and shapefiles.

- Description: Add time-budget cutover that uses best-so-far settings and transitions to inference after budget expiry.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Long LOO runs need a deterministic wall-clock stop point with interruption-safe continuation to inference.
- Problems fixed: Adds configurable runtime budget, resume-aware deadline handling, rolling checkpoint time-budget metadata, and immediate inference cutover using best completed fold artifacts when the budget is hit.

- Description: Reduce LOO tuning runtime with stronger caching defaults and XGB scoring/training efficiency improvements.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/core/features.py`, `segedge/core/banks.py`, `segedge/core/xdboost.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `docs/CHANGELOG.md`
- Reason: Address repeated DINO extraction, CPU fallback overhead, expensive roads rasterization, and oversized training samples.
- Problems fixed: Defaults to disk feature cache, adds roads-mask disk caching, switches XGB tile inference to `inplace_predict`, caps positive samples for banks/XGB datasets, hardens feature-cache validity via `feature_spec_hash`, and reduces repeated GPU retry behavior.

- Description: Add hybrid DINO plus image feature fusion, shape-filtered novel proposals, and richer inference diagnostics plots.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/core/features.py`, `segedge/core/banks.py`, `segedge/core/knn.py`, `segedge/core/xdboost.py`, `segedge/core/plotting.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Improve thin-structure segmentation boundaries and auditability by combining semantic embeddings with local patch cues and object-level heuristics.
- Problems fixed: Enables leakage-safe train-fold feature standardization for XGB, preserves kNN cosine geometry, exports feature-spec/model metadata, proposes accepted/rejected novel objects outside incomplete labels, and adds boundary/disagreement/uncertainty/importance visual diagnostics.

- Description: Add interruption-safe rolling best-config checkpoints during LOO tuning and holdout inference.
- Files touched: `segedge/pipeline/run.py`, `segedge/pipeline/common.py`, `docs/CHANGELOG.md`
- Reason: Persist the current best configuration while long runs are still in progress.
- Problems fixed: If a run stops early, `rolling_best_setting.yml` preserves the best-known config and progress counters.

### Bundles and inference-only operation
- Description: Switch persisted inference bundle to XGB-only artifacts, drop bank `.npy` files, and restore legacy `best_setting.yml` alongside `inference_best_setting.yml`.
- Files touched: `segedge/pipeline/artifacts.py`, `segedge/pipeline/run.py`, `tests/test_model_bundle.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: kNN bank arrays are too large for practical persistence, while current inference reuse only needs XGB.
- Problems fixed: Bundle save/load now requires only `xgb_model.json` plus manifest metadata, avoids writing huge `pos_bank.npy` and `neg_bank.npy`, forces XGB-only behavior for `io.training=false` loads, and writes both best-settings filenames for compatibility.

- Description: Add persisted model bundles plus inference-only runtime mode (`io.training=false`) to run massive-scale inference without retraining.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/artifacts.py`, `segedge/pipeline/run.py`, `segedge/pipeline/runtime_utils.py`, `tests/test_model_bundle.py`, `tests/test_config_loader_inference_mode.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Enable train-once/infer-many operation by reloading tuned XGB/kNN/CRF settings and feature banks.
- Problems fixed: Saves `manifest.yml` plus optional XGB model after training, validates bundle/runtime compatibility at load time, supports inference-only tile resolution through `io.inference`, and records bundle metadata in rolling and best-settings outputs.

### Workflow evolution and configuration
- Description: Add manual-mode inference directory support and switch config to directory-driven inference tiles.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `docs/ARCHITECTURE.md`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Avoid maintaining long manual `holdout_tiles` lists when inference should cover a full folder.
- Problems fixed: Manual mode now resolves inference tiles from `io.paths.inference_dir` plus `io.paths.inference_glob` with fallback to `io.paths.holdout_tiles`, and `config.yml` now points inference to `/mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt/folder_1`.

- Description: Switch runtime config to manual tile mode using the same tile filenames as `main`.
- Files touched: `config.yml`, `docs/CHANGELOG.md`
- Reason: Restore explicit train/validate/holdout tile selection for manual runs.
- Problems fixed: Sets `io.auto_split.enabled=false` and populates `io.paths.source_tiles`, `io.paths.val_tiles`, and `io.paths.holdout_tiles` with the `main` branch tile lists.

- Description: Restore optional manual tile selection flow while keeping directory-driven LOO as default.
- Files touched: `config.yml`, `segedge/pipeline/run.py`, `docs/ARCHITECTURE.md`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Allow selecting explicit source/validation/holdout tiles again without removing the LOO workflow.
- Problems fixed: Manual mode now runs source-tile training plus validation tuning plus holdout inference using explicit tile lists, and exported inference settings include `extra.mode` (`manual` or `loo`).

- Description: Switch training/search to directory-driven leave-one-out folds and export fold mean/std summaries in inference settings.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `segedge/core/xdboost.py`, `docs/ARCHITECTURE.md`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Replace manual tile split flow with GT-presence discovery plus LOO validation.
- Problems fixed: Removes dependency on manual source/validation lists, trains the final model from all GT tiles after LOO selection, and reports process variability in `inference_best_setting.yml`.

- Description: Add source-tile augmentation, compact repetitive timing logs, and richer best-settings YAML model details.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/core/timing_utils.py`, `segedge/pipeline/run.py`, `segedge/core/io_utils.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Improve source diversity during bank/XGB building, reduce log spam from per-tile timers, and export clearer model/tuning context.
- Problems fixed: Adds configurable augmentation, collapses noisy repeated timing lines into compact summaries, and records XGB/champion/model metadata in `inference_best_setting.yml`.

- Description: Remove continuity bridging and migrate runtime config from `config.py` to commented `config.yml` with typed loader access.
- Files touched: `config.yml`, `segedge/core/config_loader.py`, `segedge/pipeline/run.py`, `segedge/core/plotting.py`, `segedge/core/continuity.py`, `segedge/core/features.py`, `segedge/core/xdboost.py`, `segedge/core/io_utils.py`, `segedge/core/knn.py`, `segedge/core/banks.py`, `segedge/core/timing_utils.py`, `segedge/pipeline/common.py`, `tests/test_e2e_smoke.py`, `docs/ARCHITECTURE.md`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Simplify champion postprocessing and centralize configuration in a structured YAML format.
- Problems fixed: Removes over-connecting bridge regressions and replaces flat Python constants with a readable, validated runtime config.

- Description: Switch roads masking to per-tile spatial index rasterization.
- Files touched: `config.py`, `segedge/pipeline/run.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Avoid OOM from global raster builds while keeping tile-resolution masking.
- Problems fixed: Prevents roads raster build OOM and keeps per-tile masking fast.

- Description: Add source label, DINO similarity, and score heatmaps to the unified plot.
- Files touched: `segedge/core/plotting.py`, `segedge/pipeline/run.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Improve plot diagnostics with label context and model score maps.
- Problems fixed: Makes score variations and label alignment visible per tile.

- Description: Write a single `inference_best_setting.yml` after validation and stop per-tile settings.
- Files touched: `segedge/pipeline/run.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Freeze holdout settings once and include weighted validation metrics.
- Problems fixed: Removes redundant per-tile settings and captures run-level metrics.

- Description: Point roads mask path to `data/roads`.
- Files touched: `config.py`, `docs/CHANGELOG.md`
- Reason: Align roads penalty with the updated data layout.
- Problems fixed: Prevents missing roads-mask inputs after the folder move.

- Description: Add tunable roads mask penalty for kNN/XGB score maps.
- Files touched: `config.py`, `segedge/pipeline/run.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Downweight predictions on roads with a validated multiplier.
- Problems fixed: Allows road suppression without hard masking.

- Description: Switch tuning to weighted-mean IoU and add red phase markers.
- Files touched: `segedge/pipeline/run.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Ensure configs are chosen by GT-weighted validation performance and make phases easy to spot.
- Problems fixed: Avoids selecting configs by non-weighted aggregation and improves log readability.

- Description: Add continuity bridging, unified plots, and phase summaries for kNN/XGB/Champion.
- Files touched: `config.py`, `segedge/core/continuity.py`, `segedge/core/crf_utils.py`, `segedge/core/plotting.py`, `segedge/pipeline/run.py`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Improve topological continuity and make phase-by-phase gains visible.
- Problems fixed: Reduces broken segments and consolidates diagnostic plots.

- Description: Document rolling unions, resume, and auto split updates in the KB.
- Files touched: `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Keep docs aligned with current pipeline behavior and outputs.
- Problems fixed: Prevents outdated output paths and config guidance.

- Description: Add holdout-only rolling unions for kNN/XGB/Champion streams with resume logging.
- Files touched: `config.py`, `segedge/core/io_utils.py`, `segedge/pipeline/run.py`, `docs/CHANGELOG.md`
- Reason: Track per-stream unions during holdout inference and resume safely after interruptions.
- Problems fixed: Enables incremental union outputs and skip/restart logic based on processed tiles.

- Description: Write rolling union shapefile updates with periodic backups.
- Files touched: `config.py`, `segedge/core/io_utils.py`, `segedge/pipeline/run.py`, `docs/CHANGELOG.md`
- Reason: Keep union shapes updated during inference and protect against partial runs.
- Problems fixed: Avoids waiting for the final merge step to see union outputs.

- Description: Filter auto-split tiles to those overlapping `SOURCE_LABEL_RASTER`.
- Files touched: `segedge/pipeline/common.py`, `docs/CHANGELOG.md`
- Reason: Skip tiles outside label-raster coverage during auto split.
- Problems fixed: Prevents unnecessary GT scans for tiles without label coverage.

- Description: Add batched feature extraction for single-scale tiles.
- Files touched: `config.py`, `segedge/core/features.py`, `docs/CHANGELOG.md`
- Reason: Improve GPU utilization by processing multiple tiles per forward pass.
- Problems fixed: Reduces overhead from per-tile model calls during prefetch.

- Description: Use cached vector geometry intersection for GT presence checks.
- Files touched: `segedge/pipeline/common.py`, `docs/CHANGELOG.md`
- Reason: Avoid per-tile rasterization during auto split.
- Problems fixed: Eliminates repeated GT reprojection logs and reduces tile-scan overhead.

- Description: Parallelize GT presence checks for auto tile splitting.
- Files touched: `config.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `docs/CHANGELOG.md`
- Reason: Speed up auto split when scanning large tile directories.
- Problems fixed: Reduces wall time for GT overlap detection with many tiles.

- Description: Remove split-eval entrypoints and per-tile shapefile exports in favor of union shapefiles only.
- Files touched: `segedge/pipeline/run.py`, `split_eval.py`, `segedge/pipeline/split_eval.py`, `docs/ARCHITECTURE.md`, `docs/KB.md`, `docs/CHANGELOG.md`
- Reason: Focus on the main pipeline only and reduce shapefile clutter.
- Problems fixed: Avoids per-tile shapefiles when only union outputs are needed.

## Released Versions

### [0.2.0]
- Description: Add phase logging, diskless feature-cache mode, parallel CRF tuning in the main pipeline, shadow protect-score tuning for split evaluation, and GT-driven auto tile split.
- Files touched: `config.py`, `segedge/core/banks.py`, `segedge/core/knn.py`, `segedge/core/xdboost.py`, `segedge/pipeline/run.py`, `segedge/pipeline/split_eval.py`, `segedge/pipeline/common.py`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Reason: Allow memory-scoped feature reuse, speed up CRF search on multi-core nodes, preserve high-confidence positives under shadows, and automate source/validation/holdout selection from tiles.
- Problems fixed: Avoids redundant feature extraction while keeping memory bounded per image, reduces CRF tuning time, prevents shadow filtering from removing true positives in split evaluation, and removes manual tile list maintenance.

### [0.1.0]
- Description: Refactor pipeline scripts into a package layout and add doctests plus a smoke E2E test.
- Files touched: `segedge/`, `main.py`, `split_eval.py`, `tests/test_e2e_smoke.py`, `docs/`
- Reason: Improve structure, readability, and test coverage.
- Problems fixed: Clarifies module boundaries and adds end-to-end validation guardrails.
