# Changelog


## [Unreleased]

## [0.2.39]
- Description: Add per-trial Bayesian IoU/loss feedback logging, switch feature caching to on-disk Ceph storage, and reduce default Bayes trial budgets for faster iteration.
- file touched: `config.py`, `segedge/pipeline/tuning_bayes.py`, `segedge/core/optuna_feedback.py`, `segedge/core/run_config_logging.py`, `CHANGELOG.md`
- reason: Long tuning runs needed clearer progress telemetry and lower default budgets, while persisting tile features to shared storage to avoid repeated extraction cost.
- problems fixed: Logs stage-wise trial progress (`value`, proxy `loss`, IoU metrics, best-so-far), stores feature caches under `/mnt/ceph-hdd/projects/mthesis_davide_mattioli/dino_features`, and reduces default stage trials from `400/400/200` to `40/40/20`.

## [0.2.38]
- Description: Replace generic `test.md` loop doc with a SegEdge-specific experiment/promotion specification and enforce agent execution on feature branches only.
- file touched: `test.md`, `CHANGELOG.md`
- reason: Align the specification with this repository's real pipeline, artifacts, gating signals, and workflow constraints.
- problems fixed: Removes non-repo assumptions, defines concrete SegEdge run/metric/artifact contracts, requires human-approved promotion, and explicitly restricts agent work to dedicated feature branches.

## [0.2.37]
- Description: Reduce timing-log clutter with image-level summaries by default, add curated training ablation/config logging, and gate source feature caching to GT-overlap tiles.
- file touched: `config.py`, `segedge/core/timing_utils.py`, `segedge/core/features.py`, `segedge/core/knn.py`, `segedge/core/run_config_logging.py`, `segedge/core/__init__.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `tests/test_auto_split_modes.py`, `ARCHITECTURE.md`, `CHANGELOG.md`
- reason: Keep runtime logs readable during large runs while still surfacing actionable timing/ablation context, and avoid unnecessary source-cache work on non-GT tiles.
- problems fixed: Suppresses noisy per-tile timing lines unless explicitly enabled via `TIMING_TILE_LOGS`, logs active training/tuning settings at run start, and applies `SOURCE_PREFETCH_GT_ONLY` GT-overlap gating for source prefetch/cache usage.

## [0.2.36]
- Description: Add a comprehensive repository `README.md` that documents purpose, runtime flow, tuning strategy, outputs, telemetry, and operating guidance.
- file touched: `README.md`, `CHANGELOG.md`
- reason: Provide a clear top-level entrypoint for onboarding, operation, and artifact interpretation without requiring readers to jump directly into architecture internals.
- problems fixed: Repository lacked a dedicated root README describing how to run the pipeline and how to interpret generated outputs and CSV telemetry.

## [0.2.35]
- Description: Add Bayesian stagnation early-stop callbacks with configurable patience/delta.
- file touched: `config.py`, `segedge/core/optuna_stop.py`, `segedge/pipeline/tuning_bayes.py`, `tests/test_optuna_stop.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Stop long Bayesian runs earlier when trials plateau and no meaningful objective gains are found.
- problems fixed: Adds `BO_EARLY_STOP_PATIENCE` (default `20`) and `BO_EARLY_STOP_MIN_DELTA`, wiring callbacks into stage1/stage2/stage3 Optuna optimize loops.

## [0.2.34]
- Description: Fix black formatting and add Python cache/build artifacts to .gitignore.
- file touched: `.gitignore`, `segedge/core/optuna_csv.py`, `CHANGELOG.md`
- reason: Ensure code formatting consistency and prevent accidental commit of Python cache and build artifacts.
- problems fixed: Applies black formatting to optuna_csv.py and excludes __pycache__, *.egg-info, and related build artifacts from version control.

## [0.2.33]
- Description: Add Bayesian Optuna CSV telemetry artifacts for trial time series and parameter importances.
- file touched: `config.py`, `segedge/core/optuna_csv.py`, `segedge/pipeline/tuning.py`, `tests/test_optuna_csv.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Enable direct visualization of optimization dynamics and parameter influence without scraping logs or JSON manually.
- problems fixed: Produces run-level `bayes_trials_timeseries.csv` (per-trial objective/params/attrs) and `bayes_hyperparam_importances.csv` (stage-wise importance table) alongside existing JSON outputs.

## [0.2.32]
- Description: Propagate Bayesian-tuned `neg_alpha` into inference/runtime settings and exported best-settings YAML.
- file touched: `segedge/pipeline/tuning.py`, `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Keep deployment/evaluation behavior consistent with the hyperparameter values selected during Bayesian optimization.
- problems fixed: Eliminates train/deploy mismatch where stage-1 tuned `neg_alpha` was dropped from the return bundle and holdout inference/export silently used `cfg.NEG_ALPHA`.

## [0.2.31]
- Description: Enforce doctest integrity by forbidding `callable(...)` shortcut doctests and replace existing callable-style doctests with non-cheat assertions.
- file touched: `AGENTS.md`, `scripts/check_doctest_ratio.py`, `main.py`, `scripts/check_file_length.py`, `segedge/core/*.py`, `segedge/pipeline/*.py`, `CHANGELOG.md`
- reason: Prevent fake doctest coverage and ensure coverage signals reflect meaningful behavior checks.
- problems fixed: Adds an automated guard against callable-based doctest cheating, updates contributor policy, and removes existing callable-pattern doctests from tracked Python modules.

## [0.2.30]
- Description: Refactor Bayesian tuning to range-first sampling with stage-2 broad/refine seeding and stage-3 frozen-upstream bridge optimization.
- file touched: `config.py`, `segedge/pipeline/tuning_bayes.py`, `segedge/pipeline/tuning.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Make long-budget Optuna runs more sample-efficient and avoid redundant recomputation in late-stage topology tuning.
- problems fixed: Adds strict `BO_*_RANGE` precedence over legacy lists, implements stage-2 refinement in a fresh seeded study, tunes `NEG_ALPHA` in stage1, and freezes upstream maps for faster stage-3 bridge trials.

## [0.2.29]
- Description: Add staged Bayesian tuning (default TPE with multivariate+group; optional CMA-ES), emit run-level hyperparameter importances JSON, and wire tuned bridge/silver-core params into inference/export.
- file touched: `config.py`, `segedge/pipeline/tuning_bayes.py`, `segedge/pipeline/tuning.py`, `segedge/pipeline/run.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Replace exhaustive high-cost tuning with a smarter staged optimizer that balances GT accuracy and SH consistency under light perturbations, while exposing parameter sensitivity per stage.
- problems fixed: Enables practical optimization of large search spaces (including bridge/skeletonization), adds persistent hyperparameter-importance reporting, uses tuned post-processing params at inference time, and keeps exported settings aligned with what was actually selected.

## [0.2.28]
- Description: Fix duplicate XAI JSON write per tile and align XAI write timing with actual single-write behavior.
- file touched: `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Remove redundant disk I/O in the explainability path and keep timing telemetry trustworthy for runtime opportunity-cost analysis.
- problems fixed: Prevents writing each XAI JSON twice and avoids underreporting `xai_write_s`/`xai_total_s` caused by an unmeasured second write.

## [0.2.27]
- Description: Add Tier-1 explainability outputs (XGB+kNN) with per-tile JSON/plots and capped holdout coverage.
- file touched: `config.py`, `segedge/core/explainability.py`, `segedge/core/__init__.py`, `segedge/pipeline/run.py`, `tests/test_explainability.py`, `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Make model behavior inspectable during runtime without adding heavy dependencies, while keeping holdout overhead bounded.
- problems fixed: Adds structured explainability artifacts, run-level XAI summaries, deterministic holdout XAI capping, and explicit XAI timings per tile.

## [0.2.26]
- Description: Expand architecture and knowledge base docs into code-aligned, thesis-ready technical references.
- file touched: `ARCHITECTURE.md`, `KB.md`, `CHANGELOG.md`
- reason: Provide complete, implementation-faithful documentation of runtime flow, split semantics, telemetry contracts, and reporting guidance.
- problems fixed: Replaces outdated/incomplete docs with a structured architecture map and a reproducibility-focused knowledge base that matches current pipeline behavior.

## [0.2.25]
- Description: Retune top-p/CRF/shadow search ranges in config for less top-p cap saturation and safer post-processing defaults.
- file touched: `config.py`, `CHANGELOG.md`
- reason: Align default tuning space with recent results where XGB raw outperformed CRF-heavy settings.
- problems fixed: Reduces likelihood of CRF over-pruning and avoids overly aggressive adaptive top-p hitting max across tiles.

## [0.2.24]
- Description: Add incremental per-tile timing CSV telemetry and an opportunity-cost runtime summary CSV.
- file touched: `config.py`, `segedge/core/timing_csv.py`, `segedge/core/__init__.py`, `segedge/pipeline/run.py`, `tests/test_timing_csv.py`, `ARCHITECTURE.md`, `CHANGELOG.md`
- reason: Persist tile-level runtime phase data as the run progresses and quantify each phase's runtime opportunity cost.
- problems fixed: Avoids relying on large text logs for timing analysis and provides incremental, structured timing artifacts for long runs.

## [0.2.23]
- Description: Add auto-split mode for GT-to-validation workflow with deterministic holdout tile cap.
- file touched: `config.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `tests/test_auto_split_modes.py`, `ARCHITECTURE.md`, `CHANGELOG.md`
- reason: Support workflows that validate on all GT-overlap tiles while limiting holdout inference runtime.
- problems fixed: Prevents unwanted GT source/val splitting for this workflow and bounds inference tile count with reproducible sampling.

## [0.2.22]
- Description: Add tuning-phase preview plots for up to 10 validation tiles.
- file touched: `config.py`, `segedge/pipeline/tuning.py`, `CHANGELOG.md`
- reason: Provide visual feedback during validation tuning without waiting for holdout inference.
- problems fixed: Makes tuning progress visible while keeping plot volume bounded.

## [0.2.21]
- Description: Promote internal debug logs to info and set default log level to INFO.
- file touched: `config.py`, `segedge/core/io_utils.py`, `segedge/core/knn.py`, `segedge/pipeline/common.py`, `segedge/pipeline/inference_utils.py`, `CHANGELOG.md`
- reason: Keep our diagnostics visible without enabling third-party debug spam.
- problems fixed: Reduces noisy dependency debug logs while retaining detailed pipeline tracing.

## [0.2.20]
- Description: Clip and simplify roads geometries before rasterization to reduce mask build time.
- file touched: `config.py`, `segedge/pipeline/inference_utils.py`, `CHANGELOG.md`
- reason: Avoid long stalls when rasterizing large road geometries per tile.
- problems fixed: Cuts roads mask rasterization time during validation and holdout inference.

## [0.2.19]
- Description: Release changelog entries into versioned sections, enforce per-request versioning, and tighten file-length limits.
- file touched: `CHANGELOG.md`, `AGENTS.md`, `scripts/check_file_length.py`, `.pre-commit-config.yaml`, `segedge/pipeline/run.py`, `segedge/pipeline/inference_utils.py`, `segedge/pipeline/tuning.py`
- reason: Keep releases atomic per request and prevent oversized files.
- problems fixed: Ensures every satisfied request ships as a version and keeps long pipeline files under the new limit.

## [0.2.18]
- Description: Add per-run summary YAML with validation metrics, deltas, and timing.
- file touched: `segedge/core/io_utils.py`, `segedge/core/summary_utils.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Persist validation performance and step timing stats for each run.
- problems fixed: Captures metric deltas and runtime breakdowns without manual log scraping.

## [0.2.17]
- Description: Add always-on reprojection and coverage debug logging for label rasters.
- file touched: `config.py`, `segedge/core/io_utils.py`, `segedge/pipeline/run.py`, `segedge/pipeline/common.py`, `CHANGELOG.md`
- reason: Diagnose holdout label projection cutoffs with full CRS/bounds/coverage details.
- problems fixed: Adds traceability to confirm whether label coverage or reprojection alignment causes left-edge cuts.

## [0.2.16]
- Description: Add adaptive top-p selection with silver_core intersection outputs.
- file touched: `config.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Enable unlabeled target inference with adaptive selection and high-precision core masks.
- problems fixed: Avoids fixed-threshold drift and provides conservative silver supervision.

## [0.2.15]
- Description: Switch roads masking to per-tile spatial index rasterization.
- file touched: `config.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Avoid OOM from global raster builds while keeping tile-resolution masking.
- problems fixed: Prevents roads raster build OOM and keeps per-tile masking fast.

## [0.2.14]
- Description: Add source label, DINO similarity, and score heatmaps to unified plot.
- file touched: `segedge/core/plotting.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Improve plot diagnostics with label context and model score maps.
- problems fixed: Makes score variations and label alignment visible per tile.

## [0.2.13]
- Description: Write a single inference_best_setting.yml after validation and stop per-tile settings.
- file touched: `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Freeze holdout settings once and include weighted validation metrics.
- problems fixed: Removes redundant per-tile settings and captures run-level metrics.

## [0.2.12]
- Description: Point roads mask path to data/roads.
- file touched: `config.py`, `CHANGELOG.md`
- reason: Align roads penalty with new data layout.
- problems fixed: Prevents missing roads mask after folder move.

## [0.2.11]
- Description: Add tunable roads mask penalty for kNN/XGB score maps.
- file touched: `config.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Downweight predictions on roads with a validated multiplier.
- problems fixed: Allows road suppression without hard masking.

## [0.2.10]
- Description: Switch tuning to weighted-mean IoU and add red phase markers.
- file touched: `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Ensure configs are chosen by GT-weighted validation performance and make phases easy to spot.
- problems fixed: Avoids selecting configs by non-weighted aggregation and improves log readability.

## [0.2.9]
- Description: Add continuity bridging, unified plots, and phase summaries for kNN/XGB/Champion.
- file touched: `config.py`, `segedge/core/continuity.py`, `segedge/core/crf_utils.py`, `segedge/core/plotting.py`, `segedge/pipeline/run.py`, `KB.md`, `CHANGELOG.md`
- reason: Improve topological continuity and make phase-by-phase gains visible.
- problems fixed: Reduces broken segments and consolidates diagnostic plots.

## [0.2.8]
- Description: Document rolling unions, resume, and auto split updates in KB.
- file touched: `KB.md`, `CHANGELOG.md`
  - reason: Keep docs aligned with current pipeline behavior and outputs.
- problems fixed: Prevents outdated output paths and config guidance.

## [0.2.7]
- Description: Add holdout-only rolling unions for kNN/XGB/Champion streams with resume logging.
- file touched: `config.py`, `segedge/core/io_utils.py`, `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Track per-stream unions during holdout inference and resume safely after interruptions.
- problems fixed: Enables incremental union outputs and skip/restart logic based on processed tiles.

## [0.2.6]
- Description: Write rolling union shapefile updates with periodic backups.
- file touched: `config.py`, `segedge/core/io_utils.py`, `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Keep union shapes updated during inference and protect against partial runs.
- problems fixed: Avoids waiting for the final merge step to see union outputs.

## [0.2.5]
- Description: Filter auto-split tiles to those overlapping SOURCE_LABEL_RASTER.
- file touched: `segedge/pipeline/common.py`, `CHANGELOG.md`
- reason: Skip tiles outside label raster coverage during auto split.
- problems fixed: Prevents unnecessary GT scans for tiles without label coverage.

## [0.2.4]
- Description: Added batched feature extraction for single-scale tiles.
- file touched: `config.py`, `segedge/core/features.py`, `CHANGELOG.md`
- reason: Improve GPU utilization by processing multiple tiles per forward pass.
- problems fixed: Reduces overhead from per-tile model calls during prefetch.

## [0.2.3]
- Description: Use cached vector geometry intersection for GT presence checks.
- file touched: `segedge/pipeline/common.py`, `CHANGELOG.md`
- reason: Avoid per-tile rasterization during auto split.
- problems fixed: Eliminates repeated GT reprojection logs and reduces tile scan overhead.

## [0.2.2]
- Description: Parallelized GT presence checks for auto tile splitting.
- file touched: `config.py`, `segedge/pipeline/common.py`, `segedge/pipeline/run.py`, `CHANGELOG.md`
- reason: Speed up auto split when scanning large tile directories.
- problems fixed: Reduces wall time for GT overlap detection with many tiles.

## [0.2.1]
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
