# Knowledge Base

## What this repository is
SegEdge is a zero-shot segmentation pipeline for linear woody features. It uses DINOv3 patch features, optional XGBoost scoring, CRF refinement, and shadow filtering to produce holdout masks, plots, and rolling shapefile outputs.

For architecture details, read `docs/ARCHITECTURE.md`.
For workflow and function-level behavior, read `docs/Implementation.md`.

## Current Code Map
- `main.py`: CLI wrapper
- `segedge/pipeline/run.py`: bootstrap and workflow dispatch
- `segedge/pipeline/workflows/`: inference-only, manual, and LOO workflows
- `segedge/pipeline/runtime/`: per-concern runtime helpers
- `segedge/core/feature_ops/`: feature extraction, fusion, tiling, and cache helpers
- `segedge/pipeline/artifacts.py`: model bundle persistence
- `config.yml`: runtime configuration

## Run Modes
### Inference-only
- Set `io.training=false`.
- Uses `io.inference.model_bundle_dir` when set.
- If `io.inference.model_bundle_dir` is `null`, the pipeline resolves the newest valid previous `output/run_*/model_bundle`.
- Resolves inference tiles from `io.inference.*` or legacy inference config.
- Skips training and goes directly to holdout inference.

### Manual training
- Set `io.training=true`.
- Set `io.auto_split.enabled=false`.
- Provide explicit source, validation, and holdout tile lists.
- Builds training artifacts once, tunes once, then runs holdout inference.

### LOO training
- Set `io.training=true`.
- Set `io.auto_split.enabled=true`.
- Auto-discovers GT-positive tiles for fold construction.
- Tunes fold by fold, selects the best fold, optionally retrains on all GT tiles, then runs holdout inference.

## Outputs to inspect
- `output/run_*/run.log`
- `output/run_*/rolling_best_setting.yml`
- `output/run_*/processed_tiles.jsonl`
- `output/run_*/plots/validation/`
- `output/run_*/plots/inference/`
- `output/run_*/shapes/unions/{knn|xgb|champion}/{raw|crf|shadow}/union.shp`
- `output/run_*/inference_best_setting.yml`
- `output/run_*/best_setting.yml`
- `output/run_*/model_bundle/` when bundle saving is enabled

## Important runtime behavior
- Inference tile filtering respects `io.paths.source_label_raster` and keeps only tiles that contain at least one positive source-label pixel.
- `io.inference.score_prior` can manually scale XGB scores separately inside and outside `SOURCE_LABEL_RASTER` pixels during the final inference phase.
- `io.inference.plots` can disable individual inference plot files without changing masks, checkpoints, or `plot_every` cadence.
- The unified inference plot uses plot-only XGB raw/CRF preview masks so it can show activity outside the SH/source-label buffer without changing runtime masks, metrics, or shapefile outputs.
- The unified inference plot now merges accepted and rejected proposals into one overlay panel: accepted regions are light blue and rejected regions are light red.
- The unified inference plot labels the source-label panel as `Administrative buffered labels` and no longer includes separate RGB or GT panels.
- Each `output/run_*/` directory now contains a copy of the active `config.yml`.
- `search.crf.trimap_band_pixels_values` controls how far XGB CRF is allowed to expand/shrink the coarse XGB mask boundary when filling holes against RGB edges.
- `postprocess.fill_holes_xgb` fills enclosed holes in the thresholded XGB mask before XGB trimap CRF builds its boundary band.
- Outside-buffer novel proposals can now grant extra allowed width to strongly elongated shapes through `postprocess.novel_proposals.width_bonus_per_pca`, while `hard_width_cap_m` still blocks very wide blobs.
- `io.inference.plot_every` samples holdout plot rendering over pending tiles without changing inference masks or checkpoint cadence.
- Inference PNG exports use a higher DPI than before, so the saved plots are less pixelated.
- Holdout inference is interruption-safe at tile granularity.
- The optimized XGB scorer checks the first 3 pending holdout tiles against the legacy scorer and auto-falls back if the difference is meaningful.
- CRF tuning is guarded against unsafe CUDA multiprocessing.
- Time-budget state is persisted in the rolling checkpoint and can trigger cutover behavior.
- `runtime.cache_training_features` and `runtime.cache_inference_features` control disk persistence separately for training/validation vs final inference.
- `performance.jsonl` now records cache-cost metadata for feature prefetch, including cached vs computed tile counts and approximate feature/manifest bytes read and written.

## Major Functions to know
- `segedge.pipeline.run.main`: dispatcher and bootstrap.
- `segedge.pipeline.workflows.run_inference_only`: bundle load plus holdout inference.
- `segedge.pipeline.workflows.run_manual_training`: explicit train/validate/holdout flow.
- `segedge.pipeline.workflows.run_loo_training`: fold tuning and optional final retraining.
- `segedge.pipeline.runtime.holdout_inference.infer_on_holdout`: one-tile inference path.
- `segedge.pipeline.runtime.checkpointing.write_rolling_best_config`: rolling checkpoint writer.

## Troubleshooting
- `BrokenProcessPool` during CRF tuning:
  - reduce workers to 1
  - verify the CUDA-safe fallback path is active
- Empty holdout inference:
  - check `SOURCE_LABEL_RASTER` label-presence filtering
  - check whether `io.inference.score_prior` is enabled
  - verify inference directory/list inputs
- Very low IoU:
  - verify SH-buffer alignment and CRS
  - confirm GT clipping behavior
- Stale features:
  - delete the affected files under `FEATURE_DIR`
  - rerun to rebuild them with the current feature-spec hash

## Tooling
- `pre-commit run --all-files`: formatter, lint, doctest-ratio, file-length, and function-length checks
- `pytest --doctest-modules`: doctests plus pytest suite
- `scripts/check_file_length.py`: file-size guard
- `scripts/check_function_length.py`: function-size guard that ignores leading docstrings and doctests by excluding the full docstring block from the count
- `python scripts/analyze_performance_log.py performance.jsonl --top 10 --tile-limit 5`: summarize inference-only stage/substage timings and the hottest traced tiles from a structured performance log; mixed train+infer logs default to `phase=holdout_inference` and exclude `tile=null` rows
- `python scripts/analyze_performance_log.py performance.jsonl --phase all --include-tile-null --top 10`: inspect the full mixed log, including training/setup spans and tile-null records
