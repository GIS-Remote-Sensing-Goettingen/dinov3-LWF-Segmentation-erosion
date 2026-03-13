# ARCHITECTURE

## Goal
Document the current SegEdge runtime structure after the feature/runtime/workflow split.

## Repository Layout
- `main.py`: thin CLI wrapper; imports and calls `segedge.pipeline.run.main()`.
- `config.yml`: typed runtime configuration source.
- `deployment/`: cluster-facing orchestration helpers such as shard-file generation, shard-union merge, and the Slurm shard launcher.
- `segedge/core/`: model-facing logic, feature construction, I/O, plotting, metrics, and config loading.
- `segedge/core/feature_ops/`: feature extraction, tiling, cache I/O, and hybrid feature fusion.
- `segedge/core/features.py`: compatibility export layer for `feature_ops`.
- `segedge/pipeline/run.py`: bootstrap and dispatcher.
- `segedge/pipeline/workflows/`: mode-specific orchestration.
- `segedge/pipeline/runtime/`: heavy runtime helpers split by concern.
- `segedge/pipeline/runtime_utils.py`: compatibility export layer for `runtime`.
- `segedge/pipeline/artifacts.py`: model-bundle save/load and compatibility checks.
- `scripts/`: repo health checks used by pre-commit.
- `tests/`: doctests, smoke tests, and targeted regressions.
- `docs/`: human-maintained documentation.

## Runtime Layers
### Entrypoint
- `main.py` exists only to keep the CLI stable.
- it now also accepts `--config <path>` so generated shard workers can run against explicit config copies instead of the repo root `config.yml`
- `segedge.pipeline.run.main()` owns run-directory creation, logging setup, resume state loading, time-budget initialization, tile resolution, feature-cache mode selection, and workflow dispatch.
  - training and final inference now resolve feature-cache persistence separately, so one-shot inference can run without building disk cache while training/tuning still keeps reusable feature artifacts.

### Workflows
- `workflows/inference_only.py`:
  - loads a saved bundle, either from the explicit config path or the newest valid previous run bundle
  - validates runtime compatibility
  - writes inference settings
  - runs holdout inference with rolling checkpoints
  - can now take an exact one-tile-per-line shard file via `io.inference.tiles_file`, which bypasses folder globbing and source-label refiltering inside the worker run
- `workflows/manual_training.py`:
  - builds artifacts from explicit source tiles
  - tunes on explicit validation tiles
  - writes best settings and optional bundle output
  - runs holdout inference on the configured inference set
- `workflows/loo_training.py`:
  - builds fold-specific artifacts for GT-positive auto-split tiles
  - tunes and validates each fold
  - tracks best-so-far fold state for time-budget cutover
  - retrains on all GT tiles unless cutover skips that stage
  - writes final settings, optional bundle output, and holdout inference outputs
- `workflows/shared.py` holds cross-workflow helpers for weighted metrics, holdout wrapping, best-settings export, and feature consolidation.

### Runtime Helpers
- `runtime/holdout_inference.py`: per-tile inference loop for validation and holdout tiles.
  - when inference-side disk cache is disabled and the active stream is XGB-only, holdout inference uses a streaming extract/fuse/predict path instead of prefetching the whole image's features before scoring.
- `runtime/checkpointing.py`: writes `rolling_best_setting.yml`.
- `runtime/time_budget.py`: computes deadlines, remaining time, and serialized budget state.
- `runtime/roads.py`: cached road-mask rasterization and roads-penalty scoring.
  - roads-mask timing is now decomposed into cache lookup/load, STRtree query, candidate filtering, rasterization, resize, and cache-write spans so slow tiles can be attributed to specific roads-mask work.
- `runtime/postprocess.py`: shadow filtering and novel-proposal heuristics.
- `runtime/crf_eval.py`: CRF worker initialization and config evaluation.
  - XGB CRF can now use a trimap-band unary that treats the current XGB mask as strong interior foreground, a boundary ring as uncertain, and the far exterior as background.
  - Optional `postprocess.fill_holes_xgb` fills enclosed holes in the thresholded XGB mask before trimap CRF builds that boundary ring.
- `runtime/tile_context.py`: tile image/label/GT loading and SH-buffer preparation.
  - source-label loading now reuses cached raster handles and prefers aligned window reads over full temporary-dataset reprojection when the tile grid matches the label grid.
  - tile-context spans now emit source-label, GT, and SH-buffer coverage metadata so `load_context` outliers can be compared against actual workload size.
- `runtime/phase_metrics.py`: phase logging and summary aggregation.
  - inference now also writes structured timing spans into `performance.jsonl` so tile-level and function-internal bottlenecks can be analyzed separately from `run.log`.

### Feature Helpers
- `feature_ops/extraction.py`: DINO feature extraction and batched tile prefetch.
  - for XGB-only inference, cached feature tiles can stay lazy until the scorer actually needs each tile.
  - cache-hit validation can reuse a per-image manifest instead of reopening every tile sidecar.
  - batch feature normalization is vectorized so one-shot inference avoids per-item normalization loops after the DINO forward pass.
- `feature_ops/tiling.py`: tile iteration, patch-size cropping, and patch-label mapping.
- `feature_ops/fusion.py`: hybrid DINO + image-descriptor feature assembly and XGB feature-stat transforms.
- `feature_ops/cache.py`: per-tile feature cache persistence, validation, and per-image manifests.
- `feature_ops/spec.py`: feature layout hashing for cache compatibility.

## End-to-End Flow
1. `main.py` calls `segedge.pipeline.run.main()`.
2. `run.py` creates or resumes a run directory under `output/run_*`.
3. The bootstrap stage configures logging, loads processed holdout tiles, restores time-budget state, and opens rolling union shapefiles.
4. The dispatcher resolves one of three execution modes:
   - `io.training=false`: inference-only
   - `io.training=true` and `io.auto_split.enabled=false`: manual training
   - `io.training=true` and `io.auto_split.enabled=true`: LOO training
5. The selected workflow builds or loads model state, writes settings/checkpoint metadata, and invokes holdout inference when tiles are available.
6. Holdout inference updates union shapefiles and processed-tile logs tile by tile, so partial runs keep usable outputs.
7. Disk feature caches are consolidated after successful workflows that need them.
8. Large folder inference can be parallelized by building shard tile files once, launching isolated shard jobs, retrying incomplete shards against the same fixed run directories, and then merging the resulting per-shard union families after verification succeeds.

## Outputs
- `output/run_*/run.log`: main runtime log.
- `output/run_*/performance.jsonl`: structured per-span performance log with tile and phase context plus rolling summaries.
  - feature-prefetch spans now include cache hit counts plus approximate feature/manifest bytes read and written, so disk-cache cost is visible in profiling output.
  - holdout context loading now records child spans for `load_holdout_tile_context`, `resolve_runtime_toggles`, `load_roads_mask`, and `finalize_context`, plus workload metadata such as source-label, roads, and SH-buffer coverage.
- `output/run_*/rolling_best_setting.yml`: interruption-safe best-known config and progress state.
- `output/run_*/processed_tiles.jsonl`: append-only holdout completion log.
- `output/run_*/plots/validation/`: validation-stage plots.
- `output/run_*/plots/inference/`: holdout/inference plots.
- `output/run_*/shapes/unions/.../union.shp`: rolling union shapefiles for `raw`, `crf`, `shadow`, and `shadow_with_proposals`.
- `output/run_*/inference_best_setting.yml` and `output/run_*/best_setting.yml`: exported run settings.
- `output/run_*/model_bundle/`: optional inference bundle when bundle saving is enabled.
- `output/shards/<job_name>/`: optional shard manifests, per-shard configs, rendered Slurm scripts, retry/verification status files, and merged unions created by the Slurm orchestration flow.

## Major Runtime Functions
- `segedge.pipeline.run.main`: bootstrap, resolve runtime mode, dispatch workflow, and finalize cache consolidation.
- `segedge.pipeline.workflows.run_inference_only`: load bundle and run inference-only holdout processing.
- `segedge.pipeline.workflows.run_manual_training`: build one training set, tune once, summarize validation, then infer holdout tiles.
- `segedge.pipeline.workflows.run_loo_training`: execute per-fold tuning, best-fold selection, optional final retraining, and holdout inference.
- `segedge.pipeline.workflows.shared.run_holdout_with_checkpoint`: standard wrapper that ensures holdout inference uses phase markers and rolling checkpoint writes.
- `segedge.pipeline.runtime.holdout_inference.infer_on_holdout`: load one tile, score it, refine it, export overlays/proposals, and return masks plus metrics.
  - accepted proposals are folded into the `shadow_with_proposals` union instead of being exported as per-tile shapefiles/CSVs.
- `segedge.pipeline.runtime.checkpointing.write_rolling_best_config`: serialize the best-known stage/config/progress snapshot for resume and inspection.

## Design Constraints
- Keep the public CLI and config model stable.
- Keep compatibility exports in `segedge/core/features.py` and `segedge/pipeline/runtime_utils.py` until downstream imports are fully migrated.
- Keep per-tile holdout checkpointing behavior intact: append unions, append processed log, then write the rolling checkpoint.
- Keep structured performance logging lightweight and append-only so resumed runs still produce one continuous `performance.jsonl`.
- Prefer smaller modules with one responsibility over monolithic orchestration files.
