# Implementation Guide

## Purpose
This document explains how the current SegEdge pipeline actually runs, with emphasis on the three workflow modes and the major orchestration functions.

## Runtime Entry
`main.py` is intentionally small. The real entrypoint is `segedge.pipeline.run.main`, which does four things before any model work starts:

1. Create or resume the run directory.
2. Configure logging, processed-tile resume state, and rolling union GeoTIFF rasters.
3. Initialize the time-budget state and phase-specific feature-cache settings.
4. Resolve the tile set and dispatch the correct workflow.

The dispatcher does not train or infer by itself. It builds a shared `common` runtime payload and passes it to one workflow module.

## Workflow Modes
### Inference-only workflow
Function: `segedge.pipeline.workflows.run_inference_only`

Use this when `io.training=false`.

Execution order:
1. Resolve the model bundle directory:
   - use `io.inference.model_bundle_dir` when set
   - otherwise use the newest valid previous `output/run_*/model_bundle`
2. Validate that the bundle matches the current runtime assumptions:
   patch size, resample factor, tiling config, and feature-context radius.
3. Write `rolling_best_setting.yml` with the bundle metadata.
4. Export `inference_best_setting.yml` and the legacy `best_setting.yml`.
5. Copy the current `config.yml` into the run directory.
5. Run holdout inference on the resolved inference tile set.
6. Summarize phase metrics and consolidate disk features if inference-side caching is enabled.

Important behavior:
- No training artifacts are built in this mode.
- `main.py --config <path>` can now run the pipeline against a generated batch-specific config copy instead of the repo root `config.yml`.
- `io.inference.tiles_file` can provide an exact one-tile-per-line generated batch list; when it is set, the worker run uses that list directly instead of globbing a folder or re-running source-label filtering.
- If inference tile resolution returns an empty set after filtering out tiles with no positive `SOURCE_LABEL_RASTER` pixels inside them, holdout inference is skipped cleanly.
- The holdout step still updates rolling unions and processed-tile logs tile by tile.
- Holdout unions now track only the final stage outputs: `raw`, `crf`, `shadow`, and `shadow_with_proposals`.
- The run also writes `performance.jsonl`, which records structured spans for tile loading, cache validation, cache read/write cost, XGB scoring internals, CRF, proposal filtering, plots, and union updates.
- Holdout context loading is now broken down further in `performance.jsonl`: the parent `load_context` span contains child spans for `load_holdout_tile_context`, `resolve_runtime_toggles`, `load_roads_mask`, and `finalize_context`, so roads-mask stalls are separable from source-label or GT loading.
- When `runtime.cache_inference_features=false` and the active inference stream is XGB-only, holdout inference now skips the old full-image feature prefetch step and instead streams extraction batches directly into XGB fusion/prediction/accumulation. This keeps one-shot inference from paying the full-image `prefetch_features` cost before scoring starts.
- Source-label reprojection is optimized in the shared I/O layer: repeated tiles reuse the same source-label raster handle, aligned same-CRS grids prefer direct window reads, and the performance log now splits source-label work into open/grid/reproject/finalize substages.
- XGB CRF refinement can use a trimap-band unary: the current XGB mask is treated as strong interior foreground, a dilated ring is treated as uncertain, and CRF uses RGB edges to fill holes and expand/shrink that boundary band. The single tuning knob for this is `search.crf.trimap_band_pixels_values`.
- `postprocess.fill_holes_xgb` can fill enclosed holes in the thresholded XGB raw mask before trimap CRF, so CRF expands from the filled coarse mask instead of the original holey threshold mask.
- `io.inference.plot_every` can sample inference plots over pending tiles without changing mask generation, processed-tile logging, or union raster updates.
- In the unified inference plot, XGB raw and XGB CRF panels can now use plot-only preview masks that are not clipped to the SH/source-label buffer, so the preview shows what the score stream is doing outside the label area without changing saved masks or metrics.

### Manual training workflow
Function: `segedge.pipeline.workflows.run_manual_training`

Use this when `io.training=true` and `io.auto_split.enabled=false`.

Execution order:
1. Build training artifacts from `io.paths.source_tiles` or `io.paths.source_tile`.
2. Tune kNN/XGB/CRF/shadow configuration on `io.paths.val_tiles`.
3. Record weighted validation metrics.
4. Optionally save a model bundle.
5. Export inference settings.
6. Run holdout inference on the configured inference tile set.
7. Summarize validation and holdout metrics.
8. Consolidate disk feature caches.

Important behavior:
- Training artifacts are built once.
- Validation tiles are used only for tuning and metric reporting.
- Holdout inference uses the tuned configuration but does not re-fit the model.

### LOO workflow
Function: `segedge.pipeline.workflows.run_loo_training`

Use this when `io.training=true` and `io.auto_split.enabled=true`.

Execution order:
1. Resolve GT-positive tiles and inference-only tiles from the auto-split scan.
2. Build deterministic LOO-style folds over the GT-positive set.
3. For each fold:
   - build fold-specific training artifacts
   - tune on the fold validation tiles
   - run validation inference on those tiles
   - record per-fold phase metrics and best-so-far state
4. Select the best completed fold by `champion_shadow` IoU.
5. If the time budget still allows it, retrain on all GT-positive tiles using the selected fold configuration.
6. Otherwise apply the configured cutover policy:
   - `immediate_inference`: reuse best completed fold artifacts
   - `stop`: write state and exit before holdout inference
7. Optionally save a model bundle.
8. Export inference settings with fold summaries.
9. Run holdout inference unless cutover mode stops before that stage.
10. Consolidate disk feature caches.

Important behavior:
- The workflow preserves the best completed fold in `rolling_best_setting.yml`.
- Low-GT folds can be skipped according to `training.loo.low_gt_policy`.
- The cutover state is carried through the mutable budget payload so the exported settings and rolling checkpoints match what actually happened.

## Shared Holdout Flow
The common holdout path is wrapped by `run_holdout_with_checkpoint`.

Responsibilities:
- gate empty holdout sets
- emit `PHASE START/END` markers for holdout inference
- call the tile-by-tile inference loop
- keep checkpoint wiring consistent across workflows

The tile loop itself lives in `segedge.pipeline.inference_flow.run_holdout_inference`.
Its job is orchestration at the holdout-set level:
- skip already processed tiles on resume
- log per-tile progress as `Processing tile <path>, <current> / <total>`
- call `infer_on_holdout` for each tile
- append new masks into rolling union rasters
- append one processed record to `processed_tiles.jsonl`
- write the rolling checkpoint after each completed tile
- emit rolling summaries into `performance.jsonl` every 10 completed inference tiles

That ordering is deliberate: if the job stops after a tile finishes, the union raster and progress log already reflect that completed tile.
Accepted proposals are no longer exported as per-image shapefiles or CSVs; instead, they are folded into the rolling `shadow_with_proposals` union while `shadow` remains the final mask without proposal additions.
For large folder inference, the intended parallel pattern is:
1. launch `deployment/launch_batched_inference.py`
2. let it build batch tile files and batch-specific configs once
3. let it submit one ordinary Slurm worker job per batch
4. let its controller resubmit only incomplete batches against the same fixed batch run directories
5. let its final controller step merge the 4 stage unions only after all batches are complete

This avoids multiple jobs racing on the same run directory, `processed_tiles.jsonl`, and rolling union rasters, while still allowing crashed/incomplete batches to resume safely.

When `io.inference.score_prior.enabled=true`, the final holdout/inference phase can also apply manual XGB score multipliers separately inside and outside `SOURCE_LABEL_RASTER` pixels. This prior is not used during validation inference or tuning.
`io.inference.plots` can disable individual inference plot types while leaving `plot_every` as the outer cadence control.
The unified inference plot now renders accepted and rejected proposals in one combined subplot, with accepted regions shown as a light-blue transparent overlay and rejected regions shown as a light-red transparent overlay. It also labels the source-label panel as `Administrative buffered labels`, omits the standalone RGB and GT panels, and uses a higher DPI so the saved PNGs are less pixelated.
Outside-buffer novel proposals can also relax their width limit when `pca_ratio` is well above `min_pca_ratio`: `width_bonus_per_pca` increases the allowed width for highly elongated components, while `hard_width_cap_m` still enforces an absolute maximum width.
When the optimized XGB scorer is active, the first 3 pending holdout tiles are also compared against the legacy scorer. If the optimized and legacy score maps differ meaningfully, the run logs the mismatch and automatically falls back to the legacy scorer for the rest of that holdout phase.

## Major Functions
### `segedge.pipeline.run.main`
- Inputs: global `cfg`
- Produces: initialized run directory, workflow dispatch, optional cache consolidation
- Key decisions:
  - resume vs new run
  - inference-only vs manual vs LOO
  - independent training vs inference feature-cache persistence
  - time-budget initialization policy

### `segedge.pipeline.workflows.run_inference_only`
- Inputs: shared runtime payload, holdout tiles, bundle directory
- Produces: best-settings export, holdout outputs, phase summaries
- Main value: keeps inference-only behavior isolated from training logic

### `segedge.pipeline.workflows.run_manual_training`
- Inputs: shared runtime payload, source tiles, validation tiles, holdout tiles
- Produces: tuned settings from one explicit train/validate split
- Main value: preserves the manual workflow without carrying the LOO-specific fold machinery

### `segedge.pipeline.workflows.run_loo_training`
- Inputs: shared runtime payload, GT-positive tiles, holdout tiles, LOO and budget settings
- Produces: best-fold selection, optional final retraining, exported fold summaries
- Main value: owns the most complex control flow while keeping `run.py` small

### `segedge.pipeline.runtime.holdout_inference.infer_on_holdout`
- Inputs: one tile path plus tuned model state and runtime helpers
- Produces:
  - raw, CRF, and shadow masks
  - metrics
  - proposal exports and plots
  - per-tile metadata used by the outer holdout loop
- Main value: all expensive per-tile inference happens in one place instead of being duplicated across workflows
- Internal profiling now breaks this function down into:
  - tile context load
  - holdout-context child stages: tile load, runtime toggle resolution, roads-mask load, and context finalization
  - kNN/XGB streams
  - CRF stage
  - shadow stage
  - novel proposals
  - proposal export
  - plot export

### `segedge.pipeline.runtime.checkpointing.write_rolling_best_config`
- Inputs: stage name, tuned config, progress counts, optional fold and budget state
- Produces: `rolling_best_setting.yml`
- Main value: interruption-safe state snapshots that are consistent across workflows

### `segedge.pipeline.workflows.shared.consolidate_cached_features`
- Inputs: feature directory, image ids to consolidate, per-phase consolidation flags
- Produces: merged per-image feature arrays when disk cache mode is active for the relevant phase
- Main value: keeps feature cleanup out of the workflow bodies

## Feature and Runtime Packages
### `segedge/core/feature_ops`
- `extraction.py`: DINO feature extraction and tile prefetch
  - cached XGB-only inference tiles can now stay lazy and hand their `.npy` path to the scorer instead of loading the array up front
  - prefetch logging now records cache hits, recomputed tiles, and approximate feature/manifest bytes read and written
- `tiling.py`: tile iteration and patch-grid alignment
- `fusion.py`: hybrid feature assembly and XGB stat transforms
- `cache.py`: on-disk feature cache format
- `spec.py`: feature-layout hash used for cache validity

### `segedge/pipeline/runtime`
- `time_budget.py`: deadline math and serialized budget payloads
- `roads.py`: cached road-mask rasterization and roads penalty application
- `crf_eval.py`: CRF worker initialization and evaluation
- `postprocess.py`: shadow filtering and proposal heuristics
  - proposal filtering now works on connected-component bounding boxes and short-circuits expensive morphology for obvious failures
- `tile_context.py`: per-tile image, label, GT, and SH-buffer loading
  - GT vector geometries are cached per source path and target CRS before rasterization
- `holdout_inference.py`: per-tile inference logic
- `checkpointing.py`: rolling checkpoint persistence
- `phase_metrics.py`: weighted summaries and phase log markers

## Outputs by Stage
- Bootstrap:
  - `run.log`
  - `processed_tiles.jsonl` when resuming
  - union raster directories
- Validation:
  - `plots/validation/`
  - validation metric summaries in logs
- Holdout inference:
  - `plots/inference/`
  - rolling union GeoTIFF rasters under `shapes/unions/`
  - `processed_tiles.jsonl`
  - rolling checkpoints after every completed tile
- Final exports:
  - `inference_best_setting.yml`
  - `best_setting.yml`
  - optional `model_bundle/`

## Maintenance Notes
- If a workflow grows large again, move logic down one layer first:
  workflow -> runtime helper -> core helper.
- Keep compatibility exports thin; new behavior should live in the package modules, not in the wrappers.
- Any change that affects holdout completion ordering must preserve:
  append union -> append processed log -> write checkpoint.
