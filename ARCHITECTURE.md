# ARCHITECTURE

## Goal
Document the SegEdge zero-shot segmentation pipeline structure and entrypoints.

## Folder Structure
- `config.yml`: Primary pipeline configuration (commented, reader-first).
- `segedge/core/config_loader.py`: Typed YAML loader used by runtime modules.
- `segedge/core/`: Core modules (features, banks, kNN, XGB, CRF, I/O, metrics).
- `segedge/pipeline/`: Orchestration entrypoints and shared helpers.
- `segedge/pipeline/runtime_utils.py`: Runtime-heavy helpers split out of `run.py` (roads masks, proposal filtering, rolling checkpoint utilities, tile context loading).
- `segedge/pipeline/artifacts.py`: Train-time model-bundle persistence and inference-time bundle loading/compat checks.
- `main.py`: CLI wrapper for the full pipeline.
- `tests/`: Smoke and end-to-end tests.
- `scripts/`: Repo health checks (doctest ratio, file length).

## Design Principles
- **Modularity:** Core logic lives in `segedge/core/`, orchestration in `segedge/pipeline/`.
- **Docstrings + doctests:** Public helpers include docstrings and doctests.

## Workflow
1. Configure paths and hyperparameters in `config.yml`.
   `io.training=true` runs train+tune+inference.
   `io.training=false` skips training, loads `io.inference.model_bundle_dir`, and runs inference only.
2. Tile selection supports two modes:
   `io.auto_split.enabled=true`: tiles are discovered from `io.auto_split.tiles_dir`.
   GT-overlap tiles are first identified by vector intersection, then filtered to keep
   only tiles with effective GT positives after optional SH-buffer clipping. These tiles
   are used for leave-one-out (LOO) folds (`training.loo`), and remaining tiles are
   treated as inference-only holdout tiles. Fold validation can use multi-tile windows
   (`training.loo.val_tiles_per_fold`) and skip low-signal folds by GT-positive
   threshold (`training.loo.low_gt_policy`).
   `io.auto_split.enabled=false`: explicit manual tile lists are used from
   `io.paths.source_tiles` (fallback `io.paths.source_tile`) and
   `io.paths.val_tiles`; inference tiles come from `io.paths.inference_dir`
   (glob `io.paths.inference_glob`) or fallback `io.paths.holdout_tiles`.
3. Training artifacts are built from source tiles:
   LOO mode builds artifacts per fold; manual mode builds them once from configured
   source tiles. kNN banks and XGB data can fuse DINO patch embeddings with optional
   image patch cues (`model.hybrid_features`), with train-fold-only z-score stats for XGB.
4. Run `python main.py` for the selected mode.
   In training mode, a reusable bundle is saved (default `run_xxx/model_bundle` or `io.inference.model_bundle_dir` when set) containing:
   `manifest.yml` and `xgb_model.json` (XGB-only persisted artifact mode; no bank `.npy` files).
   Best settings are written to both `inference_best_setting.yml` and legacy `best_setting.yml`.
5. During execution, `rolling_best_setting.yml` is updated incrementally so best-known settings survive interruptions.
6. Optional runtime time-budget cutover (`runtime.time_budget`) can stop training
   phases after the configured wall-clock budget and switch directly to holdout
   inference using best-so-far fold settings.
7. In inference, champion masks can spawn `postprocess.novel_proposals` outside the
   incomplete source label raster; connected components are filtered by shape
   heuristics and exported as accepted/rejected proposal layers. Candidate scope can be
   SH-buffer constrained or whole-tile (`postprocess.novel_proposals.search_scope`).
8. Plot outputs are split by stage under `plots/validation/` and
   `plots/inference/`. Inference panels render only the active model stream
   (XGB or kNN), and proposal overlays include per-component IDs with 0-1
   acceptance scores. Proposal filtering auto-accepts components inside the SH
   buffer and applies shape heuristics only to outside-buffer components.
