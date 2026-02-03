# Changelog


## [Unreleased]
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
