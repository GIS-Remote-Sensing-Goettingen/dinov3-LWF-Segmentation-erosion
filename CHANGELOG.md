# Changelog

## [Unreleased]
- Description: Added phase logging, diskless feature-cache mode, and parallel CRF tuning in the main pipeline.
- file touched: `config.py`, `segedge/core/banks.py`, `segedge/core/knn.py`, `segedge/core/xdboost.py`, `segedge/pipeline/run.py`, `segedge/pipeline/split_eval.py`, `CHANGELOG.md`
- reason: Allow memory-scoped feature reuse and speed up CRF search on multi-core nodes.
- problems fixed: Avoids redundant feature extraction while keeping memory bounded per image and reduces CRF tuning time.

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
