# Changelog


## [0.2.0]
- Description: Added phase logging, diskless feature-cache mode, parallel CRF tuning in the main pipeline, and shadow protect score tuning for split evaluation.
- file touched: `config.py`, `segedge/core/banks.py`, `segedge/core/knn.py`, `segedge/core/xdboost.py`, `segedge/pipeline/run.py`, `segedge/pipeline/split_eval.py`, `CHANGELOG.md`
- reason: Allow memory-scoped feature reuse, speed up CRF search on multi-core nodes, and preserve high-confidence positives under shadows in split evaluation.
- problems fixed: Avoids redundant feature extraction while keeping memory bounded per image, reduces CRF tuning time, and prevents shadow filtering from removing true positives in split evaluation.

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
