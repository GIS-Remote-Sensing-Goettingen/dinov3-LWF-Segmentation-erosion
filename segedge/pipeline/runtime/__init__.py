"""Runtime helper modules split from the legacy runtime_utils module."""

from .checkpointing import write_rolling_best_config
from .crf_eval import _eval_crf_config, _init_crf_parallel
from .holdout_inference import infer_on_holdout
from .phase_metrics import (
    _log_phase,
    _summarize_phase_metrics,
    _update_phase_metrics,
    _weighted_mean,
    summarize_phase_metrics_mean_std,
)
from .postprocess import (
    _apply_shadow_filter,
    _compute_component_shape_metrics,
    filter_novel_proposals,
)
from .roads import _apply_roads_penalty, _get_roads_mask
from .tile_context import load_b_tile_context
from .time_budget import (
    build_time_budget_status,
    compute_budget_deadline,
    deadline_ts_to_utc_iso,
    is_budget_exceeded,
    parse_utc_iso_to_epoch,
    remaining_budget_s,
)

__all__ = [
    "_apply_roads_penalty",
    "_apply_shadow_filter",
    "_compute_component_shape_metrics",
    "_eval_crf_config",
    "_get_roads_mask",
    "_init_crf_parallel",
    "_log_phase",
    "_summarize_phase_metrics",
    "_update_phase_metrics",
    "_weighted_mean",
    "build_time_budget_status",
    "compute_budget_deadline",
    "deadline_ts_to_utc_iso",
    "filter_novel_proposals",
    "infer_on_holdout",
    "is_budget_exceeded",
    "load_b_tile_context",
    "parse_utc_iso_to_epoch",
    "remaining_budget_s",
    "summarize_phase_metrics_mean_std",
    "write_rolling_best_config",
]
