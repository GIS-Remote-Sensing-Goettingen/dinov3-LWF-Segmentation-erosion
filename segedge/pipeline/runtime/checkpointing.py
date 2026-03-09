"""Rolling checkpoint persistence."""

from __future__ import annotations

import os
from datetime import datetime

import yaml

from ...core.config_loader import cfg
from ...core.features import hybrid_feature_spec_hash


def write_rolling_best_config(
    out_path: str,
    stage: str,
    tuned: dict,
    fold_done: int,
    fold_total: int,
    holdout_done: int,
    holdout_total: int,
    best_fold: dict | None = None,
    time_budget: dict | None = None,
    model_bundle: dict | None = None,
) -> None:
    """Write rolling best config checkpoint for interruption-safe resume context.

    Examples:
        >>> callable(write_rolling_best_config)
        True
    """
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "stage": stage,
        "progress": {
            "loo_folds_done": int(fold_done),
            "loo_folds_total": int(fold_total),
            "holdout_done": int(holdout_done),
            "holdout_total": int(holdout_total),
        },
        "best_raw_config": tuned.get("best_raw_config"),
        "best_xgb_config": tuned.get("best_xgb_config"),
        "best_crf_config": tuned.get("best_crf_config"),
        "best_shadow_config": tuned.get("shadow_cfg"),
        "champion_source": tuned.get("champion_source"),
        "roads_penalty": tuned.get("roads_penalty"),
        "inference_score_prior": {
            "enabled": bool(cfg.io.inference.score_prior.enabled),
            "apply_to": cfg.io.inference.score_prior.apply_to,
            "target": cfg.io.inference.score_prior.target,
            "mode": cfg.io.inference.score_prior.mode,
            "factor": float(cfg.io.inference.score_prior.factor),
            "clip_max": float(cfg.io.inference.score_prior.clip_max),
        },
        "feature_spec_hash": hybrid_feature_spec_hash(),
        "hybrid_features_enabled": bool(cfg.model.hybrid_features.enabled),
    }
    if best_fold is not None:
        payload["selected_fold"] = {
            "fold_index": int(best_fold["fold_index"]),
            "val_tile": best_fold["val_tile"],
            "val_champion_shadow_iou": float(best_fold["val_champion_shadow_iou"]),
        }
    if time_budget is not None:
        payload["time_budget"] = time_budget
    if model_bundle is not None:
        payload["model_bundle"] = model_bundle
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
