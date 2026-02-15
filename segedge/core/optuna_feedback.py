"""Optuna callback helpers for trial feedback logging."""

from __future__ import annotations

import logging

from .optuna_stop import build_no_improvement_callbacks_from_config

logger = logging.getLogger(__name__)


def _trial_attr_float(trial, keys: tuple[str, ...]) -> float | None:
    attrs = getattr(trial, "user_attrs", {}) or {}
    for key in keys:
        if key in attrs:
            try:
                return float(attrs[key])
            except Exception:
                return None
    return None


def _trial_feedback_callback(stage_name: str):
    def _callback(study, trial) -> None:
        trial_num = int(getattr(trial, "number", -1))
        state_name = str(getattr(getattr(trial, "state", None), "name", "UNKNOWN"))
        if state_name != "COMPLETE":
            logger.info("bayes %s trial=%s state=%s", stage_name, trial_num, state_name)
            return

        value = float(trial.value if trial.value is not None else 0.0)
        loss_proxy = float(1.0 - value)
        iou_gt = _trial_attr_float(trial, ("weighted_iou_gt_core", "weighted_iou_gt"))
        iou_sh = _trial_attr_float(trial, ("weighted_iou_sh_core", "weighted_iou_sh"))
        knn_iou = _trial_attr_float(trial, ("knn_iou",))
        xgb_iou = _trial_attr_float(trial, ("xgb_iou",))
        try:
            best_trial = study.best_trial
            best_value = float(
                best_trial.value if best_trial.value is not None else value
            )
            best_number = int(best_trial.number)
        except Exception:
            best_value = value
            best_number = trial_num
        logger.info(
            "bayes %s trial=%d value=%.4f loss=%.4f iou_gt=%s iou_sh=%s "
            "knn_iou=%s xgb_iou=%s best=%.4f best_trial=%d",
            stage_name,
            trial_num,
            value,
            loss_proxy,
            "n/a" if iou_gt is None else f"{iou_gt:.4f}",
            "n/a" if iou_sh is None else f"{iou_sh:.4f}",
            "n/a" if knn_iou is None else f"{knn_iou:.4f}",
            "n/a" if xgb_iou is None else f"{xgb_iou:.4f}",
            best_value,
            best_number,
        )

    return _callback


def build_optuna_callbacks_with_feedback(
    cfg_module,
    *,
    stage_name: str,
    default_patience: int = 20,
):
    """Return stagnation-stop + per-trial feedback callbacks."""
    callbacks = []
    early_stop_callbacks = build_no_improvement_callbacks_from_config(
        cfg_module,
        default_patience=default_patience,
    )
    if early_stop_callbacks:
        callbacks.extend(early_stop_callbacks)
    callbacks.append(_trial_feedback_callback(stage_name))
    return callbacks


def objective_weights(cfg_module) -> tuple[float, float]:
    """Return normalized GT/SH objective weights from config."""
    w_gt = float(getattr(cfg_module, "BO_OBJECTIVE_W_GT", 0.8))
    w_sh = float(getattr(cfg_module, "BO_OBJECTIVE_W_SH", 0.2))
    w_sum = w_gt + w_sh
    if w_sum <= 0:
        return 1.0, 0.0
    return w_gt / w_sum, w_sh / w_sum
