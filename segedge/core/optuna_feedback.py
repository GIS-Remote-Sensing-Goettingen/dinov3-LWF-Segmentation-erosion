"""Optuna callback helpers for trial feedback logging."""

from __future__ import annotations

import logging

from .optuna_stop import build_no_improvement_callbacks_from_config

logger = logging.getLogger(__name__)


def _trial_attr_float(trial, keys: tuple[str, ...]) -> float | None:
    """Return first parseable float from trial attrs by priority key.

    Examples:
        >>> isinstance(_trial_attr_float.__name__, str)
        True
    """
    attrs = getattr(trial, "user_attrs", {}) or {}
    for key in keys:
        if key in attrs:
            try:
                return float(attrs[key])
            except Exception:
                return None
    return None


def _safe_best_trial(
    study, fallback_value: float, fallback_number: int
) -> tuple[float, int]:
    """Return best trial value/number with safe fallback.

    Examples:
        >>> isinstance(_safe_best_trial.__name__, str)
        True
    """
    try:
        best_trial = study.best_trial
        best_value = float(
            best_trial.value if best_trial.value is not None else fallback_value
        )
        return best_value, int(best_trial.number)
    except Exception:
        return fallback_value, fallback_number


def _trial_feedback_callback(
    cfg_module,
    *,
    stage_label: str,
    trial_total: int | None = None,
    trial_offset: int = 0,
):
    """Build a readable per-trial feedback logger callback.

    Examples:
        >>> isinstance(_trial_feedback_callback.__name__, str)
        True
    """
    verbose_separators = bool(getattr(cfg_module, "BO_VERBOSE_TRIAL_SEPARATORS", True))
    seen = {"count": 0}

    def _callback(study, trial) -> None:
        seen["count"] += 1
        progress_idx = int(trial_offset + seen["count"])
        progress_total = int(trial_total or progress_idx)
        if progress_total < progress_idx:
            progress_total = progress_idx
        if verbose_separators:
            logger.info(
                "==== %s Trial %d/%d ====",
                stage_label,
                progress_idx,
                progress_total,
            )
        trial_num = int(getattr(trial, "number", -1))
        state_name = str(getattr(getattr(trial, "state", None), "name", "UNKNOWN"))
        if state_name != "COMPLETE":
            logger.info(
                "bayes %s trial=%s state=%s progress=%d/%d",
                stage_label,
                trial_num,
                state_name,
                progress_idx,
                progress_total,
            )
            if verbose_separators:
                logger.info(
                    "==== End %s Trial %d/%d ====",
                    stage_label,
                    progress_idx,
                    progress_total,
                )
            return

        value = float(trial.value if trial.value is not None else 0.0)
        loss_proxy = float(1.0 - value)
        iou_gt = _trial_attr_float(trial, ("weighted_iou_gt_core", "weighted_iou_gt"))
        iou_sh = _trial_attr_float(trial, ("weighted_iou_sh_core", "weighted_iou_sh"))
        knn_iou = _trial_attr_float(trial, ("knn_iou",))
        xgb_iou = _trial_attr_float(trial, ("xgb_iou",))
        best_value, best_number = _safe_best_trial(
            study,
            fallback_value=value,
            fallback_number=trial_num,
        )
        logger.info(
            "bayes %s trial=%d value=%.4f loss=%.4f iou_gt=%s iou_sh=%s "
            "knn_iou=%s xgb_iou=%s best=%.4f best_trial=%d progress=%d/%d",
            stage_label,
            trial_num,
            value,
            loss_proxy,
            "n/a" if iou_gt is None else f"{iou_gt:.4f}",
            "n/a" if iou_sh is None else f"{iou_sh:.4f}",
            "n/a" if knn_iou is None else f"{knn_iou:.4f}",
            "n/a" if xgb_iou is None else f"{xgb_iou:.4f}",
            best_value,
            best_number,
            progress_idx,
            progress_total,
        )
        if verbose_separators:
            logger.info(
                "==== End %s Trial %d/%d ====",
                stage_label,
                progress_idx,
                progress_total,
            )

    return _callback


def build_optuna_callbacks_with_feedback(
    cfg_module,
    *,
    stage_name: str,
    trial_total: int | None = None,
    trial_offset: int = 0,
    default_patience: int = 20,
):
    """Return stagnation-stop + per-trial feedback callbacks.

    Examples:
        >>> class _Cfg:
        ...     BO_EARLY_STOP_PATIENCE = 0
        ...     BO_EARLY_STOP_MIN_DELTA = 0.0
        ...     BO_VERBOSE_TRIAL_SEPARATORS = False
        >>> len(build_optuna_callbacks_with_feedback(_Cfg, stage_name="stage")) >= 1
        True
    """
    callbacks = []
    early_stop_callbacks = build_no_improvement_callbacks_from_config(
        cfg_module,
        default_patience=default_patience,
    )
    if early_stop_callbacks:
        callbacks.extend(early_stop_callbacks)
    callbacks.append(
        _trial_feedback_callback(
            cfg_module,
            stage_label=stage_name,
            trial_total=trial_total,
            trial_offset=trial_offset,
        )
    )
    return callbacks


def objective_weights(cfg_module) -> tuple[float, float]:
    """Return normalized GT/SH objective weights from config.

    Examples:
        >>> class _Cfg:
        ...     BO_OBJECTIVE_W_GT = 0.8
        ...     BO_OBJECTIVE_W_SH = 0.2
        >>> objective_weights(_Cfg)
        (0.8, 0.2)
    """
    w_gt = float(getattr(cfg_module, "BO_OBJECTIVE_W_GT", 0.8))
    w_sh = float(getattr(cfg_module, "BO_OBJECTIVE_W_SH", 0.2))
    w_sum = w_gt + w_sh
    if w_sum <= 0:
        return 1.0, 0.0
    return w_gt / w_sum, w_sh / w_sum
