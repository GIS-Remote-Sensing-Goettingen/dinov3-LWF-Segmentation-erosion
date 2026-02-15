"""Shared helpers for Bayesian tuning objectives.

Examples:
    >>> robust_objective(0.5, 0.25, 0.8, 0.2)
    0.45
"""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import gaussian_filter

import config as cfg

from ..core.bayes_timing import TrialPhaseTimer


def mask_iou(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    region_mask: np.ndarray | None = None,
) -> float:
    """Compute IoU between binary masks inside an optional region.

    Examples:
        >>> import numpy as np
        >>> p = np.array([[1, 0], [1, 0]], dtype=bool)
        >>> r = np.array([[1, 0], [0, 1]], dtype=bool)
        >>> round(mask_iou(p, r), 2)
        0.33
    """
    p = np.asarray(pred_mask).astype(bool)
    r = np.asarray(ref_mask).astype(bool)
    if region_mask is not None:
        reg = np.asarray(region_mask).astype(bool)
        p = np.logical_and(p, reg)
        r = np.logical_and(r, reg)
    tp = float(np.logical_and(p, r).sum())
    fp = float(np.logical_and(p, ~r).sum())
    fn = float(np.logical_and(~p, r).sum())
    return tp / (tp + fp + fn + 1e-8)


def robust_objective(iou_gt: float, iou_sh: float, w_gt: float, w_sh: float) -> float:
    """Weighted robust objective used by Bayesian stages.

    Examples:
        >>> robust_objective(0.5, 0.25, 0.8, 0.2)
        0.45
    """
    return float(w_gt * iou_gt + w_sh * iou_sh)


def f1_optimal_threshold(
    score_map: np.ndarray,
    gt_mask: np.ndarray,
    region_mask: np.ndarray,
    bins: int = 64,
) -> float:
    """Approximate F1-optimal threshold on a score map.

    Examples:
        >>> import numpy as np
        >>> score = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
        >>> gt = np.array([[0, 1], [0, 1]], dtype=bool)
        >>> reg = np.ones_like(gt, dtype=bool)
        >>> thr = f1_optimal_threshold(score, gt, reg, bins=8)
        >>> 0.1 <= thr <= 0.9
        True
    """
    reg = np.asarray(region_mask).astype(bool)
    if reg.sum() == 0:
        return 0.5
    y_true = np.asarray(gt_mask).astype(bool)[reg]
    y_score = np.asarray(score_map, dtype=np.float32)[reg]
    if y_score.size == 0:
        return 0.5
    lo = float(np.min(y_score))
    hi = float(np.max(y_score))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return float(lo if np.isfinite(lo) else 0.5)
    thresholds = np.linspace(lo, hi, int(max(8, bins)))
    best_thr = thresholds[0]
    best_f1 = -1.0
    for thr in thresholds:
        pred = y_score >= thr
        tp = float(np.logical_and(pred, y_true).sum())
        fp = float(np.logical_and(pred, ~y_true).sum())
        fn = float(np.logical_and(~pred, y_true).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def light_deterministic_perturbations(
    img_rgb: np.ndarray,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    """Generate deterministic light perturbations for robustness scoring.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((4, 4, 3), dtype=np.uint8)
        >>> outs = light_deterministic_perturbations(img, count=1, seed=1)
        >>> len(outs)
        1
    """
    if count <= 0:
        return []
    base = img_rgb.astype(np.float32) / 255.0
    rng = np.random.default_rng(int(seed))
    outs: list[np.ndarray] = []
    for _ in range(int(count)):
        contrast = float(rng.uniform(0.95, 1.05))
        brightness = float(rng.uniform(0.95, 1.05))
        noise_sigma = float(rng.uniform(0.0, 0.01))
        blur_sigma = float(rng.uniform(0.0, 0.6))
        arr = (base - 0.5) * contrast + 0.5
        arr = arr * brightness
        if noise_sigma > 0.0:
            arr = arr + rng.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        if blur_sigma > 1e-6:
            arr = gaussian_filter(arr, sigma=(blur_sigma, blur_sigma, 0.0))
        outs.append(np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8))
    return outs


def attach_perturbations_to_contexts(
    val_contexts: list[dict],
    count: int,
    seed: int,
) -> None:
    """Attach deterministic perturbed images to validation contexts.

    Examples:
        >>> import numpy as np
        >>> contexts = [{"img_b": np.zeros((2, 2, 3), dtype=np.uint8)}]
        >>> attach_perturbations_to_contexts(contexts, count=1, seed=7)
        >>> len(contexts[0]["perturbed_imgs"])
        1
    """
    for idx, ctx in enumerate(val_contexts):
        ctx["perturbed_imgs"] = light_deterministic_perturbations(
            ctx["img_b"],
            count=count,
            seed=seed + idx * 1009,
        )


def set_trial_timing_attrs(trial, timer: TrialPhaseTimer, *, trial_t0: float) -> None:
    """Persist compact phase timings into trial attrs for CSV and feedback logs.

    Examples:
        >>> class _Trial:
        ...     def __init__(self):
        ...         self.user_attrs = {}
        ...     def set_user_attr(self, key, value):
        ...         self.user_attrs[key] = value
        >>> tr = _Trial()
        >>> timer = TrialPhaseTimer()
        >>> timer.add_duration("score_base_knn_xgb_s", 1.0)
        >>> set_trial_timing_attrs(tr, timer, trial_t0=time.perf_counter() - 2.0)
        >>> "trial_total_s" in tr.user_attrs
        True
    """
    snap = timer.finalize(total_s=time.perf_counter() - trial_t0)
    for key, value in snap.items():
        trial.set_user_attr(key, float(value))
    summary = timer.summary(
        top_n=int(getattr(cfg, "BO_TIMING_LOG_TOP_PHASES", 1) or 1),
        min_phase_s=float(getattr(cfg, "BO_TIMING_LOG_MIN_PHASE_S", 0.5) or 0.0),
    )
    trial.set_user_attr("timing_summary", str(summary))
