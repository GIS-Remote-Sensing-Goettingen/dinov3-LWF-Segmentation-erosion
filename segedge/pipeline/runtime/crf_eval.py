"""CRF process-pool evaluation helpers."""

from __future__ import annotations

from ...core.crf_utils import refine_with_densecrf
from ...core.metrics_utils import compute_metrics
from .phase_metrics import _weighted_mean

_CRF_PARALLEL_CONTEXTS: list[dict] | None = None


def _init_crf_parallel(contexts: list[dict]) -> None:
    """Initialize process-global CRF evaluation contexts.

    Examples:
        >>> _init_crf_parallel([])
    """
    global _CRF_PARALLEL_CONTEXTS
    _CRF_PARALLEL_CONTEXTS = contexts


def _eval_crf_config(cfg, n_iters: int = 5) -> tuple[float, tuple[float, ...]]:
    """Evaluate one CRF config against preloaded validation contexts.

    Examples:
        >>> callable(_eval_crf_config)
        True
    """
    if _CRF_PARALLEL_CONTEXTS is None:
        raise RuntimeError("CRF contexts not initialized")
    prob_soft, trimap_band, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
    ious = []
    weights = []
    for ctx in _CRF_PARALLEL_CONTEXTS:
        mask_crf_local = refine_with_densecrf(
            ctx["img_b"],
            ctx["score_full"],
            ctx["thr_center"],
            ctx["sh_buffer_mask"],
            prob_softness=prob_soft,
            n_iters=n_iters,
            pos_w=pos_w,
            pos_xy_std=pos_xy,
            bilateral_w=bi_w,
            bilateral_xy_std=bi_xy,
            bilateral_rgb_std=bi_rgb,
            trimap_band_pixels=(
                int(trimap_band) if bool(ctx.get("crf_use_trimap", False)) else None
            ),
        )
        ious.append(compute_metrics(mask_crf_local, ctx["gt_mask_eval"])["iou"])
        weights.append(float(ctx["gt_weight"]))
    return _weighted_mean(ious, weights), cfg
