"""Per-tile inference for holdout and validation tiles.

This module is intentionally organized as a small stage pipeline:
- load and annotate tile context
- score the enabled model streams
- refine masks with CRF and shadow filtering
- export plots and return the union-ready masks
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.ndimage import binary_fill_holes, median_filter

from ...core.config_loader import cfg
from ...core.crf_utils import refine_with_densecrf
from ...core.features import prefetch_features_single_scale_image
from ...core.knn import zero_shot_knn_single_scale_B_with_saliency
from ...core.metrics_utils import compute_metrics
from ...core.plotting import (
    save_core_qualitative_plot,
    save_disagreement_entropy_plot,
    save_proposal_overlay_plot,
    save_score_threshold_plot,
    save_unified_plot,
)
from ...core.timing_utils import perf_call, perf_metadata, perf_span
from ...core.xdboost import (
    xgb_score_image_b,
    xgb_score_image_b_legacy,
    xgb_score_image_b_streaming,
)
from .postprocess import _apply_shadow_filter, filter_novel_proposals
from .roads import _apply_roads_penalty, _get_roads_mask
from .tile_context import load_b_tile_context

logger = logging.getLogger(__name__)


def _postprocess_stream_mask(
    mask: np.ndarray,
    sh_buffer_mask: np.ndarray,
    *,
    fill_holes: bool = False,
    image_id: str | None = None,
    stream_name: str | None = None,
) -> np.ndarray:
    """Finalize a thresholded stream mask before metrics and CRF.

    Examples:
        >>> base = np.ones((7, 7), dtype=bool)
        >>> base[2:5, 2:5] = False
        >>> out = _postprocess_stream_mask(base, np.ones((7, 7), dtype=bool), fill_holes=True)
        >>> int(out[3, 3])
        1
    """
    filtered_mask = median_filter(mask.astype(np.uint8), size=3) > 0
    filtered_mask = np.logical_and(filtered_mask, sh_buffer_mask)
    if not fill_holes:
        return filtered_mask
    filled_mask = perf_call(
        binary_fill_holes,
        filtered_mask,
        stage="postprocess_stream_mask",
        substage="fill_holes",
    )
    filled_mask = np.logical_and(filled_mask, sh_buffer_mask)
    filled_hole_pixels = max(0, int(filled_mask.sum()) - int(filtered_mask.sum()))
    perf_metadata(
        "postprocess_stream_mask",
        substage="fill_holes_metadata",
        extra={
            "image_id": image_id or "",
            "stream": stream_name or "",
            "raw_mask_pixels_before_fill": int(filtered_mask.sum()),
            "raw_mask_pixels_after_fill": int(filled_mask.sum()),
            "filled_hole_pixels": filled_hole_pixels,
        },
    )
    logger.info(
        "hole fill: tile=%s stream=%s before=%s after=%s filled=%s",
        image_id or "<unknown>",
        stream_name or "<unknown>",
        int(filtered_mask.sum()),
        int(filled_mask.sum()),
        filled_hole_pixels,
    )
    return filled_mask


def _build_xgb_plot_preview_masks(
    *,
    context: dict[str, object],
    champion_score: np.ndarray,
    champion_thr: float,
    active_crf_mask: np.ndarray,
    tuned: dict | None,
) -> dict[str, np.ndarray]:
    """Build plot-only XGB preview masks without SH clipping.

    Examples:
        >>> context = {
        ...     "sh_buffer_mask": np.ones((1, 1), dtype=bool),
        ...     "img_b": np.zeros((1, 1, 3), dtype=np.uint8),
        ... }
        >>> out = _build_xgb_plot_preview_masks(
        ...     context=context,
        ...     champion_score=np.array([[0.9]], dtype=np.float32),
        ...     champion_thr=0.5,
        ...     active_crf_mask=np.array([[True]]),
        ...     tuned={},
        ... )
        >>> sorted(out.keys())
        ['xgb_crf_plot', 'xgb_raw_plot']
    """
    full_mask = np.ones_like(context["sh_buffer_mask"], dtype=bool)
    raw_plot_mask = _postprocess_stream_mask(
        champion_score >= champion_thr,
        full_mask,
        fill_holes=bool(cfg.postprocess.fill_holes_xgb),
        image_id=context.get("image_id_b"),
        stream_name="xgb_raw_plot",
    )
    crf_cfg = dict((tuned or {}).get("best_crf_config") or {})
    required_crf_keys = {
        "prob_softness",
        "pos_w",
        "pos_xy_std",
        "bilateral_w",
        "bilateral_xy_std",
        "bilateral_rgb_std",
    }
    if (not bool(crf_cfg.get("enabled", True))) or (
        required_crf_keys - set(crf_cfg.keys())
    ):
        crf_plot_mask = active_crf_mask
    else:
        crf_plot_mask = _apply_crf_to_stream(
            context["img_b"],
            champion_score,
            champion_thr,
            full_mask,
            crf_cfg,
            use_trimap_band=True,
            base_mask_override=raw_plot_mask,
        )
    return {
        "xgb_raw_plot": raw_plot_mask,
        "xgb_crf_plot": crf_plot_mask,
    }


def _load_holdout_tile_context(
    holdout_path: str,
    gt_vector_paths: list[str] | None,
    tuned: dict,
) -> dict[str, object]:
    """Load tile context and runtime toggles.

    Examples:
        >>> callable(_load_holdout_tile_context)
        True
    """
    (
        img_b,
        labels_sh,
        _,
        gt_mask_eval,
        sh_buffer_mask,
        buffer_m,
        pixel_size_m,
    ) = perf_call(
        load_b_tile_context,
        holdout_path,
        gt_vector_paths,
        stage="infer_on_holdout",
        substage="load_holdout_tile_context",
        extra={"tile_path": holdout_path},
    )
    gt_available = gt_mask_eval is not None
    if gt_mask_eval is None:
        logger.warning("Holdout has no GT; metrics will be reported as 0.0.")
        gt_mask_eval = np.zeros(img_b.shape[:2], dtype=bool)
    image_id_b = os.path.splitext(os.path.basename(holdout_path))[0]
    perf_metadata(
        "infer_on_holdout",
        substage="resolve_runtime_toggles",
        extra={"tile_path": holdout_path},
    )
    knn_enabled = bool(tuned.get("knn_enabled", cfg.search.knn.enabled))
    xgb_enabled = bool(tuned.get("xgb_enabled", cfg.search.xgb.enabled))
    ds = int(cfg.model.backbone.resample_factor or 1)
    roads_mask = perf_call(
        _get_roads_mask,
        holdout_path,
        ds,
        stage="infer_on_holdout",
        substage="load_roads_mask",
        extra={
            "tile_path": holdout_path,
            "target_shape": list(img_b.shape[:2]),
            "downsample_factor": ds,
        },
        target_shape=img_b.shape[:2],
    )
    perf_metadata(
        "infer_on_holdout",
        substage="finalize_context",
        extra={
            "tile_path": holdout_path,
            "image_shape": list(img_b.shape[:2]),
            "source_label_positive_pixels": int((labels_sh > 0).sum()),
            "source_label_coverage_ratio": (
                float((labels_sh > 0).mean()) if labels_sh.size else 0.0
            ),
            "buffer_positive_pixels": int(sh_buffer_mask.sum()),
            "buffer_coverage_ratio": (
                float(sh_buffer_mask.mean()) if sh_buffer_mask.size else 0.0
            ),
            "roads_positive_pixels": (
                int(roads_mask.sum()) if roads_mask is not None else 0
            ),
            "roads_coverage_ratio": (
                float(roads_mask.mean())
                if roads_mask is not None and roads_mask.size
                else 0.0
            ),
        },
    )
    return {
        "img_b": img_b,
        "labels_sh": labels_sh,
        "gt_mask_eval": gt_mask_eval,
        "gt_available": gt_available,
        "gt_weight": float(gt_mask_eval.sum()),
        "sh_buffer_mask": sh_buffer_mask,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
        "image_id_b": image_id_b,
        "roads_mask": roads_mask,
        "roads_penalty": float(tuned.get("roads_penalty", 1.0)),
        "xgb_feature_stats": tuned.get("xgb_feature_stats"),
        "knn_enabled": knn_enabled,
        "xgb_enabled": xgb_enabled,
        "crf_enabled": bool(tuned.get("crf_enabled", cfg.search.crf.enabled)),
    }


def _prefetch_holdout_features(
    context: dict[str, object],
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
) -> dict[tuple[int, int], dict[str, object]]:
    """Prefetch or compute per-tile feature caches for one holdout image.

    Examples:
        >>> callable(_prefetch_holdout_features)
        True
    """
    return prefetch_features_single_scale_image(
        context["img_b"],
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        None,
        feature_dir,
        context["image_id_b"],
        materialize_cached=bool(context["knn_enabled"] or not context["xgb_enabled"]),
    )


def _should_stream_xgb_features(
    context: dict[str, object],
    feature_dir: str | None,
) -> bool:
    """Return whether one-shot XGB inference should stream features.

    Examples:
        >>> _should_stream_xgb_features({"xgb_enabled": True, "knn_enabled": False}, None)
        True
    """
    return bool(
        feature_dir is None
        and bool(context.get("xgb_enabled", False))
        and not bool(context.get("knn_enabled", False))
    )


def _streaming_xgb_enabled(
    context: dict[str, object],
    feature_dir: str | None,
) -> bool:
    """Return the cached streaming-XGB decision when available.

    Examples:
        >>> _streaming_xgb_enabled({"streaming_xgb": True}, None)
        True
    """
    if "streaming_xgb" in context:
        return bool(context["streaming_xgb"])
    return _should_stream_xgb_features(context, feature_dir)


def _prefetch_context_features(
    context: dict[str, object],
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
) -> dict[tuple[int, int], dict[str, object]] | None:
    """Return prefetched features unless the XGB-only streaming fast path is active.

    Examples:
        >>> context = {"xgb_enabled": True, "knn_enabled": False}
        >>> _prefetch_context_features(
        ...     context, None, None, None, 16, 32, 32, None
        ... ) is None
        True
    """
    if _streaming_xgb_enabled(context, feature_dir):
        return None
    return _prefetch_holdout_features(
        context,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        feature_dir,
    )


def _compute_shadow_stage_masks(
    context: dict[str, object],
    champion_source: str,
    champion_score: np.ndarray,
    knn_result: dict[str, object],
    xgb_result: dict[str, object],
    mask_crf_knn: np.ndarray,
    mask_crf_xgb: np.ndarray,
    shadow_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the champion and per-stream shadow-filtered masks.

    Examples:
        >>> callable(_compute_shadow_stage_masks)
        True
    """
    active_crf_mask = mask_crf_knn if champion_source == "raw" else mask_crf_xgb
    active_shadow_mask = _apply_shadow_mask(
        context["img_b"], active_crf_mask, shadow_cfg, champion_score
    )
    shadow_mask_knn = (
        _apply_shadow_mask(
            context["img_b"], mask_crf_knn, shadow_cfg, knn_result["score"]
        )
        if context["knn_enabled"]
        else np.zeros_like(context["sh_buffer_mask"], dtype=bool)
    )
    shadow_mask_xgb = (
        _apply_shadow_mask(
            context["img_b"], mask_crf_xgb, shadow_cfg, xgb_result["score"]
        )
        if context["xgb_enabled"]
        else np.zeros_like(context["sh_buffer_mask"], dtype=bool)
    )
    return active_shadow_mask, shadow_mask_knn, shadow_mask_xgb


def _tile_workload_metadata(context: dict[str, object], ps: int) -> dict[str, object]:
    """Return the derived workload metadata attached to each holdout tile record.

    Examples:
        >>> context = {
        ...     "img_b": np.zeros((16, 16, 3), dtype=np.uint8),
        ...     "labels_sh": np.zeros((16, 16), dtype=np.uint8),
        ...     "sh_buffer_mask": np.zeros((16, 16), dtype=bool),
        ...     "roads_mask": None,
        ... }
        >>> _tile_workload_metadata(context, 16)["xgb_patch_rows"]
        1
    """
    return {
        "image_shape": list(context["img_b"].shape[:2]),
        "source_label_positive_pixels": int((context["labels_sh"] > 0).sum()),
        "source_label_coverage_ratio": (
            float((context["labels_sh"] > 0).mean())
            if context["labels_sh"].size
            else 0.0
        ),
        "sh_buffer_positive_pixels": int(context["sh_buffer_mask"].sum()),
        "sh_buffer_coverage_ratio": (
            float(context["sh_buffer_mask"].mean())
            if context["sh_buffer_mask"].size
            else 0.0
        ),
        "roads_positive_pixels": (
            int(context["roads_mask"].sum()) if context["roads_mask"] is not None else 0
        ),
        "roads_coverage_ratio": (
            float(context["roads_mask"].mean())
            if context["roads_mask"] is not None and context["roads_mask"].size
            else 0.0
        ),
        "xgb_patch_rows": int(context["img_b"].shape[0] // ps)
        * int(context["img_b"].shape[1] // ps),
    }


def _compute_knn_stream(
    context: dict[str, object],
    tuned: dict,
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
) -> dict[str, object]:
    """Return kNN scores, masks, and metrics for one tile.

    Examples:
        >>> callable(_compute_knn_stream)
        True
    """
    sh_buffer_mask = context["sh_buffer_mask"]
    gt_mask_eval = context["gt_mask_eval"]
    output: dict[str, object] = {
        "score_raw": None,
        "score": None,
        "threshold": None,
        "mask": np.zeros_like(sh_buffer_mask, dtype=bool),
        "metrics": None,
    }
    if not context["knn_enabled"]:
        return output
    k = int(tuned["best_raw_config"]["k"])
    threshold = float(tuned["best_raw_config"]["threshold"])
    score_map, _ = zero_shot_knn_single_scale_B_with_saliency(
        context["img_b"],
        pos_bank,
        neg_bank,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        k=k,
        aggregate_layers=None,
        feature_dir=feature_dir,
        image_id=context["image_id_b"],
        neg_alpha=cfg.model.banks.neg_alpha,
        prefetched_tiles=context["prefetched_b"],
        use_fp16_matmul=cfg.search.knn.use_fp16_knn,
        context_radius=context_radius,
    )
    score_penalized = _apply_roads_penalty(
        score_map,
        context["roads_mask"],
        context["roads_penalty"],
    )
    mask = _postprocess_stream_mask(
        score_penalized >= threshold,
        sh_buffer_mask,
        image_id=context["image_id_b"],
        stream_name="knn_raw",
    )
    output.update(
        {
            "score_raw": score_map,
            "score": score_penalized,
            "threshold": threshold,
            "mask": mask,
            "metrics": compute_metrics(mask, gt_mask_eval),
        }
    )
    return output


def _compute_xgb_stream(
    context: dict[str, object],
    tuned: dict,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    final_inference_phase: bool,
    xgb_guard_state: dict[str, object] | None,
) -> dict[str, object]:
    """Return XGB scores, masks, and metrics for one tile.

    Examples:
        >>> callable(_compute_xgb_stream)
        True
    """
    sh_buffer_mask = context["sh_buffer_mask"]
    gt_mask_eval = context["gt_mask_eval"]
    output: dict[str, object] = {
        "score": None,
        "threshold": None,
        "mask": np.zeros_like(sh_buffer_mask, dtype=bool),
        "metrics": None,
    }
    if not context["xgb_enabled"]:
        return output
    bst = tuned.get("bst")
    if bst is None:
        raise ValueError("XGB enabled but no trained booster is available")
    threshold = float(tuned["best_xgb_config"]["threshold"])
    score_map = _score_xgb_with_guard(
        context=context,
        tuned=tuned,
        bst=bst,
        model=model,
        processor=processor,
        device=device,
        ps=ps,
        tile_size=tile_size,
        stride=stride,
        feature_dir=feature_dir,
        context_radius=context_radius,
        xgb_guard_state=xgb_guard_state,
    )
    score_penalized = _apply_roads_penalty(
        score_map,
        context["roads_mask"],
        context["roads_penalty"],
    )
    score_penalized = _apply_inference_score_prior(
        score_penalized,
        context["labels_sh"],
        context["image_id_b"],
        final_inference_phase=final_inference_phase,
    )
    mask = _postprocess_stream_mask(
        score_penalized >= threshold,
        sh_buffer_mask,
        fill_holes=bool(cfg.postprocess.fill_holes_xgb),
        image_id=context["image_id_b"],
        stream_name="xgb_raw",
    )
    output.update(
        {
            "score": score_penalized,
            "threshold": threshold,
            "mask": mask,
            "metrics": compute_metrics(mask, gt_mask_eval),
        }
    )
    return output


def _score_xgb_with_guard(
    *,
    context: dict[str, object],
    tuned: dict,
    bst,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    xgb_guard_state: dict[str, object] | None,
) -> np.ndarray:
    """Score one tile with the optimized XGB path and optional legacy guard."""
    use_streaming = _streaming_xgb_enabled(context, feature_dir)

    def _score_xgb_optimized() -> np.ndarray:
        if use_streaming:
            return xgb_score_image_b_streaming(
                context["img_b"],
                bst,
                model,
                processor,
                device,
                ps,
                tile_size,
                stride,
                context_radius=context_radius,
                xgb_feature_stats=context["xgb_feature_stats"],
            )
        return xgb_score_image_b(
            context["img_b"],
            bst,
            ps,
            tile_size,
            stride,
            feature_dir,
            context["image_id_b"],
            prefetched_tiles=context["prefetched_b"],
            context_radius=context_radius,
            xgb_feature_stats=context["xgb_feature_stats"],
        )

    if not xgb_guard_state or not bool(xgb_guard_state.get("enabled", False)):
        return _score_xgb_optimized()
    if bool(xgb_guard_state.get("fallback_to_legacy", False)):
        return _score_xgb_legacy(
            context=context,
            model=model,
            processor=processor,
            device=device,
            bst=bst,
            ps=ps,
            tile_size=tile_size,
            stride=stride,
            feature_dir=feature_dir,
            context_radius=context_radius,
        )
    checked_tiles = int(xgb_guard_state.get("checked_tiles", 0))
    guard_tiles = int(xgb_guard_state.get("guard_tiles", 0))
    if checked_tiles >= guard_tiles:
        return _score_xgb_optimized()
    score_map = _score_xgb_optimized()
    legacy_score = _score_xgb_legacy(
        context=context,
        model=model,
        processor=processor,
        device=device,
        bst=bst,
        ps=ps,
        tile_size=tile_size,
        stride=stride,
        feature_dir=feature_dir,
        context_radius=context_radius,
    )
    meaningful_diff, mean_abs_diff, max_abs_diff = _xgb_guard_diff_is_meaningful(
        optimized_score=score_map,
        legacy_score=legacy_score,
        threshold=float(tuned["best_xgb_config"]["threshold"]),
        atol=float(xgb_guard_state.get("atol", 1e-5)),
        rtol=float(xgb_guard_state.get("rtol", 1e-4)),
    )
    xgb_guard_state["checked_tiles"] = checked_tiles + 1
    logger.info(
        "xgb scorer guard: tile=%s compared=%s/%s mean_abs_diff=%.6f max_abs_diff=%.6f fallback=%s",
        context["image_id_b"],
        int(xgb_guard_state["checked_tiles"]),
        guard_tiles,
        mean_abs_diff,
        max_abs_diff,
        meaningful_diff,
    )
    if meaningful_diff:
        xgb_guard_state["fallback_to_legacy"] = True
        logger.warning(
            "xgb scorer guard triggered fallback to legacy scorer after meaningful difference on tile=%s",
            context["image_id_b"],
        )
        return legacy_score
    return score_map


def _score_xgb_legacy(
    *,
    context: dict[str, object],
    model,
    processor,
    device,
    bst,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
) -> np.ndarray:
    """Run the legacy XGB scorer for guard comparisons and fallback."""
    prefetched_b = context["prefetched_b"]
    if prefetched_b is None:
        prefetched_b = _prefetch_holdout_features(
            context,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            feature_dir,
        )
    return xgb_score_image_b_legacy(
        context["img_b"],
        bst,
        ps,
        tile_size,
        stride,
        feature_dir,
        context["image_id_b"],
        prefetched_tiles=prefetched_b,
        context_radius=context_radius,
        xgb_feature_stats=context["xgb_feature_stats"],
    )


def _xgb_guard_diff_is_meaningful(
    *,
    optimized_score: np.ndarray,
    legacy_score: np.ndarray,
    threshold: float,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float]:
    """Return whether optimized and legacy XGB scores differ meaningfully."""
    abs_diff = np.abs(
        optimized_score.astype(np.float32) - legacy_score.astype(np.float32)
    )
    mean_abs_diff = float(abs_diff.mean())
    max_abs_diff = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
    masks_match = np.array_equal(
        optimized_score >= threshold, legacy_score >= threshold
    )
    return (
        (not masks_match)
        or (not np.allclose(optimized_score, legacy_score, atol=atol, rtol=rtol)),
        mean_abs_diff,
        max_abs_diff,
    )


def _apply_inference_score_prior(
    score_map: np.ndarray,
    labels_sh: np.ndarray,
    image_id_b: str,
    *,
    final_inference_phase: bool,
) -> np.ndarray:
    """Apply the manual inference-only score prior inside/outside source-label pixels.

    Examples:
        >>> score = np.array([[0.5, 0.5]], dtype=np.float32)
        >>> labels = np.array([[1, 0]], dtype=np.uint8)
        >>> out = _apply_inference_score_prior(
        ...     score,
        ...     labels,
        ...     "tile",
        ...     final_inference_phase=False,
        ... )
        >>> np.allclose(out, score)
        True
    """
    prior_cfg = cfg.io.inference.score_prior
    if not final_inference_phase or not prior_cfg.enabled:
        return score_map
    inside_mask = labels_sh > 0
    outside_mask = ~inside_mask
    if not np.any(inside_mask) and not np.any(outside_mask):
        return score_map
    score_with_prior = score_map.astype(np.float32, copy=True)
    inside_before = (
        score_with_prior[inside_mask].copy() if np.any(inside_mask) else None
    )
    outside_before = (
        score_with_prior[outside_mask].copy() if np.any(outside_mask) else None
    )
    if np.any(inside_mask):
        score_with_prior[inside_mask] *= float(prior_cfg.inside_factor)
    if np.any(outside_mask):
        score_with_prior[outside_mask] *= float(prior_cfg.outside_factor)
    score_with_prior = np.clip(score_with_prior, 0.0, float(prior_cfg.clip_max))
    inside_after = score_with_prior[inside_mask] if np.any(inside_mask) else None
    outside_after = score_with_prior[outside_mask] if np.any(outside_mask) else None
    logger.info(
        (
            "inference score prior: tile=%s inside_factor=%.3f outside_factor=%.3f "
            "inside_pixels=%s outside_pixels=%s inside_mean_before=%.4f "
            "inside_mean_after=%.4f outside_mean_before=%.4f outside_mean_after=%.4f"
        ),
        image_id_b,
        float(prior_cfg.inside_factor),
        float(prior_cfg.outside_factor),
        int(inside_mask.sum()),
        int(outside_mask.sum()),
        float(inside_before.mean()) if inside_before is not None else 0.0,
        float(inside_after.mean()) if inside_after is not None else 0.0,
        float(outside_before.mean()) if outside_before is not None else 0.0,
        float(outside_after.mean()) if outside_after is not None else 0.0,
    )
    return score_with_prior


def _resolve_champion_stream(
    tuned: dict,
    knn_enabled: bool,
    xgb_enabled: bool,
) -> str:
    """Return the active champion stream key.

    Examples:
        >>> _resolve_champion_stream({"champion_source": "raw"}, True, False)
        'raw'
    """
    champion_source = str(tuned.get("champion_source", "xgb"))
    if knn_enabled and xgb_enabled:
        return champion_source if champion_source in {"raw", "xgb"} else "xgb"
    return "xgb" if xgb_enabled else "raw"


def _apply_crf_to_stream(
    img_b: np.ndarray,
    score_map: np.ndarray | None,
    threshold: float | None,
    sh_buffer_mask: np.ndarray,
    crf_cfg: dict,
    *,
    use_trimap_band: bool = False,
    base_mask_override: np.ndarray | None = None,
) -> np.ndarray:
    """Return CRF-refined mask when inputs are available.

    Examples:
        >>> callable(_apply_crf_to_stream)
        True
    """
    if score_map is None or threshold is None:
        return np.zeros_like(sh_buffer_mask, dtype=bool)
    return refine_with_densecrf(
        img_b,
        score_map,
        threshold,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
        trimap_band_pixels=(
            int(crf_cfg.get("trimap_band_pixels", 0)) if use_trimap_band else None
        ),
        base_mask_override=base_mask_override if use_trimap_band else None,
    )


def _run_crf_stage(
    context: dict[str, object],
    tuned: dict,
    knn_result: dict[str, object],
    xgb_result: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Return CRF masks for the kNN and XGB streams.

    Examples:
        >>> callable(_run_crf_stage)
        True
    """
    if not context["crf_enabled"]:
        return knn_result["mask"], xgb_result["mask"]
    crf_cfg = dict(tuned.get("best_crf_config") or {})
    if not bool(crf_cfg.get("enabled", True)):
        return knn_result["mask"], xgb_result["mask"]
    mask_crf_knn = (
        _apply_crf_to_stream(
            context["img_b"],
            knn_result["score"],
            knn_result["threshold"],
            context["sh_buffer_mask"],
            crf_cfg,
            use_trimap_band=False,
        )
        if context["knn_enabled"]
        else knn_result["mask"]
    )
    mask_crf_xgb = (
        _apply_crf_to_stream(
            context["img_b"],
            xgb_result["score"],
            xgb_result["threshold"],
            context["sh_buffer_mask"],
            crf_cfg,
            use_trimap_band=True,
            base_mask_override=xgb_result["mask"],
        )
        if context["xgb_enabled"]
        else xgb_result["mask"]
    )
    return mask_crf_knn, mask_crf_xgb


def _apply_shadow_mask(
    img_b: np.ndarray,
    base_mask: np.ndarray,
    shadow_cfg: dict,
    score_map: np.ndarray | None,
) -> np.ndarray:
    """Apply shadow filtering to a single stream.

    Examples:
        >>> callable(_apply_shadow_mask)
        True
    """
    if score_map is None:
        return np.zeros_like(base_mask, dtype=bool)
    return _apply_shadow_filter(
        img_b,
        base_mask,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        score_map,
        shadow_cfg.get("protect_score"),
    )


def _build_metrics_map(
    gt_mask_eval: np.ndarray,
    gt_weight: float,
    champion_raw_mask: np.ndarray,
    champion_crf_mask: np.ndarray,
    champion_shadow_mask: np.ndarray,
    knn_enabled: bool,
    xgb_enabled: bool,
    knn_result: dict[str, object],
    xgb_result: dict[str, object],
    mask_crf_knn: np.ndarray,
    mask_crf_xgb: np.ndarray,
    shadow_mask_knn: np.ndarray,
    shadow_mask_xgb: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Build GT-weighted metrics for every active stage.

    Examples:
        >>> callable(_build_metrics_map)
        True
    """
    metrics_map: dict[str, dict[str, float]] = {
        "champion_raw": compute_metrics(champion_raw_mask, gt_mask_eval),
        "champion_crf": compute_metrics(champion_crf_mask, gt_mask_eval),
        "champion_shadow": compute_metrics(champion_shadow_mask, gt_mask_eval),
    }
    if knn_enabled:
        metrics_map["knn_raw"] = knn_result["metrics"]
        metrics_map["knn_crf"] = compute_metrics(mask_crf_knn, gt_mask_eval)
        metrics_map["knn_shadow"] = compute_metrics(shadow_mask_knn, gt_mask_eval)
    if xgb_enabled:
        metrics_map["xgb_raw"] = xgb_result["metrics"]
        metrics_map["xgb_crf"] = compute_metrics(mask_crf_xgb, gt_mask_eval)
        metrics_map["xgb_shadow"] = compute_metrics(shadow_mask_xgb, gt_mask_eval)
    return {
        key: {**metrics, "_weight": gt_weight}
        for key, metrics in metrics_map.items()
        if metrics is not None
    }


def _build_proposal_masks(
    proposal_bundle: dict,
    template_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return proposal masks with zero defaults for missing keys.

    Examples:
        >>> bundle = {
        ...     "accepted_mask": np.zeros((1, 1), dtype=bool),
        ...     "rejected_mask": np.zeros((1, 1), dtype=bool),
        ...     "candidate_mask": np.zeros((1, 1), dtype=bool),
        ... }
        >>> sorted(_build_proposal_masks(bundle, bundle["candidate_mask"]).keys())[:2]
        ['accepted_inside_mask', 'accepted_mask']
    """
    return {
        "candidate_mask": proposal_bundle["candidate_mask"],
        "candidate_inside_mask": proposal_bundle.get(
            "candidate_inside_mask", np.zeros_like(template_mask, dtype=bool)
        ),
        "evaluated_outside_mask": proposal_bundle.get(
            "evaluated_outside_mask", np.zeros_like(template_mask, dtype=bool)
        ),
        "accepted_inside_mask": proposal_bundle.get(
            "accepted_inside_mask", np.zeros_like(template_mask, dtype=bool)
        ),
        "accepted_outside_mask": proposal_bundle.get(
            "accepted_outside_mask", np.zeros_like(template_mask, dtype=bool)
        ),
        "accepted_mask": proposal_bundle["accepted_mask"],
        "rejected_mask": proposal_bundle["rejected_mask"],
    }


def _compute_probability_and_diagnostics(
    active_stream: str,
    champion_score: np.ndarray,
    score_knn: np.ndarray | None,
    score_xgb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Build probability, disagreement, and entropy maps for plots.

    Examples:
        >>> prob, disagreement, entropy = _compute_probability_and_diagnostics(
        ...     "xgb",
        ...     np.ones((1, 1), dtype=np.float32) * 0.5,
        ...     None,
        ...     None,
        ... )
        >>> prob.shape, disagreement, entropy.shape
        ((1, 1), None, (1, 1))
    """
    if active_stream == "xgb":
        champion_prob = np.clip(champion_score.astype(np.float32), 1e-6, 1.0 - 1e-6)
    else:
        score_float = champion_score.astype(np.float32)
        score_min = float(np.min(score_float))
        score_max = float(np.max(score_float))
        champion_prob = (score_float - score_min) / (score_max - score_min + 1e-8)
        champion_prob = np.clip(champion_prob, 1e-6, 1.0 - 1e-6)
    disagreement_map = (
        np.abs(score_xgb - score_knn).astype(np.float32)
        if score_xgb is not None and score_knn is not None
        else None
    )
    entropy_map = (
        -champion_prob * np.log(champion_prob)
        - (1.0 - champion_prob) * np.log(1.0 - champion_prob)
    ).astype(np.float32)
    return champion_prob, disagreement_map, entropy_map


def _save_holdout_plots(
    *,
    context: dict[str, object],
    plot_dir: str,
    plot_with_metrics: bool,
    active_stream: str,
    active_label: str,
    champion_score: np.ndarray,
    champion_thr: float,
    active_raw_mask: np.ndarray,
    active_crf_mask: np.ndarray,
    active_shadow_mask: np.ndarray,
    crf_cfg: dict,
    metrics_map: dict[str, dict[str, float]],
    proposal_masks: dict[str, np.ndarray],
    proposal_bundle: dict,
    score_knn_raw: np.ndarray | None,
    disagreement_map: np.ndarray | None,
    entropy_map: np.ndarray,
) -> None:
    """Write the standard plot bundle for one inferred tile.

    Examples:
        >>> callable(_save_holdout_plots)
        True
    """
    image_id_b = context["image_id_b"]
    plot_cfg = cfg.io.inference.plots
    if plot_cfg.unified:
        plot_masks = {
            f"{active_stream}_raw": active_raw_mask,
            f"{active_stream}_crf": active_crf_mask,
            f"{active_stream}_shadow": active_shadow_mask,
        }
        if active_stream == "xgb":
            plot_masks.update(
                _build_xgb_plot_preview_masks(
                    context=context,
                    champion_score=champion_score,
                    champion_thr=champion_thr,
                    active_crf_mask=active_crf_mask,
                    tuned={"best_crf_config": crf_cfg},
                )
            )
        with perf_span("save_holdout_plots", substage="unified"):
            save_unified_plot(
                img_b=context["img_b"],
                gt_mask=context["gt_mask_eval"],
                labels_sh=context["labels_sh"],
                masks=plot_masks,
                metrics=metrics_map,
                plot_dir=plot_dir,
                image_id_b=image_id_b,
                show_metrics=plot_with_metrics and context["gt_available"],
                gt_available=context["gt_available"],
                similarity_map=score_knn_raw if active_stream == "knn" else None,
                score_maps={active_stream: champion_score},
                proposal_masks={
                    "accepted": proposal_masks["accepted_mask"],
                    "rejected": proposal_masks["rejected_mask"],
                },
            )
    if plot_cfg.qualitative_core:
        with perf_span("save_holdout_plots", substage="qualitative_core"):
            save_core_qualitative_plot(
                img_b=context["img_b"],
                gt_mask=context["gt_mask_eval"],
                pred_mask=active_shadow_mask,
                plot_dir=plot_dir,
                image_id_b=image_id_b,
                gt_available=context["gt_available"],
                model_label=active_label,
                boundary_band_px=int(cfg.runtime.plotting.boundary_band_px),
            )
    if plot_cfg.score_threshold:
        with perf_span("save_holdout_plots", substage="score_threshold"):
            save_score_threshold_plot(
                score_map=champion_score,
                threshold=champion_thr,
                sh_buffer_mask=context["sh_buffer_mask"],
                plot_dir=plot_dir,
                image_id_b=image_id_b,
                model_label=active_label,
            )
    if plot_cfg.disagreement_entropy:
        with perf_span("save_holdout_plots", substage="disagreement_entropy"):
            save_disagreement_entropy_plot(
                disagreement_map=disagreement_map,
                entropy_map=entropy_map,
                candidate_mask=proposal_masks["candidate_mask"],
                plot_dir=plot_dir,
                image_id_b=image_id_b,
                model_label=active_label,
            )
    if plot_cfg.proposal_overlay:
        with perf_span("save_holdout_plots", substage="proposal_overlay"):
            save_proposal_overlay_plot(
                img_b=context["img_b"],
                prediction_mask=active_shadow_mask,
                candidate_mask=proposal_masks["candidate_mask"],
                candidate_inside_mask=proposal_masks["candidate_inside_mask"],
                evaluated_outside_mask=proposal_masks["evaluated_outside_mask"],
                accepted_mask=proposal_masks["accepted_mask"],
                accepted_inside_mask=proposal_masks["accepted_inside_mask"],
                accepted_outside_mask=proposal_masks["accepted_outside_mask"],
                rejected_mask=proposal_masks["rejected_mask"],
                proposal_records=proposal_bundle.get("records"),
                plot_dir=plot_dir,
                image_id_b=image_id_b,
                model_label=active_label,
                accept_rgb=tuple(cfg.runtime.plotting.proposal_accept_rgb),
                reject_rgb=tuple(cfg.runtime.plotting.proposal_reject_rgb),
                candidate_rgb=tuple(cfg.runtime.plotting.proposal_candidate_rgb),
            )


def infer_on_holdout(
    holdout_path: str,
    gt_vector_paths: list[str] | None,
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    tuned: dict,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    shape_dir: str,
    plot_dir: str,
    context_radius: int,
    plot_with_metrics: bool = True,
    final_inference_phase: bool = False,
    save_plots: bool = True,
    xgb_guard_state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run inference on a holdout tile using tuned settings.

    Examples:
        >>> callable(infer_on_holdout)
        True
    """
    logger.info("inference: holdout tile %s", holdout_path)
    # Stage 1: load immutable tile context and optional reusable features.
    context = perf_call(
        _load_holdout_tile_context,
        holdout_path,
        gt_vector_paths,
        tuned,
        stage="infer_on_holdout",
        substage="load_context",
    )
    context["streaming_xgb"] = _should_stream_xgb_features(context, feature_dir)
    context["prefetched_b"] = perf_call(
        _prefetch_context_features,
        context,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        feature_dir,
        stage="infer_on_holdout",
        substage="prefetch_features",
        extra={"streaming_xgb": context["streaming_xgb"]},
    )
    if not (context["knn_enabled"] or context["xgb_enabled"]):
        raise ValueError("both kNN and XGB are disabled for inference")

    # Stage 2: score the enabled streams before downstream refinement.
    knn_result = perf_call(
        _compute_knn_stream,
        context,
        tuned,
        model,
        processor,
        device,
        pos_bank,
        neg_bank,
        ps,
        tile_size,
        stride,
        feature_dir,
        context_radius,
        stage="infer_on_holdout",
        substage="knn_stream",
    )
    xgb_result = perf_call(
        _compute_xgb_stream,
        context,
        tuned,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        feature_dir,
        context_radius,
        final_inference_phase,
        xgb_guard_state,
        stage="infer_on_holdout",
        substage="xgb_stream",
    )
    champion_source = _resolve_champion_stream(
        tuned,
        bool(context["knn_enabled"]),
        bool(context["xgb_enabled"]),
    )
    active_stream = "knn" if champion_source == "raw" else "xgb"
    active_label = "kNN" if champion_source == "raw" else "XGB"
    champion_result = knn_result if champion_source == "raw" else xgb_result
    champion_score = champion_result["score"]
    champion_thr = float(champion_result["threshold"])
    active_raw_mask = champion_result["mask"]
    mask_crf_knn, mask_crf_xgb = perf_call(
        _run_crf_stage,
        context,
        tuned,
        knn_result,
        xgb_result,
        stage="infer_on_holdout",
        substage="crf_stage",
    )
    active_crf_mask = mask_crf_knn if champion_source == "raw" else mask_crf_xgb
    shadow_cfg = tuned["shadow_cfg"]
    active_shadow_mask, shadow_mask_knn, shadow_mask_xgb = perf_call(
        _compute_shadow_stage_masks,
        context,
        champion_source,
        champion_score,
        knn_result,
        xgb_result,
        mask_crf_knn,
        mask_crf_xgb,
        shadow_cfg,
        stage="infer_on_holdout",
        substage="shadow_stage",
    )

    # Stage 3: choose champion outputs, derive diagnostics, then export artifacts.
    metrics_map = _build_metrics_map(
        context["gt_mask_eval"],
        float(context["gt_weight"]),
        active_raw_mask,
        active_crf_mask,
        active_shadow_mask,
        bool(context["knn_enabled"]),
        bool(context["xgb_enabled"]),
        knn_result,
        xgb_result,
        mask_crf_knn,
        mask_crf_xgb,
        shadow_mask_knn,
        shadow_mask_xgb,
    )
    proposal_cfg = cfg.postprocess.novel_proposals
    proposal_source_mask = (
        champion_score
        >= (
            float(proposal_cfg.score_threshold)
            if proposal_cfg.score_threshold is not None
            else champion_thr
        )
        if proposal_cfg.source == "champion_score"
        else active_shadow_mask
    )
    proposal_bundle = perf_call(
        filter_novel_proposals,
        active_shadow_mask,
        context["labels_sh"],
        champion_score,
        context["roads_mask"],
        context["pixel_size_m"],
        stage="infer_on_holdout",
        substage="novel_proposals",
        sh_buffer_mask=context["sh_buffer_mask"],
        proposal_source_mask=proposal_source_mask,
    )
    proposal_masks = _build_proposal_masks(
        proposal_bundle,
        proposal_bundle["candidate_mask"],
    )
    _, disagreement_map, entropy_map = _compute_probability_and_diagnostics(
        active_stream,
        champion_score,
        knn_result["score"],
        xgb_result["score"],
    )
    if save_plots:
        perf_call(
            _save_holdout_plots,
            context=context,
            plot_dir=plot_dir,
            plot_with_metrics=plot_with_metrics,
            active_stream=active_stream,
            active_label=active_label,
            champion_score=champion_score,
            champion_thr=champion_thr,
            active_raw_mask=active_raw_mask,
            active_crf_mask=active_crf_mask,
            active_shadow_mask=active_shadow_mask,
            crf_cfg=dict(tuned.get("best_crf_config") or {}),
            metrics_map=metrics_map,
            proposal_masks=proposal_masks,
            proposal_bundle=proposal_bundle,
            score_knn_raw=knn_result["score_raw"],
            disagreement_map=disagreement_map,
            entropy_map=entropy_map,
            stage="infer_on_holdout",
            substage="plot_exports",
        )

    perf_metadata(
        "infer_on_holdout",
        substage="tile_workload_metadata",
        extra=_tile_workload_metadata(context, ps),
    )

    shadow_with_proposals_mask = np.logical_or(
        active_shadow_mask,
        proposal_masks["accepted_mask"],
    )

    return {
        "ref_path": holdout_path,
        "image_id": context["image_id_b"],
        "active_stream": active_stream,
        "gt_available": context["gt_available"],
        "buffer_m": context["buffer_m"],
        "pixel_size_m": context["pixel_size_m"],
        "metrics": metrics_map,
        "masks": {
            "raw": active_raw_mask,
            "crf": active_crf_mask,
            "shadow": active_shadow_mask,
            "shadow_with_proposals": shadow_with_proposals_mask,
        },
        "proposals": proposal_bundle,
    }
