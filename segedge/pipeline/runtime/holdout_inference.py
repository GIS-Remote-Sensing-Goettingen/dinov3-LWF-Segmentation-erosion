"""Per-tile inference for holdout and validation tiles."""

from __future__ import annotations

import csv
import logging
import os

import numpy as np
from scipy.ndimage import median_filter

from ...core.config_loader import cfg
from ...core.crf_utils import refine_with_densecrf
from ...core.features import prefetch_features_single_scale_image
from ...core.io_utils import export_mask_to_shapefile
from ...core.knn import zero_shot_knn_single_scale_B_with_saliency
from ...core.metrics_utils import compute_metrics
from ...core.plotting import (
    save_core_qualitative_plot,
    save_disagreement_entropy_plot,
    save_proposal_overlay_plot,
    save_score_threshold_plot,
    save_unified_plot,
)
from ...core.timing_utils import perf_span
from ...core.xdboost import xgb_score_image_b
from .postprocess import _apply_shadow_filter, filter_novel_proposals
from .roads import _apply_roads_penalty, _get_roads_mask
from .tile_context import load_b_tile_context

logger = logging.getLogger(__name__)


def _load_holdout_tile(
    holdout_path: str,
    gt_vector_paths: list[str] | None,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    tuned: dict,
) -> dict[str, object]:
    """Load tile context, prefetched features, and runtime toggles.

    Examples:
        >>> callable(_load_holdout_tile)
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
    ) = load_b_tile_context(holdout_path, gt_vector_paths)
    gt_available = gt_mask_eval is not None
    if gt_mask_eval is None:
        logger.warning("Holdout has no GT; metrics will be reported as 0.0.")
        gt_mask_eval = np.zeros(img_b.shape[:2], dtype=bool)
    image_id_b = os.path.splitext(os.path.basename(holdout_path))[0]
    knn_enabled = bool(tuned.get("knn_enabled", cfg.search.knn.enabled))
    xgb_enabled = bool(tuned.get("xgb_enabled", cfg.search.xgb.enabled))
    prefetched_b = prefetch_features_single_scale_image(
        img_b,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        None,
        feature_dir,
        image_id_b,
        materialize_cached=bool(knn_enabled or not xgb_enabled),
    )
    ds = int(cfg.model.backbone.resample_factor or 1)
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
        "prefetched_b": prefetched_b,
        "roads_mask": _get_roads_mask(holdout_path, ds, target_shape=img_b.shape[:2]),
        "roads_penalty": float(tuned.get("roads_penalty", 1.0)),
        "xgb_feature_stats": tuned.get("xgb_feature_stats"),
        "knn_enabled": knn_enabled,
        "xgb_enabled": xgb_enabled,
        "crf_enabled": bool(tuned.get("crf_enabled", cfg.search.crf.enabled)),
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
    mask = (score_penalized >= threshold) & sh_buffer_mask
    mask = median_filter(mask.astype(np.uint8), size=3) > 0
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
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    final_inference_phase: bool,
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
    score_map = xgb_score_image_b(
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
    mask = (score_penalized >= threshold) & sh_buffer_mask
    mask = median_filter(mask.astype(np.uint8), size=3) > 0
    output.update(
        {
            "score": score_penalized,
            "threshold": threshold,
            "mask": mask,
            "metrics": compute_metrics(mask, gt_mask_eval),
        }
    )
    return output


def _apply_inference_score_prior(
    score_map: np.ndarray,
    labels_sh: np.ndarray,
    image_id_b: str,
    *,
    final_inference_phase: bool,
) -> np.ndarray:
    """Apply the manual inference-only score prior inside source-label pixels.

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
    if not np.any(inside_mask):
        return score_map
    score_with_prior = score_map.astype(np.float32, copy=True)
    score_before = score_with_prior[inside_mask]
    score_with_prior[inside_mask] *= float(prior_cfg.factor)
    score_with_prior = np.clip(score_with_prior, 0.0, float(prior_cfg.clip_max))
    score_after = score_with_prior[inside_mask]
    logger.info(
        "inference score prior: tile=%s factor=%.3f boosted_pixels=%s mean_before=%.4f mean_after=%.4f",
        image_id_b,
        float(prior_cfg.factor),
        int(inside_mask.sum()),
        float(score_before.mean()),
        float(score_after.mean()),
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


def _export_proposal_artifacts(
    proposal_bundle: dict,
    proposal_masks: dict[str, np.ndarray],
    holdout_path: str,
    image_id_b: str,
    shape_dir: str,
) -> None:
    """Write proposal shapefiles and CSV exports.

    Examples:
        >>> callable(_export_proposal_artifacts)
        True
    """
    if not cfg.postprocess.novel_proposals.enabled:
        return
    proposal_dir = os.path.join(shape_dir, "proposals")
    os.makedirs(proposal_dir, exist_ok=True)
    if proposal_masks["accepted_mask"].any():
        export_mask_to_shapefile(
            proposal_masks["accepted_mask"],
            holdout_path,
            os.path.join(proposal_dir, f"{image_id_b}_accepted.shp"),
        )
    if proposal_masks["rejected_mask"].any():
        export_mask_to_shapefile(
            proposal_masks["rejected_mask"],
            holdout_path,
            os.path.join(proposal_dir, f"{image_id_b}_rejected.shp"),
        )
    records_path = os.path.join(proposal_dir, f"{image_id_b}_proposal_records.csv")
    records = list(proposal_bundle.get("records", []))
    fieldnames = [
        "component_id",
        "zone",
        "accepted",
        "acceptance_score",
        "reject_reasons",
        "area_px",
        "length_m",
        "mean_width_m",
        "skeleton_ratio",
        "pca_ratio",
        "circularity",
        "mean_score",
        "road_overlap",
        "centroid_row",
        "centroid_col",
    ]
    with open(records_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "component_id": rec.get("component_id"),
                    "zone": rec.get("zone"),
                    "accepted": rec.get("accepted"),
                    "acceptance_score": rec.get("acceptance_score"),
                    "reject_reasons": ";".join(rec.get("reject_reasons", [])),
                    "area_px": rec.get("area_px"),
                    "length_m": rec.get("length_m"),
                    "mean_width_m": rec.get("mean_width_m"),
                    "skeleton_ratio": rec.get("skeleton_ratio"),
                    "pca_ratio": rec.get("pca_ratio"),
                    "circularity": rec.get("circularity"),
                    "mean_score": rec.get("mean_score"),
                    "road_overlap": rec.get("road_overlap"),
                    "centroid_row": rec.get("centroid_row"),
                    "centroid_col": rec.get("centroid_col"),
                }
            )


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
    save_unified_plot(
        img_b=context["img_b"],
        gt_mask=context["gt_mask_eval"],
        labels_sh=context["labels_sh"],
        masks={
            f"{active_stream}_raw": active_raw_mask,
            f"{active_stream}_crf": active_crf_mask,
            f"{active_stream}_shadow": active_shadow_mask,
        },
        metrics=metrics_map,
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        show_metrics=plot_with_metrics and context["gt_available"],
        gt_available=context["gt_available"],
        similarity_map=score_knn_raw if active_stream == "knn" else None,
        score_maps={active_stream: champion_score},
        proposal_masks={
            "candidate": proposal_masks["candidate_mask"],
            "accepted": proposal_masks["accepted_mask"],
            "rejected": proposal_masks["rejected_mask"],
        },
    )
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
    save_score_threshold_plot(
        score_map=champion_score,
        threshold=champion_thr,
        sh_buffer_mask=context["sh_buffer_mask"],
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        model_label=active_label,
    )
    save_disagreement_entropy_plot(
        disagreement_map=disagreement_map,
        entropy_map=entropy_map,
        candidate_mask=proposal_masks["candidate_mask"],
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        model_label=active_label,
    )
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
) -> dict[str, object]:
    """Run inference on a holdout tile using tuned settings.

    Examples:
        >>> callable(infer_on_holdout)
        True
    """
    logger.info("inference: holdout tile %s", holdout_path)
    with perf_span("infer_on_holdout", substage="load_context"):
        context = _load_holdout_tile(
            holdout_path,
            gt_vector_paths,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            feature_dir,
            tuned,
        )
    if not (context["knn_enabled"] or context["xgb_enabled"]):
        raise ValueError("both kNN and XGB are disabled for inference")
    with perf_span("infer_on_holdout", substage="knn_stream"):
        knn_result = _compute_knn_stream(
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
        )
    with perf_span("infer_on_holdout", substage="xgb_stream"):
        xgb_result = _compute_xgb_stream(
            context,
            tuned,
            ps,
            tile_size,
            stride,
            feature_dir,
            context_radius,
            final_inference_phase,
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
    with perf_span("infer_on_holdout", substage="crf_stage"):
        mask_crf_knn, mask_crf_xgb = _run_crf_stage(
            context,
            tuned,
            knn_result,
            xgb_result,
        )
    active_crf_mask = mask_crf_knn if champion_source == "raw" else mask_crf_xgb
    shadow_cfg = tuned["shadow_cfg"]
    with perf_span("infer_on_holdout", substage="shadow_stage"):
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
    with perf_span("infer_on_holdout", substage="novel_proposals"):
        proposal_bundle = filter_novel_proposals(
            active_shadow_mask,
            context["labels_sh"],
            champion_score,
            context["roads_mask"],
            context["pixel_size_m"],
            sh_buffer_mask=context["sh_buffer_mask"],
            proposal_source_mask=proposal_source_mask,
        )
    proposal_masks = _build_proposal_masks(
        proposal_bundle,
        proposal_bundle["candidate_mask"],
    )
    with perf_span("infer_on_holdout", substage="proposal_exports"):
        _export_proposal_artifacts(
            proposal_bundle,
            proposal_masks,
            holdout_path,
            context["image_id_b"],
            shape_dir,
        )
    _, disagreement_map, entropy_map = _compute_probability_and_diagnostics(
        active_stream,
        champion_score,
        knn_result["score"],
        xgb_result["score"],
    )
    with perf_span("infer_on_holdout", substage="plot_exports"):
        _save_holdout_plots(
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
            metrics_map=metrics_map,
            proposal_masks=proposal_masks,
            proposal_bundle=proposal_bundle,
            score_knn_raw=knn_result["score_raw"],
            disagreement_map=disagreement_map,
            entropy_map=entropy_map,
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
            f"{active_stream}_raw": active_raw_mask,
            f"{active_stream}_crf": active_crf_mask,
            f"{active_stream}_shadow": active_shadow_mask,
        },
        "proposals": proposal_bundle,
    }
