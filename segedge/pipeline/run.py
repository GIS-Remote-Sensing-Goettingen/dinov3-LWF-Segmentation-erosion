"""Primary pipeline entrypoint for SegEdge."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import binary_dilation, disk

import config as cfg

from ..core.banks import build_banks_single_scale
from ..core.continuity import bridge_skeleton_gaps, skeletonize_with_endpoints
from ..core.crf_utils import refine_with_densecrf
from ..core.explainability import (
    append_xai_summary_csv,
    build_dim_activation_map,
    build_xai_summary_row,
    distribution_stats,
    emit_training_xgb_pca_plot,
    get_xgb_importance_dict,
    save_xai_tile_plot,
    select_holdout_tiles_for_xai,
    summarize_knn_signals,
    summarize_tile_latent_dims,
    write_xai_tile_json,
)
from ..core.features import prefetch_features_single_scale_image
from ..core.io_utils import (
    append_mask_to_union_shapefile,
    backup_union_shapefile,
    consolidate_features_for_image,
    count_shapefile_features,
    export_best_settings,
    export_run_summary,
    load_dop20_image,
)
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from ..core.logging_utils import setup_logging
from ..core.metrics_utils import compute_metrics
from ..core.plotting import save_unified_plot
from ..core.run_config_logging import log_training_ablation_summary
from ..core.summary_utils import phase_delta_summary as _phase_delta_summary
from ..core.summary_utils import phase_metrics_summary as _phase_metrics_summary
from ..core.summary_utils import timing_summary as _timing_summary
from ..core.summary_utils import weighted_mean as _weighted_mean
from ..core.timing_csv import (
    append_tile_timing_csv_rows,
    build_tile_timing_rows,
    read_timing_detail_csv,
    write_timing_summary_csv,
)
from ..core.timing_utils import time_end, time_start
from ..core.xdboost import build_xgb_dataset, xgb_score_image_b
from .common import (
    AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT,
    AUTO_SPLIT_MODE_LEGACY,
    init_model,
    resolve_source_training_labels,
    resolve_tile_splits_from_gt,
    run_source_validation_anti_leak_checks,
    tile_has_gt_overlap,
)
from .inference_utils import (
    _apply_roads_penalty,
    _apply_shadow_filter,
    _compute_top_p,
    _get_roads_mask,
    _top_p_threshold,
    load_b_tile_context,
)
from .tuning import tune_on_validation_multi

# Config-driven flags
USE_FP16_KNN = getattr(cfg, "USE_FP16_KNN", True)
logger = logging.getLogger(__name__)


def _log_phase(kind: str, name: str) -> None:
    """Log a phase marker with ANSI color.

    Args:
        kind (str): Phase kind.
        name (str): Phase name.

    Examples:
        >>> isinstance(_log_phase.__name__, str)
        True
    """
    msg = f"PHASE {kind}: {name}".upper()
    logger.info("\033[31m%s\033[0m", msg)


def _update_phase_metrics(acc: dict[str, list[dict]], metrics_map: dict) -> None:
    for key, metrics in metrics_map.items():
        acc.setdefault(key, []).append(metrics)


def _summarize_phase_metrics(
    acc: dict[str, list[dict]], label: str, bridge_enabled: bool
) -> None:
    if not acc:
        logger.info("summary %s: no metrics", label)
        return
    metric_keys = ["iou", "f1", "precision", "recall"]
    phase_order = [
        "knn_raw",
        "knn_crf",
        "knn_shadow",
        "xgb_raw",
        "xgb_crf",
        "xgb_shadow",
        "silver_core",
        "champion_raw",
        "champion_crf",
    ]
    if bridge_enabled:
        phase_order.append("champion_bridge")
    phase_order.append("champion_shadow")

    logger.info("summary %s: phase metrics", label)
    for phase in phase_order:
        if phase not in acc or not acc[phase]:
            continue
        weights = [float(m.get("_weight", 0.0)) for m in acc[phase]]
        vals = {k: [m.get(k, 0.0) for m in acc[phase]] for k in metric_keys}
        mean_vals = {k: _weighted_mean(v, weights) for k, v in vals.items()}
        med_vals = {k: float(np.median(v)) for k, v in vals.items()}
        logger.info(
            "summary %s %s wmean IoU=%.3f F1=%.3f P=%.3f R=%.3f | median IoU=%.3f F1=%.3f",
            label,
            phase,
            mean_vals["iou"],
            mean_vals["f1"],
            mean_vals["precision"],
            mean_vals["recall"],
            med_vals["iou"],
            med_vals["f1"],
        )

    champ_chain = ["champion_raw", "champion_crf"]
    if bridge_enabled:
        champ_chain.append("champion_bridge")
    champ_chain.append("champion_shadow")
    for prev, curr in zip(champ_chain, champ_chain[1:]):
        if prev not in acc or curr not in acc:
            continue
        prev_weights = [float(m.get("_weight", 0.0)) for m in acc[prev]]
        curr_weights = [float(m.get("_weight", 0.0)) for m in acc[curr]]
        prev_iou = _weighted_mean([m.get("iou", 0.0) for m in acc[prev]], prev_weights)
        curr_iou = _weighted_mean([m.get("iou", 0.0) for m in acc[curr]], curr_weights)
        prev_f1 = _weighted_mean([m.get("f1", 0.0) for m in acc[prev]], prev_weights)
        curr_f1 = _weighted_mean([m.get("f1", 0.0) for m in acc[curr]], curr_weights)
        logger.info(
            "summary %s delta %s→%s IoU=%.3f F1=%.3f",
            label,
            prev,
            curr,
            float(curr_iou - prev_iou),
            float(curr_f1 - prev_f1),
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
    context_radius: int,
    plot_with_metrics: bool = True,
    xai_options: dict | None = None,
):
    """Run inference on a holdout tile using tuned settings.

    Args:
        holdout_path (str): Holdout tile path.
        gt_vector_paths (list[str] | None): Vector GT paths.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        pos_bank (np.ndarray): Positive bank.
        neg_bank (np.ndarray | None): Negative bank.
        tuned (dict): Tuned configuration bundle.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str | None): Feature cache directory.
        shape_dir (str): Output shapefile directory.
        context_radius (int): Feature context radius.
        plot_with_metrics (bool): Whether to show metrics on plots.
        xai_options (dict | None): Optional explainability configuration.

    Returns:
        dict: Masks, metrics, and metadata for the tile.

    Examples:
        >>> isinstance(infer_on_holdout.__name__, str)
        True
    """
    logger.info("inference: holdout tile %s", holdout_path)
    timings: dict[str, float] = {}
    t0_total = time.perf_counter()
    t0 = time.perf_counter()
    (
        img_b,
        labels_sh,
        _,
        gt_mask_eval,
        sh_buffer_mask,
        buffer_m,
        pixel_size_m,
    ) = load_b_tile_context(holdout_path, gt_vector_paths)
    timings["load_context_s"] = time.perf_counter() - t0
    gt_available = gt_mask_eval is not None
    if gt_mask_eval is None:
        logger.warning("Holdout has no GT; metrics will be reported as 0.0.")
        gt_mask_eval = np.zeros(img_b.shape[:2], dtype=bool)
    gt_weight = float(gt_mask_eval.sum())
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    t0 = time.perf_counter()
    roads_mask = _get_roads_mask(holdout_path, ds, target_shape=img_b.shape[:2])
    timings["roads_mask_s"] = time.perf_counter() - t0
    roads_penalty = float(tuned.get("roads_penalty", 1.0))

    image_id_b = os.path.splitext(os.path.basename(holdout_path))[0]
    t0 = time.perf_counter()
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
    )
    timings["prefetch_features_s"] = time.perf_counter() - t0

    k = tuned["best_raw_config"]["k"]
    tuned_neg_alpha = float(tuned.get("neg_alpha", getattr(cfg, "NEG_ALPHA", 1.0)))
    top_p_a = float(tuned.get("top_p_a", getattr(cfg, "TOP_P_A", 0.0)))
    top_p_b = float(tuned.get("top_p_b", getattr(cfg, "TOP_P_B", 0.05)))
    top_p_min = float(tuned.get("top_p_min", getattr(cfg, "TOP_P_MIN", 0.02)))
    top_p_max = float(tuned.get("top_p_max", getattr(cfg, "TOP_P_MAX", 0.08)))
    buffer_density = float(sh_buffer_mask.mean())
    top_p_val = _compute_top_p(buffer_density, top_p_a, top_p_b, top_p_min, top_p_max)
    t0 = time.perf_counter()
    score_knn, saliency_knn = zero_shot_knn_single_scale_B_with_saliency(
        img_b,
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
        image_id=image_id_b,
        neg_alpha=tuned_neg_alpha,
        prefetched_tiles=prefetched_b,
        use_fp16_matmul=USE_FP16_KNN,
        context_radius=context_radius,
    )
    timings["knn_score_s"] = time.perf_counter() - t0
    score_knn_raw = score_knn
    t0 = time.perf_counter()
    score_knn = _apply_roads_penalty(score_knn, roads_mask, roads_penalty)
    knn_thr, mask_knn = _top_p_threshold(score_knn, sh_buffer_mask, top_p_val)
    mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0
    timings["knn_threshold_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics_knn = compute_metrics(mask_knn, gt_mask_eval)
    timings["knn_metrics_s"] = time.perf_counter() - t0
    timings["knn_s"] = (
        timings["knn_score_s"] + timings["knn_threshold_s"] + timings["knn_metrics_s"]
    )

    bst = tuned["bst"]
    t0 = time.perf_counter()
    score_xgb = xgb_score_image_b(
        img_b,
        bst,
        ps,
        tile_size,
        stride,
        feature_dir,
        image_id_b,
        prefetched_tiles=prefetched_b,
        context_radius=context_radius,
    )
    timings["xgb_score_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    score_xgb = _apply_roads_penalty(score_xgb, roads_mask, roads_penalty)
    xgb_thr, mask_xgb = _top_p_threshold(score_xgb, sh_buffer_mask, top_p_val)
    mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
    timings["xgb_threshold_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics_xgb = compute_metrics(mask_xgb, gt_mask_eval)
    timings["xgb_metrics_s"] = time.perf_counter() - t0
    timings["xgb_s"] = (
        timings["xgb_score_s"] + timings["xgb_threshold_s"] + timings["xgb_metrics_s"]
    )

    t0 = time.perf_counter()
    core_mask = np.logical_and(mask_knn, mask_xgb)
    dilate_px = int(
        tuned.get(
            "silver_core_dilate_px",
            getattr(cfg, "SILVER_CORE_DILATE_PX", 1),
        )
    )
    if dilate_px > 0:
        core_mask = binary_dilation(core_mask, disk(dilate_px))
    metrics_core = compute_metrics(core_mask, gt_mask_eval)
    timings["silver_core_s"] = time.perf_counter() - t0

    champion_source = tuned["champion_source"]
    if champion_source == "raw":
        champion_score = score_knn
    else:
        champion_score = score_xgb

    crf_cfg = tuned["best_crf_config"]
    t0 = time.perf_counter()
    mask_crf_knn, prob_crf_knn = refine_with_densecrf(
        img_b,
        score_knn,
        knn_thr,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
        return_prob=True,
    )
    timings["crf_knn_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mask_crf_xgb, prob_crf_xgb = refine_with_densecrf(
        img_b,
        score_xgb,
        xgb_thr,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
        return_prob=True,
    )
    timings["crf_xgb_s"] = time.perf_counter() - t0
    timings["crf_s"] = timings["crf_knn_s"] + timings["crf_xgb_s"]
    if champion_source == "raw":
        best_crf_mask = mask_crf_knn
    else:
        best_crf_mask = mask_crf_xgb

    bridge_enabled = bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False))
    bridge_mask = best_crf_mask
    if bridge_enabled:
        t0 = time.perf_counter()
        prob_crf = prob_crf_knn if champion_source == "raw" else prob_crf_xgb
        bridge_cfg = tuned.get("best_bridge_config", {}) or {}
        bridge_mask = bridge_skeleton_gaps(
            best_crf_mask,
            prob_crf,
            max_gap_px=int(
                bridge_cfg.get(
                    "bridge_max_gap_px", getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)
                )
            ),
            max_pairs_per_endpoint=int(
                bridge_cfg.get("bridge_max_pairs", getattr(cfg, "BRIDGE_MAX_PAIRS", 3))
            ),
            max_avg_cost=float(
                bridge_cfg.get(
                    "bridge_max_avg_cost",
                    getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0),
                )
            ),
            bridge_width_px=int(
                bridge_cfg.get("bridge_width_px", getattr(cfg, "BRIDGE_WIDTH_PX", 2))
            ),
            min_component_area_px=int(
                bridge_cfg.get(
                    "bridge_min_component_px",
                    getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300),
                )
            ),
            spur_prune_iters=int(
                bridge_cfg.get(
                    "bridge_spur_prune_iters",
                    getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15),
                )
            ),
        )
        timings["bridge_s"] = time.perf_counter() - t0

    shadow_cfg = tuned["shadow_cfg"]
    protect_score = shadow_cfg.get("protect_score")
    t0 = time.perf_counter()
    shadow_mask = _apply_shadow_filter(
        img_b,
        bridge_mask,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        champion_score,
        protect_score,
    )
    shadow_mask_knn = _apply_shadow_filter(
        img_b,
        mask_crf_knn,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        score_knn,
        protect_score,
    )
    shadow_mask_xgb = _apply_shadow_filter(
        img_b,
        mask_crf_xgb,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        score_xgb,
        protect_score,
    )
    timings["shadow_s"] = time.perf_counter() - t0
    metrics_knn_crf = compute_metrics(mask_crf_knn, gt_mask_eval)
    metrics_knn_shadow = compute_metrics(shadow_mask_knn, gt_mask_eval)
    metrics_xgb_crf = compute_metrics(mask_crf_xgb, gt_mask_eval)
    metrics_xgb_shadow = compute_metrics(shadow_mask_xgb, gt_mask_eval)
    champ_raw_mask = mask_knn if champion_source == "raw" else mask_xgb
    metrics_champion_raw = compute_metrics(champ_raw_mask, gt_mask_eval)
    metrics_champion_crf = compute_metrics(best_crf_mask, gt_mask_eval)
    metrics_champion_bridge = compute_metrics(bridge_mask, gt_mask_eval)
    shadow_metrics = compute_metrics(shadow_mask, gt_mask_eval)

    t0 = time.perf_counter()
    skel, endpoints = skeletonize_with_endpoints(bridge_mask)
    timings["skeleton_s"] = time.perf_counter() - t0
    metrics_map = {
        "knn_raw": metrics_knn,
        "knn_crf": metrics_knn_crf,
        "knn_shadow": metrics_knn_shadow,
        "xgb_raw": metrics_xgb,
        "xgb_crf": metrics_xgb_crf,
        "xgb_shadow": metrics_xgb_shadow,
        "silver_core": metrics_core,
        "champion_raw": metrics_champion_raw,
        "champion_crf": metrics_champion_crf,
        "champion_bridge": metrics_champion_bridge,
        "champion_shadow": shadow_metrics,
    }
    metrics_map = {
        key: {**metrics, "_weight": gt_weight} for key, metrics in metrics_map.items()
    }
    masks_map = {
        "knn_raw": mask_knn,
        "knn_crf": mask_crf_knn,
        "knn_shadow": shadow_mask_knn,
        "xgb_raw": mask_xgb,
        "xgb_crf": mask_crf_xgb,
        "xgb_shadow": shadow_mask_xgb,
        "silver_core": core_mask,
        "champion_raw": champ_raw_mask,
        "champion_crf": best_crf_mask,
        "champion_bridge": bridge_mask,
        "champion_shadow": shadow_mask,
    }
    t0 = time.perf_counter()
    save_unified_plot(
        img_b=img_b,
        gt_mask=gt_mask_eval,
        labels_sh=labels_sh,
        masks=masks_map,
        metrics=metrics_map,
        plot_dir=cfg.PLOT_DIR,
        image_id_b=image_id_b,
        show_metrics=plot_with_metrics and gt_available,
        gt_available=gt_available,
        similarity_map=score_knn_raw,
        score_maps={"knn": score_knn, "xgb": score_xgb},
        skeleton=skel,
        endpoints=endpoints,
        bridge_enabled=bridge_enabled,
    )
    timings["plot_s"] = time.perf_counter() - t0

    xai_cfg = xai_options or {}
    xai_enabled = bool(xai_cfg.get("enabled", False))
    if xai_enabled:
        include_xgb = bool(xai_cfg.get("include_xgb", True))
        include_knn = bool(xai_cfg.get("include_knn", True))
        top_features = int(xai_cfg.get("top_features", 20) or 20)
        top_patches = int(xai_cfg.get("top_patches", 50) or 50)
        stage_name = str(xai_cfg.get("stage", "holdout")).strip().lower()
        xai_root = str(xai_cfg.get("xai_dir", cfg.PLOT_DIR))
        summary_csv_path = xai_cfg.get("summary_csv_path")
        save_json = bool(xai_cfg.get("save_json", True))
        save_plots = bool(xai_cfg.get("save_plots", True))

        t0 = time.perf_counter()
        xgb_importance = (
            get_xgb_importance_dict(tuned.get("bst"), top_k=top_features)
            if include_xgb
            else {"top_dims": [], "importance": [], "xgb_gain_share_top5": 0.0}
        )
        latent_summary = summarize_tile_latent_dims(
            prefetched_b,
            xgb_importance["top_dims"],
            top_patches=top_patches,
        )
        dim_activation_map = (
            build_dim_activation_map(
                prefetched_b,
                img_b.shape[:2],
                latent_summary.get("selected_dims", []),
            )
            if include_xgb
            else np.zeros(img_b.shape[:2], dtype=np.float32)
        )
        knn_summary = (
            summarize_knn_signals(
                score_knn,
                saliency_knn,
                sh_buffer_mask,
                threshold=knn_thr,
            )
            if include_knn
            else {
                "score_distribution": distribution_stats(score_knn),
                "saliency_distribution": None,
                "buffered_score_mean": 0.0,
                "buffered_positive_ratio": None,
            }
        )
        xgb_score_stats = distribution_stats(score_xgb)
        timings["xai_prepare_s"] = time.perf_counter() - t0

        xai_stage_dir = os.path.join(xai_root, stage_name)
        os.makedirs(xai_stage_dir, exist_ok=True)
        payload = {
            "run_dir": os.path.dirname(cfg.PLOT_DIR),
            "stage": stage_name,
            "tile_path": holdout_path,
            "image_id": image_id_b,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "xgb": {
                "global_feature_importance": xgb_importance["importance"],
                "top_dims": xgb_importance["top_dims"],
                "xgb_gain_share_top5": xgb_importance["xgb_gain_share_top5"],
                "tile_feature_activation_stats": latent_summary,
                "score_distribution": xgb_score_stats,
            },
            "knn": knn_summary,
            "masks": {
                "knn_raw_nonzero": int(np.count_nonzero(mask_knn)),
                "xgb_raw_nonzero": int(np.count_nonzero(mask_xgb)),
                "champion_raw_nonzero": int(np.count_nonzero(champ_raw_mask)),
                "champion_shadow_nonzero": int(np.count_nonzero(shadow_mask)),
            },
            "runtime": {},
        }

        t0 = time.perf_counter()
        if save_plots:
            xai_plot_path = os.path.join(xai_stage_dir, f"{image_id_b}_xai.png")
            save_xai_tile_plot(
                img_b=img_b,
                score_xgb=score_xgb,
                score_knn=score_knn,
                dim_activation_map=dim_activation_map,
                champion_mask=shadow_mask,
                sh_buffer_mask=sh_buffer_mask,
                out_path=xai_plot_path,
                title_suffix=stage_name,
            )
            payload["xai_plot_path"] = xai_plot_path
        timings["xai_plot_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        xai_json_path = None
        payload["runtime"] = {
            "xai_prepare_s": float(timings["xai_prepare_s"]),
            "xai_plot_s": float(timings["xai_plot_s"]),
        }
        if save_json:
            xai_json_path = os.path.join(xai_stage_dir, f"{image_id_b}.json")
            payload["xai_json_path"] = xai_json_path
            write_xai_tile_json(xai_json_path, payload)
        timings["xai_write_s"] = time.perf_counter() - t0
        timings["xai_total_s"] = (
            timings.get("xai_prepare_s", 0.0)
            + timings.get("xai_plot_s", 0.0)
            + timings.get("xai_write_s", 0.0)
        )
        payload["runtime"] = {
            "xai_prepare_s": float(timings["xai_prepare_s"]),
            "xai_plot_s": float(timings["xai_plot_s"]),
            "xai_write_s": float(timings["xai_write_s"]),
            "xai_total_s": float(timings["xai_total_s"]),
        }
        if summary_csv_path:
            summary_row = build_xai_summary_row(
                stage=stage_name,
                image_id=image_id_b,
                tile_path=holdout_path,
                top_dims=xgb_importance["top_dims"],
                xgb_gain_share_top5=xgb_importance["xgb_gain_share_top5"],
                knn_score_stats=knn_summary["score_distribution"],
                xgb_score_stats=xgb_score_stats,
                champion_mask=shadow_mask,
                xai_total_s=float(timings["xai_total_s"]),
            )
            append_xai_summary_csv(str(summary_csv_path), summary_row)
    timings["total_s"] = time.perf_counter() - t0_total

    champ_raw_mask = mask_knn if champion_source == "raw" else mask_xgb
    return {
        "ref_path": holdout_path,
        "image_id": image_id_b,
        "gt_available": gt_available,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
        "metrics": metrics_map,
        "timings": timings,
        "masks": {
            "knn_raw": mask_knn,
            "knn_crf": mask_crf_knn,
            "knn_shadow": shadow_mask_knn,
            "xgb_raw": mask_xgb,
            "xgb_crf": mask_crf_xgb,
            "xgb_shadow": shadow_mask_xgb,
            "silver_core": core_mask,
            "champion_raw": champ_raw_mask,
            "champion_crf": best_crf_mask,
            "champion_bridge": bridge_mask,
            "champion_shadow": shadow_mask,
        },
    }


def main():
    """Run the full segmentation pipeline for configured tiles.

    Examples:
        >>> isinstance(main.__name__, str)
        True
    """

    t0_main = time_start()
    run_start = time.perf_counter()
    model_name = cfg.MODEL_NAME

    # ------------------------------------------------------------
    # Output organization (one folder per run)
    # ------------------------------------------------------------
    output_root = getattr(cfg, "OUTPUT_DIR", "output")
    os.makedirs(output_root, exist_ok=True)
    resume_run = bool(getattr(cfg, "RESUME_RUN", False))
    resume_dir = getattr(cfg, "RESUME_RUN_DIR", None)
    if resume_run:
        if not resume_dir:
            raise ValueError("RESUME_RUN_DIR must be set when RESUME_RUN=True")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"RESUME_RUN_DIR not found: {resume_dir}")
        run_dir = resume_dir
        logger.info("resume run: %s", run_dir)
    else:
        existing = sorted(d for d in os.listdir(output_root) if d.startswith("run_"))
        next_idx = 1
        if existing:
            try:
                next_idx = (
                    max(
                        int(d.split("_")[1])
                        for d in existing
                        if d.split("_")[1].isdigit()
                    )
                    + 1
                )
            except ValueError:
                next_idx = len(existing) + 1
        run_dir = os.path.join(output_root, f"run_{next_idx:03d}")
    plot_dir = os.path.join(run_dir, "plots")
    shape_dir = os.path.join(run_dir, "shapes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(shape_dir, exist_ok=True)
    cfg.PLOT_DIR = plot_dir
    cfg.BEST_SETTINGS_PATH = os.path.join(run_dir, "best_settings.yml")
    cfg.LOG_PATH = os.path.join(run_dir, "run.log")
    log_level = getattr(cfg, "LOG_LEVEL", logging.INFO)
    setup_logging(getattr(cfg, "LOG_PATH", None), level=log_level)
    logger.info("logging configured: level=%s", log_level)
    timing_csv_enabled = bool(getattr(cfg, "TIMING_CSV_ENABLED", True))
    timing_csv_path = os.path.join(
        run_dir, str(getattr(cfg, "TIMING_CSV_FILENAME", "tile_phase_timing.csv"))
    )
    timing_summary_csv_path = os.path.join(
        run_dir,
        str(getattr(cfg, "TIMING_SUMMARY_CSV_FILENAME", "timing_opportunity_cost.csv")),
    )
    timing_flush_every = max(1, int(getattr(cfg, "TIMING_CSV_FLUSH_EVERY", 1) or 1))
    timing_rows: list[dict[str, object]] = []
    pending_timing_rows: list[dict[str, object]] = []
    if resume_run and timing_csv_enabled:
        timing_rows = read_timing_detail_csv(timing_csv_path)
    processed_log_path = os.path.join(run_dir, "processed_tiles.jsonl")
    processed_tiles: set[str] = set()
    if resume_run and os.path.exists(processed_log_path):
        with open(processed_log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "done" and record.get("tile_path"):
                    processed_tiles.add(record["tile_path"])
        logger.info("resume: loaded %s processed tiles", len(processed_tiles))

    union_backup_every = int(getattr(cfg, "UNION_BACKUP_EVERY", 10) or 0)
    union_root = os.path.join(shape_dir, "unions")
    union_variants_by_stream = {
        "knn": ["raw", "crf", "shadow"],
        "xgb": ["raw", "crf", "shadow"],
        "champion": ["raw", "crf", "shadow"],
        "silver_core": ["raw"],
    }
    union_states: dict[tuple[str, str], dict[str, str | int]] = {}
    for stream, variants in union_variants_by_stream.items():
        for variant in variants:
            union_dir = os.path.join(union_root, stream, variant)
            os.makedirs(union_dir, exist_ok=True)
            union_path = os.path.join(union_dir, "union.shp")
            feature_id = count_shapefile_features(union_path) if resume_run else 0
            union_states[(stream, variant)] = {
                "path": union_path,
                "backup_dir": os.path.join(union_dir, "backups"),
                "feature_id": feature_id,
            }
            if resume_run and feature_id:
                logger.info(
                    "resume union: %s/%s features=%s",
                    stream,
                    variant,
                    feature_id,
                )
    logger.info(
        "rolling unions: root=%s backup_every=%s",
        union_root,
        union_backup_every,
    )

    def _append_union(
        stream: str, variant: str, mask, ref_path: str, step: int
    ) -> None:
        state = union_states[(stream, variant)]
        union_path = str(state["path"])
        backup_dir = str(state["backup_dir"])
        feature_id = int(state["feature_id"])
        state["feature_id"] = append_mask_to_union_shapefile(
            mask,
            ref_path,
            union_path,
            start_id=feature_id,
        )
        if union_backup_every > 0 and step % union_backup_every == 0:
            backup_union_shapefile(union_path, backup_dir, step)

    # ------------------------------------------------------------
    # Init DINOv3 model & processor
    # ------------------------------------------------------------
    _log_phase("START", "init_model")
    model, processor, device = init_model(model_name)
    _log_phase("END", "init_model")
    ps = getattr(cfg, "PATCH_SIZE", model.config.patch_size)
    tile_size = getattr(cfg, "TILE_SIZE", 1024)
    stride = getattr(cfg, "STRIDE", tile_size)

    # ------------------------------------------------------------
    # Resolve paths to imagery + SH_2022 + GT vector labels
    # ------------------------------------------------------------
    source_tile_default = cfg.SOURCE_TILE
    source_label_raster = cfg.SOURCE_LABEL_RASTER
    gt_vector_paths = cfg.EVAL_GT_VECTORS
    auto_split_tiles = getattr(cfg, "AUTO_SPLIT_TILES", False)
    auto_split_mode_runtime = "disabled"
    source_mode = "manual"

    # ------------------------------------------------------------
    # Resolve one or more labeled source images (Image A list)
    # ------------------------------------------------------------
    if auto_split_tiles:
        tiles_dir = getattr(cfg, "TILES_DIR", "data/tiles")
        tile_glob = getattr(cfg, "TILE_GLOB", "*.tif")
        split_mode = str(
            getattr(cfg, "AUTO_SPLIT_MODE", AUTO_SPLIT_MODE_LEGACY)
        ).strip()
        auto_split_mode_runtime = split_mode
        val_fraction = float(getattr(cfg, "VAL_SPLIT_FRACTION", 0.2))
        seed = int(getattr(cfg, "SPLIT_SEED", 42))
        downsample_factor = getattr(cfg, "GT_PRESENCE_DOWNSAMPLE", None)
        num_workers = getattr(cfg, "GT_PRESENCE_WORKERS", None)
        split_source_tiles, val_tiles, holdout_tiles = resolve_tile_splits_from_gt(
            tiles_dir,
            tile_glob,
            gt_vector_paths,
            val_fraction,
            seed,
            downsample_factor=downsample_factor,
            num_workers=num_workers,
            mode=split_mode,
            inference_tile_cap_enabled=getattr(
                cfg, "INFERENCE_TILE_CAP_ENABLED", False
            ),
            inference_tile_cap=getattr(cfg, "INFERENCE_TILE_CAP", None),
            inference_tile_cap_seed=getattr(cfg, "INFERENCE_TILE_CAP_SEED", seed),
        )
        if split_mode == AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT:
            img_a_paths = getattr(cfg, "SOURCE_TILES", None) or [source_tile_default]
            source_mode = "manual"
            if not img_a_paths:
                raise ValueError(
                    "SOURCE_TILES must be set when "
                    "AUTO_SPLIT_MODE='gt_to_val_cap_holdout'"
                )
            logger.info(
                "auto split tiles mode=%s: source(manual)=%s val(gt)=%s holdout=%s",
                split_mode,
                len(img_a_paths),
                len(val_tiles),
                len(holdout_tiles),
            )
        else:
            img_a_paths = split_source_tiles
            source_mode = "auto"
            logger.info(
                "auto split tiles mode=%s: source=%s val=%s holdout=%s",
                split_mode,
                len(img_a_paths),
                len(val_tiles),
                len(holdout_tiles),
            )
    else:
        img_a_paths = getattr(cfg, "SOURCE_TILES", None) or [source_tile_default]
        val_tiles = cfg.VAL_TILES
        holdout_tiles = cfg.HOLDOUT_TILES
    lab_a_paths = [source_label_raster] * len(img_a_paths)

    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)

    def _flush_timing_rows(force: bool = False) -> None:
        if not timing_csv_enabled:
            return
        if not pending_timing_rows:
            return
        if not force and len(pending_timing_rows) < timing_flush_every:
            return
        append_tile_timing_csv_rows(timing_csv_path, pending_timing_rows)
        timing_rows.extend(pending_timing_rows)
        pending_timing_rows.clear()
        write_timing_summary_csv(timing_summary_csv_path, timing_rows)

    def _record_tile_timings(
        stage: str,
        tile_role: str,
        tile_path: str,
        image_id: str,
        timings: dict[str, float],
        gt_available: bool,
        status: str = "done",
    ) -> None:
        if not timing_csv_enabled or not timings:
            return
        rows = build_tile_timing_rows(
            run_dir=run_dir,
            stage=stage,
            tile_role=tile_role,
            tile_path=tile_path,
            image_id=image_id,
            timings=timings,
            gt_available=gt_available,
            source_mode=source_mode,
            auto_split_mode=auto_split_mode_runtime,
            resample_factor=int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1),
            tile_size=int(tile_size),
            stride=int(stride),
            status=status,
            timestamp_utc=datetime.utcnow().isoformat(),
        )
        pending_timing_rows.extend(rows)
        _flush_timing_rows(force=False)

    # Resolve validation + holdout tiles (required).
    if not val_tiles:
        raise ValueError("VAL_TILES must be set for main.py.")
    if not holdout_tiles:
        logger.warning("no holdout tiles resolved; skipping holdout inference")
    if bool(getattr(cfg, "ANTI_LEAK_CHECKS_ENABLED", True)):
        _ = run_source_validation_anti_leak_checks(
            source_tiles=img_a_paths,
            val_tiles=val_tiles,
            eval_gt_vector_paths=gt_vector_paths,
        )

    xai_enabled = bool(getattr(cfg, "XAI_ENABLED", True))
    xai_save_json = bool(getattr(cfg, "XAI_SAVE_JSON", True))
    xai_save_plots = bool(getattr(cfg, "XAI_SAVE_PLOTS", True))
    xai_include_xgb = bool(getattr(cfg, "XAI_INCLUDE_XGB", True))
    xai_include_knn = bool(getattr(cfg, "XAI_INCLUDE_KNN", True))
    xai_top_features = int(getattr(cfg, "XAI_TOP_FEATURES", 20) or 20)
    xai_top_patches = int(getattr(cfg, "XAI_TOP_PATCHES", 50) or 50)
    xai_dir = os.path.join(run_dir, str(getattr(cfg, "XAI_DIRNAME", "xai")))
    xai_summary_csv_path = os.path.join(
        run_dir,
        str(getattr(cfg, "XAI_SUMMARY_FILENAME", "xai_summary.csv")),
    )
    xai_holdout_cap_enabled = bool(getattr(cfg, "XAI_HOLDOUT_CAP_ENABLED", True))
    xai_holdout_cap = int(getattr(cfg, "XAI_HOLDOUT_CAP", 10) or 0)
    xai_holdout_cap_seed = int(getattr(cfg, "XAI_HOLDOUT_CAP_SEED", 42) or 42)
    xai_holdout_selected = (
        select_holdout_tiles_for_xai(
            holdout_tiles,
            xai_holdout_cap_enabled,
            xai_holdout_cap,
            xai_holdout_cap_seed,
        )
        if xai_enabled
        else set()
    )
    logger.info(
        "xai: enabled=%s val=all holdout_selected=%s/%s cap_enabled=%s cap=%s",
        xai_enabled,
        len(xai_holdout_selected),
        len(holdout_tiles),
        xai_holdout_cap_enabled,
        xai_holdout_cap,
    )

    # Feature caching.
    feature_cache_mode = getattr(cfg, "FEATURE_CACHE_MODE", "disk")
    if feature_cache_mode not in {"disk", "memory"}:
        raise ValueError("FEATURE_CACHE_MODE must be 'disk' or 'memory'")
    if feature_cache_mode == "disk":
        feature_dir = cfg.FEATURE_DIR
        os.makedirs(feature_dir, exist_ok=True)
    else:
        feature_dir = None
    logger.info("feature cache mode: %s", feature_cache_mode)
    source_prefetch_gt_only = bool(getattr(cfg, "SOURCE_PREFETCH_GT_ONLY", True))
    log_training_ablation_summary(
        source_count=len(img_a_paths),
        val_count=len(val_tiles),
        holdout_count=len(holdout_tiles),
        feature_cache_mode=feature_cache_mode,
        source_prefetch_gt_only=source_prefetch_gt_only,
        auto_split_mode_legacy=AUTO_SPLIT_MODE_LEGACY,
    )

    image_id_a_list = [os.path.splitext(os.path.basename(p))[0] for p in img_a_paths]

    # Build DINOv3 banks + XGBoost training data from Image A sources.
    _log_phase("START", "image_a_processing")
    pos_banks = []
    neg_banks = []
    X_list = []
    y_list = []
    source_cache_eligible = 0
    source_cache_skipped_no_gt = 0
    source_prefetch_attempted = 0
    source_prefetch_completed = 0
    source_gt_overlap_cache: dict[str, bool] = {}
    source_prefetch_warned_no_gt_vectors = False
    for img_a_path, lab_a_path, image_id_a in zip(
        img_a_paths, lab_a_paths, image_id_a_list, strict=True
    ):
        source_timings: dict[str, float] = {}
        t0_source_total = time.perf_counter()
        logger.info("source A: %s (labels: %s)", img_a_path, lab_a_path)
        ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
        source_has_gt = True
        if source_prefetch_gt_only:
            if gt_vector_paths:
                source_has_gt = source_gt_overlap_cache.get(img_a_path, False)
                if img_a_path not in source_gt_overlap_cache:
                    source_has_gt = tile_has_gt_overlap(
                        img_a_path,
                        gt_vector_paths,
                        downsample_factor=ds,
                    )
                    source_gt_overlap_cache[img_a_path] = source_has_gt
            else:
                if not source_prefetch_warned_no_gt_vectors:
                    logger.warning(
                        "SOURCE_PREFETCH_GT_ONLY=True but EVAL_GT_VECTORS is "
                        "empty; source cache gating disabled for this run"
                    )
                    source_prefetch_warned_no_gt_vectors = True
                source_has_gt = True
        if source_has_gt:
            source_cache_eligible += 1
        else:
            source_cache_skipped_no_gt += 1

        t0 = time.perf_counter()
        img_a = load_dop20_image(img_a_path, downsample_factor=ds)
        source_timings["load_source_image_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        labels_A, supervision = resolve_source_training_labels(
            img_a_path,
            lab_a_path,
            gt_vector_paths,
            downsample_factor=ds,
        )
        logger.info("source supervision=%s image=%s", supervision, image_id_a)
        source_timings["reproject_source_labels_s"] = time.perf_counter() - t0
        source_feature_dir = feature_dir if source_has_gt else None
        prefetched_a = None
        if feature_cache_mode == "memory":
            source_prefetch_attempted += 1
            if source_has_gt:
                logger.info("prefetch: Image A %s", image_id_a)
                t0 = time.perf_counter()
                prefetched_a = prefetch_features_single_scale_image(
                    img_a,
                    model,
                    processor,
                    device,
                    ps,
                    tile_size,
                    stride,
                    None,
                    None,
                    image_id_a,
                )
                source_timings["prefetch_source_features_s"] = time.perf_counter() - t0
                source_prefetch_completed += 1
            else:
                logger.info("prefetch: skip Image A %s (no GT overlap)", image_id_a)

        t0 = time.perf_counter()
        pos_bank_i, neg_bank_i = build_banks_single_scale(
            img_a,
            labels_A,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            getattr(cfg, "POS_FRAC_THRESH", 0.1),
            None,
            source_feature_dir,
            image_id_a,
            cfg.BANK_CACHE_DIR,
            context_radius=context_radius,
            prefetched_tiles=prefetched_a,
        )
        source_timings["build_banks_s"] = time.perf_counter() - t0
        if pos_bank_i.size > 0:
            pos_banks.append(pos_bank_i)
        if neg_bank_i is not None and len(neg_bank_i) > 0:
            neg_banks.append(neg_bank_i)

        t0 = time.perf_counter()
        X_i, y_i = build_xgb_dataset(
            img_a,
            labels_A,
            ps,
            tile_size,
            stride,
            source_feature_dir,
            image_id_a,
            pos_frac=getattr(cfg, "XGB_POS_FRAC_THRESH", None) or cfg.POS_FRAC_THRESH,
            neg_frac_max=float(getattr(cfg, "XGB_NEG_FRAC_MAX", 0.0) or 0.0),
            max_neg=getattr(cfg, "MAX_NEG_BANK", 8000),
            context_radius=context_radius,
            prefetched_tiles=prefetched_a,
        )
        source_timings["build_xgb_dataset_s"] = time.perf_counter() - t0
        source_timings["source_tile_total_s"] = time.perf_counter() - t0_source_total
        _record_tile_timings(
            stage="source_training",
            tile_role="source",
            tile_path=img_a_path,
            image_id=image_id_a,
            timings=source_timings,
            gt_available=True,
            status="done",
        )
        if X_i.size > 0 and y_i.size > 0:
            X_list.append(X_i)
            y_list.append(y_i)
    if source_prefetch_gt_only:
        logger.info(
            "source cache gate: eligible=%s skipped_no_gt=%s total=%s",
            source_cache_eligible,
            source_cache_skipped_no_gt,
            len(img_a_paths),
        )
    if feature_cache_mode == "memory":
        logger.info(
            "source prefetch summary: attempted=%s completed=%s skipped_no_gt=%s",
            source_prefetch_attempted,
            source_prefetch_completed,
            source_prefetch_attempted - source_prefetch_completed,
        )

    if not pos_banks:
        raise ValueError("no positive banks were built; check SOURCE_TILES and labels")
    pos_bank = np.concatenate(pos_banks, axis=0)
    neg_bank = np.concatenate(neg_banks, axis=0) if neg_banks else None
    logger.info(
        "combined banks: pos=%s, neg=%s",
        len(pos_bank),
        0 if neg_bank is None else len(neg_bank),
    )

    X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=np.float32)
    if X.size == 0 or y.size == 0:
        raise ValueError("XGBoost dataset is empty; check SOURCE_TILES and labels")
    _log_phase("END", "image_a_processing")

    # Tune on validation tile, then infer on holdout tiles.
    _log_phase("START", "validation_tuning")
    tuned = tune_on_validation_multi(
        val_tiles,
        gt_vector_paths,
        model,
        processor,
        device,
        pos_bank,
        neg_bank,
        X,
        y,
        ps,
        tile_size,
        stride,
        feature_dir,
        context_radius,
    )
    _log_phase("END", "validation_tuning")
    if bool(getattr(cfg, "XAI_PCA_XGB_ENABLED", True)) and img_a_paths:
        _ = emit_training_xgb_pca_plot(
            source_tiles=img_a_paths,
            bst=tuned.get("bst"),
            X_train=X,
            model=model,
            processor=processor,
            device=device,
            ps=ps,
            tile_size=tile_size,
            stride=stride,
            feature_dir=feature_dir,
            context_radius=context_radius,
            out_dir=xai_dir,
            downsample_factor=int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1),
            source_tile_index=int(getattr(cfg, "XAI_PCA_SOURCE_TILE_INDEX", 0) or 0),
            top_n=int(getattr(cfg, "XAI_PCA_TOP_COMPONENTS", 5) or 5),
            max_pca_components=int(getattr(cfg, "XAI_PCA_MAX_COMPONENTS", 32) or 32),
        )

    val_phase_metrics: dict[str, list[dict]] = {}
    holdout_phase_metrics: dict[str, list[dict]] = {}
    val_timings: list[dict] = []
    holdout_timings: list[dict] = []
    val_buffer_m = None
    val_pixel_size_m = None

    # Run inference on validation tiles with fixed settings (for plots/metrics)
    _log_phase("START", "validation_inference")
    for val_path in val_tiles:
        result = infer_on_holdout(
            val_path,
            gt_vector_paths,
            model,
            processor,
            device,
            pos_bank,
            neg_bank,
            tuned,
            ps,
            tile_size,
            stride,
            feature_dir,
            shape_dir,
            context_radius,
            plot_with_metrics=True,
            xai_options={
                "enabled": xai_enabled,
                "stage": "validation",
                "xai_dir": xai_dir,
                "summary_csv_path": xai_summary_csv_path,
                "top_features": xai_top_features,
                "top_patches": xai_top_patches,
                "include_xgb": xai_include_xgb,
                "include_knn": xai_include_knn,
                "save_json": xai_save_json,
                "save_plots": xai_save_plots,
            },
        )
        if result["gt_available"]:
            _update_phase_metrics(val_phase_metrics, result["metrics"])
        if isinstance(result.get("timings"), dict):
            val_timings.append(result["timings"])
            _record_tile_timings(
                stage="validation_inference",
                tile_role="validation",
                tile_path=val_path,
                image_id=result["image_id"],
                timings=result["timings"],
                gt_available=bool(result["gt_available"]),
                status="done",
            )
        if val_buffer_m is None:
            val_buffer_m = result["buffer_m"]
            val_pixel_size_m = result["pixel_size_m"]
    _log_phase("END", "validation_inference")

    weighted_phase_metrics: dict[str, dict[str, float]] = {}
    metric_keys = ["iou", "f1", "precision", "recall"]
    for phase, metrics_list in val_phase_metrics.items():
        weights = [float(m.get("_weight", 0.0)) for m in metrics_list]
        weighted_phase_metrics[phase] = {
            key: _weighted_mean([m.get(key, 0.0) for m in metrics_list], weights)
            for key in metric_keys
        }

    inference_best_settings_path = os.path.join(run_dir, "inference_best_setting.yml")
    tuned_bridge_cfg = tuned.get("best_bridge_config", {}) or {}
    export_best_settings(
        tuned["best_raw_config"],
        tuned["best_crf_config"],
        cfg.MODEL_NAME,
        img_a_paths if len(img_a_paths) > 1 else img_a_paths[0],
        f"holdout_tiles={len(holdout_tiles)}",
        float(val_buffer_m) if val_buffer_m is not None else 0.0,
        float(val_pixel_size_m) if val_pixel_size_m is not None else 0.0,
        shadow_cfg=tuned["shadow_cfg"],
        extra_settings={
            "tile_size": tile_size,
            "stride": stride,
            "patch_size": ps,
            "feat_context_radius": context_radius,
            "neg_alpha": float(tuned.get("neg_alpha", getattr(cfg, "NEG_ALPHA", 1.0))),
            "pos_frac_thresh": getattr(cfg, "POS_FRAC_THRESH", 0.1),
            "roads_penalty": tuned.get("roads_penalty", 1.0),
            "roads_mask_path": getattr(cfg, "ROADS_MASK_PATH", None),
            "top_p_a": tuned.get("top_p_a", getattr(cfg, "TOP_P_A", 0.0)),
            "top_p_b": tuned.get("top_p_b", getattr(cfg, "TOP_P_B", 0.05)),
            "top_p_min": tuned.get("top_p_min", getattr(cfg, "TOP_P_MIN", 0.02)),
            "top_p_max": tuned.get("top_p_max", getattr(cfg, "TOP_P_MAX", 0.08)),
            "silver_core_dilate_px": int(
                tuned.get(
                    "silver_core_dilate_px", getattr(cfg, "SILVER_CORE_DILATE_PX", 1)
                )
            ),
            "gap_bridging": bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False)),
            "bridge_max_gap_px": int(
                tuned_bridge_cfg.get(
                    "bridge_max_gap_px", getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)
                )
            ),
            "bridge_max_pairs": int(
                tuned_bridge_cfg.get(
                    "bridge_max_pairs", getattr(cfg, "BRIDGE_MAX_PAIRS", 3)
                )
            ),
            "bridge_max_avg_cost": float(
                tuned_bridge_cfg.get(
                    "bridge_max_avg_cost",
                    getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0),
                )
            ),
            "bridge_width_px": int(
                tuned_bridge_cfg.get(
                    "bridge_width_px", getattr(cfg, "BRIDGE_WIDTH_PX", 2)
                )
            ),
            "bridge_min_component_px": int(
                tuned_bridge_cfg.get(
                    "bridge_min_component_px",
                    getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300),
                )
            ),
            "bridge_spur_prune_iters": int(
                tuned_bridge_cfg.get(
                    "bridge_spur_prune_iters",
                    getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15),
                )
            ),
            "auto_split_tiles": bool(auto_split_tiles),
            "auto_split_mode": str(
                getattr(cfg, "AUTO_SPLIT_MODE", AUTO_SPLIT_MODE_LEGACY)
            ),
            "inference_tile_cap_enabled": bool(
                getattr(cfg, "INFERENCE_TILE_CAP_ENABLED", False)
            ),
            "inference_tile_cap": getattr(cfg, "INFERENCE_TILE_CAP", None),
            "timing_csv_enabled": timing_csv_enabled,
            "timing_csv_filename": str(
                getattr(cfg, "TIMING_CSV_FILENAME", "tile_phase_timing.csv")
            ),
            "timing_summary_csv_filename": str(
                getattr(
                    cfg, "TIMING_SUMMARY_CSV_FILENAME", "timing_opportunity_cost.csv"
                )
            ),
            "xai_enabled": xai_enabled,
            "xai_summary_filename": str(
                getattr(cfg, "XAI_SUMMARY_FILENAME", "xai_summary.csv")
            ),
            "xai_holdout_cap_enabled": xai_holdout_cap_enabled,
            "xai_holdout_cap": xai_holdout_cap,
            "val_tiles_count": len(val_tiles),
            "holdout_tiles_count": len(holdout_tiles),
            "weighted_phase_metrics": weighted_phase_metrics,
        },
        best_settings_path=inference_best_settings_path,
    )
    logger.info("wrote inference best settings: %s", inference_best_settings_path)

    _log_phase("START", "holdout_inference")
    holdout_tiles_processed = len(processed_tiles)
    for b_path in holdout_tiles:
        if b_path in processed_tiles:
            logger.info("holdout skip (already processed): %s", b_path)
            continue
        holdout_xai_enabled = xai_enabled and b_path in xai_holdout_selected
        result = infer_on_holdout(
            b_path,
            gt_vector_paths,
            model,
            processor,
            device,
            pos_bank,
            neg_bank,
            tuned,
            ps,
            tile_size,
            stride,
            feature_dir,
            shape_dir,
            context_radius,
            plot_with_metrics=False,
            xai_options={
                "enabled": holdout_xai_enabled,
                "stage": "holdout",
                "xai_dir": xai_dir,
                "summary_csv_path": xai_summary_csv_path,
                "top_features": xai_top_features,
                "top_patches": xai_top_patches,
                "include_xgb": xai_include_xgb,
                "include_knn": xai_include_knn,
                "save_json": xai_save_json,
                "save_plots": xai_save_plots,
            },
        )
        if result["gt_available"]:
            _update_phase_metrics(holdout_phase_metrics, result["metrics"])
        if isinstance(result.get("timings"), dict):
            holdout_timings.append(result["timings"])
            _record_tile_timings(
                stage="holdout_inference",
                tile_role="holdout",
                tile_path=b_path,
                image_id=result["image_id"],
                timings=result["timings"],
                gt_available=bool(result["gt_available"]),
                status="done",
            )
        holdout_tiles_processed += 1
        ref_path = result["ref_path"]
        masks = result["masks"]
        for stream, variants in union_variants_by_stream.items():
            for variant in variants:
                if stream == "silver_core":
                    mask_key = "silver_core"
                else:
                    mask_key = f"{stream}_{variant}"
                _append_union(
                    stream,
                    variant,
                    masks[mask_key],
                    ref_path,
                    holdout_tiles_processed,
                )
        record = {
            "tile_path": b_path,
            "image_id": result["image_id"],
            "status": "done",
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(processed_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    _log_phase("END", "holdout_inference")

    bridge_enabled = bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False))
    _summarize_phase_metrics(val_phase_metrics, "validation", bridge_enabled)
    _summarize_phase_metrics(holdout_phase_metrics, "holdout", bridge_enabled)

    # ------------------------------------------------------------
    # Consolidate tile-level feature files (.npy) → one per image
    # ------------------------------------------------------------
    if feature_cache_mode == "disk":
        if feature_dir is None:
            raise ValueError("feature_dir must be set for disk cache mode")
        _log_phase("START", "feature_consolidation")
        for image_id_a in image_id_a_list:
            consolidate_features_for_image(feature_dir, image_id_a)
        for b_path in val_tiles + holdout_tiles:
            image_id_b = os.path.splitext(os.path.basename(b_path))[0]
            consolidate_features_for_image(feature_dir, image_id_b)
        _log_phase("END", "feature_consolidation")

    _flush_timing_rows(force=True)
    if timing_csv_enabled:
        logger.info("timing detail csv: %s", timing_csv_path)
        logger.info("timing summary csv: %s", timing_summary_csv_path)

    run_total_s = time.perf_counter() - run_start
    phase_summary_val = _phase_metrics_summary(val_phase_metrics, bridge_enabled)
    phase_deltas_val = _phase_delta_summary(phase_summary_val, bridge_enabled)
    timing_summary_val = _timing_summary(val_timings)
    summary_payload = {
        "run": {
            "run_dir": run_dir,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "tile_size": int(tile_size),
            "stride": int(stride),
            "patch_size": int(ps),
            "resample_factor": int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1),
            "buffer_m": float(cfg.BUFFER_M),
            "context_radius": int(context_radius),
            "val_tiles_count": int(len(val_tiles)),
            "holdout_tiles_count": int(len(holdout_tiles)),
            "runtime_s": float(run_total_s),
            "timing_csv_path": timing_csv_path if timing_csv_enabled else None,
            "timing_summary_csv_path": (
                timing_summary_csv_path if timing_csv_enabled else None
            ),
            "xai_enabled": bool(xai_enabled),
            "xai_dir": xai_dir if xai_enabled else None,
            "xai_summary_csv_path": xai_summary_csv_path if xai_enabled else None,
        },
        "validation": {
            "metrics": phase_summary_val,
            "deltas": phase_deltas_val,
            "timing": timing_summary_val,
        },
    }
    if holdout_phase_metrics:
        phase_summary_holdout = _phase_metrics_summary(
            holdout_phase_metrics, bridge_enabled
        )
        phase_deltas_holdout = _phase_delta_summary(
            phase_summary_holdout, bridge_enabled
        )
        timing_summary_holdout = _timing_summary(holdout_timings)
        summary_payload["holdout"] = {
            "metrics": phase_summary_holdout,
            "deltas": phase_deltas_holdout,
            "timing": timing_summary_holdout,
        }
    run_summary_path = os.path.join(run_dir, "run_summary.yml")
    export_run_summary(summary_payload, run_summary_path)

    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()
