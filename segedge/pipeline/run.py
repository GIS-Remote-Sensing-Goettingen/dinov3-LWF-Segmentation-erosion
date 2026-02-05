"""Primary pipeline entrypoint for SegEdge."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import torch
from scipy.ndimage import median_filter
from skimage.transform import resize

import config as cfg

from ..core.banks import build_banks_single_scale
from ..core.continuity import bridge_skeleton_gaps, skeletonize_with_endpoints
from ..core.crf_utils import refine_with_densecrf
from ..core.features import prefetch_features_single_scale_image
from ..core.io_utils import (
    append_mask_to_union_shapefile,
    backup_union_shapefile,
    build_sh_buffer_mask,
    consolidate_features_for_image,
    count_shapefile_features,
    export_best_settings,
    load_dop20_image,
    rasterize_vector_labels,
    reproject_labels_to_image,
)
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from ..core.logging_utils import setup_logging
from ..core.metrics_utils import (
    compute_metrics,
    compute_metrics_batch_cpu,
    compute_metrics_batch_gpu,
    compute_oracle_upper_bound,
)
from ..core.plotting import save_unified_plot
from ..core.timing_utils import time_end, time_start
from ..core.xdboost import (
    build_xgb_dataset,
    hyperparam_search_xgb_iou,
    train_xgb_classifier,
    xgb_score_image_b,
)
from .common import init_model, resolve_tile_splits_from_gt

# Config-driven flags
USE_FP16_KNN = getattr(cfg, "USE_FP16_KNN", True)
CRF_MAX_CONFIGS = getattr(cfg, "CRF_MAX_CONFIGS", 64)

logger = logging.getLogger(__name__)

_CRF_PARALLEL_CONTEXTS: list[dict] | None = None
_ROADS_MASK_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _get_roads_mask(tile_path: str, downsample_factor: int) -> np.ndarray | None:
    """Load or cache a roads mask rasterized to the tile grid.

    Args:
        tile_path (str): Tile path.
        downsample_factor (int): Downsample factor for rasterization.

    Returns:
        np.ndarray | None: Boolean mask if available.

    Examples:
        >>> callable(_get_roads_mask)
        True
    """
    roads_path = getattr(cfg, "ROADS_MASK_PATH", None)
    if not roads_path:
        return None
    if not os.path.exists(roads_path):
        logger.warning("roads mask not found: %s", roads_path)
        return None
    key = (tile_path, downsample_factor)
    if key in _ROADS_MASK_CACHE:
        return _ROADS_MASK_CACHE[key]
    mask = rasterize_vector_labels(
        roads_path, tile_path, downsample_factor=downsample_factor
    ).astype(bool)
    _ROADS_MASK_CACHE[key] = mask
    return mask


def _apply_roads_penalty(
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    penalty: float,
) -> np.ndarray:
    """Apply a multiplicative penalty on road pixels.

    Args:
        score_map (np.ndarray): Score map.
        roads_mask (np.ndarray | None): Roads mask.
        penalty (float): Multiplicative penalty in [0, 1].

    Returns:
        np.ndarray: Penalized score map.

    Examples:
        >>> import numpy as np
        >>> score = np.array([[1.0, 2.0]], dtype=np.float32)
        >>> mask = np.array([[True, False]])
        >>> _apply_roads_penalty(score, mask, 0.5).tolist()
        [[0.5, 2.0]]
    """
    if roads_mask is None or penalty >= 1.0:
        return score_map
    if roads_mask.shape != score_map.shape:
        raise ValueError("roads_mask must match score_map shape")
    penalty_map = np.where(roads_mask, penalty, 1.0).astype(score_map.dtype)
    return score_map * penalty_map


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Compute weighted mean with safe fallback.

    Args:
        values (list[float]): Values.
        weights (list[float]): Weights.

    Returns:
        float: Weighted mean or simple mean if total weight is zero.

    Examples:
        >>> _weighted_mean([1.0, 3.0], [1.0, 1.0])
        2.0
    """
    total_w = float(np.sum(weights))
    if total_w <= 0:
        return float(np.mean(values)) if values else 0.0
    return float(np.sum(np.array(values) * np.array(weights)) / total_w)


def _log_phase(kind: str, name: str) -> None:
    """Log a phase marker with ANSI color.

    Args:
        kind (str): Phase kind.
        name (str): Phase name.

    Examples:
        >>> callable(_log_phase)
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
    for prev, curr in zip(champ_chain, champ_chain[1:], strict=True):
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


def _init_crf_parallel(contexts: list[dict]) -> None:
    global _CRF_PARALLEL_CONTEXTS
    _CRF_PARALLEL_CONTEXTS = contexts


def _eval_crf_config(cfg, n_iters: int = 5) -> tuple[float, tuple[float, ...]]:
    if _CRF_PARALLEL_CONTEXTS is None:
        raise RuntimeError("CRF contexts not initialized")
    prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
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
        )
        ious.append(compute_metrics(mask_crf_local, ctx["gt_mask_eval"])["iou"])
        weights.append(float(ctx["gt_weight"]))
    return _weighted_mean(ious, weights), cfg


def _apply_shadow_filter(
    img_b: np.ndarray,
    base_mask: np.ndarray,
    weights,
    threshold: float,
    score_full: np.ndarray,
    protect_score: float | None,
) -> np.ndarray:
    img_float = img_b.astype(np.float32)
    w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
    wsum = (img_float * w).sum(axis=2)
    shadow_pass = wsum >= threshold
    if protect_score is None:
        return np.logical_and(base_mask, shadow_pass)
    return np.logical_and(base_mask, shadow_pass | (score_full >= protect_score))


def load_b_tile_context(img_path: str, gt_vector_paths: list[str] | None):
    """Load B tile, SH raster, GT (optional), and buffer mask.

    Args:
        img_path (str): Image B path.
        gt_vector_paths (list[str] | None): Vector GT paths.

    Returns:
        tuple: (img_b, labels_sh, gt_mask, gt_mask_eval, sh_buffer_mask, buffer_m, pixel_size_m)

    Examples:
        >>> callable(load_b_tile_context)
        True
    """
    logger.info("loading tile: %s", img_path)
    t0_data = time_start()
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    img_b = load_dop20_image(img_path, downsample_factor=ds)
    labels_sh = reproject_labels_to_image(
        img_path, cfg.SOURCE_LABEL_RASTER, downsample_factor=ds
    )
    gt_mask = (
        rasterize_vector_labels(gt_vector_paths, img_path, downsample_factor=ds)
        if gt_vector_paths
        else None
    )
    time_end("data_loading_and_reprojection", t0_data)
    target_shape = img_b.shape[:2]
    if labels_sh.shape != target_shape:
        logger.warning(
            "labels_sh shape %s != image shape %s; resizing to match",
            labels_sh.shape,
            target_shape,
        )
        labels_sh = resize(
            labels_sh,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(labels_sh.dtype)
    if gt_mask is not None and gt_mask.shape != target_shape:
        logger.warning(
            "gt_mask shape %s != image shape %s; resizing to match",
            gt_mask.shape,
            target_shape,
        )
        gt_mask = resize(
            gt_mask,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(gt_mask.dtype)

    if gt_mask is not None:
        logger.debug("GT positives on B: %s", gt_mask.sum())
    logger.debug("SH_2022 positives on B: %s", (labels_sh > 0).sum())

    with __import__("rasterio").open(img_path) as src:
        pixel_size_m = abs(src.transform.a)
    pixel_size_m = pixel_size_m * ds
    buffer_m = cfg.BUFFER_M
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    logger.info(
        "tile=%s pixel_size=%.3f m, buffer_m=%s, buffer_pixels=%s",
        img_path,
        pixel_size_m,
        buffer_m,
        buffer_pixels,
    )

    sh_buffer_mask = build_sh_buffer_mask(labels_sh, buffer_pixels)
    if gt_mask is not None and getattr(cfg, "CLIP_GT_TO_BUFFER", False):
        gt_mask_eval = np.logical_and(gt_mask, sh_buffer_mask)
        logger.info(
            "CLIP_GT_TO_BUFFER enabled: GT positives -> %s (was %s)",
            gt_mask_eval.sum(),
            gt_mask.sum(),
        )
    else:
        gt_mask_eval = gt_mask
    return (
        img_b,
        labels_sh,
        gt_mask,
        gt_mask_eval,
        sh_buffer_mask,
        buffer_m,
        pixel_size_m,
    )


def tune_on_validation_multi(
    val_paths: list[str],
    gt_vector_paths: list[str],
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    X: np.ndarray,
    y: np.ndarray,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
):
    """Tune hyperparameters using weighted-mean IoU across validation tiles.

    Args:
        val_paths (list[str]): Validation tile paths.
        gt_vector_paths (list[str]): Vector GT paths.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        pos_bank (np.ndarray): Positive bank.
        neg_bank (np.ndarray | None): Negative bank.
        X (np.ndarray): XGB feature matrix.
        y (np.ndarray): XGB labels.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str | None): Feature cache directory.
        context_radius (int): Feature context radius.

    Returns:
        dict: Tuned configurations and models.

    Examples:
        >>> callable(tune_on_validation_multi)
        True
    """
    if not val_paths:
        raise ValueError("VAL_TILES is empty.")

    val_contexts = []
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    for val_path in val_paths:
        logger.info("tune: preparing validation tile %s", val_path)
        (
            img_b,
            labels_sh,
            gt_mask_B,
            gt_mask_eval,
            sh_buffer_mask,
            buffer_m,
            pixel_size_m,
        ) = load_b_tile_context(val_path, gt_vector_paths)
        if gt_mask_eval is None:
            raise ValueError("Validation requires GT vectors for metric-based tuning.")
        _ = compute_oracle_upper_bound(gt_mask_eval, sh_buffer_mask)
        gt_weight = float(gt_mask_eval.sum())
        image_id_b = os.path.splitext(os.path.basename(val_path))[0]
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
        roads_mask = _get_roads_mask(val_path, ds)
        val_contexts.append(
            {
                "path": val_path,
                "image_id": image_id_b,
                "img_b": img_b,
                "labels_sh": labels_sh,
                "gt_mask_B": gt_mask_B,
                "gt_mask_eval": gt_mask_eval,
                "sh_buffer_mask": sh_buffer_mask,
                "gt_weight": gt_weight,
                "roads_mask": roads_mask,
                "prefetched_b": prefetched_b,
                "buffer_m": buffer_m,
                "pixel_size_m": pixel_size_m,
            }
        )

    # XGB training (shared across road penalties)
    use_gpu_xgb = getattr(cfg, "XGB_USE_GPU", True)
    param_grid = getattr(cfg, "XGB_PARAM_GRID", None)
    num_boost_round = getattr(cfg, "XGB_NUM_BOOST_ROUND", 300)
    early_stop = getattr(cfg, "XGB_EARLY_STOP", 40)
    verbose_eval = getattr(cfg, "XGB_VERBOSE_EVAL", 50)
    val_fraction = getattr(cfg, "XGB_VAL_FRACTION", 0.2)
    if param_grid is None:
        param_grid = [None]

    xgb_candidates = []
    for overrides in param_grid:
        if overrides is None:
            bst = train_xgb_classifier(
                X,
                y,
                use_gpu=use_gpu_xgb,
                num_boost_round=num_boost_round,
                verbose_eval=verbose_eval,
            )
            params_used = None
        else:
            bst, params_used, _, _, _ = hyperparam_search_xgb_iou(
                X,
                y,
                [0.5],
                val_contexts[0]["sh_buffer_mask"],
                val_contexts[0]["gt_mask_eval"],
                val_contexts[0]["img_b"],
                ps,
                tile_size,
                stride,
                feature_dir,
                val_contexts[0]["image_id"],
                prefetched_tiles=val_contexts[0]["prefetched_b"],
                device=device,
                use_gpu=use_gpu_xgb,
                param_grid=[overrides],
                num_boost_round=num_boost_round,
                val_fraction=val_fraction,
                early_stopping_rounds=early_stop,
                verbose_eval=verbose_eval,
                seed=42,
                context_radius=context_radius,
            )
        xgb_candidates.append({"bst": bst, "params": params_used})

    roads_penalties = [float(p) for p in getattr(cfg, "ROADS_PENALTY_VALUES", [1.0])]
    best_bundle = None
    best_champion_iou = None

    for penalty in roads_penalties:
        logger.info("tune: roads penalty=%s", penalty)

        # kNN tuning (weighted-mean IoU across val tiles)
        best_raw_config = None
        for k in cfg.K_VALUES:
            stats_by_thr = {
                thr: {
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "w": 0.0,
                    "n": 0,
                }
                for thr in cfg.THRESHOLDS
            }
            for ctx in val_contexts:
                logger.info("tune: kNN scoring on %s (k=%s)", ctx["path"], k)
                score_full, _ = zero_shot_knn_single_scale_B_with_saliency(
                    img_b=ctx["img_b"],
                    pos_bank=pos_bank,
                    neg_bank=neg_bank,
                    model=model,
                    processor=processor,
                    device=device,
                    ps=ps,
                    tile_size=tile_size,
                    stride=stride,
                    k=k,
                    aggregate_layers=None,
                    feature_dir=feature_dir,
                    image_id=ctx["image_id"],
                    neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
                    prefetched_tiles=ctx["prefetched_b"],
                    use_fp16_matmul=USE_FP16_KNN,
                    context_radius=context_radius,
                )
                score_full = _apply_roads_penalty(
                    score_full, ctx["roads_mask"], penalty
                )
                try:
                    metrics_list = compute_metrics_batch_gpu(
                        score_full,
                        cfg.THRESHOLDS,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    metrics_list = compute_metrics_batch_cpu(
                        score_full,
                        cfg.THRESHOLDS,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                    )
                weight = float(ctx["gt_weight"])
                for m in metrics_list:
                    stats = stats_by_thr[m["threshold"]]
                    stats["iou"] += float(m["iou"]) * weight
                    stats["f1"] += float(m["f1"]) * weight
                    stats["precision"] += float(m["precision"]) * weight
                    stats["recall"] += float(m["recall"]) * weight
                    stats["w"] += weight
                    stats["n"] += 1

            for thr, stats in stats_by_thr.items():
                if stats["w"] > 0:
                    weighted_iou = float(stats["iou"] / stats["w"])
                    weighted_f1 = float(stats["f1"] / stats["w"])
                    weighted_precision = float(stats["precision"] / stats["w"])
                    weighted_recall = float(stats["recall"] / stats["w"])
                else:
                    weighted_iou = 0.0
                    weighted_f1 = 0.0
                    weighted_precision = 0.0
                    weighted_recall = 0.0
                if best_raw_config is None or weighted_iou > best_raw_config["iou"]:
                    best_raw_config = {
                        "k": k,
                        "threshold": thr,
                        "source": "raw",
                        "iou": weighted_iou,
                        "f1": weighted_f1,
                        "precision": weighted_precision,
                        "recall": weighted_recall,
                    }
        if best_raw_config is None:
            raise ValueError("kNN tuning returned no results")

        # XGB tuning (weighted-mean IoU across val tiles)
        best_xgb_config = None
        best_bst = None
        for candidate in xgb_candidates:
            bst = candidate["bst"]
            params_used = candidate["params"]
            stats_by_thr = {
                thr: {
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "w": 0.0,
                    "n": 0,
                }
                for thr in cfg.THRESHOLDS
            }
            for ctx in val_contexts:
                logger.info("tune: XGB scoring on %s", ctx["path"])
                score_full = xgb_score_image_b(
                    ctx["img_b"],
                    bst,
                    ps,
                    tile_size,
                    stride,
                    feature_dir,
                    ctx["image_id"],
                    prefetched_tiles=ctx["prefetched_b"],
                    context_radius=context_radius,
                )
                score_full = _apply_roads_penalty(
                    score_full, ctx["roads_mask"], penalty
                )
                try:
                    metrics_list = compute_metrics_batch_gpu(
                        score_full,
                        cfg.THRESHOLDS,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    metrics_list = compute_metrics_batch_cpu(
                        score_full,
                        cfg.THRESHOLDS,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                    )
                weight = float(ctx["gt_weight"])
                for m in metrics_list:
                    stats = stats_by_thr[m["threshold"]]
                    stats["iou"] += float(m["iou"]) * weight
                    stats["f1"] += float(m["f1"]) * weight
                    stats["precision"] += float(m["precision"]) * weight
                    stats["recall"] += float(m["recall"]) * weight
                    stats["w"] += weight
                    stats["n"] += 1

            for thr, stats in stats_by_thr.items():
                if stats["w"] > 0:
                    weighted_iou = float(stats["iou"] / stats["w"])
                    weighted_f1 = float(stats["f1"] / stats["w"])
                    weighted_precision = float(stats["precision"] / stats["w"])
                    weighted_recall = float(stats["recall"] / stats["w"])
                else:
                    weighted_iou = 0.0
                    weighted_f1 = 0.0
                    weighted_precision = 0.0
                    weighted_recall = 0.0
                cand = {
                    "k": -1,
                    "threshold": thr,
                    "source": "xgb",
                    "iou": weighted_iou,
                    "f1": weighted_f1,
                    "precision": weighted_precision,
                    "recall": weighted_recall,
                    "params": params_used,
                }
                if best_xgb_config is None or weighted_iou > best_xgb_config["iou"]:
                    best_xgb_config = cand
                    best_bst = bst
        if best_xgb_config is None or best_bst is None:
            raise ValueError("XGB tuning returned no results")

        champion_source = (
            "raw" if best_raw_config["iou"] >= best_xgb_config["iou"] else "xgb"
        )
        champion_iou = (
            best_raw_config["iou"]
            if champion_source == "raw"
            else best_xgb_config["iou"]
        )
        if best_champion_iou is None or champion_iou > best_champion_iou:
            best_champion_iou = champion_iou
            best_bundle = {
                "roads_penalty": penalty,
                "best_raw_config": best_raw_config,
                "best_xgb_config": best_xgb_config,
                "best_bst": best_bst,
                "champion_source": champion_source,
            }

    if best_bundle is None:
        raise ValueError("roads penalty tuning returned no results")

    roads_penalty = best_bundle["roads_penalty"]
    best_raw_config = best_bundle["best_raw_config"]
    best_xgb_config = best_bundle["best_xgb_config"]
    best_bst = best_bundle["best_bst"]
    champion_source = best_bundle["champion_source"]

    thr_center = (
        best_raw_config["threshold"]
        if champion_source == "raw"
        else best_xgb_config["threshold"]
    )
    for ctx in val_contexts:
        if champion_source == "raw":
            score_full, _ = zero_shot_knn_single_scale_B_with_saliency(
                img_b=ctx["img_b"],
                pos_bank=pos_bank,
                neg_bank=neg_bank,
                model=model,
                processor=processor,
                device=device,
                ps=ps,
                tile_size=tile_size,
                stride=stride,
                k=best_raw_config["k"],
                aggregate_layers=None,
                feature_dir=feature_dir,
                image_id=ctx["image_id"],
                neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
                prefetched_tiles=ctx["prefetched_b"],
                use_fp16_matmul=USE_FP16_KNN,
                context_radius=context_radius,
            )
        else:
            score_full = xgb_score_image_b(
                ctx["img_b"],
                best_bst,
                ps,
                tile_size,
                stride,
                feature_dir,
                ctx["image_id"],
                prefetched_tiles=ctx["prefetched_b"],
                context_radius=context_radius,
            )
        score_full = _apply_roads_penalty(
            score_full, ctx["roads_mask"], float(roads_penalty)
        )
        ctx["score_full"] = score_full
        ctx["thr_center"] = thr_center

    # CRF tuning across val tiles
    crf_candidates = [
        (psf, pw, pxy, bw, bxy, brgb)
        for psf in cfg.PROB_SOFTNESS_VALUES
        for pw in cfg.POS_W_VALUES
        for pxy in cfg.POS_XY_STD_VALUES
        for bw in cfg.BILATERAL_W_VALUES
        for bxy in cfg.BILATERAL_XY_STD_VALUES
        for brgb in cfg.BILATERAL_RGB_STD_VALUES
    ]
    best_crf_cfg = None
    best_crf_iou = None
    crf_candidates = crf_candidates[:CRF_MAX_CONFIGS]
    num_workers = int(getattr(cfg, "CRF_NUM_WORKERS", 1) or 1)
    logger.info(
        "tune: CRF grid search configs=%s, workers=%s",
        len(crf_candidates),
        num_workers,
    )
    _init_crf_parallel(val_contexts)
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for med_iou, cand in ex.map(_eval_crf_config, crf_candidates):
                if best_crf_iou is None or med_iou > best_crf_iou:
                    best_crf_iou = med_iou
                    best_crf_cfg = {
                        "prob_softness": cand[0],
                        "pos_w": cand[1],
                        "pos_xy_std": cand[2],
                        "bilateral_w": cand[3],
                        "bilateral_xy_std": cand[4],
                        "bilateral_rgb_std": cand[5],
                    }
    else:
        for cand in crf_candidates:
            med_iou, _ = _eval_crf_config(cand)
            if best_crf_iou is None or med_iou > best_crf_iou:
                best_crf_iou = med_iou
                best_crf_cfg = {
                    "prob_softness": cand[0],
                    "pos_w": cand[1],
                    "pos_xy_std": cand[2],
                    "bilateral_w": cand[3],
                    "bilateral_xy_std": cand[4],
                    "bilateral_rgb_std": cand[5],
                }
    if best_crf_cfg is None:
        raise ValueError("CRF tuning returned no results")

    # Shadow tuning across val tiles
    best_shadow_cfg = None
    best_shadow_iou = None
    protect_scores = getattr(cfg, "SHADOW_PROTECT_SCORES", [0.5])
    for weights in cfg.SHADOW_WEIGHT_SETS:
        iou_by_key = {
            (thr, protect_score): {"sum": 0.0, "w": 0.0}
            for thr in cfg.SHADOW_THRESHOLDS
            for protect_score in protect_scores
        }
        for ctx in val_contexts:
            logger.info("tune: shadow scoring on %s", ctx["path"])
            score_full = ctx["score_full"]
            thr_center = ctx["thr_center"]

            mask_crf = refine_with_densecrf(
                ctx["img_b"],
                score_full,
                thr_center,
                ctx["sh_buffer_mask"],
                prob_softness=best_crf_cfg["prob_softness"],
                n_iters=5,
                pos_w=best_crf_cfg["pos_w"],
                pos_xy_std=best_crf_cfg["pos_xy_std"],
                bilateral_w=best_crf_cfg["bilateral_w"],
                bilateral_xy_std=best_crf_cfg["bilateral_xy_std"],
                bilateral_rgb_std=best_crf_cfg["bilateral_rgb_std"],
            )
            img_float = ctx["img_b"].astype(np.float32)
            w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
            wsum = (img_float * w).sum(axis=2)
            flat_base = mask_crf.reshape(-1)
            flat_gt = ctx["gt_mask_eval"].reshape(-1).astype(bool)
            vals = wsum.reshape(-1)[flat_base]
            gt_vals = flat_gt[flat_base]
            if vals.size == 0:
                continue
            thr_arr = np.array(cfg.SHADOW_THRESHOLDS, dtype=np.float32).reshape(-1, 1)
            mask_thr = vals[None, :] >= thr_arr
            gt_bool = gt_vals.astype(bool)
            score_vals = score_full.reshape(-1)[flat_base]
            weight = float(ctx["gt_weight"])
            for protect_score in protect_scores:
                protect_mask = score_vals >= protect_score
                mask_keep = mask_thr | protect_mask[None, :]
                tp = (
                    np.logical_and(mask_keep, gt_bool[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                fp = (
                    np.logical_and(mask_keep, ~gt_bool[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                fn = (
                    np.logical_and(~mask_keep, gt_bool[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                iou = tp / (tp + fp + fn + 1e-8)
                for i, thr in enumerate(cfg.SHADOW_THRESHOLDS):
                    stats = iou_by_key[(thr, protect_score)]
                    stats["sum"] += float(iou[i]) * weight
                    stats["w"] += weight

        for (thr, protect_score), stats in iou_by_key.items():
            if stats["w"] <= 0:
                continue
            weighted_iou = float(stats["sum"] / stats["w"])
            if best_shadow_iou is None or weighted_iou > best_shadow_iou:
                best_shadow_iou = weighted_iou
                best_shadow_cfg = {
                    "weights": weights,
                    "threshold": thr,
                    "protect_score": protect_score,
                    "iou": weighted_iou,
                }
    if best_shadow_cfg is None:
        raise ValueError("shadow tuning returned no results")

    logger.info("tune: roads penalty selected=%s", roads_penalty)
    return {
        "bst": best_bst,
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "best_crf_config": {**best_crf_cfg, "k": best_raw_config["k"]},
        "shadow_cfg": best_shadow_cfg,
        "roads_penalty": float(roads_penalty),
    }


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

    Returns:
        dict: Masks, metrics, and metadata for the tile.

    Examples:
        >>> callable(infer_on_holdout)
        True
    """
    logger.info("inference: holdout tile %s", holdout_path)
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
    gt_weight = float(gt_mask_eval.sum())
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    roads_mask = _get_roads_mask(holdout_path, ds)
    roads_penalty = float(tuned.get("roads_penalty", 1.0))

    image_id_b = os.path.splitext(os.path.basename(holdout_path))[0]
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

    k = tuned["best_raw_config"]["k"]
    knn_thr = tuned["best_raw_config"]["threshold"]
    score_knn, _ = zero_shot_knn_single_scale_B_with_saliency(
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
        neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
        prefetched_tiles=prefetched_b,
        use_fp16_matmul=USE_FP16_KNN,
        context_radius=context_radius,
    )
    score_knn_raw = score_knn
    score_knn = _apply_roads_penalty(score_knn, roads_mask, roads_penalty)
    mask_knn = (score_knn >= knn_thr) & sh_buffer_mask
    mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0
    metrics_knn = compute_metrics(mask_knn, gt_mask_eval)

    bst = tuned["bst"]
    xgb_thr = tuned["best_xgb_config"]["threshold"]
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
    score_xgb = _apply_roads_penalty(score_xgb, roads_mask, roads_penalty)
    mask_xgb = (score_xgb >= xgb_thr) & sh_buffer_mask
    mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
    metrics_xgb = compute_metrics(mask_xgb, gt_mask_eval)

    champion_source = tuned["champion_source"]
    if champion_source == "raw":
        champion_score = score_knn
    else:
        champion_score = score_xgb

    crf_cfg = tuned["best_crf_config"]
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
    if champion_source == "raw":
        best_crf_mask = mask_crf_knn
    else:
        best_crf_mask = mask_crf_xgb

    bridge_enabled = bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False))
    bridge_mask = best_crf_mask
    if bridge_enabled:
        prob_crf = prob_crf_knn if champion_source == "raw" else prob_crf_xgb
        bridge_mask = bridge_skeleton_gaps(
            best_crf_mask,
            prob_crf,
            max_gap_px=int(getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)),
            max_pairs_per_endpoint=int(getattr(cfg, "BRIDGE_MAX_PAIRS", 3)),
            max_avg_cost=float(getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)),
            bridge_width_px=int(getattr(cfg, "BRIDGE_WIDTH_PX", 2)),
            min_component_area_px=int(getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300)),
            spur_prune_iters=int(getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15)),
        )

    shadow_cfg = tuned["shadow_cfg"]
    protect_score = shadow_cfg.get("protect_score")
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
    metrics_knn_crf = compute_metrics(mask_crf_knn, gt_mask_eval)
    metrics_knn_shadow = compute_metrics(shadow_mask_knn, gt_mask_eval)
    metrics_xgb_crf = compute_metrics(mask_crf_xgb, gt_mask_eval)
    metrics_xgb_shadow = compute_metrics(shadow_mask_xgb, gt_mask_eval)
    champ_raw_mask = mask_knn if champion_source == "raw" else mask_xgb
    metrics_champion_raw = compute_metrics(champ_raw_mask, gt_mask_eval)
    metrics_champion_crf = compute_metrics(best_crf_mask, gt_mask_eval)
    metrics_champion_bridge = compute_metrics(bridge_mask, gt_mask_eval)
    shadow_metrics = compute_metrics(shadow_mask, gt_mask_eval)

    skel, endpoints = skeletonize_with_endpoints(bridge_mask)
    metrics_map = {
        "knn_raw": metrics_knn,
        "knn_crf": metrics_knn_crf,
        "knn_shadow": metrics_knn_shadow,
        "xgb_raw": metrics_xgb,
        "xgb_crf": metrics_xgb_crf,
        "xgb_shadow": metrics_xgb_shadow,
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
        "champion_raw": champ_raw_mask,
        "champion_crf": best_crf_mask,
        "champion_bridge": bridge_mask,
        "champion_shadow": shadow_mask,
    }
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

    champ_raw_mask = mask_knn if champion_source == "raw" else mask_xgb
    return {
        "ref_path": holdout_path,
        "image_id": image_id_b,
        "gt_available": gt_available,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
        "metrics": metrics_map,
        "masks": {
            "knn_raw": mask_knn,
            "knn_crf": mask_crf_knn,
            "knn_shadow": shadow_mask_knn,
            "xgb_raw": mask_xgb,
            "xgb_crf": mask_crf_xgb,
            "xgb_shadow": shadow_mask_xgb,
            "champion_raw": champ_raw_mask,
            "champion_crf": best_crf_mask,
            "champion_bridge": bridge_mask,
            "champion_shadow": shadow_mask,
        },
    }


def main():
    """Run the full segmentation pipeline for configured tiles.

    Examples:
        >>> callable(main)
        True
    """

    t0_main = time_start()
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
    setup_logging(getattr(cfg, "LOG_PATH", None))
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
    union_streams = ["knn", "xgb", "champion"]
    union_variants = ["raw", "crf", "shadow"]
    union_states: dict[tuple[str, str], dict[str, str | int]] = {}
    for stream in union_streams:
        for variant in union_variants:
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

    # ------------------------------------------------------------
    # Resolve one or more labeled source images (Image A list)
    # ------------------------------------------------------------
    if auto_split_tiles:
        tiles_dir = getattr(cfg, "TILES_DIR", "data/tiles")
        tile_glob = getattr(cfg, "TILE_GLOB", "*.tif")
        val_fraction = float(getattr(cfg, "VAL_SPLIT_FRACTION", 0.2))
        seed = int(getattr(cfg, "SPLIT_SEED", 42))
        downsample_factor = getattr(cfg, "GT_PRESENCE_DOWNSAMPLE", None)
        num_workers = getattr(cfg, "GT_PRESENCE_WORKERS", None)
        img_a_paths, val_tiles, holdout_tiles = resolve_tile_splits_from_gt(
            tiles_dir,
            tile_glob,
            gt_vector_paths,
            val_fraction,
            seed,
            downsample_factor=downsample_factor,
            num_workers=num_workers,
        )
        logger.info(
            "auto split tiles: source=%s val=%s holdout=%s",
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

    # ------------------------------------------------------------
    # Resolve validation + holdout tiles (required)
    # ------------------------------------------------------------
    if not val_tiles:
        raise ValueError("VAL_TILES must be set for main.py.")
    if not holdout_tiles:
        logger.warning("no holdout tiles resolved; skipping holdout inference")

    # ------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------
    feature_cache_mode = getattr(cfg, "FEATURE_CACHE_MODE", "disk")
    if feature_cache_mode not in {"disk", "memory"}:
        raise ValueError("FEATURE_CACHE_MODE must be 'disk' or 'memory'")
    if feature_cache_mode == "disk":
        feature_dir = cfg.FEATURE_DIR
        os.makedirs(feature_dir, exist_ok=True)
    else:
        feature_dir = None
    logger.info("feature cache mode: %s", feature_cache_mode)

    image_id_a_list = [os.path.splitext(os.path.basename(p))[0] for p in img_a_paths]

    # ------------------------------------------------------------
    # Build DINOv3 banks + XGBoost training data from Image A sources
    # ------------------------------------------------------------
    _log_phase("START", "image_a_processing")
    pos_banks = []
    neg_banks = []
    X_list = []
    y_list = []
    for img_a_path, lab_a_path, image_id_a in zip(
        img_a_paths, lab_a_paths, image_id_a_list, strict=True
    ):
        logger.info("source A: %s (labels: %s)", img_a_path, lab_a_path)
        ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
        img_a = load_dop20_image(img_a_path, downsample_factor=ds)
        labels_A = reproject_labels_to_image(
            img_a_path, lab_a_path, downsample_factor=ds
        )
        prefetched_a = None
        if feature_cache_mode == "memory":
            logger.info("prefetch: Image A %s", image_id_a)
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
            feature_dir,
            image_id_a,
            cfg.BANK_CACHE_DIR,
            context_radius=context_radius,
            prefetched_tiles=prefetched_a,
        )
        if pos_bank_i.size > 0:
            pos_banks.append(pos_bank_i)
        if neg_bank_i is not None and len(neg_bank_i) > 0:
            neg_banks.append(neg_bank_i)

        X_i, y_i = build_xgb_dataset(
            img_a,
            labels_A,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id_a,
            pos_frac=cfg.POS_FRAC_THRESH,
            max_neg=getattr(cfg, "MAX_NEG_BANK", 8000),
            context_radius=context_radius,
            prefetched_tiles=prefetched_a,
        )
        if X_i.size > 0 and y_i.size > 0:
            X_list.append(X_i)
            y_list.append(y_i)

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

    # ------------------------------------------------------------
    # Tune on validation tile, then infer on holdout tiles
    # ------------------------------------------------------------
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

    val_phase_metrics: dict[str, list[dict]] = {}
    holdout_phase_metrics: dict[str, list[dict]] = {}
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
        )
        if result["gt_available"]:
            _update_phase_metrics(val_phase_metrics, result["metrics"])
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
    export_best_settings(
        tuned["best_raw_config"],
        tuned["best_crf_config"],
        cfg.MODEL_NAME,
        getattr(cfg, "SOURCE_TILES", None) or cfg.SOURCE_TILE,
        f"holdout_tiles={len(holdout_tiles)}",
        float(val_buffer_m) if val_buffer_m is not None else 0.0,
        float(val_pixel_size_m) if val_pixel_size_m is not None else 0.0,
        shadow_cfg=tuned["shadow_cfg"],
        extra_settings={
            "tile_size": tile_size,
            "stride": stride,
            "patch_size": ps,
            "feat_context_radius": context_radius,
            "neg_alpha": getattr(cfg, "NEG_ALPHA", 1.0),
            "pos_frac_thresh": getattr(cfg, "POS_FRAC_THRESH", 0.1),
            "roads_penalty": tuned.get("roads_penalty", 1.0),
            "roads_mask_path": getattr(cfg, "ROADS_MASK_PATH", None),
            "gap_bridging": bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False)),
            "bridge_max_gap_px": int(getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)),
            "bridge_max_pairs": int(getattr(cfg, "BRIDGE_MAX_PAIRS", 3)),
            "bridge_max_avg_cost": float(getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)),
            "bridge_width_px": int(getattr(cfg, "BRIDGE_WIDTH_PX", 2)),
            "bridge_min_component_px": int(
                getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300)
            ),
            "bridge_spur_prune_iters": int(getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15)),
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
        )
        if result["gt_available"]:
            _update_phase_metrics(holdout_phase_metrics, result["metrics"])
        holdout_tiles_processed += 1
        ref_path = result["ref_path"]
        masks = result["masks"]
        for stream in ("knn", "xgb", "champion"):
            for variant in ("raw", "crf", "shadow"):
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

    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()
