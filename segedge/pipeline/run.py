"""Primary pipeline entrypoint for SegEdge."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.ndimage import median_filter
from skimage.transform import resize

import config as cfg

from ..core.banks import build_banks_single_scale
from ..core.crf_utils import refine_with_densecrf
from ..core.features import prefetch_features_single_scale_image
from ..core.io_utils import (
    build_sh_buffer_mask,
    consolidate_features_for_image,
    export_best_settings,
    export_mask_to_shapefile,
    export_masks_to_shapefile_union,
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
from ..core.plotting import save_best_model_plot, save_knn_xgb_gt_plot, save_plot
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


def _init_crf_parallel(contexts: list[dict]) -> None:
    global _CRF_PARALLEL_CONTEXTS
    _CRF_PARALLEL_CONTEXTS = contexts


def _eval_crf_config(cfg, n_iters: int = 5) -> tuple[float, tuple[float, ...]]:
    if _CRF_PARALLEL_CONTEXTS is None:
        raise RuntimeError("CRF contexts not initialized")
    prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
    ious = []
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
    return float(np.median(ious)), cfg


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
    """Tune hyperparameters using median IoU across validation tiles.

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
        val_contexts.append(
            {
                "path": val_path,
                "image_id": image_id_b,
                "img_b": img_b,
                "labels_sh": labels_sh,
                "gt_mask_B": gt_mask_B,
                "gt_mask_eval": gt_mask_eval,
                "sh_buffer_mask": sh_buffer_mask,
                "prefetched_b": prefetched_b,
                "buffer_m": buffer_m,
                "pixel_size_m": pixel_size_m,
            }
        )

    # kNN tuning (median IoU across val tiles)
    best_raw_config = None
    for k in cfg.K_VALUES:
        iou_by_thr = {thr: [] for thr in cfg.THRESHOLDS}
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
            for m in metrics_list:
                iou_by_thr[m["threshold"]].append(m["iou"])

        for thr, ious in iou_by_thr.items():
            med_iou = float(np.median(ious))
            if best_raw_config is None or med_iou > best_raw_config["iou"]:
                best_raw_config = {
                    "k": k,
                    "threshold": thr,
                    "source": "raw",
                    "iou": med_iou,
                }
    if best_raw_config is None:
        raise ValueError("kNN tuning returned no results")

    # XGB tuning (median IoU across val tiles)
    use_gpu_xgb = getattr(cfg, "XGB_USE_GPU", True)
    param_grid = getattr(cfg, "XGB_PARAM_GRID", None)
    num_boost_round = getattr(cfg, "XGB_NUM_BOOST_ROUND", 300)
    early_stop = getattr(cfg, "XGB_EARLY_STOP", 40)
    verbose_eval = getattr(cfg, "XGB_VERBOSE_EVAL", 50)
    val_fraction = getattr(cfg, "XGB_VAL_FRACTION", 0.2)
    if param_grid is None:
        param_grid = [None]

    best_xgb_config = None
    best_bst = None
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

        iou_by_thr = {thr: [] for thr in cfg.THRESHOLDS}
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
            for m in metrics_list:
                iou_by_thr[m["threshold"]].append(m["iou"])

        for thr, ious in iou_by_thr.items():
            med_iou = float(np.median(ious))
            cand = {
                "k": -1,
                "threshold": thr,
                "source": "xgb",
                "iou": med_iou,
                "params": params_used,
            }
            if best_xgb_config is None or med_iou > best_xgb_config["iou"]:
                best_xgb_config = cand
                best_bst = bst
    if best_xgb_config is None or best_bst is None:
        raise ValueError("XGB tuning returned no results")

    champion_source = (
        "raw" if best_raw_config["iou"] >= best_xgb_config["iou"] else "xgb"
    )

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
            (thr, protect_score): []
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
                    iou_by_key[(thr, protect_score)].append(float(iou[i]))

        for (thr, protect_score), ious in iou_by_key.items():
            if not ious:
                continue
            med_iou = float(np.median(ious))
            if best_shadow_iou is None or med_iou > best_shadow_iou:
                best_shadow_iou = med_iou
                best_shadow_cfg = {
                    "weights": weights,
                    "threshold": thr,
                    "protect_score": protect_score,
                }
    if best_shadow_cfg is None:
        raise ValueError("shadow tuning returned no results")

    return {
        "bst": best_bst,
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "best_crf_config": {**best_crf_cfg, "k": best_raw_config["k"]},
        "shadow_cfg": best_shadow_cfg,
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

    Returns:
        tuple[np.ndarray, str]: Shadow mask and reference path.

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
    if gt_mask_eval is None:
        logger.warning("Holdout has no GT; metrics will be reported as 0.0.")
        gt_mask_eval = np.zeros(img_b.shape[:2], dtype=bool)

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
    mask_xgb = (score_xgb >= xgb_thr) & sh_buffer_mask
    mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
    metrics_xgb = compute_metrics(mask_xgb, gt_mask_eval)

    champion_source = tuned["champion_source"]
    if champion_source == "raw":
        champion_score = score_knn
        thr_center_for_crf = knn_thr
        best_raw_config = {**tuned["best_raw_config"], **metrics_knn}
    else:
        champion_score = score_xgb
        thr_center_for_crf = xgb_thr
        best_raw_config = {**tuned["best_xgb_config"], **metrics_xgb}

    crf_cfg = tuned["best_crf_config"]
    best_crf_mask = refine_with_densecrf(
        img_b,
        champion_score,
        thr_center_for_crf,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
    )
    best_crf_config = {**crf_cfg, **compute_metrics(best_crf_mask, gt_mask_eval)}

    shadow_cfg = tuned["shadow_cfg"]
    shadow_mask = _apply_shadow_filter(
        img_b,
        best_crf_mask,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        champion_score,
        shadow_cfg.get("protect_score"),
    )
    shadow_metrics = compute_metrics(shadow_mask, gt_mask_eval)
    shadow_cfg_full = {**shadow_cfg, **shadow_metrics}

    save_knn_xgb_gt_plot(
        img_b,
        gt_mask_eval,
        mask_knn,
        mask_xgb,
        cfg.PLOT_DIR,
        image_id_b,
        title_knn=f"kNN IoU={metrics_knn['iou']:.3f}",
        title_xgb=f"XGB IoU={metrics_xgb['iou']:.3f}",
        filename_suffix="knn_vs_xgb.png",
    )
    save_best_model_plot(
        img_b,
        gt_mask_eval,
        mask_knn if champion_source == "raw" else mask_xgb,
        title=f"Champion ({champion_source}) IoU={best_raw_config['iou']:.3f}",
        plot_dir=cfg.PLOT_DIR,
        image_id_b=image_id_b,
        filename_suffix="champion_pre_crf.png",
    )
    save_plot(
        img_b,
        gt_mask_eval,
        mask_knn if champion_source == "raw" else mask_xgb,
        best_raw_config,
        best_crf_mask,
        best_crf_config,
        thr_center_for_crf,
        cfg.PLOT_DIR,
        image_id_b,
        best_shadow={"cfg": shadow_cfg_full, "mask": shadow_mask},
        labels_sh=labels_sh,
    )

    export_mask_to_shapefile(
        mask_knn if champion_source == "raw" else mask_xgb,
        holdout_path,
        os.path.join(shape_dir, f"{image_id_b}_pred_mask_best_raw.shp"),
    )
    export_mask_to_shapefile(
        best_crf_mask,
        holdout_path,
        os.path.join(shape_dir, f"{image_id_b}_pred_mask_best_crf.shp"),
    )
    export_mask_to_shapefile(
        shadow_mask,
        holdout_path,
        os.path.join(shape_dir, f"{image_id_b}_pred_mask_best_shadow.shp"),
    )

    export_best_settings(
        best_raw_config,
        best_crf_config,
        cfg.MODEL_NAME,
        getattr(cfg, "SOURCE_TILES", None) or cfg.SOURCE_TILE,
        holdout_path,
        buffer_m,
        pixel_size_m,
        shadow_cfg=shadow_cfg_full,
        extra_settings={
            "tile_size": tile_size,
            "stride": stride,
            "patch_size": ps,
            "feat_context_radius": context_radius,
            "neg_alpha": getattr(cfg, "NEG_ALPHA", 1.0),
            "pos_frac_thresh": getattr(cfg, "POS_FRAC_THRESH", 0.1),
        },
        best_settings_path=os.path.join(
            os.path.dirname(cfg.BEST_SETTINGS_PATH), f"best_settings_{image_id_b}.yml"
        ),
    )

    return shadow_mask, holdout_path


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
    existing = sorted(d for d in os.listdir(output_root) if d.startswith("run_"))
    next_idx = 1
    if existing:
        try:
            next_idx = (
                max(int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit())
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

    # ------------------------------------------------------------
    # Init DINOv3 model & processor
    # ------------------------------------------------------------
    logger.info("phase:start init_model")
    model, processor, device = init_model(model_name)
    logger.info("phase:end init_model")
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
        img_a_paths, val_tiles, holdout_tiles = resolve_tile_splits_from_gt(
            tiles_dir,
            tile_glob,
            gt_vector_paths,
            val_fraction,
            seed,
            downsample_factor=downsample_factor,
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
    logger.info("phase:start image_a_processing")
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
    logger.info("phase:end image_a_processing")

    # ------------------------------------------------------------
    # Tune on validation tile, then infer on holdout tiles
    # ------------------------------------------------------------
    logger.info("phase:start validation_tuning")
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
    logger.info("phase:end validation_tuning")

    masks_for_union = []
    # Run inference on validation tiles with fixed settings (for plots/metrics)
    logger.info("phase:start validation_inference")
    for val_path in val_tiles:
        infer_on_holdout(
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
        )
    logger.info("phase:end validation_inference")

    logger.info("phase:start holdout_inference")
    for b_path in holdout_tiles:
        shadow_mask, ref_path = infer_on_holdout(
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
        )
        masks_for_union.append((shadow_mask, ref_path))
    logger.info("phase:end holdout_inference")

    if masks_for_union:
        logger.info("phase:start union_export")
        export_masks_to_shapefile_union(
            masks_for_union, os.path.join(shape_dir, "pred_mask_best_shadow_merged.shp")
        )
        logger.info("phase:end union_export")

    # ------------------------------------------------------------
    # Consolidate tile-level feature files (.npy) → one per image
    # ------------------------------------------------------------
    if feature_cache_mode == "disk":
        if feature_dir is None:
            raise ValueError("feature_dir must be set for disk cache mode")
        logger.info("phase:start feature_consolidation")
        for image_id_a in image_id_a_list:
            consolidate_features_for_image(feature_dir, image_id_a)
        for b_path in val_tiles + holdout_tiles:
            image_id_b = os.path.splitext(os.path.basename(b_path))[0]
            consolidate_features_for_image(feature_dir, image_id_b)
        logger.info("phase:end feature_consolidation")

    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()
