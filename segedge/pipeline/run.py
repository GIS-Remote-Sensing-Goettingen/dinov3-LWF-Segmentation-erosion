"""Primary pipeline entrypoint for SegEdge."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import fiona
import numpy as np
import rasterio
import rasterio.features as rfeatures
import torch
import yaml
from pyproj import CRS, Transformer
from scipy.ndimage import median_filter
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from skimage.transform import resize

from ..core.config_loader import cfg
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
    hyperparam_search_xgb_iou,
    train_xgb_classifier,
    xgb_score_image_b,
)
from .common import (
    build_training_artifacts_for_tiles,
    init_model,
    resolve_tiles_from_gt_presence,
)

# Config-driven flags
USE_FP16_KNN = cfg.search.knn.use_fp16_knn
CRF_MAX_CONFIGS = cfg.search.crf.max_configs

logger = logging.getLogger(__name__)

_CRF_PARALLEL_CONTEXTS: list[dict] | None = None
_ROADS_MASK_CACHE: dict[tuple[str, int], np.ndarray] = {}
_ROADS_INDEX_CACHE: dict[tuple[str, str], tuple[STRtree | None, list]] = {}


def _get_roads_index(tile_crs) -> tuple[STRtree | None, list]:
    """Load and cache road geometries in a spatial index.

    Args:
        tile_crs: Target CRS for geometries.

    Returns:
        tuple[STRtree | None, list]: Index and geometry list.

    Examples:
        >>> callable(_get_roads_index)
        True
    """
    roads_path = cfg.io.paths.roads_mask_path
    if not roads_path or not os.path.exists(roads_path):
        return None, []
    crs_key = CRS.from_user_input(tile_crs).to_string() if tile_crs else "<none>"
    cache_key = (roads_path, crs_key)
    if cache_key in _ROADS_INDEX_CACHE:
        return _ROADS_INDEX_CACHE[cache_key]

    t0 = time_start()
    geoms = []
    with fiona.open(roads_path, "r") as shp:
        vec_crs = shp.crs
        transformer = None
        if vec_crs and tile_crs is not None:
            vec_crs_obj = CRS.from_user_input(vec_crs)
            tile_crs_obj = CRS.from_user_input(tile_crs)
            if vec_crs_obj != tile_crs_obj:
                logger.info(
                    "reprojecting road geometries from %s -> %s for %s",
                    vec_crs_obj.to_string(),
                    tile_crs_obj.to_string(),
                    roads_path,
                )
                transformer = Transformer.from_crs(
                    vec_crs_obj, tile_crs_obj, always_xy=True
                )
        for feat in shp:
            geom = feat.get("geometry")
            if not geom:
                continue
            geom_obj = shape(geom)
            if geom_obj.is_empty:
                continue
            if transformer is not None:
                geom_obj = shp_transform(transformer.transform, geom_obj)
            geoms.append(geom_obj)

    tree = STRtree(geoms) if geoms else None
    _ROADS_INDEX_CACHE[cache_key] = (tree, geoms)
    time_end("roads_index_build", t0)
    logger.info("roads index built: %s geometries", len(geoms))
    return tree, geoms


def _get_roads_mask(
    tile_path: str,
    downsample_factor: int,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
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
    key = (tile_path, downsample_factor)
    if key in _ROADS_MASK_CACHE:
        return _ROADS_MASK_CACHE[key]

    with rasterio.open(tile_path) as tile_src:
        if downsample_factor > 1:
            out_shape = (
                tile_src.height // downsample_factor,
                tile_src.width // downsample_factor,
            )
            transform = tile_src.transform * tile_src.transform.scale(
                tile_src.width / out_shape[1],
                tile_src.height / out_shape[0],
            )
        else:
            out_shape = (tile_src.height, tile_src.width)
            transform = tile_src.transform
        tile_bounds = tile_src.bounds
        tile_crs = tile_src.crs

    tree, geoms = _get_roads_index(tile_crs)
    if tree is None or not geoms:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        return mask_empty

    tile_box = box(
        tile_bounds.left,
        tile_bounds.bottom,
        tile_bounds.right,
        tile_bounds.top,
    )
    hits = tree.query(tile_box)
    if len(hits) == 0:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        return mask_empty

    if isinstance(hits[0], (int, np.integer)):
        candidates = [geoms[int(idx)] for idx in hits]
    else:
        candidates = list(hits)
    shapes = [mapping(g) for g in candidates if g.intersects(tile_box)]
    if not shapes:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        return mask_empty

    t0 = time_start()
    mask = rfeatures.rasterize(
        shapes=[(geom, 1) for geom in shapes],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    )
    if target_shape is not None and mask.shape != target_shape:
        mask = resize(
            mask,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype("uint8")
    mask_bool = mask.astype(bool)
    time_end("roads_mask_rasterize", t0)
    logger.info(
        "roads mask rasterized: shapes=%s coverage=%.4f",
        len(shapes),
        float(mask_bool.mean()),
    )
    _ROADS_MASK_CACHE[key] = mask_bool
    return mask_bool


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


def _summarize_phase_metrics(acc: dict[str, list[dict]], label: str) -> None:
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

    champ_chain = ["champion_raw", "champion_crf", "champion_shadow"]
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


def summarize_phase_metrics_mean_std(
    phase_metrics: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Summarize phase metrics as mean/std pairs across runs.

    Args:
        phase_metrics (dict[str, list[dict]]): Phase metrics keyed by phase.

    Returns:
        dict[str, dict[str, float]]: Per-phase metric summary.

    Examples:
        >>> summarize_phase_metrics_mean_std({}) == {}
        True
    """
    out: dict[str, dict[str, float]] = {}
    metric_keys = ["iou", "f1", "precision", "recall"]
    for phase, rows in phase_metrics.items():
        if not rows:
            continue
        phase_summary: dict[str, float] = {}
        for key in metric_keys:
            vals = [float(r.get(key, 0.0)) for r in rows]
            phase_summary[f"{key}_mean"] = float(np.mean(vals))
            phase_summary[f"{key}_std"] = float(np.std(vals))
        out[phase] = phase_summary
    return out


def write_rolling_best_config(
    out_path: str,
    stage: str,
    tuned: dict,
    fold_done: int,
    fold_total: int,
    holdout_done: int,
    holdout_total: int,
    best_fold: dict | None = None,
) -> None:
    """Write rolling best config checkpoint for interruption-safe resume context.

    Args:
        out_path (str): Destination YAML path.
        stage (str): Current pipeline stage.
        tuned (dict): Tuned settings bundle.
        fold_done (int): Completed LOO folds.
        fold_total (int): Total LOO folds.
        holdout_done (int): Processed holdout tiles.
        holdout_total (int): Total holdout tiles.
        best_fold (dict | None): Optional best-fold metadata.

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
    }
    if best_fold is not None:
        payload["selected_fold"] = {
            "fold_index": int(best_fold["fold_index"]),
            "val_tile": best_fold["val_tile"],
            "val_champion_shadow_iou": float(best_fold["val_champion_shadow_iou"]),
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)


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
    ds = int(cfg.model.backbone.resample_factor or 1)
    img_b = load_dop20_image(img_path, downsample_factor=ds)
    labels_sh = reproject_labels_to_image(
        img_path, cfg.io.paths.source_label_raster, downsample_factor=ds
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
    buffer_m = cfg.model.priors.buffer_m
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    logger.info(
        "tile=%s pixel_size=%.3f m, buffer_m=%s, buffer_pixels=%s",
        img_path,
        pixel_size_m,
        buffer_m,
        buffer_pixels,
    )

    sh_buffer_mask = build_sh_buffer_mask(labels_sh, buffer_pixels)
    if gt_mask is not None and cfg.model.priors.clip_gt_to_buffer:
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
    ds = int(cfg.model.backbone.resample_factor or 1)
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
        roads_mask = _get_roads_mask(val_path, ds, target_shape=img_b.shape[:2])
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
    use_gpu_xgb = cfg.search.xgb.use_gpu
    param_grid = cfg.search.xgb.param_grid
    num_boost_round = cfg.search.xgb.num_boost_round
    early_stop = cfg.search.xgb.early_stop
    verbose_eval = cfg.search.xgb.verbose_eval
    val_fraction = cfg.search.xgb.val_fraction
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

    roads_penalties = [float(p) for p in cfg.postprocess.roads.penalty_values]
    best_bundle = None
    best_champion_iou = None

    for penalty in roads_penalties:
        logger.info("tune: roads penalty=%s", penalty)

        # kNN tuning (weighted-mean IoU across val tiles)
        best_raw_config = None
        for k in cfg.search.knn.k_values:
            stats_by_thr = {
                thr: {
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "w": 0.0,
                    "n": 0,
                }
                for thr in cfg.search.knn.thresholds.values
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
                    neg_alpha=cfg.model.banks.neg_alpha,
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
                        cfg.search.knn.thresholds.values,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    metrics_list = compute_metrics_batch_cpu(
                        score_full,
                        cfg.search.knn.thresholds.values,
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
                for thr in cfg.search.knn.thresholds.values
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
                        cfg.search.knn.thresholds.values,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    metrics_list = compute_metrics_batch_cpu(
                        score_full,
                        cfg.search.knn.thresholds.values,
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
                neg_alpha=cfg.model.banks.neg_alpha,
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
        for psf in cfg.search.crf.prob_softness_values
        for pw in cfg.search.crf.pos_w_values
        for pxy in cfg.search.crf.pos_xy_std_values
        for bw in cfg.search.crf.bilateral_w_values
        for bxy in cfg.search.crf.bilateral_xy_std_values
        for brgb in cfg.search.crf.bilateral_rgb_std_values
    ]
    best_crf_cfg = None
    best_crf_iou = None
    crf_candidates = crf_candidates[:CRF_MAX_CONFIGS]
    num_workers = int(cfg.search.crf.num_workers or 1)
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
    protect_scores = cfg.postprocess.shadow.protect_scores
    for weights in cfg.postprocess.shadow.weight_sets:
        iou_by_key = {
            (thr, protect_score): {"sum": 0.0, "w": 0.0}
            for thr in cfg.postprocess.shadow.thresholds
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
            thr_arr = np.array(
                cfg.postprocess.shadow.thresholds, dtype=np.float32
            ).reshape(-1, 1)
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
                for i, thr in enumerate(cfg.postprocess.shadow.thresholds):
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
    ds = int(cfg.model.backbone.resample_factor or 1)
    roads_mask = _get_roads_mask(holdout_path, ds, target_shape=img_b.shape[:2])
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
        neg_alpha=cfg.model.banks.neg_alpha,
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
    mask_crf_knn = refine_with_densecrf(
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
    )
    mask_crf_xgb = refine_with_densecrf(
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
    )
    if champion_source == "raw":
        best_crf_mask = mask_crf_knn
    else:
        best_crf_mask = mask_crf_xgb

    shadow_cfg = tuned["shadow_cfg"]
    protect_score = shadow_cfg.get("protect_score")
    shadow_mask = _apply_shadow_filter(
        img_b,
        best_crf_mask,
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
    shadow_metrics = compute_metrics(shadow_mask, gt_mask_eval)
    metrics_map = {
        "knn_raw": metrics_knn,
        "knn_crf": metrics_knn_crf,
        "knn_shadow": metrics_knn_shadow,
        "xgb_raw": metrics_xgb,
        "xgb_crf": metrics_xgb_crf,
        "xgb_shadow": metrics_xgb_shadow,
        "champion_raw": metrics_champion_raw,
        "champion_crf": metrics_champion_crf,
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
        "champion_shadow": shadow_mask,
    }
    save_unified_plot(
        img_b=img_b,
        gt_mask=gt_mask_eval,
        labels_sh=labels_sh,
        masks=masks_map,
        metrics=metrics_map,
        plot_dir=cfg.io.paths.plot_dir,
        image_id_b=image_id_b,
        show_metrics=plot_with_metrics and gt_available,
        gt_available=gt_available,
        similarity_map=score_knn_raw,
        score_maps={"knn": score_knn, "xgb": score_xgb},
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
    model_name = cfg.model.backbone.name

    # ------------------------------------------------------------
    # Output organization (one folder per run)
    # ------------------------------------------------------------
    output_root = cfg.io.paths.output_dir
    os.makedirs(output_root, exist_ok=True)
    resume_run = bool(cfg.runtime.resume_run)
    resume_dir = cfg.runtime.resume_run_dir
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
    cfg.io.paths.plot_dir = plot_dir
    cfg.io.paths.best_settings_path = os.path.join(run_dir, "best_settings.yml")
    cfg.io.paths.log_path = os.path.join(run_dir, "run.log")
    setup_logging(cfg.io.paths.log_path)
    processed_log_path = os.path.join(run_dir, "processed_tiles.jsonl")
    processed_tiles: set[str] = set()
    rolling_best_settings_path = os.path.join(run_dir, "rolling_best_setting.yml")
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

    union_backup_every = int(cfg.runtime.union_backup_every or 0)
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
    ps = cfg.model.backbone.patch_size
    tile_size = cfg.model.tiling.tile_size
    stride = cfg.model.tiling.stride

    # ------------------------------------------------------------
    # Resolve paths to imagery + SH_2022 + GT vector labels
    # ------------------------------------------------------------
    source_label_raster = cfg.io.paths.source_label_raster
    gt_vector_paths = cfg.io.paths.eval_gt_vectors
    auto_split_tiles = cfg.io.auto_split.enabled

    # ------------------------------------------------------------
    # Resolve GT-positive tiles (LOO set) and inference-only tiles
    # ------------------------------------------------------------
    if not auto_split_tiles:
        raise ValueError(
            "io.auto_split.enabled must be true: directory-driven LOO training is required"
        )
    tiles_dir = cfg.io.auto_split.tiles_dir
    tile_glob = cfg.io.auto_split.tile_glob
    downsample_factor = cfg.io.auto_split.gt_presence_downsample
    num_workers = cfg.io.auto_split.gt_presence_workers
    gt_tiles, holdout_tiles = resolve_tiles_from_gt_presence(
        tiles_dir,
        tile_glob,
        gt_vector_paths,
        downsample_factor=downsample_factor,
        num_workers=num_workers,
    )
    val_tiles = list(gt_tiles)
    logger.info(
        "auto split tiles: gt_tiles=%s inference_tiles=%s",
        len(val_tiles),
        len(holdout_tiles),
    )

    context_radius = int(cfg.model.banks.feat_context_radius or 0)
    min_train_tiles = int(cfg.training.loo.min_train_tiles or 1)

    # Resolve required tile sets
    # ------------------------------------------------------------
    if not val_tiles:
        raise ValueError("no GT-positive tiles resolved for LOO training")
    if not holdout_tiles:
        logger.warning("no inference-only tiles resolved; skipping holdout inference")

    # ------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------
    feature_cache_mode = cfg.runtime.feature_cache_mode
    if feature_cache_mode not in {"disk", "memory"}:
        raise ValueError("FEATURE_CACHE_MODE must be 'disk' or 'memory'")
    if feature_cache_mode == "disk":
        feature_dir = cfg.io.paths.feature_dir
        os.makedirs(feature_dir, exist_ok=True)
    else:
        feature_dir = None
    logger.info("feature cache mode: %s", feature_cache_mode)

    val_phase_metrics: dict[str, list[dict]] = {}
    holdout_phase_metrics: dict[str, list[dict]] = {}
    val_buffer_m = None
    val_pixel_size_m = None
    loo_fold_records = []

    # ------------------------------------------------------------
    # LOO tuning/search: train on N-1 GT tiles, validate on the left-out tile
    # ------------------------------------------------------------
    if not cfg.training.loo.enabled:
        raise ValueError("training.loo.enabled must be true for this pipeline")
    _log_phase("START", "loo_validation_tuning")
    for fold_idx, val_path in enumerate(val_tiles, start=1):
        train_tiles = [p for p in gt_tiles if p != val_path]
        if not train_tiles:
            logger.warning(
                "LOO fold %s has no train tiles; reusing validation tile as source",
                fold_idx,
            )
            train_tiles = [val_path]
        if len(train_tiles) < min_train_tiles:
            logger.warning(
                "LOO fold %s train tiles=%s < min_train_tiles=%s",
                fold_idx,
                len(train_tiles),
                min_train_tiles,
            )
        logger.info(
            "LOO fold %s/%s train_tiles=%s val_tile=%s",
            fold_idx,
            len(val_tiles),
            len(train_tiles),
            val_path,
        )
        pos_bank_fold, neg_bank_fold, x_fold, y_fold, _, _ = (
            build_training_artifacts_for_tiles(
                train_tiles,
                source_label_raster,
                model,
                processor,
                device,
                ps,
                tile_size,
                stride,
                feature_cache_mode,
                feature_dir,
                context_radius,
            )
        )
        tuned_fold = tune_on_validation_multi(
            [val_path],
            gt_vector_paths,
            model,
            processor,
            device,
            pos_bank_fold,
            neg_bank_fold,
            x_fold,
            y_fold,
            ps,
            tile_size,
            stride,
            feature_dir,
            context_radius,
        )
        fold_result = infer_on_holdout(
            val_path,
            gt_vector_paths,
            model,
            processor,
            device,
            pos_bank_fold,
            neg_bank_fold,
            tuned_fold,
            ps,
            tile_size,
            stride,
            feature_dir,
            shape_dir,
            context_radius,
            plot_with_metrics=True,
        )
        if fold_result["gt_available"]:
            _update_phase_metrics(val_phase_metrics, fold_result["metrics"])
        if val_buffer_m is None:
            val_buffer_m = fold_result["buffer_m"]
            val_pixel_size_m = fold_result["pixel_size_m"]
        loo_fold_records.append(
            {
                "fold_index": fold_idx,
                "val_tile": val_path,
                "train_tiles_count": len(train_tiles),
                "val_champion_shadow_iou": float(
                    fold_result["metrics"]["champion_shadow"]["iou"]
                ),
                "roads_penalty": float(tuned_fold.get("roads_penalty", 1.0)),
                "champion_source": tuned_fold["champion_source"],
                "best_raw_config": tuned_fold["best_raw_config"],
                "best_xgb_config": tuned_fold["best_xgb_config"],
                "best_crf_config": tuned_fold["best_crf_config"],
                "best_shadow_config": tuned_fold["shadow_cfg"],
                "phase_metrics": fold_result["metrics"],
                "tuned": tuned_fold,
            }
        )
        current_best_fold = max(
            loo_fold_records,
            key=lambda r: float(r["val_champion_shadow_iou"]),
        )
        write_rolling_best_config(
            rolling_best_settings_path,
            stage="loo_validation_tuning",
            tuned=current_best_fold["tuned"],
            fold_done=len(loo_fold_records),
            fold_total=len(val_tiles),
            holdout_done=len(processed_tiles),
            holdout_total=len(holdout_tiles),
            best_fold=current_best_fold,
        )
    _log_phase("END", "loo_validation_tuning")
    if not loo_fold_records:
        raise ValueError("LOO tuning produced no fold records")

    best_fold = max(
        loo_fold_records,
        key=lambda r: float(r["val_champion_shadow_iou"]),
    )
    selected_tuned = best_fold["tuned"]
    logger.info(
        "LOO selected fold=%s val_tile=%s champion_shadow_iou=%.3f",
        best_fold["fold_index"],
        best_fold["val_tile"],
        float(best_fold["val_champion_shadow_iou"]),
    )

    # ------------------------------------------------------------
    # Final training on all GT tiles with selected LOO hyperparameters
    # ------------------------------------------------------------
    _log_phase("START", "final_all_gt_training")
    pos_bank, neg_bank, X, y, image_id_a_list, aug_modes = (
        build_training_artifacts_for_tiles(
            gt_tiles,
            source_label_raster,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            feature_cache_mode,
            feature_dir,
            context_radius,
        )
    )
    best_xgb_params = selected_tuned["best_xgb_config"].get("params")
    final_bst = train_xgb_classifier(
        X,
        y,
        use_gpu=cfg.search.xgb.use_gpu,
        num_boost_round=cfg.search.xgb.num_boost_round,
        verbose_eval=cfg.search.xgb.verbose_eval,
        param_overrides=best_xgb_params,
    )
    tuned = {**selected_tuned, "bst": final_bst}
    _log_phase("END", "final_all_gt_training")
    write_rolling_best_config(
        rolling_best_settings_path,
        stage="final_model_ready",
        tuned=tuned,
        fold_done=len(loo_fold_records),
        fold_total=len(val_tiles),
        holdout_done=len(processed_tiles),
        holdout_total=len(holdout_tiles),
        best_fold=best_fold,
    )

    loo_fold_export = []
    for fold in loo_fold_records:
        loo_fold_export.append(
            {
                "fold_index": int(fold["fold_index"]),
                "val_tile": fold["val_tile"],
                "train_tiles_count": int(fold["train_tiles_count"]),
                "val_champion_shadow_iou": float(fold["val_champion_shadow_iou"]),
                "roads_penalty": float(fold["roads_penalty"]),
                "champion_source": fold["champion_source"],
                "best_raw_config": fold["best_raw_config"],
                "best_xgb_config": fold["best_xgb_config"],
                "best_crf_config": fold["best_crf_config"],
                "best_shadow_config": fold["best_shadow_config"],
            }
        )
    loo_phase_mean_std = summarize_phase_metrics_mean_std(val_phase_metrics)

    weighted_phase_metrics: dict[str, dict[str, float]] = {}
    metric_keys = ["iou", "f1", "precision", "recall"]
    for phase, metrics_list in val_phase_metrics.items():
        weights = [float(m.get("_weight", 0.0)) for m in metrics_list]
        weighted_phase_metrics[phase] = {
            key: _weighted_mean([m.get(key, 0.0) for m in metrics_list], weights)
            for key in metric_keys
        }

    inference_best_settings_path = os.path.join(run_dir, "inference_best_setting.yml")
    bst = tuned["bst"]
    xgb_model_info: dict[str, object] = {}
    if bst is not None:
        best_iter = getattr(bst, "best_iteration", None)
        best_score = getattr(bst, "best_score", None)
        xgb_model_info = {
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "best_score": float(best_score) if best_score is not None else None,
            "num_features": int(bst.num_features()),
            "attributes": bst.attributes(),
        }
    model_info = {
        "backbone": {
            "name": cfg.model.backbone.name,
            "patch_size": cfg.model.backbone.patch_size,
            "resample_factor": cfg.model.backbone.resample_factor,
        },
        "tiling": {"tile_size": tile_size, "stride": stride},
        "augmentation": {
            "enabled": cfg.model.augmentation.enabled,
            "modes": aug_modes,
        },
        "xgb_search": {
            "use_gpu": cfg.search.xgb.use_gpu,
            "num_boost_round": cfg.search.xgb.num_boost_round,
            "early_stop": cfg.search.xgb.early_stop,
            "param_grid_size": len(cfg.search.xgb.param_grid),
        },
        "knn_search": {
            "k_values": cfg.search.knn.k_values,
            "threshold_range": {
                "start": cfg.search.knn.thresholds.start,
                "stop": cfg.search.knn.thresholds.stop,
                "count": cfg.search.knn.thresholds.count,
            },
        },
    }
    export_best_settings(
        tuned["best_raw_config"],
        tuned["best_crf_config"],
        cfg.model.backbone.name,
        gt_tiles,
        f"inference_tiles={len(holdout_tiles)}",
        float(val_buffer_m) if val_buffer_m is not None else 0.0,
        float(val_pixel_size_m) if val_pixel_size_m is not None else 0.0,
        shadow_cfg=tuned["shadow_cfg"],
        best_xgb_config=tuned["best_xgb_config"],
        champion_source=tuned["champion_source"],
        xgb_model_info=xgb_model_info,
        model_info=model_info,
        extra_settings={
            "tile_size": tile_size,
            "stride": stride,
            "patch_size": ps,
            "feat_context_radius": context_radius,
            "neg_alpha": cfg.model.banks.neg_alpha,
            "pos_frac_thresh": cfg.model.banks.pos_frac_thresh,
            "roads_penalty": tuned.get("roads_penalty", 1.0),
            "roads_mask_path": cfg.io.paths.roads_mask_path,
            "val_tiles_count": len(val_tiles),
            "holdout_tiles_count": len(holdout_tiles),
            "weighted_phase_metrics": weighted_phase_metrics,
            "loo": {
                "enabled": True,
                "fold_count": len(loo_fold_export),
                "min_train_tiles": min_train_tiles,
                "selected_fold_index": int(best_fold["fold_index"]),
                "selected_val_tile": best_fold["val_tile"],
                "phase_metrics_mean_std": loo_phase_mean_std,
                "folds": loo_fold_export,
            },
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
        write_rolling_best_config(
            rolling_best_settings_path,
            stage="holdout_inference",
            tuned=tuned,
            fold_done=len(loo_fold_records),
            fold_total=len(val_tiles),
            holdout_done=holdout_tiles_processed,
            holdout_total=len(holdout_tiles),
            best_fold=best_fold,
        )
    _log_phase("END", "holdout_inference")

    _summarize_phase_metrics(val_phase_metrics, "loo_validation")
    _summarize_phase_metrics(holdout_phase_metrics, "holdout")

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
