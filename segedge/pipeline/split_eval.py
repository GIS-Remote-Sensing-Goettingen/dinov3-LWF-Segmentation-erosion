"""Split evaluation entrypoint for SegEdge."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from scipy.ndimage import median_filter

import config as cfg

from ..core.banks import build_banks_single_scale
from ..core.crf_utils import crf_grid_search, refine_with_densecrf
from ..core.features import prefetch_features_single_scale_image
from ..core.io_utils import load_dop20_image, reproject_labels_to_image
from ..core.knn import (
    fine_tune_threshold,
    grid_search_k_threshold,
    zero_shot_knn_single_scale_B_with_saliency,
)
from ..core.logging_utils import setup_logging
from ..core.metrics_utils import (
    compute_metrics,
    compute_metrics_batch_cpu,
    compute_metrics_batch_gpu,
)
from ..core.xdboost import (
    build_xgb_dataset,
    hyperparam_search_xgb_iou,
    train_xgb_classifier,
    xgb_score_image_b,
)
from .common import build_xgb_training_data, init_model, log_metrics, prep_b_tile

logger = logging.getLogger(__name__)


def _shadow_weighted_sum(
    img_rgb: np.ndarray,
    weights,
) -> np.ndarray:
    """Compute weighted RGB sum for shadow filtering.

    Args:
        img_rgb (np.ndarray): RGB image array.
        weights (Iterable[float]): RGB weights.

    Returns:
        np.ndarray: Weighted sum image (H, W) float32.

    Examples:
        >>> import numpy as np
        >>> img = np.ones((1, 1, 3), dtype=np.uint8) * 10
        >>> _shadow_weighted_sum(img, (1.0, 2.0, 3.0)).item()
        60.0
    """
    img_float = img_rgb.astype(np.float32)
    w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
    return (img_float * w).sum(axis=2)


def _apply_shadow_filter(
    base_mask: np.ndarray,
    shadow_pass: np.ndarray,
    score_full: np.ndarray,
    protect_score: float | None,
) -> np.ndarray:
    """Apply shadow filtering with optional protect score override.

    Args:
        base_mask (np.ndarray): Base prediction mask.
        shadow_pass (np.ndarray): Shadow-pass mask from weighted sum threshold.
        score_full (np.ndarray): Full-resolution score map.
        protect_score (float | None): Score threshold to override shadows.

    Returns:
        np.ndarray: Filtered mask.

    Examples:
        >>> import numpy as np
        >>> base = np.array([[1, 0]], dtype=bool)
        >>> shadow = np.array([[0, 1]], dtype=bool)
        >>> score = np.array([[0.6, 0.2]], dtype=np.float32)
        >>> _apply_shadow_filter(base, shadow, score, 0.5).tolist()
        [[True, False]]
    """
    base_mask_bool = base_mask.astype(bool)
    shadow_pass_bool = shadow_pass.astype(bool)
    if protect_score is None:
        return base_mask_bool & shadow_pass_bool
    return base_mask_bool & (shadow_pass_bool | (score_full >= protect_score))


def tune_on_validation(
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
    feature_dir: str | None,
    img_b_path,
    gt_paths,
):
    """Tune pipeline settings on a validation tile.

    Args:
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        pos_bank (np.ndarray): Positive bank.
        neg_bank (np.ndarray | None): Negative bank.
        X (np.ndarray | None): Feature matrix from Image A.
        y (np.ndarray | None): Binary labels from Image A.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str | None): Feature cache directory.
        img_b_path (str): Validation tile path.
        gt_paths (list[str]): Vector GT paths.

    Returns:
        dict: Tuned configurations and models.

    Examples:
        >>> callable(tune_on_validation)
        True
    """
    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    img_b, _labels_sh_b, gt_mask_eval, sh_buffer_mask = prep_b_tile(
        img_b_path, gt_paths
    )
    image_id_b = os.path.splitext(os.path.basename(img_b_path))[0]

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

    best_raw_config, best_raw_score_full, _ = grid_search_k_threshold(
        img_b,
        pos_bank,
        neg_bank,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        cfg.K_VALUES,
        cfg.THRESHOLDS,
        feature_dir,
        image_id_b,
        sh_buffer_mask,
        gt_mask_eval,
        prefetched_b,
        getattr(cfg, "USE_FP16_KNN", True),
        context_radius=context_radius,
    )
    if best_raw_config is None or best_raw_score_full is None:
        raise ValueError("kNN grid search returned no results")
    thr_refined, metrics_refined, mask_raw_best = fine_tune_threshold(
        best_raw_score_full,
        best_raw_config["threshold"],
        sh_buffer_mask,
        gt_mask_eval,
    )
    if metrics_refined["iou"] >= best_raw_config["iou"]:
        best_raw_config = {
            **best_raw_config,
            "threshold": thr_refined,
            **metrics_refined,
        }
    else:
        mask_raw_best = (
            best_raw_score_full >= best_raw_config["threshold"]
        ) & sh_buffer_mask
    mask_raw_best = median_filter(mask_raw_best.astype(np.uint8), size=3) > 0
    metrics_raw_filtered = compute_metrics(mask_raw_best, gt_mask_eval)
    best_raw_config = {**best_raw_config, **metrics_raw_filtered}
    log_metrics("val kNN", best_raw_config)

    if X is None or y is None:
        if feature_dir is None:
            raise ValueError("feature_dir must be set when X/y are not provided")
        X, y = build_xgb_training_data(ps, tile_size, stride, feature_dir)
    use_gpu_xgb = getattr(cfg, "XGB_USE_GPU", True)
    param_grid = getattr(cfg, "XGB_PARAM_GRID", None)
    num_boost_round = getattr(cfg, "XGB_NUM_BOOST_ROUND", 300)
    early_stop = getattr(cfg, "XGB_EARLY_STOP", 40)
    verbose_eval = getattr(cfg, "XGB_VERBOSE_EVAL", 50)
    val_fraction = getattr(cfg, "XGB_VAL_FRACTION", 0.2)

    if param_grid:
        bst, best_params_xgb, _, best_thr_xgb, best_metrics_xgb = (
            hyperparam_search_xgb_iou(
                X,
                y,
                cfg.THRESHOLDS,
                sh_buffer_mask,
                gt_mask_eval,
                img_b,
                ps,
                tile_size,
                stride,
                feature_dir,
                image_id_b,
                prefetched_tiles=prefetched_b,
                device=device,
                use_gpu=use_gpu_xgb,
                param_grid=param_grid,
                num_boost_round=num_boost_round,
                val_fraction=val_fraction,
                early_stopping_rounds=early_stop,
                verbose_eval=verbose_eval,
                seed=42,
                context_radius=context_radius,
            )
        )
        if best_thr_xgb is None or best_metrics_xgb is None:
            raise ValueError("XGB search returned no results")
        best_xgb_config = {
            "k": -1,
            "threshold": best_thr_xgb,
            "source": "xgb",
            **best_metrics_xgb,
            "params": best_params_xgb,
        }
    else:
        bst = train_xgb_classifier(
            X,
            y,
            use_gpu=use_gpu_xgb,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
        )
        score_full_xgb = xgb_score_image_b(
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
        try:
            metrics_list = compute_metrics_batch_gpu(
                score_full_xgb,
                cfg.THRESHOLDS,
                sh_buffer_mask,
                gt_mask_eval,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            metrics_list = compute_metrics_batch_cpu(
                score_full_xgb, cfg.THRESHOLDS, sh_buffer_mask, gt_mask_eval
            )
        best_xgb = max(metrics_list, key=lambda m: m["iou"])
        mask_xgb = (score_full_xgb >= best_xgb["threshold"]) & sh_buffer_mask
        mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
        metrics_xgb_filtered = compute_metrics(mask_xgb, gt_mask_eval)
        best_xgb_config = {
            "k": -1,
            "threshold": best_xgb["threshold"],
            "source": "xgb",
            **metrics_xgb_filtered,
            "params": None,
        }

    log_metrics("val XGB", best_xgb_config)

    champion_config = (
        best_raw_config
        if best_raw_config["iou"] >= best_xgb_config["iou"]
        else best_xgb_config
    )
    champion_source = champion_config["source"]
    champion_score = (
        best_raw_score_full
        if champion_source == "raw"
        else xgb_score_image_b(
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
    )
    thr_center = champion_config["threshold"]

    best_crf_cfg, best_crf_mask = crf_grid_search(
        img_b,
        champion_score,
        thr_center,
        sh_buffer_mask,
        gt_mask_eval,
        cfg.PROB_SOFTNESS_VALUES,
        cfg.POS_W_VALUES,
        cfg.POS_XY_STD_VALUES,
        cfg.BILATERAL_W_VALUES,
        cfg.BILATERAL_XY_STD_VALUES,
        cfg.BILATERAL_RGB_STD_VALUES,
        5,
        getattr(cfg, "CRF_MAX_CONFIGS", 64),
        2,
        cfg.CRF_NUM_WORKERS,
        "process",
    )
    if best_crf_cfg is None or best_crf_mask is None:
        raise ValueError("CRF grid search returned no results")
    log_metrics("val CRF", best_crf_cfg)

    protect_scores = getattr(cfg, "SHADOW_PROTECT_SCORES", [0.5])
    best_shadow_cfg = None
    best_shadow_iou = -1.0
    for weights in cfg.SHADOW_WEIGHT_SETS:
        wsum = _shadow_weighted_sum(img_b, weights)
        for thr in cfg.SHADOW_THRESHOLDS:
            shadow_pass = wsum >= thr
            for protect_score in protect_scores:
                mask_shadow = _apply_shadow_filter(
                    best_crf_mask,
                    shadow_pass,
                    champion_score,
                    protect_score,
                )
                metrics_shadow = compute_metrics(mask_shadow, gt_mask_eval)
                if metrics_shadow["iou"] > best_shadow_iou:
                    best_shadow_iou = metrics_shadow["iou"]
                    best_shadow_cfg = {
                        "weights": tuple(float(x) for x in weights),
                        "threshold": float(thr),
                        "protect_score": float(protect_score),
                        **metrics_shadow,
                    }
    if best_shadow_cfg is None:
        raise ValueError("shadow filter search returned no results")
    logger.info("val shadow best: %s", best_shadow_cfg)

    return {
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "best_crf_config": best_crf_cfg,
        "best_shadow_cfg": best_shadow_cfg,
        "bst": bst,
    }


def eval_holdout_tile(
    model,
    processor,
    device,
    pos_bank,
    neg_bank,
    ps,
    tile_size,
    stride,
    feature_dir: str | None,
    img_b_path,
    gt_paths,
    tuned,
):
    """Evaluate tuned settings on a holdout tile.

    Args:
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        pos_bank (np.ndarray): Positive bank.
        neg_bank (np.ndarray | None): Negative bank.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str | None): Feature cache directory.
        img_b_path (str): Holdout tile path.
        gt_paths (list[str]): Vector GT paths.
        tuned (dict): Tuned configuration bundle.

    Returns:
        None

    Examples:
        >>> callable(eval_holdout_tile)
        True
    """
    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    img_b, _labels_sh_b, gt_mask_eval, sh_buffer_mask = prep_b_tile(
        img_b_path, gt_paths
    )
    image_id_b = os.path.splitext(os.path.basename(img_b_path))[0]
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
        use_fp16_matmul=getattr(cfg, "USE_FP16_KNN", True),
        context_radius=context_radius,
    )
    knn_thr = tuned["best_raw_config"]["threshold"]
    mask_knn = (score_knn >= knn_thr) & sh_buffer_mask
    mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0
    metrics_knn = compute_metrics(mask_knn, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} kNN", metrics_knn)

    bst = tuned["bst"]
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
    xgb_thr = tuned["best_xgb_config"]["threshold"]
    mask_xgb = (score_xgb >= xgb_thr) & sh_buffer_mask
    mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
    metrics_xgb = compute_metrics(mask_xgb, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} XGB", metrics_xgb)

    champion_source = tuned["champion_source"]
    if champion_source == "raw":
        champion_score = score_knn
        thr_center = knn_thr
    else:
        champion_score = score_xgb
        thr_center = xgb_thr

    crf_cfg = tuned["best_crf_config"]
    mask_crf = refine_with_densecrf(
        img_b,
        champion_score,
        thr_center,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
    )
    metrics_crf = compute_metrics(mask_crf, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} CRF", metrics_crf)

    shadow_cfg = tuned["best_shadow_cfg"]
    shadow_pass = (
        _shadow_weighted_sum(img_b, shadow_cfg["weights"]) >= shadow_cfg["threshold"]
    )
    protect_score = shadow_cfg.get("protect_score")
    mask_shadow = _apply_shadow_filter(
        mask_crf,
        shadow_pass,
        champion_score,
        protect_score,
    )
    metrics_shadow = compute_metrics(mask_shadow, gt_mask_eval)
    log_metrics(f"holdout {image_id_b} shadow", metrics_shadow)


def main():
    """Run split evaluation (tune on validation, eval on holdouts).

    Examples:
        >>> callable(main)
        True
    """
    setup_logging(getattr(cfg, "LOG_PATH", None))
    model, processor, device = init_model(cfg.MODEL_NAME)
    ps = getattr(cfg, "PATCH_SIZE", model.config.patch_size)
    tile_size = getattr(cfg, "TILE_SIZE", 1024)
    stride = getattr(cfg, "STRIDE", tile_size)
    feature_cache_mode = getattr(cfg, "FEATURE_CACHE_MODE", "disk")
    if feature_cache_mode not in {"disk", "memory"}:
        raise ValueError("FEATURE_CACHE_MODE must be 'disk' or 'memory'")
    if feature_cache_mode == "disk":
        feature_dir = cfg.FEATURE_DIR
        os.makedirs(feature_dir, exist_ok=True)
    else:
        feature_dir = None
    logger.info("feature cache mode: %s", feature_cache_mode)

    val_b_path = getattr(cfg, "VAL_TILE", None) or cfg.TARGET_TILE
    gt_paths = cfg.EVAL_GT_VECTORS

    holdout_b_paths = getattr(cfg, "HOLDOUT_TILES", None)
    if not holdout_b_paths:
        raise ValueError("HOLDOUT_TILES must be set for split evaluation.")

    img_a_paths = getattr(cfg, "SOURCE_TILES", None) or [cfg.SOURCE_TILE]
    lab_a_paths = [cfg.SOURCE_LABEL_RASTER] * len(img_a_paths)
    image_id_a_list = [os.path.splitext(os.path.basename(p))[0] for p in img_a_paths]
    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)

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
        labels_a = reproject_labels_to_image(
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
            labels_a,
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
            labels_a,
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
    tuned = tune_on_validation(
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
        val_b_path,
        gt_paths,
    )

    for i, b_path in enumerate(holdout_b_paths):
        logger.info("holdout eval on %s (gt=%s)", b_path, gt_paths)
        eval_holdout_tile(
            model,
            processor,
            device,
            pos_bank,
            neg_bank,
            ps,
            tile_size,
            stride,
            feature_dir,
            b_path,
            gt_paths,
            tuned,
        )


if __name__ == "__main__":
    main()
