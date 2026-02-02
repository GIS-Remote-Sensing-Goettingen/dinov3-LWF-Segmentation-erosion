"""XGBoost training and scoring utilities for SegEdge."""

from __future__ import annotations

import logging
import os

import numpy as np
import xgboost as xgb
from skimage.transform import resize

from .features import (
    add_local_context_mean,
    crop_to_multiple_of_ps,
    labels_to_patch_masks,
    load_tile_features_if_valid,
    tile_feature_path,
    tile_iterator,
)
from .metrics_utils import compute_metrics_batch_cpu, compute_metrics_batch_gpu

logger = logging.getLogger(__name__)


def build_xgb_dataset(
    img,
    labels,
    ps,
    tile_size,
    stride,
    feature_dir,
    image_id,
    pos_frac,
    max_neg=8000,
    context_radius: int = 0,
    prefetched_tiles: dict | None = None,
):
    """Build an XGBoost dataset from tiled features and labels.

    Args:
        img (np.ndarray): Input image.
        labels (np.ndarray): Label image aligned to img.
        ps (int): Patch size for cropping.
        tile_size (int): Size of each tile.
        stride (int): Stride for tiling.
        feature_dir (str | None): Directory containing precomputed features.
        image_id (str): Identifier for the image.
        pos_frac (float): Fraction threshold for positive patches.
        max_neg (int): Maximum number of negative samples.
        context_radius (int): Feature context radius.
        prefetched_tiles (dict | None): Optional in-memory tile feature cache.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix X and labels y.

    Examples:
        >>> callable(build_xgb_dataset)
        True
    """
    if feature_dir is None and prefetched_tiles is None:
        raise ValueError("feature_dir or prefetched_tiles must be provided")

    X_pos, X_neg = [], []
    missing_feature_tiles = 0
    resample_factor = int(getattr(__import__("config"), "RESAMPLE_FACTOR", 1) or 1)
    for y, x, img_tile, lab_tile in tile_iterator(img, labels, tile_size, stride):
        prefetched = prefetched_tiles.get((y, x)) if prefetched_tiles else None
        if prefetched is not None:
            h_eff = prefetched["h_eff"]
            w_eff = prefetched["w_eff"]
            if h_eff < ps or w_eff < ps:
                continue
            lab_c = lab_tile[:h_eff, :w_eff] if lab_tile is not None else None
            feats_tile = prefetched["feats"]
            hp = prefetched["hp"]
            wp = prefetched["wp"]
        else:
            if prefetched_tiles is not None and feature_dir is None:
                missing_feature_tiles += 1
                continue
            img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, lab_tile, ps)
            if h_eff < ps or w_eff < ps:
                continue
            fpath = tile_feature_path(feature_dir, image_id, y, x)
            if not os.path.exists(fpath):
                missing_feature_tiles += 1
                continue
            hp = h_eff // ps
            wp = w_eff // ps
            feats_tile = load_tile_features_if_valid(
                feature_dir,
                image_id,
                y,
                x,
                expected_hp=hp,
                expected_wp=wp,
                ps=ps,
                resample_factor=resample_factor,
            )
            if feats_tile is None:
                missing_feature_tiles += 1
                continue

        if lab_c is None:
            logger.warning("missing labels for tile y=%s x=%s; skipping", y, x)
            continue
        if context_radius and context_radius > 0:
            feats_tile = add_local_context_mean(feats_tile, int(context_radius))
        if hp is None or wp is None:
            logger.warning(
                "missing patch dimensions for tile y=%s x=%s; skipping", y, x
            )
            continue
        pos_mask, neg_mask = labels_to_patch_masks(
            lab_c, hp, wp, pos_frac_thresh=pos_frac
        )
        if pos_mask.any():
            X_pos.append(feats_tile[pos_mask])
        if neg_mask.any():
            X_neg.append(feats_tile[neg_mask])

    if missing_feature_tiles > 0:
        logger.warning(
            "build_xgb_dataset: skipped %s tiles with missing cached features for image_id=%s",
            missing_feature_tiles,
            image_id,
        )

    if not X_pos:
        logger.warning(
            "build_xgb_dataset produced no positive samples for image_id=%s; skipping this source tile",
            image_id,
        )
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_pos = np.concatenate(X_pos, axis=0)
    X_neg = np.concatenate(X_neg, axis=0) if X_neg else np.empty((0, X_pos.shape[1]))
    # Subsample negatives
    if len(X_neg) > max_neg:
        idx = np.random.default_rng(42).choice(len(X_neg), size=max_neg, replace=False)
        X_neg = X_neg[idx]

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    return X, y


def train_xgb_classifier(X, y, use_gpu=False, num_boost_round=300, verbose_eval=50):
    """Train a binary XGBoost classifier for patch embeddings.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary labels.
        use_gpu (bool): Use GPU histogram algorithm when available.
        num_boost_round (int): Number of boosting rounds.
        verbose_eval (int): Verbosity interval.

    Returns:
        xgb.Booster: Trained XGBoost booster.

    Examples:
        >>> callable(train_xgb_classifier)
        True
    """
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    try:
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train")],
            verbose_eval=verbose_eval,
        )
    except xgb.core.XGBoostError as e:
        if use_gpu and "gpu_hist" in str(e):
            logger.warning(
                "xgboost build does not support GPU; falling back to CPU hist."
            )
            params["tree_method"] = "hist"
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train")],
                verbose_eval=verbose_eval,
            )
        else:
            raise
    return bst


def hyperparam_search_xgb(
    X,
    y,
    use_gpu=False,
    param_grid=None,
    num_boost_round=300,
    val_fraction=0.2,
    early_stopping_rounds=40,
    verbose_eval=50,
    seed: int = 42,
):
    """Run hyperparameter search, selecting by validation logloss.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary labels.
        use_gpu (bool): Use GPU histogram algorithm when available.
        param_grid (list[dict] | None): Parameter overrides.
        num_boost_round (int): Number of boosting rounds.
        val_fraction (float): Validation fraction for early stopping.
        early_stopping_rounds (int): Early stopping patience.
        verbose_eval (int): Verbosity interval.
        seed (int): RNG seed.

    Returns:
        tuple[xgb.Booster | None, dict | None, float]: Best model, params, logloss.

    Examples:
        >>> callable(hyperparam_search_xgb)
        True
    """
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset for XGBoost.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - val_fraction)))
    train_idx, val_idx = idx[:split], idx[split:]

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    if param_grid is None:
        param_grid = [base_params]

    best_model = None
    best_params = None
    best_score = float("inf")  # logloss: lower is better

    for i, overrides in enumerate(param_grid):
        params = base_params.copy()
        params.update(overrides)
        params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        logger.info("xgb-search cfg %s/%s: %s", i + 1, len(param_grid), params)
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )
        except xgb.core.XGBoostError as e:
            if use_gpu and "gpu_hist" in str(e):
                logger.warning("xgboost build lacks GPU; retrying with CPU hist.")
                params["tree_method"] = "hist"
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                )
            else:
                raise

        score = (
            float(model.best_score)
            if hasattr(model, "best_score")
            else model.evals_result()["val"]["logloss"][-1]
        )
        best_ntree = getattr(
            model, "best_ntree_limit", getattr(model, "best_iteration", None)
        )
        logger.info(
            "xgb-search cfg %s: val_logloss=%.4f (best_ntree=%s)",
            i + 1,
            score,
            best_ntree,
        )

        if score < best_score:
            best_score = score
            best_model = model
            best_params = params.copy()

    return best_model, best_params, best_score


def hyperparam_search_xgb_iou(
    X,
    y,
    thresholds,
    sh_buffer_mask_b,
    gt_mask_b,
    img_b,
    ps,
    tile_size,
    stride,
    feature_dir,
    image_id_b,
    prefetched_tiles=None,
    device=None,
    use_gpu=False,
    param_grid=None,
    num_boost_round=300,
    val_fraction=0.2,
    early_stopping_rounds=40,
    verbose_eval=50,
    seed: int = 42,
    context_radius: int = 0,
):
    """Search hyperparameters using IoU on Image B as the selection metric.

    Args:
        X (np.ndarray): Feature matrix from Image A.
        y (np.ndarray): Binary labels from Image A.
        thresholds (list[float]): Thresholds to sweep on Image B scores.
        sh_buffer_mask_b (np.ndarray): SH buffer mask for Image B.
        gt_mask_b (np.ndarray): Ground-truth mask for Image B.
        img_b (np.ndarray): Image B array.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str): Feature cache directory.
        image_id_b (str): Image B identifier.
        prefetched_tiles (dict | None): Optional in-memory cache.
        device: Torch device for metrics (optional).
        use_gpu (bool): Use GPU histogram algorithm when available.
        param_grid (list[dict] | None): Parameter overrides.
        num_boost_round (int): Number of boosting rounds.
        val_fraction (float): Validation fraction for early stopping.
        early_stopping_rounds (int): Early stopping patience.
        verbose_eval (int): Verbosity interval.
        seed (int): RNG seed.
        context_radius (int): Feature context radius.

    Returns:
        tuple[xgb.Booster | None, dict | None, float, float | None, dict | None]:
            Best model, params, IoU, threshold, metrics.

    Examples:
        >>> callable(hyperparam_search_xgb_iou)
        True
    """
    best_model = None
    best_params = None
    best_iou = -1.0
    best_metrics = None
    best_thr = None

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "tree_method": "gpu_hist" if use_gpu else "hist",
    }
    if param_grid is None:
        param_grid = [base_params]

    # simple train/val split (used only for early stopping)
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset for XGBoost.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - val_fraction)))
    train_idx, val_idx = idx[:split], idx[split:]
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

    for i, overrides in enumerate(param_grid):
        params = base_params.copy()
        params.update(overrides)
        params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        logger.info("xgb-search-iou cfg %s/%s: %s", i + 1, len(param_grid), params)
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
            )
        except xgb.core.XGBoostError as e:
            if use_gpu and "gpu_hist" in str(e):
                logger.warning("xgboost build lacks GPU; retrying with CPU hist.")
                params["tree_method"] = "hist"
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                )
            else:
                raise

        # Evaluate IoU on Image B
        score_full_xgb = xgb_score_image_b(
            img_b,
            model,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id_b,
            prefetched_tiles=prefetched_tiles,
            context_radius=context_radius,
        )
        if device is not None:
            try:
                metrics_list = compute_metrics_batch_gpu(
                    score_full_xgb,
                    thresholds,
                    sh_buffer_mask_b,
                    gt_mask_b,
                    device=device,
                )
            except Exception:
                metrics_list = compute_metrics_batch_cpu(
                    score_full_xgb,
                    thresholds,
                    sh_buffer_mask_b,
                    gt_mask_b,
                )
        else:
            metrics_list = compute_metrics_batch_cpu(
                score_full_xgb,
                thresholds,
                sh_buffer_mask_b,
                gt_mask_b,
            )

        best_local = max(metrics_list, key=lambda m: m["iou"])
        logger.info(
            "xgb-search-iou cfg %s: thr=%.3f, IoU=%.3f, F1=%.3f",
            i + 1,
            best_local["threshold"],
            best_local["iou"],
            best_local["f1"],
        )

        if best_local["iou"] > best_iou:
            best_iou = best_local["iou"]
            best_thr = best_local["threshold"]
            best_metrics = best_local
            best_params = params.copy()
            best_model = model

    return best_model, best_params, best_iou, best_thr, best_metrics


def xgb_score_image_b(
    img_b,
    bst,
    ps,
    tile_size,
    stride,
    feature_dir,
    image_id_b,
    prefetched_tiles=None,
    context_radius: int = 0,
):
    """Apply a trained XGBoost model to Image B and return a score map.

    Args:
        img_b (np.ndarray): Image B array.
        bst (xgb.Booster): Trained booster.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str): Feature cache directory.
        image_id_b (str): Image B identifier.
        prefetched_tiles (dict | None): Optional in-memory cache.
        context_radius (int): Feature context radius.

    Returns:
        np.ndarray: Score map at pixel resolution.

    Examples:
        >>> callable(xgb_score_image_b)
        True
    """
    if feature_dir is None and prefetched_tiles is None:
        raise ValueError("feature_dir or prefetched_tiles must be provided")
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)

    if prefetched_tiles is not None:
        tile_items = sorted(prefetched_tiles.items())
        tile_iter = ((y, x, info) for (y, x), info in tile_items)
    else:
        tile_iter = (
            (y, x, img_tile)
            for y, x, img_tile, _ in tile_iterator(img_b, None, tile_size, stride)
        )

    for tile_entry in tile_iter:
        if prefetched_tiles is not None:
            y, x, feat_info = tile_entry
            feats_tile = feat_info["feats"]
            h_eff = feat_info["h_eff"]
            w_eff = feat_info["w_eff"]
            hp = feat_info["hp"]
            wp = feat_info["wp"]
        else:
            y, x, img_tile = tile_entry
            img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
            if h_eff < ps or w_eff < ps:
                continue
            fpath = tile_feature_path(feature_dir, image_id_b, y, x)
            if not os.path.exists(fpath):
                continue
            feats_tile = np.load(fpath)
            hp, wp = feats_tile.shape[:2]

        if context_radius and context_radius > 0:
            feats_tile = add_local_context_mean(feats_tile, int(context_radius))

        if hp is None or wp is None:
            logger.warning(
                "missing patch dimensions for tile y=%s x=%s; skipping", y, x
            )
            continue
        dtest = xgb.DMatrix(feats_tile.reshape(-1, feats_tile.shape[-1]))
        scores_patch = bst.predict(dtest).reshape(hp, wp)
        scores_tile = resize(
            scores_patch,
            (h_eff, w_eff),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        score_full[y : y + h_eff, x : x + w_eff] += scores_tile
        weight_full[y : y + h_eff, x : x + w_eff] += 1.0

    mask_nonzero = weight_full > 0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    return score_full
