"""XGBoost training and scoring utilities for SegEdge."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from skimage.transform import resize

import config as cfg

from .features import (
    add_local_context_mean,
    crop_to_multiple_of_ps,
    labels_to_patch_masks,
    load_tile_features_if_valid,
    tile_feature_path,
    tile_iterator,
)
from .metrics_utils import compute_metrics_batch_cpu, compute_metrics_batch_gpu
from .timing_utils import DEBUG_TIMING, time_end, time_start

logger = logging.getLogger(__name__)


def _resize_patch_map(
    patch_map: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    resize_device: torch.device | None,
) -> np.ndarray:
    """Resize a patch-grid map to pixel space, preferring CUDA interpolation.

    Examples:
        >>> isinstance(_resize_patch_map.__name__, str)
        True
    """
    if resize_device is not None and resize_device.type == "cuda":
        with torch.no_grad():
            patch_t = torch.from_numpy(patch_map).to(
                device=resize_device,
                dtype=torch.float32,
            )
            resized_t = F.interpolate(
                patch_t.unsqueeze(0).unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
        return resized_t.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    return resize(
        patch_map,
        (out_h, out_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def _class_balance(y: np.ndarray) -> tuple[int, int]:
    """Return positive and negative sample counts for binary labels.

    Examples:
        >>> _class_balance(np.array([1, 0, 1, 0], dtype=np.float32))
        (2, 2)
    """
    yb = np.asarray(y) > 0.5
    pos = int(np.count_nonzero(yb))
    neg = int(yb.size - pos)
    return pos, neg


def _auto_scale_pos_weight(y: np.ndarray) -> float:
    """Compute a clamped negative/positive ratio for XGBoost class balancing.

    Examples:
        >>> round(_auto_scale_pos_weight(np.array([1, 0, 0], dtype=np.float32)), 2)
        2.0
    """
    pos, neg = _class_balance(y)
    if pos <= 0:
        return 1.0
    ratio = float(neg) / float(max(pos, 1))
    max_ratio = float(getattr(cfg, "XGB_CLASS_WEIGHT_MAX", 25.0) or 25.0)
    return float(np.clip(ratio, 1.0, max_ratio))


def _build_binary_sample_weights(y: np.ndarray) -> np.ndarray | None:
    """Build per-sample weights for imbalanced binary training.

    Examples:
        >>> w = _build_binary_sample_weights(np.array([1, 0, 0], dtype=np.float32))
        >>> bool(w is None or (w[0] > 1.0 and w[1] == 1.0))
        True
    """
    if not bool(getattr(cfg, "XGB_USE_SAMPLE_WEIGHTS", True)):
        return None
    yb = np.asarray(y) > 0.5
    if yb.size == 0:
        return None
    w_pos = _auto_scale_pos_weight(y)
    weights = np.ones(yb.size, dtype=np.float32)
    weights[yb] = np.float32(w_pos)
    return weights


def _kfold_index_splits(
    n_samples: int,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic shuffled k-fold index splits.

    Examples:
        >>> splits = _kfold_index_splits(5, 3, 42)
        >>> len(splits)
        3
        >>> all(len(np.intersect1d(tr, va)) == 0 for tr, va in splits)
        True
    """
    if n_samples < 2:
        return []
    k = max(2, int(n_splits or 2))
    k = min(k, int(n_samples))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(k, n_samples // k, dtype=np.int64)
    fold_sizes[: n_samples % k] += 1
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for fold_size in fold_sizes.tolist():
        end = start + int(fold_size)
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
        start = end
    return splits


def build_xgb_dataset(
    img,
    labels,
    ps,
    tile_size,
    stride,
    feature_dir,
    image_id,
    pos_frac,
    neg_frac_max: float = 0.0,
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
        neg_frac_max (float): Maximum positive fraction for negative patches.
        max_neg (int): Maximum number of negative samples.
        context_radius (int): Feature context radius.
        prefetched_tiles (dict | None): Optional in-memory tile feature cache.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix X and labels y.

    Examples:
        >>> isinstance(build_xgb_dataset.__name__, str)
        True
    """
    if feature_dir is None and prefetched_tiles is None:
        raise ValueError("feature_dir or prefetched_tiles must be provided")

    X_pos, X_neg = [], []
    rng = np.random.default_rng(42)
    tile_neg_cap = int(getattr(cfg, "XGB_MAX_NEG_PER_TILE", 0) or 0)
    patch_pos_total = 0
    patch_neg_total = 0
    patch_ignored_total = 0
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
            lab_c,
            hp,
            wp,
            pos_frac_thresh=float(pos_frac),
            neg_frac_thresh=float(neg_frac_max),
        )
        patch_count = int(pos_mask.size)
        pos_count = int(np.count_nonzero(pos_mask))
        neg_count = int(np.count_nonzero(neg_mask))
        ignored_count = max(0, patch_count - pos_count - neg_count)
        patch_pos_total += pos_count
        patch_neg_total += neg_count
        patch_ignored_total += ignored_count
        if pos_mask.any():
            X_pos.append(feats_tile[pos_mask])
        if neg_mask.any():
            neg_tile = feats_tile[neg_mask]
            if tile_neg_cap > 0 and len(neg_tile) > tile_neg_cap:
                idx = rng.choice(len(neg_tile), size=tile_neg_cap, replace=False)
                neg_tile = neg_tile[idx]
            X_neg.append(neg_tile)

    if missing_feature_tiles > 0:
        logger.warning(
            "build_xgb_dataset: skipped %s tiles with missing cached features for image_id=%s",
            missing_feature_tiles,
            image_id,
        )

    if not X_pos:
        logger.warning(
            "build_xgb_dataset produced no positive samples for image_id=%s; "
            "skipping this source tile (patches: pos=%s neg=%s ignored=%s)",
            image_id,
            patch_pos_total,
            patch_neg_total,
            patch_ignored_total,
        )
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_pos = np.concatenate(X_pos, axis=0)
    X_neg = np.concatenate(X_neg, axis=0) if X_neg else np.empty((0, X_pos.shape[1]))
    # Subsample negatives
    if len(X_neg) > max_neg:
        idx = np.random.default_rng(42).choice(len(X_neg), size=max_neg, replace=False)
        X_neg = X_neg[idx]

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(np.float32)
    pos_count, neg_count = _class_balance(y)
    total = int(y.size)
    pos_ratio = float(pos_count / max(total, 1))
    min_pos = int(getattr(cfg, "XGB_MIN_POS_SAMPLES_WARN", 200) or 200)
    min_pos_ratio = float(getattr(cfg, "XGB_MIN_POS_RATIO_WARN", 0.02) or 0.02)
    if pos_count < min_pos or pos_ratio < min_pos_ratio:
        logger.warning(
            "xgb-dataset sparsity image=%s samples(pos=%s neg=%s pos_ratio=%.4f) "
            "patches(pos=%s neg=%s ignored=%s) thresholds(min_pos=%s min_ratio=%.4f) "
            "neg_caps(tile=%s total=%s)",
            image_id,
            pos_count,
            neg_count,
            pos_ratio,
            patch_pos_total,
            patch_neg_total,
            patch_ignored_total,
            min_pos,
            min_pos_ratio,
            tile_neg_cap,
            max_neg,
        )
    else:
        logger.info(
            "xgb-dataset image=%s samples(pos=%s neg=%s pos_ratio=%.4f) "
            "patches(pos=%s neg=%s ignored=%s) neg_caps(tile=%s total=%s)",
            image_id,
            pos_count,
            neg_count,
            pos_ratio,
            patch_pos_total,
            patch_neg_total,
            patch_ignored_total,
            tile_neg_cap,
            max_neg,
        )
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
        >>> isinstance(train_xgb_classifier.__name__, str)
        True
    """
    y_arr = np.asarray(y, dtype=np.float32)
    dtrain = xgb.DMatrix(X, label=y_arr, weight=_build_binary_sample_weights(y_arr))
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
    if bool(getattr(cfg, "XGB_USE_SCALE_POS_WEIGHT", True)):
        params["scale_pos_weight"] = _auto_scale_pos_weight(y_arr)
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
        >>> isinstance(hyperparam_search_xgb.__name__, str)
        True
    """
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset for XGBoost.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - val_fraction)))
    train_idx, val_idx = idx[:split], idx[split:]
    y_train = np.asarray(y[train_idx], dtype=np.float32)
    y_val = np.asarray(y[val_idx], dtype=np.float32)
    dtrain = xgb.DMatrix(
        X[train_idx],
        label=y_train,
        weight=_build_binary_sample_weights(y_train),
    )
    dval = xgb.DMatrix(
        X[val_idx],
        label=y_val,
        weight=_build_binary_sample_weights(y_val),
    )

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
        if bool(getattr(cfg, "XGB_USE_SCALE_POS_WEIGHT", True)):
            params.setdefault("scale_pos_weight", _auto_scale_pos_weight(y_train))
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
    use_kfold: bool | None = None,
    kfold_splits: int | None = None,
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
        use_kfold (bool | None): Enable k-fold selection (None uses config).
        kfold_splits (int | None): Number of folds when k-fold is enabled.

    Returns:
        tuple[xgb.Booster | None, dict | None, float, float | None, dict | None]:
            Best model, params, IoU, threshold, metrics.

    Examples:
        >>> isinstance(hyperparam_search_xgb_iou.__name__, str)
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

    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset for XGBoost.")
    if n < 2:
        raise ValueError("Need at least 2 samples for XGBoost validation splits.")
    if use_kfold is None:
        use_kfold = bool(getattr(cfg, "XGB_USE_KFOLD", False))
    if kfold_splits is None:
        kfold_splits = int(getattr(cfg, "XGB_KFOLD_SPLITS", 3) or 3)
    fold_splits: list[tuple[np.ndarray, np.ndarray]] = []
    if use_kfold:
        fold_splits = _kfold_index_splits(n, kfold_splits, seed)
        if len(fold_splits) > 1:
            logger.info(
                "xgb-search-iou: kfold enabled (folds=%s, samples=%s, seed=%s)",
                len(fold_splits),
                n,
                seed,
            )
    if not fold_splits:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split = max(1, int(n * (1 - val_fraction)))
        if split >= n:
            split = n - 1
        train_idx, val_idx = idx[:split], idx[split:]
        fold_splits = [(train_idx, val_idx)]
        logger.info(
            "xgb-search-iou: using single split (val_fraction=%.3f, train=%s, val=%s)",
            float(val_fraction),
            len(train_idx),
            len(val_idx),
        )

    for i, overrides in enumerate(param_grid):
        params = base_params.copy()
        params.update(overrides)
        params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        logger.info("xgb-search-iou cfg %s/%s: %s", i + 1, len(param_grid), params)
        fold_metrics: list[dict] = []
        fold_models = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            y_train = np.asarray(y[train_idx], dtype=np.float32)
            y_val = np.asarray(y[val_idx], dtype=np.float32)
            dtrain = xgb.DMatrix(
                X[train_idx],
                label=y_train,
                weight=_build_binary_sample_weights(y_train),
            )
            dval = xgb.DMatrix(
                X[val_idx],
                label=y_val,
                weight=_build_binary_sample_weights(y_val),
            )
            params_fold = params.copy()
            if bool(getattr(cfg, "XGB_USE_SCALE_POS_WEIGHT", True)):
                params_fold.setdefault(
                    "scale_pos_weight", _auto_scale_pos_weight(y_train)
                )
            try:
                model = xgb.train(
                    params_fold,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, "train"), (dval, "val")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval,
                )
            except xgb.core.XGBoostError as e:
                if use_gpu and "gpu_hist" in str(e):
                    logger.warning("xgboost build lacks GPU; retrying with CPU hist.")
                    params_fold["tree_method"] = "hist"
                    model = xgb.train(
                        params_fold,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=[(dtrain, "train"), (dval, "val")],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=verbose_eval,
                    )
                else:
                    raise

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
            if len(fold_splits) > 1:
                logger.info(
                    "xgb-search-iou cfg %s fold %s/%s: thr=%.3f, IoU=%.3f, F1=%.3f",
                    i + 1,
                    fold_idx,
                    len(fold_splits),
                    best_local["threshold"],
                    best_local["iou"],
                    best_local["f1"],
                )
            else:
                logger.info(
                    "xgb-search-iou cfg %s (single-tile proxy): thr=%.3f, IoU=%.3f, F1=%.3f",
                    i + 1,
                    best_local["threshold"],
                    best_local["iou"],
                    best_local["f1"],
                )
            fold_metrics.append(best_local)
            fold_models.append(model)

        fold_ious = np.asarray(
            [float(m["iou"]) for m in fold_metrics], dtype=np.float32
        )
        fold_f1s = np.asarray([float(m["f1"]) for m in fold_metrics], dtype=np.float32)
        score_iou = float(fold_ious.mean())
        if len(fold_splits) > 1:
            logger.info(
                "xgb-search-iou cfg %s kfold summary: folds=%s mean_IoU=%.3f mean_F1=%.3f",
                i + 1,
                len(fold_splits),
                score_iou,
                float(fold_f1s.mean()),
            )
        best_fold_idx = int(np.argmax(fold_ious))
        if score_iou > best_iou:
            best_iou = score_iou
            best_model = fold_models[best_fold_idx]
            best_params = params.copy()
            best_thr = fold_metrics[best_fold_idx]["threshold"]
            best_metrics = fold_metrics[best_fold_idx]

    # Refit the best configuration on all source samples when k-fold is enabled.
    if len(fold_splits) > 1 and best_params is not None:
        y_full = np.asarray(y, dtype=np.float32)
        params_full = best_params.copy()
        if bool(getattr(cfg, "XGB_USE_SCALE_POS_WEIGHT", True)):
            params_full["scale_pos_weight"] = _auto_scale_pos_weight(y_full)
        dtrain_full = xgb.DMatrix(
            X,
            label=y_full,
            weight=_build_binary_sample_weights(y_full),
        )
        logger.info("xgb-search-iou: refit best config on full source dataset")
        try:
            model_full = xgb.train(
                params_full,
                dtrain_full,
                num_boost_round=num_boost_round,
                verbose_eval=False,
            )
        except xgb.core.XGBoostError as e:
            if use_gpu and "gpu_hist" in str(e):
                logger.warning("xgboost build lacks GPU; retrying with CPU hist.")
                params_full["tree_method"] = "hist"
                model_full = xgb.train(
                    params_full,
                    dtrain_full,
                    num_boost_round=num_boost_round,
                    verbose_eval=False,
                )
            else:
                raise
        score_full_xgb = xgb_score_image_b(
            img_b,
            model_full,
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
            "xgb-search-iou final-refit: thr=%.3f, IoU=%.3f, F1=%.3f",
            best_local["threshold"],
            best_local["iou"],
            best_local["f1"],
        )
        best_model = model_full
        best_thr = best_local["threshold"]
        best_metrics = best_local
        best_iou = float(best_local["iou"])

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
    emit_timing_logs: bool = True,
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
        emit_timing_logs (bool): If False, suppress image-level timing/info logs.

    Returns:
        np.ndarray: Score map at pixel resolution.

    Examples:
        >>> isinstance(xgb_score_image_b.__name__, str)
        True
    """
    t0 = time_start()
    t0_wall = time.perf_counter()
    if feature_dir is None and prefetched_tiles is None:
        raise ValueError("feature_dir or prefetched_tiles must be provided")
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)
    predict_time = resize_time = 0.0
    tile_total = cached_tiles = loaded_tiles = skipped_tiles = 0
    use_gpu_resize = bool(getattr(cfg, "USE_GPU_RESIZE", True))
    resize_device = (
        torch.device("cuda") if use_gpu_resize and torch.cuda.is_available() else None
    )

    if prefetched_tiles is not None:
        tile_items = sorted(prefetched_tiles.items())
        tile_iter = ((y, x, info) for (y, x), info in tile_items)
    else:
        tile_iter = (
            (y, x, img_tile)
            for y, x, img_tile, _ in tile_iterator(img_b, None, tile_size, stride)
        )

    for tile_entry in tile_iter:
        tile_total += 1
        if prefetched_tiles is not None:
            y, x, feat_info = tile_entry
            feats_tile = feat_info["feats"]
            h_eff = feat_info["h_eff"]
            w_eff = feat_info["w_eff"]
            hp = feat_info["hp"]
            wp = feat_info["wp"]
            cached_tiles += 1
        else:
            y, x, img_tile = tile_entry
            img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
            if h_eff < ps or w_eff < ps:
                skipped_tiles += 1
                continue
            fpath = tile_feature_path(feature_dir, image_id_b, y, x)
            if not os.path.exists(fpath):
                skipped_tiles += 1
                continue
            feats_tile = np.load(fpath)
            hp, wp = feats_tile.shape[:2]
            loaded_tiles += 1

        if context_radius and context_radius > 0:
            feats_tile = add_local_context_mean(feats_tile, int(context_radius))

        if hp is None or wp is None:
            logger.warning(
                "missing patch dimensions for tile y=%s x=%s; skipping", y, x
            )
            skipped_tiles += 1
            continue
        t_predict0 = time.perf_counter() if DEBUG_TIMING else None
        dtest = xgb.DMatrix(feats_tile.reshape(-1, feats_tile.shape[-1]))
        scores_patch = bst.predict(dtest).reshape(hp, wp)
        if DEBUG_TIMING and t_predict0 is not None:
            predict_time += time.perf_counter() - t_predict0
        t_resize0 = time.perf_counter() if DEBUG_TIMING else None
        scores_tile = _resize_patch_map(
            scores_patch,
            h_eff,
            w_eff,
            resize_device=resize_device,
        )
        if DEBUG_TIMING and t_resize0 is not None:
            resize_time += time.perf_counter() - t_resize0
        score_full[y : y + h_eff, x : x + w_eff] += scores_tile
        weight_full[y : y + h_eff, x : x + w_eff] += 1.0

    mask_nonzero = weight_full > 0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    total_s = time.perf_counter() - t0_wall
    if emit_timing_logs:
        time_end("xgb_score_image_b", t0)
        logger.info(
            "xgb_score image=%s total=%.3fs tiles=%s (cached=%s, loaded=%s, skipped=%s)",
            image_id_b,
            total_s,
            tile_total,
            cached_tiles,
            loaded_tiles,
            skipped_tiles,
        )
        if DEBUG_TIMING:
            logger.info(
                "xgb image=%s predict_time=%.2fs, resize_time=%.2fs",
                image_id_b,
                predict_time,
                resize_time,
            )
    return score_full
