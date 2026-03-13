"""XGBoost training and scoring utilities for SegEdge."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import xgboost as xgb
from skimage.transform import resize

from .config_loader import cfg
from .features import (
    add_local_context_mean,
    crop_to_multiple_of_ps,
    extract_patch_features_batch_single_scale,
    extract_patch_features_single_scale,
    fuse_patch_features,
    labels_to_patch_masks,
    load_tile_features_if_valid,
    tile_iterator,
)
from .metrics_utils import compute_metrics_batch_cpu, compute_metrics_batch_gpu
from .timing_utils import perf_span

logger = logging.getLogger(__name__)
_XGB_BATCH_ROW_BUDGET_PER_UNIT = 1024
_XGB_BATCH_TILE_MULTIPLIER = 32


def _gpu_error_message(exc: Exception) -> str:
    return str(exc).lower()


def _is_gpu_unavailable_error(exc: Exception) -> bool:
    msg = _gpu_error_message(exc)
    markers = [
        "cuda",
        "gpu",
        "gpu_hist",
        "no visible gpu",
        "not compiled with",
        "device",
    ]
    return any(m in msg for m in markers)


def _base_xgb_params(use_gpu: bool) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
        "tree_method": "hist",
    }
    if use_gpu:
        params["device"] = "cuda"
    return params


def build_xgb_dataset(
    img,
    labels,
    ps,
    tile_size,
    stride,
    feature_dir,
    image_id,
    pos_frac,
    max_pos=120000,
    max_neg=8000,
    context_radius: int = 0,
    prefetched_tiles: dict | None = None,
    xgb_feature_stats: dict | None = None,
    return_layout: bool = False,
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
        max_pos (int): Maximum number of positive samples.
        max_neg (int): Maximum number of negative samples.
        context_radius (int): Feature context radius.
        prefetched_tiles (dict | None): Optional in-memory tile feature cache.

    Returns:
        tuple[np.ndarray, np.ndarray, dict | None]: Feature matrix X, labels y, and
            optional feature layout info.

    Examples:
        >>> callable(build_xgb_dataset)
        True
    """
    if feature_dir is None and prefetched_tiles is None:
        raise ValueError("feature_dir or prefetched_tiles must be provided")

    X_pos, X_neg = [], []
    feature_layout: dict | None = None
    missing_feature_tiles = 0
    resample_factor = int(cfg.model.backbone.resample_factor or 1)
    for y, x, img_tile, lab_tile in tile_iterator(img, labels, tile_size, stride):
        prefetched = prefetched_tiles.get((y, x)) if prefetched_tiles else None
        if prefetched is not None:
            h_eff = prefetched["h_eff"]
            w_eff = prefetched["w_eff"]
            if h_eff < ps or w_eff < ps:
                continue
            img_c = img_tile[:h_eff, :w_eff]
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
            img_c = img_c[:h_eff, :w_eff]

        if lab_c is None:
            logger.warning("missing labels for tile y=%s x=%s; skipping", y, x)
            continue
        if context_radius and context_radius > 0:
            feats_tile = add_local_context_mean(feats_tile, int(context_radius))
        feats_tile, layout_i = fuse_patch_features(
            feats_tile,
            img_c,
            ps,
            mode="xgb",
            xgb_feature_stats=xgb_feature_stats,
            return_layout=return_layout and feature_layout is None,
        )
        if feature_layout is None and layout_i is not None:
            feature_layout = layout_i
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
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            feature_layout,
        )

    X_pos = np.concatenate(X_pos, axis=0)
    if len(X_pos) > max_pos:
        idx = np.random.default_rng(42).choice(len(X_pos), size=max_pos, replace=False)
        X_pos = X_pos[idx]
        logger.info(
            "build_xgb_dataset: subsampled positives to %s (max_pos=%s) for image_id=%s",
            len(X_pos),
            max_pos,
            image_id,
        )
    X_neg = np.concatenate(X_neg, axis=0) if X_neg else np.empty((0, X_pos.shape[1]))
    # Subsample negatives
    if len(X_neg) > max_neg:
        idx = np.random.default_rng(42).choice(len(X_neg), size=max_neg, replace=False)
        X_neg = X_neg[idx]

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    return X, y, feature_layout


def train_xgb_classifier(
    X,
    y,
    use_gpu=False,
    num_boost_round=300,
    verbose_eval=50,
    param_overrides: dict | None = None,
    feature_names: list[str] | None = None,
):
    """Train a binary XGBoost classifier for patch embeddings.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary labels.
        use_gpu (bool): Use GPU histogram algorithm when available.
        num_boost_round (int): Number of boosting rounds.
        verbose_eval (int): Verbosity interval.
        param_overrides (dict | None): Optional parameter overrides.

    Returns:
        xgb.Booster: Trained XGBoost booster.

    Examples:
        >>> callable(train_xgb_classifier)
        True
    """
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    params = _base_xgb_params(use_gpu=use_gpu)
    if param_overrides:
        params.update(param_overrides)
        params["tree_method"] = "hist"
        if use_gpu:
            params["device"] = "cuda"
        else:
            params.pop("device", None)
    try:
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train")],
            verbose_eval=verbose_eval,
        )
    except xgb.core.XGBoostError as e:
        if use_gpu and _is_gpu_unavailable_error(e):
            logger.warning("xgboost GPU unavailable; falling back to CPU hist.")
            params["tree_method"] = "hist"
            params.pop("device", None)
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
    feature_names: list[str] | None = None,
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

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=feature_names)
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=feature_names)

    base_params = _base_xgb_params(use_gpu=use_gpu)
    if param_grid is None:
        param_grid = [base_params]

    best_model = None
    best_params = None
    best_score = float("inf")  # logloss: lower is better
    gpu_enabled = bool(use_gpu)

    for i, overrides in enumerate(param_grid):
        params = _base_xgb_params(use_gpu=gpu_enabled)
        params.update(overrides)
        params["tree_method"] = "hist"
        if gpu_enabled:
            params["device"] = "cuda"
        else:
            params.pop("device", None)
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
            if gpu_enabled and _is_gpu_unavailable_error(e):
                logger.warning("xgboost GPU unavailable; retrying with CPU hist.")
                gpu_enabled = False
                params["tree_method"] = "hist"
                params.pop("device", None)
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
    xgb_feature_stats: dict | None = None,
    feature_names: list[str] | None = None,
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

    base_params = _base_xgb_params(use_gpu=use_gpu)
    if param_grid is None:
        param_grid = [base_params]
    gpu_enabled = bool(use_gpu)

    # simple train/val split (used only for early stopping)
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset for XGBoost.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = max(1, int(n * (1 - val_fraction)))
    train_idx, val_idx = idx[:split], idx[split:]
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=feature_names)
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=feature_names)

    for i, overrides in enumerate(param_grid):
        params = _base_xgb_params(use_gpu=gpu_enabled)
        params.update(overrides)
        params["tree_method"] = "hist"
        if gpu_enabled:
            params["device"] = "cuda"
        else:
            params.pop("device", None)
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
            if gpu_enabled and _is_gpu_unavailable_error(e):
                logger.warning("xgboost GPU unavailable; retrying with CPU hist.")
                gpu_enabled = False
                params["tree_method"] = "hist"
                params.pop("device", None)
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
            xgb_feature_stats=xgb_feature_stats,
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
    xgb_feature_stats: dict | None = None,
):
    """Apply a trained XGBoost model to Image B and return a score map.

    Examples:
        >>> callable(xgb_score_image_b)
        True
    """
    batch_tile_limit = max(1, int(cfg.runtime.feature_batch_size or 1))
    batch_row_limit = max(1, batch_tile_limit * _XGB_BATCH_ROW_BUDGET_PER_UNIT)
    pending_tile_limit = max(1, batch_tile_limit * _XGB_BATCH_TILE_MULTIPLIER)
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)
    pending_tiles: list[dict[str, np.ndarray | int]] = []
    pending_rows = 0

    for tile_info in _iter_xgb_tile_payloads(
        img_b,
        ps,
        tile_size,
        stride,
        feature_dir,
        image_id_b,
        prefetched_tiles=prefetched_tiles,
        context_radius=context_radius,
        xgb_feature_stats=xgb_feature_stats,
    ):
        tile_rows = int(tile_info["row_count"])
        if pending_tiles and (
            pending_rows + tile_rows > batch_row_limit
            or len(pending_tiles) >= pending_tile_limit
        ):
            _flush_xgb_prediction_batch(pending_tiles, bst, score_full, weight_full)
            pending_tiles.clear()
            pending_rows = 0
        pending_tiles.append(tile_info)
        pending_rows += tile_rows

    if pending_tiles:
        _flush_xgb_prediction_batch(pending_tiles, bst, score_full, weight_full)

    mask_nonzero = weight_full > 0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    return score_full


def xgb_score_image_b_streaming(
    img_b,
    bst,
    model,
    processor,
    device,
    ps,
    tile_size,
    stride,
    *,
    context_radius: int = 0,
    xgb_feature_stats: dict | None = None,
):
    """Apply XGBoost to Image B by streaming feature extraction and scoring.

    Examples:
        >>> callable(xgb_score_image_b_streaming)
        True
    """
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)
    batch_tile_limit = max(1, int(cfg.runtime.feature_batch_size or 1))
    extract_batch_limit = batch_tile_limit
    predict_batch_row_limit = max(1, batch_tile_limit * _XGB_BATCH_ROW_BUDGET_PER_UNIT)
    predict_tile_limit = max(1, batch_tile_limit * _XGB_BATCH_TILE_MULTIPLIER)
    pending_extract: dict[
        tuple[int, int], list[tuple[int, int, np.ndarray, int, int]]
    ] = {}
    pending_predict: list[dict[str, np.ndarray | int]] = []
    pending_predict_rows = 0

    def flush_predict_batch() -> None:
        nonlocal pending_predict_rows
        if not pending_predict:
            return
        _flush_xgb_prediction_batch(pending_predict, bst, score_full, weight_full)
        pending_predict.clear()
        pending_predict_rows = 0

    def append_predict_tile(payload: dict[str, np.ndarray | int]) -> None:
        nonlocal pending_predict_rows
        tile_rows = int(payload["row_count"])
        if pending_predict and (
            pending_predict_rows + tile_rows > predict_batch_row_limit
            or len(pending_predict) >= predict_tile_limit
        ):
            flush_predict_batch()
        pending_predict.append(payload)
        pending_predict_rows += tile_rows

    def flush_extract_batch(
        items: list[tuple[int, int, np.ndarray, int, int]],
    ) -> None:
        if not items:
            return
        imgs = [item[2] for item in items]
        with perf_span(
            "xgb_score_image_b",
            substage="streaming_extract_features",
            extra={"batch_size": len(items)},
        ):
            if len(items) == 1:
                feats_tile, hp_i, wp_i = extract_patch_features_single_scale(
                    imgs[0],
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=None,
                )
                feats_list = [feats_tile]
            else:
                feats_list, hp_i, wp_i = extract_patch_features_batch_single_scale(
                    imgs,
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=None,
                )
        for (y_i, x_i, img_i, h_i, w_i), feats_tile in zip(
            items, feats_list, strict=True
        ):
            if context_radius and context_radius > 0:
                with perf_span(
                    "xgb_score_image_b",
                    substage="local_context_mean",
                    extra={"y": y_i, "x": x_i},
                ):
                    feats_tile = add_local_context_mean(feats_tile, int(context_radius))
            with perf_span(
                "xgb_score_image_b",
                substage="streaming_fuse_features",
                extra={"y": y_i, "x": x_i},
            ):
                fused_tile, _ = fuse_patch_features(
                    feats_tile,
                    img_i,
                    ps,
                    mode="xgb",
                    xgb_feature_stats=xgb_feature_stats,
                    return_layout=False,
                )
            with perf_span(
                "xgb_score_image_b",
                substage="flatten_features",
                extra={"y": y_i, "x": x_i},
            ):
                flat_feats = fused_tile.reshape(-1, fused_tile.shape[-1])
            append_predict_tile(
                {
                    "y": y_i,
                    "x": x_i,
                    "h_eff": h_i,
                    "w_eff": w_i,
                    "hp": hp_i,
                    "wp": wp_i,
                    "row_count": int(hp_i * wp_i),
                    "flat_feats": flat_feats,
                }
            )

    for y, x, img_tile, _ in tile_iterator(img_b, None, tile_size, stride):
        with perf_span("xgb_score_image_b", substage="tile_loop"):
            img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
            if h_eff < ps or w_eff < ps:
                continue
            img_c = img_c[:h_eff, :w_eff]
            key = (h_eff, w_eff)
            pending_extract.setdefault(key, []).append((y, x, img_c, h_eff, w_eff))
            if len(pending_extract[key]) >= extract_batch_limit:
                flush_extract_batch(pending_extract.pop(key))

    for items in pending_extract.values():
        flush_extract_batch(items)
    flush_predict_batch()
    mask_nonzero = weight_full > 0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    return score_full


def _iter_xgb_tile_payloads(
    img_b,
    ps,
    tile_size,
    stride,
    feature_dir,
    image_id_b,
    *,
    prefetched_tiles=None,
    context_radius: int = 0,
    xgb_feature_stats: dict | None = None,
) -> Iterator[dict[str, np.ndarray | int]]:
    """Yield flattened tile payloads for batched XGB prediction.

    Examples:
        >>> callable(_iter_xgb_tile_payloads)
        True
    """
    if feature_dir is None and prefetched_tiles is None:
        raise ValueError("feature_dir or prefetched_tiles must be provided")

    if prefetched_tiles is not None:
        tile_items = sorted(prefetched_tiles.items())
        tile_iter = ((y, x, info) for (y, x), info in tile_items)
    else:
        tile_iter = (
            (y, x, img_tile)
            for y, x, img_tile, _ in tile_iterator(img_b, None, tile_size, stride)
        )

    for tile_entry in tile_iter:
        with perf_span("xgb_score_image_b", substage="tile_loop"):
            if prefetched_tiles is not None:
                y, x, feat_info = tile_entry
                feats_tile = feat_info["feats"]
                h_eff = feat_info["h_eff"]
                w_eff = feat_info["w_eff"]
                hp = feat_info["hp"]
                wp = feat_info["wp"]
                img_c = img_b[y : y + h_eff, x : x + w_eff]
                if feats_tile is None and feat_info.get("feature_path"):
                    with perf_span(
                        "xgb_score_image_b",
                        substage="load_cached_feature_array",
                        extra={"y": y, "x": x},
                    ):
                        feats_tile = np.load(
                            feat_info["feature_path"], mmap_mode="r"
                        ).astype(np.float32, copy=False)
            else:
                y, x, img_tile = tile_entry
                img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
                if h_eff < ps or w_eff < ps:
                    continue
                hp = h_eff // ps
                wp = w_eff // ps
                with perf_span(
                    "xgb_score_image_b",
                    substage="load_cached_feature_array",
                    extra={"y": y, "x": x},
                ):
                    feats_tile = load_tile_features_if_valid(
                        feature_dir,
                        image_id_b,
                        y,
                        x,
                        expected_hp=hp,
                        expected_wp=wp,
                        ps=ps,
                        resample_factor=int(cfg.model.backbone.resample_factor or 1),
                    )
                if feats_tile is None:
                    continue
                img_c = img_c[:h_eff, :w_eff]

            if feats_tile is None:
                continue
            if feats_tile.dtype != np.float32:
                feats_tile = feats_tile.astype(np.float32, copy=False)
            if feats_tile.shape[:2] != (hp, wp):
                logger.warning(
                    (
                        "xgb_score_image_b: cached feature shape mismatch for "
                        "image_id=%s y=%s x=%s expected=(%s,%s) actual=%s"
                    ),
                    image_id_b,
                    y,
                    x,
                    hp,
                    wp,
                    feats_tile.shape[:2],
                )
                continue
            if context_radius and context_radius > 0:
                with perf_span(
                    "xgb_score_image_b",
                    substage="local_context_mean",
                    extra={"y": y, "x": x},
                ):
                    feats_tile = add_local_context_mean(feats_tile, int(context_radius))
            with perf_span(
                "xgb_score_image_b",
                substage="feature_fusion",
                extra={"y": y, "x": x},
            ):
                feats_tile, _ = fuse_patch_features(
                    feats_tile,
                    img_c,
                    ps,
                    mode="xgb",
                    xgb_feature_stats=xgb_feature_stats,
                    return_layout=False,
                )

            if hp is None or wp is None:
                logger.warning(
                    "missing patch dimensions for tile y=%s x=%s; skipping", y, x
                )
                continue
            with perf_span(
                "xgb_score_image_b",
                substage="flatten_features",
                extra={"y": y, "x": x},
            ):
                flat_feats = feats_tile.reshape(-1, feats_tile.shape[-1])
            yield {
                "y": y,
                "x": x,
                "h_eff": h_eff,
                "w_eff": w_eff,
                "hp": hp,
                "wp": wp,
                "row_count": int(hp * wp),
                "flat_feats": flat_feats,
            }


def _flush_xgb_prediction_batch(
    tile_payloads: list[dict[str, np.ndarray | int]],
    bst,
    score_full: np.ndarray,
    weight_full: np.ndarray,
) -> None:
    """Run one batched XGB predict call and accumulate outputs.

    Examples:
        >>> callable(_flush_xgb_prediction_batch)
        True
    """
    if not tile_payloads:
        return
    batch_rows = [payload["flat_feats"] for payload in tile_payloads]
    total_rows = int(sum(int(payload["row_count"]) for payload in tile_payloads))
    if len(batch_rows) == 1:
        flat_feats = batch_rows[0]
    else:
        with perf_span(
            "xgb_score_image_b",
            substage="concat_predict_batch",
            extra={
                "tile_count": len(tile_payloads),
                "row_count": total_rows,
            },
        ):
            flat_feats = np.concatenate(batch_rows, axis=0)
    try:
        with perf_span(
            "xgb_score_image_b",
            substage="predict_inplace",
            extra={
                "tile_count": len(tile_payloads),
                "row_count": total_rows,
            },
        ):
            scores_batch = bst.inplace_predict(flat_feats)
    except Exception:
        with perf_span(
            "xgb_score_image_b",
            substage="predict_dmatrix_fallback",
            extra={
                "tile_count": len(tile_payloads),
                "row_count": total_rows,
            },
        ):
            dtest = xgb.DMatrix(flat_feats)
            scores_batch = bst.predict(dtest)
    scores_batch = np.asarray(scores_batch, dtype=np.float32)
    offset = 0
    for payload in tile_payloads:
        hp = int(payload["hp"])
        wp = int(payload["wp"])
        h_eff = int(payload["h_eff"])
        w_eff = int(payload["w_eff"])
        y = int(payload["y"])
        x = int(payload["x"])
        rows = hp * wp
        scores_patch = scores_batch[offset : offset + rows].reshape(hp, wp)
        offset += rows
        with perf_span(
            "xgb_score_image_b",
            substage="resize_patch_scores",
            extra={"y": y, "x": x},
        ):
            scores_tile = resize(
                scores_patch,
                (h_eff, w_eff),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
        with perf_span(
            "xgb_score_image_b",
            substage="accumulate_scores",
            extra={"y": y, "x": x},
        ):
            score_full[y : y + h_eff, x : x + w_eff] += scores_tile
            weight_full[y : y + h_eff, x : x + w_eff] += 1.0


def xgb_score_image_b_legacy(
    img_b,
    bst,
    ps,
    tile_size,
    stride,
    feature_dir,
    image_id_b,
    prefetched_tiles=None,
    context_radius: int = 0,
    xgb_feature_stats: dict | None = None,
):
    """Legacy tile-by-tile XGB scorer kept for guard comparisons and fallback.

    Examples:
        >>> callable(xgb_score_image_b_legacy)
        True
    """
    if feature_dir is None and prefetched_tiles is None:
        raise ValueError("feature_dir or prefetched_tiles must be provided")
    h_full, w_full = img_b.shape[:2]
    score_full = np.zeros((h_full, w_full), dtype=np.float32)
    weight_full = np.zeros((h_full, w_full), dtype=np.float32)
    for payload in _iter_xgb_tile_payloads(
        img_b,
        ps,
        tile_size,
        stride,
        feature_dir,
        image_id_b,
        prefetched_tiles=prefetched_tiles,
        context_radius=context_radius,
        xgb_feature_stats=xgb_feature_stats,
    ):
        flat_feats = np.asarray(payload["flat_feats"], dtype=np.float32)
        try:
            with perf_span(
                "xgb_score_image_b_legacy",
                substage="predict_inplace",
                extra={"y": int(payload["y"]), "x": int(payload["x"])},
            ):
                scores_flat = bst.inplace_predict(flat_feats)
        except Exception:
            with perf_span(
                "xgb_score_image_b_legacy",
                substage="predict_dmatrix_fallback",
                extra={"y": int(payload["y"]), "x": int(payload["x"])},
            ):
                dtest = xgb.DMatrix(flat_feats)
                scores_flat = bst.predict(dtest)
        scores_patch = np.asarray(scores_flat, dtype=np.float32).reshape(
            int(payload["hp"]),
            int(payload["wp"]),
        )
        with perf_span(
            "xgb_score_image_b_legacy",
            substage="resize_patch_scores",
            extra={"y": int(payload["y"]), "x": int(payload["x"])},
        ):
            scores_tile = resize(
                scores_patch,
                (int(payload["h_eff"]), int(payload["w_eff"])),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
        with perf_span(
            "xgb_score_image_b_legacy",
            substage="accumulate_scores",
            extra={"y": int(payload["y"]), "x": int(payload["x"])},
        ):
            score_full[
                int(payload["y"]) : int(payload["y"]) + int(payload["h_eff"]),
                int(payload["x"]) : int(payload["x"]) + int(payload["w_eff"]),
            ] += scores_tile
            weight_full[
                int(payload["y"]) : int(payload["y"]) + int(payload["h_eff"]),
                int(payload["x"]) : int(payload["x"]) + int(payload["w_eff"]),
            ] += 1.0
    mask_nonzero = weight_full > 0
    score_full[mask_nonzero] /= weight_full[mask_nonzero]
    return score_full
