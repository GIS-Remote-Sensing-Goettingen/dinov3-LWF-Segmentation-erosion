"""Feature extraction and tiling utilities for SegEdge."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from .config_loader import cfg
from .timing_utils import DEBUG_TIMING, DEBUG_TIMING_VERBOSE, time_end, time_start

logger = logging.getLogger(__name__)


def l2_normalize(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize feature vectors along the last dimension.

    Args:
        feats (np.ndarray): Feature array with last axis as channels.
        eps (float): Small epsilon for numerical stability.

    Returns:
        np.ndarray: L2-normalized feature array.

    Examples:
        >>> import numpy as np
        >>> feats = np.array([[3.0, 4.0]])
        >>> out = l2_normalize(feats)
        >>> np.allclose(out, np.array([[0.6, 0.8]]))
        True
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    norms = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
    out = feats / norms
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("l2_normalize", t0)
    return out


def add_local_context_mean(feats_hwc: np.ndarray, radius: int) -> np.ndarray:
    """Add local spatial context to patch embeddings by averaging neighbors.

    Operates on patch-grid features (Hp x Wp x C) without mixing channels.

    Args:
        feats_hwc (np.ndarray): Patch-grid features.
        radius (int): Context radius in patch units.

    Returns:
        np.ndarray: Context-smoothed, L2-normalized features.

    Examples:
        >>> import numpy as np
        >>> feats = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        >>> out = add_local_context_mean(feats, radius=0)
        >>> np.array_equal(out, feats)
        True
    """
    if radius <= 0:
        return feats_hwc
    if feats_hwc.ndim != 3:
        raise ValueError(
            f"expected feats with shape (Hp, Wp, C), got {feats_hwc.shape}"
        )
    k = 2 * int(radius) + 1
    feats = feats_hwc.astype(np.float32, copy=False)
    feats_ctx = uniform_filter(feats, size=(k, k, 1), mode="reflect")
    return l2_normalize(feats_ctx)


def hybrid_feature_spec_dict() -> dict:
    """Build a JSON-serializable hybrid feature specification.

    Returns:
        dict: Hybrid feature spec used for reproducibility/hash.

    Examples:
        >>> isinstance(hybrid_feature_spec_dict(), dict)
        True
    """
    hybrid = cfg.model.hybrid_features
    blocks = hybrid.blocks
    return {
        "enabled": bool(hybrid.enabled),
        "knn_l2_normalize": bool(hybrid.knn_l2_normalize),
        "xgb_zscore": bool(hybrid.xgb_zscore),
        "zscore_eps": float(hybrid.zscore_eps),
        "blocks": {
            "dino": {
                "enabled": bool(blocks.dino.enabled),
                "weight_knn": float(blocks.dino.weight_knn),
                "weight_xgb": float(blocks.dino.weight_xgb),
            },
            "rgb_stats": {
                "enabled": bool(blocks.rgb_stats.enabled),
                "weight_knn": float(blocks.rgb_stats.weight_knn),
                "weight_xgb": float(blocks.rgb_stats.weight_xgb),
            },
            "hsv_mean": {
                "enabled": bool(blocks.hsv_mean.enabled),
                "weight_knn": float(blocks.hsv_mean.weight_knn),
                "weight_xgb": float(blocks.hsv_mean.weight_xgb),
            },
            "grad_stats": {
                "enabled": bool(blocks.grad_stats.enabled),
                "weight_knn": float(blocks.grad_stats.weight_knn),
                "weight_xgb": float(blocks.grad_stats.weight_xgb),
            },
            "grad_orient_hist": {
                "enabled": bool(blocks.grad_orient_hist.enabled),
                "bins": int(blocks.grad_orient_hist.bins or 8),
                "weight_knn": float(blocks.grad_orient_hist.weight_knn),
                "weight_xgb": float(blocks.grad_orient_hist.weight_xgb),
            },
            "lbp_hist": {
                "enabled": bool(blocks.lbp_hist.enabled),
                "bins": int(blocks.lbp_hist.bins or 16),
                "weight_knn": float(blocks.lbp_hist.weight_knn),
                "weight_xgb": float(blocks.lbp_hist.weight_xgb),
            },
        },
    }


def hybrid_feature_spec_hash() -> str:
    """Return a stable short hash of the hybrid feature specification.

    Returns:
        str: Short SHA1 hash.

    Examples:
        >>> len(hybrid_feature_spec_hash()) >= 8
        True
    """
    payload = json.dumps(
        hybrid_feature_spec_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _rgb_to_hsv_image(img_rgb_01: np.ndarray) -> np.ndarray:
    """Convert RGB image in [0, 1] to HSV.

    Args:
        img_rgb_01 (np.ndarray): RGB image normalized to [0, 1].

    Returns:
        np.ndarray: HSV image with channels in [0, 1].

    Examples:
        >>> import numpy as np
        >>> hsv = _rgb_to_hsv_image(np.zeros((2, 2, 3), dtype=np.float32))
        >>> hsv.shape
        (2, 2, 3)
    """
    r = img_rgb_01[..., 0]
    g = img_rgb_01[..., 1]
    b = img_rgb_01[..., 2]
    cmax = np.max(img_rgb_01, axis=2)
    cmin = np.min(img_rgb_01, axis=2)
    delta = cmax - cmin

    hue = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-8
    r_mask = mask & (cmax == r)
    g_mask = mask & (cmax == g)
    b_mask = mask & (cmax == b)
    hue[r_mask] = ((g[r_mask] - b[r_mask]) / (delta[r_mask] + 1e-8)) % 6.0
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / (delta[g_mask] + 1e-8)) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / (delta[b_mask] + 1e-8)) + 4.0
    hue = hue / 6.0

    sat = np.zeros_like(cmax, dtype=np.float32)
    nz = cmax > 1e-8
    sat[nz] = delta[nz] / (cmax[nz] + 1e-8)
    val = cmax.astype(np.float32)
    return np.stack([hue.astype(np.float32), sat, val], axis=-1)


def _compute_lbp_codes(gray_01: np.ndarray) -> np.ndarray:
    """Compute 8-neighborhood LBP codes on a grayscale image.

    Examples:
        >>> import numpy as np
        >>> _compute_lbp_codes(np.zeros((4, 4), dtype=np.float32)).shape
        (2, 2)
    """
    if gray_01.shape[0] < 3 or gray_01.shape[1] < 3:
        return np.zeros((0, 0), dtype=np.uint8)
    c = gray_01[1:-1, 1:-1]
    codes = np.zeros_like(c, dtype=np.uint8)
    neigh = [
        gray_01[:-2, :-2],
        gray_01[:-2, 1:-1],
        gray_01[:-2, 2:],
        gray_01[1:-1, 2:],
        gray_01[2:, 2:],
        gray_01[2:, 1:-1],
        gray_01[2:, :-2],
        gray_01[1:-1, :-2],
    ]
    for bit, n in enumerate(neigh):
        codes |= ((n >= c).astype(np.uint8) << bit).astype(np.uint8)
    return codes


def _mode_weight(block_cfg, mode: str) -> float:
    """Return per-model block weight from config block settings.

    Examples:
        >>> class _B:
        ...     weight_knn = 0.5
        ...     weight_xgb = 1.0
        >>> _mode_weight(_B(), "knn")
        0.5
    """
    return float(block_cfg.weight_knn) if mode == "knn" else float(block_cfg.weight_xgb)


def _apply_xgb_stats_hwc(
    feats_hwc: np.ndarray,
    xgb_feature_stats: dict | None,
) -> np.ndarray:
    """Apply per-feature z-score stats to an HxWxC feature tensor.

    Examples:
        >>> import numpy as np
        >>> feats = np.zeros((1, 1, 2), dtype=np.float32)
        >>> stats = {"mean": np.zeros(2, dtype=np.float32), "std": np.ones(2, dtype=np.float32)}
        >>> _apply_xgb_stats_hwc(feats, stats).shape
        (1, 1, 2)
    """
    if xgb_feature_stats is None:
        return feats_hwc
    mean = np.asarray(xgb_feature_stats.get("mean"), dtype=np.float32)
    std = np.asarray(xgb_feature_stats.get("std"), dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError("invalid xgb_feature_stats mean/std shapes")
    if feats_hwc.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"xgb_feature_stats dim mismatch: feats={feats_hwc.shape[-1]} stats={mean.shape[0]}"
        )
    return ((feats_hwc - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(
        np.float32
    )


def fit_xgb_feature_stats(X: np.ndarray, eps: float = 1e-6) -> dict:
    """Fit per-feature z-score statistics on training data only.

    Args:
        X (np.ndarray): Feature matrix (N, C).
        eps (float): Minimum std floor.

    Returns:
        dict: Mean/std arrays and metadata.

    Examples:
        >>> import numpy as np
        >>> stats = fit_xgb_feature_stats(np.array([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32))
        >>> stats["mean"].shape[0]
        2
    """
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("fit_xgb_feature_stats expects non-empty (N, C) matrix")
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.maximum(std, float(eps)).astype(np.float32)
    return {
        "mean": mean,
        "std": std,
        "eps": float(eps),
        "feature_spec_hash": hybrid_feature_spec_hash(),
    }


def apply_xgb_feature_stats(
    X: np.ndarray, xgb_feature_stats: dict | None
) -> np.ndarray:
    """Apply fitted z-score statistics to an (N, C) feature matrix.

    Args:
        X (np.ndarray): Feature matrix.
        xgb_feature_stats (dict | None): Stats from fit_xgb_feature_stats.

    Returns:
        np.ndarray: Standardized features.

    Examples:
        >>> import numpy as np
        >>> X = np.array([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32)
        >>> stats = fit_xgb_feature_stats(X)
        >>> apply_xgb_feature_stats(X, stats).shape
        (2, 2)
    """
    if xgb_feature_stats is None:
        return X
    mean = np.asarray(xgb_feature_stats.get("mean"), dtype=np.float32)
    std = np.asarray(xgb_feature_stats.get("std"), dtype=np.float32)
    if X.shape[1] != mean.shape[0]:
        raise ValueError(
            f"xgb feature dim mismatch: X={X.shape[1]} stats={mean.shape[0]}"
        )
    return ((X - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def serialize_xgb_feature_stats(xgb_feature_stats: dict | None) -> dict | None:
    """Convert xgb feature stats to YAML/JSON-safe payload.

    Examples:
        >>> serialize_xgb_feature_stats(None) is None
        True
    """
    if xgb_feature_stats is None:
        return None
    return {
        "mean": np.asarray(xgb_feature_stats["mean"], dtype=np.float32).tolist(),
        "std": np.asarray(xgb_feature_stats["std"], dtype=np.float32).tolist(),
        "eps": float(xgb_feature_stats.get("eps", 1e-6)),
        "feature_spec_hash": str(
            xgb_feature_stats.get("feature_spec_hash", hybrid_feature_spec_hash())
        ),
    }


def deserialize_xgb_feature_stats(payload: dict | None) -> dict | None:
    """Convert serialized xgb feature stats back to numpy payload.

    Examples:
        >>> deserialize_xgb_feature_stats(None) is None
        True
    """
    if payload is None:
        return None
    return {
        "mean": np.asarray(payload.get("mean", []), dtype=np.float32),
        "std": np.asarray(payload.get("std", []), dtype=np.float32),
        "eps": float(payload.get("eps", 1e-6)),
        "feature_spec_hash": str(
            payload.get("feature_spec_hash", hybrid_feature_spec_hash())
        ),
    }


def fuse_patch_features(
    dino_feats_hwc: np.ndarray,
    img_tile_hw3: np.ndarray | None,
    ps: int,
    *,
    mode: str,
    xgb_feature_stats: dict | None = None,
    return_layout: bool = False,
) -> tuple[np.ndarray, dict | None]:
    """Fuse DINO patch embeddings with optional image patch descriptors.

    Args:
        dino_feats_hwc (np.ndarray): DINO features (Hp, Wp, C).
        img_tile_hw3 (np.ndarray | None): Cropped tile image aligned to dino grid.
        ps (int): Patch size.
        mode (str): ``knn`` or ``xgb``.
        xgb_feature_stats (dict | None): Optional z-score stats for XGB inference.
        return_layout (bool): Whether to return feature names and block ranges.

    Returns:
        tuple[np.ndarray, dict | None]: Fused features and optional layout.

    Examples:
        >>> import numpy as np
        >>> d = np.zeros((2, 2, 4), dtype=np.float32)
        >>> f, layout = fuse_patch_features(
        ...     d,
        ...     np.zeros((32, 32, 3), dtype=np.uint8),
        ...     16,
        ...     mode="knn",
        ...     return_layout=True,
        ... )
        >>> f.shape[:2], isinstance(layout, dict)
        ((2, 2), True)
    """
    if mode not in {"knn", "xgb"}:
        raise ValueError(f"mode must be 'knn' or 'xgb', got {mode}")
    if dino_feats_hwc.ndim != 3:
        raise ValueError("dino_feats_hwc must be (Hp, Wp, C)")

    hf = cfg.model.hybrid_features
    hp, wp, dino_dim = dino_feats_hwc.shape
    feat_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    block_slices: dict[str, list[int]] = {}
    offset = 0

    def add_block(
        name: str,
        values_hwc: np.ndarray,
        channels: list[str],
        weight: float,
    ) -> None:
        nonlocal offset
        if values_hwc.size == 0 or values_hwc.shape[-1] == 0:
            return
        weighted = (values_hwc.astype(np.float32, copy=False) * float(weight)).astype(
            np.float32
        )
        feat_blocks.append(weighted)
        start = offset
        offset += weighted.shape[-1]
        block_slices[name] = [int(start), int(offset)]
        if return_layout:
            feature_names.extend(channels)

    use_hybrid = bool(hf.enabled)
    if (not use_hybrid) or hf.blocks.dino.enabled:
        dino_weight = _mode_weight(hf.blocks.dino, mode) if use_hybrid else 1.0
        add_block(
            "dino",
            dino_feats_hwc,
            [f"dino_{i}" for i in range(dino_dim)] if return_layout else [],
            dino_weight,
        )

    if use_hybrid and img_tile_hw3 is not None:
        h_eff = hp * ps
        w_eff = wp * ps
        img_c = img_tile_hw3[:h_eff, :w_eff, :3]
        img_f = img_c.astype(np.float32)
        if img_f.max() > 1.5:
            img_f /= 255.0
        patch_rgb = img_f.reshape(hp, ps, wp, ps, 3)
        gray = (
            0.2989 * img_f[:, :, 0] + 0.5870 * img_f[:, :, 1] + 0.1140 * img_f[:, :, 2]
        ).astype(np.float32)
        gy, gx = np.gradient(gray)
        grad_mag = np.hypot(gx, gy).astype(np.float32)
        grad_ori = np.mod(np.arctan2(gy, gx), 2.0 * np.pi).astype(np.float32)

        rows = (np.arange(h_eff, dtype=np.int32) // ps).reshape(-1, 1)
        cols = (np.arange(w_eff, dtype=np.int32) // ps).reshape(1, -1)
        patch_idx_map = rows * wp + cols

        if hf.blocks.rgb_stats.enabled:
            rgb_mean = patch_rgb.mean(axis=(1, 3)).astype(np.float32)
            rgb_std = patch_rgb.std(axis=(1, 3)).astype(np.float32)
            add_block(
                "rgb_stats",
                np.concatenate([rgb_mean, rgb_std], axis=-1),
                (
                    [
                        "rgb_mean_r",
                        "rgb_mean_g",
                        "rgb_mean_b",
                        "rgb_std_r",
                        "rgb_std_g",
                        "rgb_std_b",
                    ]
                    if return_layout
                    else []
                ),
                _mode_weight(hf.blocks.rgb_stats, mode),
            )

        if hf.blocks.hsv_mean.enabled:
            hsv = _rgb_to_hsv_image(img_f)
            patch_hsv = hsv.reshape(hp, ps, wp, ps, 3)
            hsv_mean = patch_hsv.mean(axis=(1, 3)).astype(np.float32)
            add_block(
                "hsv_mean",
                hsv_mean,
                ["hsv_mean_h", "hsv_mean_s", "hsv_mean_v"] if return_layout else [],
                _mode_weight(hf.blocks.hsv_mean, mode),
            )

        if hf.blocks.grad_stats.enabled:
            grad_blocks = grad_mag.reshape(hp, ps, wp, ps)
            grad_mean = grad_blocks.mean(axis=(1, 3))
            grad_std = grad_blocks.std(axis=(1, 3))
            grad_stats = np.stack([grad_mean, grad_std], axis=-1).astype(np.float32)
            add_block(
                "grad_stats",
                grad_stats,
                ["grad_mag_mean", "grad_mag_std"] if return_layout else [],
                _mode_weight(hf.blocks.grad_stats, mode),
            )

        if hf.blocks.grad_orient_hist.enabled:
            bins = int(hf.blocks.grad_orient_hist.bins or 8)
            ori_bins = np.floor((grad_ori / (2.0 * np.pi)) * bins).astype(np.int32)
            ori_bins = np.clip(ori_bins, 0, bins - 1)
            hist = np.zeros((hp * wp, bins), dtype=np.float32)
            np.add.at(
                hist,
                (patch_idx_map.reshape(-1), ori_bins.reshape(-1)),
                grad_mag.reshape(-1),
            )
            hist /= hist.sum(axis=1, keepdims=True) + 1e-8
            hist = hist.reshape(hp, wp, bins)
            add_block(
                "grad_orient_hist",
                hist,
                (
                    [f"grad_orient_hist_{i}" for i in range(bins)]
                    if return_layout
                    else []
                ),
                _mode_weight(hf.blocks.grad_orient_hist, mode),
            )

        if hf.blocks.lbp_hist.enabled:
            lbp_bins = int(hf.blocks.lbp_hist.bins or 16)
            lbp_codes = _compute_lbp_codes(gray)
            if lbp_codes.size > 0:
                lbp_idx = ((lbp_codes.astype(np.int32) * lbp_bins) // 256).astype(
                    np.int32
                )
                lbp_idx = np.clip(lbp_idx, 0, lbp_bins - 1)
                patch_idx_inner = patch_idx_map[1:-1, 1:-1]
                hist = np.zeros((hp * wp, lbp_bins), dtype=np.float32)
                np.add.at(
                    hist,
                    (patch_idx_inner.reshape(-1), lbp_idx.reshape(-1)),
                    1.0,
                )
                hist /= hist.sum(axis=1, keepdims=True) + 1e-8
                hist = hist.reshape(hp, wp, lbp_bins)
            else:
                hist = np.zeros((hp, wp, lbp_bins), dtype=np.float32)
            add_block(
                "lbp_hist",
                hist,
                [f"lbp_hist_{i}" for i in range(lbp_bins)] if return_layout else [],
                _mode_weight(hf.blocks.lbp_hist, mode),
            )

    if not feat_blocks:
        fused = dino_feats_hwc.astype(np.float32, copy=False)
        block_slices = {"dino": [0, int(fused.shape[-1])]}
        if return_layout:
            feature_names = [f"dino_{i}" for i in range(fused.shape[-1])]
    else:
        fused = np.concatenate(feat_blocks, axis=-1).astype(np.float32)

    if use_hybrid and mode == "knn" and hf.knn_l2_normalize:
        fused = l2_normalize(fused)
    if use_hybrid and mode == "xgb" and hf.xgb_zscore and xgb_feature_stats is not None:
        fused = _apply_xgb_stats_hwc(fused, xgb_feature_stats)

    layout = None
    if return_layout:
        layout = {
            "feature_names": feature_names,
            "block_slices": block_slices,
            "feature_dim": int(fused.shape[-1]),
            "feature_spec_hash": hybrid_feature_spec_hash(),
            "hybrid_enabled": bool(use_hybrid),
            "dino_slice": block_slices.get("dino"),
        }
    return fused, layout


def tile_iterator(
    image_hw3: np.ndarray,
    labels_hw: np.ndarray | None = None,
    tile_size: int = 1024,
    stride: int | None = None,
):
    """Yield (y, x, img_tile, label_tile) windows over an image.

    Args:
        image_hw3 (np.ndarray): HxWxC image array.
        labels_hw (np.ndarray | None): Optional label mask.
        tile_size (int): Tile size in pixels.
        stride (int | None): Stride in pixels; defaults to tile_size.

    Yields:
        tuple: (y, x, img_tile, label_tile) for each tile.

    Examples:
        >>> import numpy as np
        >>> tiles = list(tile_iterator(np.zeros((3, 4, 3)), tile_size=2, stride=2))
        >>> [(y, x, t.shape[:2]) for y, x, t, _ in tiles]
        [(0, 0, (2, 2)), (0, 2, (2, 2)), (2, 0, (1, 2)), (2, 2, (1, 2))]
    """
    h, w = image_hw3.shape[:2]
    if stride is None:
        stride = tile_size
    y = 0
    while y < h:
        x = 0
        y_end = min(y + tile_size, h)
        while x < w:
            x_end = min(x + tile_size, w)
            img_tile = image_hw3[y:y_end, x:x_end]
            lab_tile = labels_hw[y:y_end, x:x_end] if labels_hw is not None else None
            yield y, x, img_tile, lab_tile
            x += stride
        y += stride


def crop_to_multiple_of_ps(
    img_tile_hw3: np.ndarray, labels_tile_hw: np.ndarray | None, ps: int
):
    """Crop a tile so height/width are multiples of patch size.

    Args:
        img_tile_hw3 (np.ndarray): Image tile (H, W, C).
        labels_tile_hw (np.ndarray | None): Optional label tile.
        ps (int): Patch size in pixels.

    Returns:
        tuple: Cropped image, cropped labels, effective height, effective width.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((5, 7, 3))
        >>> labels = np.ones((5, 7))
        >>> img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(img, labels, 4)
        >>> img_c.shape, lab_c.shape, (h_eff, w_eff)
        ((4, 4, 3), (4, 4), (4, 4))
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h, w = img_tile_hw3.shape[:2]
    h_eff = (h // ps) * ps
    w_eff = (w // ps) * ps
    img_c = img_tile_hw3[:h_eff, :w_eff]
    lab_c = labels_tile_hw[:h_eff, :w_eff] if labels_tile_hw is not None else None
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("crop_to_multiple_of_ps", t0)
    return img_c, lab_c, h_eff, w_eff


def labels_to_patch_masks(
    labels_tile: np.ndarray, hp: int, wp: int, pos_frac_thresh: float = 0.1
):
    """Convert pixel labels to patch-level positive/negative masks.

    Args:
        labels_tile (np.ndarray): Label mask at pixel resolution.
        hp (int): Patch grid height.
        wp (int): Patch grid width.
        pos_frac_thresh (float): Fraction of positive pixels to mark a patch as positive.

    Returns:
        tuple[np.ndarray, np.ndarray]: Boolean positive mask, boolean negative mask.

    Examples:
        >>> import numpy as np
        >>> labels = np.array([
        ...     [1, 1, 0, 0],
        ...     [1, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ... ])
        >>> pos, neg = labels_to_patch_masks(labels, hp=2, wp=2, pos_frac_thresh=0.5)
        >>> pos.tolist(), neg.tolist()
        ([[True, False], [False, False]], [[False, True], [True, True]])
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h_eff, w_eff = labels_tile.shape
    patch_h = h_eff // hp
    patch_w = w_eff // wp
    labels_c = labels_tile[: hp * patch_h, : wp * patch_w]
    labels_bin = (labels_c > 0).astype(np.float32)
    blocks = labels_bin.reshape(hp, patch_h, wp, patch_w)
    frac_pos = blocks.mean(axis=(1, 3))
    pos_mask = frac_pos >= pos_frac_thresh
    neg_mask = frac_pos == 0.0
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("labels_to_patch_masks", t0)
    return pos_mask, neg_mask


def tile_feature_path(feature_dir: str, image_id: str, y: int, x: int) -> str:
    """Return the canonical path for a tile's feature array.

    Args:
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.

    Returns:
        str: Absolute or relative path for the feature file.

    Examples:
        >>> tile_feature_path("feat", "img", 1, 2)
        'feat/img_y1_x2_features.npy'
    """
    fname = f"{image_id}_y{y}_x{x}_features.npy"
    return os.path.join(feature_dir, fname)


def tile_feature_meta_path(feature_dir: str, image_id: str, y: int, x: int) -> str:
    """Return the sidecar JSON path for feature metadata.

    Args:
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.

    Returns:
        str: Absolute or relative path for the metadata JSON.

    Examples:
        >>> tile_feature_meta_path("feat", "img", 1, 2)
        'feat/img_y1_x2_features.json'
    """
    fname = f"{image_id}_y{y}_x{x}_features.json"
    return os.path.join(feature_dir, fname)


def save_tile_features(
    feats_tile: np.ndarray,
    feature_dir: str,
    image_id: str,
    y: int,
    x: int,
    meta: dict | None = None,
):
    """Persist a tile's features to disk (and optional metadata).

    Args:
        feats_tile (np.ndarray): Feature tile array.
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.
        meta (dict | None): Optional metadata to serialize as JSON.

    Examples:
        >>> import numpy as np
        >>> import os
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     feats = np.zeros((2, 2, 3), dtype=np.float32)
        ...     save_tile_features(
        ...         feats,
        ...         d,
        ...         "img",
        ...         0,
        ...         0,
        ...         meta={
        ...             "ps": 16,
        ...             "resample_factor": 1,
        ...             "feature_spec_hash": hybrid_feature_spec_hash(),
        ...         },
        ...     )
        ...     os.path.exists(tile_feature_path(d, "img", 0, 0))
        True
    """
    os.makedirs(feature_dir, exist_ok=True)
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    np.save(fpath, feats_tile.astype(np.float32))
    if meta is not None:
        mpath = tile_feature_meta_path(feature_dir, image_id, y, x)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f)


def load_tile_features_if_valid(
    feature_dir: str,
    image_id: str,
    y: int,
    x: int,
    expected_hp: int,
    expected_wp: int,
    ps: int,
    resample_factor: int,
) -> np.ndarray | None:
    """Load cached features if valid, otherwise return None.

    If metadata is missing or mismatched, the cache is removed.

    Args:
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.
        expected_hp (int): Expected patch grid height.
        expected_wp (int): Expected patch grid width.
        ps (int): Patch size in pixels.
        resample_factor (int): Resample factor used for cached data.

    Returns:
        np.ndarray | None: Cached feature tile if valid, else None.

    Examples:
        >>> import numpy as np
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     feats = np.zeros((2, 2, 3), dtype=np.float32)
        ...     save_tile_features(
        ...         feats,
        ...         d,
        ...         "img",
        ...         0,
        ...         0,
        ...         meta={
        ...             "ps": 16,
        ...             "resample_factor": 1,
        ...             "feature_spec_hash": hybrid_feature_spec_hash(),
        ...         },
        ...     )
        ...     out = load_tile_features_if_valid(d, "img", 0, 0, 2, 2, ps=16, resample_factor=1)
        ...     out.shape
        (2, 2, 3)
    """
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    if not os.path.exists(fpath):
        return None
    mpath = tile_feature_meta_path(feature_dir, image_id, y, x)
    if not os.path.exists(mpath):
        try:
            os.remove(fpath)
        except OSError:
            pass
        return None
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        try:
            os.remove(fpath)
        except OSError:
            pass
        return None

    if meta.get("ps") != ps or meta.get("resample_factor") != resample_factor:
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None
    expected_spec_hash = hybrid_feature_spec_hash()
    if meta.get("feature_spec_hash") != expected_spec_hash:
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None

    feats = np.load(fpath)
    if feats.shape[0] != expected_hp or feats.shape[1] != expected_wp:
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None
    return feats


def extract_patch_features_single_scale(
    image_hw3: np.ndarray, model, processor, device, ps: int = 16, aggregate_layers=None
):
    """Extract single-scale DINO patch features (Hp×Wp×C) from an RGB image.

    Args:
        image_hw3 (np.ndarray): RGB image array.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        aggregate_layers (list[int] | None): Optional layer indices to average.

    Returns:
        tuple[np.ndarray, int, int]: Feature grid, patch height, patch width.

    Examples:
        >>> callable(extract_patch_features_single_scale)
        True
    """
    t0 = time_start()
    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, h_proc, w_proc = pixel_values.shape
    with torch.no_grad():
        if aggregate_layers is None:
            out = model(**inputs)
            tokens = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states
            layers = [hidden_states[i] for i in aggregate_layers]
            tokens = torch.stack(layers, dim=0).mean(0)
    reg_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = tokens[:, 1 + reg_tokens :, :]
    num_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[2]
    hp = h_proc // ps
    wp = w_proc // ps
    assert hp * wp == num_tokens, f"patch-grid mismatch: {hp} * {wp} != {num_tokens}"
    feats = patch_tokens[0].cpu().numpy().reshape(hp, wp, dim)
    feats = l2_normalize(feats)
    time_end("extract_patch_features_single_scale", t0)
    return feats, hp, wp


def extract_patch_features_batch_single_scale(
    images_hw3: list[np.ndarray],
    model,
    processor,
    device,
    ps: int = 16,
    aggregate_layers=None,
):
    """Extract DINO patch features for a batch of same-sized RGB tiles.

    Args:
        images_hw3 (list[np.ndarray]): List of RGB tile arrays (same H/W).
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        aggregate_layers (list[int] | None): Optional layer indices to average.

    Returns:
        tuple[list[np.ndarray], int, int]: Feature grids, patch height, patch width.

    Examples:
        >>> callable(extract_patch_features_batch_single_scale)
        True
    """
    t0 = time_start()
    inputs = processor(
        images=images_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, h_proc, w_proc = pixel_values.shape
    with torch.no_grad():
        if aggregate_layers is None:
            out = model(**inputs)
            tokens = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states
            layers = [hidden_states[i] for i in aggregate_layers]
            tokens = torch.stack(layers, dim=0).mean(0)
    reg_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = tokens[:, 1 + reg_tokens :, :]
    num_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[2]
    hp = h_proc // ps
    wp = w_proc // ps
    assert hp * wp == num_tokens, f"patch-grid mismatch: {hp} * {wp} != {num_tokens}"
    feats_np = patch_tokens.cpu().numpy().reshape(len(images_hw3), hp, wp, dim)
    feats_list = [l2_normalize(feats_np[i]) for i in range(feats_np.shape[0])]
    time_end("extract_patch_features_batch_single_scale", t0)
    return feats_list, hp, wp


def prefetch_features_single_scale_image(
    img_hw3: np.ndarray,
    model,
    processor,
    device,
    ps: int = 16,
    tile_size: int = 1024,
    stride: int | None = None,
    aggregate_layers=None,
    feature_dir: str | None = None,
    image_id: str | None = None,
):
    """Precompute and cache all tile features for an image.

    Args:
        img_hw3 (np.ndarray): RGB image array.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int | None): Tile stride.
        aggregate_layers (list[int] | None): Optional layer indices to average.
        feature_dir (str | None): Optional cache directory.
        image_id (str | None): Optional image id for cache naming.

    Returns:
        dict: Cache keyed by (y, x) with feature arrays and tile shapes.

    Examples:
        >>> callable(prefetch_features_single_scale_image)
        True
    """
    t0 = time_start()
    cache = {}
    cached_tiles = computed_tiles = skipped_tiles = 0
    resample_factor = int(cfg.model.backbone.resample_factor or 1)
    batch_size = int(cfg.runtime.feature_batch_size or 1)
    batch_size = max(1, batch_size)
    pending: dict[tuple[int, int], list[tuple[int, int, np.ndarray, int, int]]] = {}

    def flush_pending(
        items: list[tuple[int, int, np.ndarray, int, int]],
    ) -> None:
        nonlocal computed_tiles
        if not items:
            return
        if batch_size <= 1:
            for y_i, x_i, img_i, h_i, w_i in items:
                feats_tile, hp_i, wp_i = extract_patch_features_single_scale(
                    img_i,
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=aggregate_layers,
                )
                computed_tiles += 1
                if feature_dir is not None and image_id is not None:
                    meta = {
                        "ps": ps,
                        "resample_factor": resample_factor,
                        "h_eff": h_i,
                        "w_eff": w_i,
                        "feature_spec_hash": hybrid_feature_spec_hash(),
                    }
                    save_tile_features(
                        feats_tile, feature_dir, image_id, y_i, x_i, meta=meta
                    )
                cache[(y_i, x_i)] = {
                    "feats": feats_tile,
                    "h_eff": h_i,
                    "w_eff": w_i,
                    "hp": hp_i,
                    "wp": wp_i,
                }
            return

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            imgs = [item[2] for item in chunk]
            if len(chunk) == 1:
                feats_tile, hp_i, wp_i = extract_patch_features_single_scale(
                    imgs[0],
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=aggregate_layers,
                )
                feats_list = [feats_tile]
            else:
                feats_list, hp_i, wp_i = extract_patch_features_batch_single_scale(
                    imgs,
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=aggregate_layers,
                )
            for (y_i, x_i, _img_i, h_i, w_i), feats_tile in zip(
                chunk, feats_list, strict=True
            ):
                computed_tiles += 1
                if feature_dir is not None and image_id is not None:
                    meta = {
                        "ps": ps,
                        "resample_factor": resample_factor,
                        "h_eff": h_i,
                        "w_eff": w_i,
                        "feature_spec_hash": hybrid_feature_spec_hash(),
                    }
                    save_tile_features(
                        feats_tile, feature_dir, image_id, y_i, x_i, meta=meta
                    )
                cache[(y_i, x_i)] = {
                    "feats": feats_tile,
                    "h_eff": h_i,
                    "w_eff": w_i,
                    "hp": hp_i,
                    "wp": wp_i,
                }

    for y, x, img_tile, _ in tile_iterator(img_hw3, None, tile_size, stride):
        img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
        if h_eff < ps or w_eff < ps:
            skipped_tiles += 1
            continue
        feats_tile = None
        hp = wp = None
        if feature_dir is not None and image_id is not None:
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
            if feats_tile is not None:
                cached_tiles += 1
        if feats_tile is None:
            key = (h_eff, w_eff)
            pending.setdefault(key, []).append((y, x, img_c, h_eff, w_eff))
            if len(pending[key]) >= batch_size:
                flush_pending(pending.pop(key))
        else:
            cache[(y, x)] = {
                "feats": feats_tile,
                "h_eff": h_eff,
                "w_eff": w_eff,
                "hp": hp,
                "wp": wp,
            }

    for items in pending.values():
        flush_pending(items)
    time_end("prefetch_features_single_scale_image", t0)
    logger.info(
        "prefetch tiles=%s (cached=%s, computed=%s, skipped=%s)",
        len(cache),
        cached_tiles,
        computed_tiles,
        skipped_tiles,
    )
    return cache
