"""Feature normalization, stats, and fusion helpers."""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.ndimage import uniform_filter

from ..config_loader import cfg
from ..timing_utils import DEBUG_TIMING, DEBUG_TIMING_VERBOSE, perf_span, time_end
from .spec import hybrid_feature_spec_hash

logger = logging.getLogger(__name__)


def l2_normalize(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize feature vectors along the last dimension.

    Args:
        feats (np.ndarray): Feature array with last axis as channels.
        eps (float): Small epsilon for numerical stability.

    Returns:
        np.ndarray: L2-normalized feature array.

    Examples:
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

    Examples:
        >>> feats = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        >>> np.array_equal(add_local_context_mean(feats, 0), feats)
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


def _rgb_to_hsv_image(img_rgb_01: np.ndarray) -> np.ndarray:
    """Convert RGB image in [0, 1] to HSV.

    Examples:
        >>> _rgb_to_hsv_image(np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)).shape
        (1, 1, 3)
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
        >>> _compute_lbp_codes(np.ones((3, 3), dtype=np.float32)).shape
        (1, 1)
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
        >>> class _Cfg:
        ...     weight_knn = 1.5
        ...     weight_xgb = 2.5
        >>> _mode_weight(_Cfg(), "knn")
        1.5
    """
    return float(block_cfg.weight_knn) if mode == "knn" else float(block_cfg.weight_xgb)


def _apply_xgb_stats_hwc(
    feats_hwc: np.ndarray,
    xgb_feature_stats: dict | None,
) -> np.ndarray:
    """Apply per-feature z-score stats to an HxWxC feature tensor.

    Examples:
        >>> feats = np.ones((1, 1, 2), dtype=np.float32)
        >>> stats = {"mean": [1.0, 1.0], "std": [1.0, 2.0]}
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

    Examples:
        >>> stats = fit_xgb_feature_stats(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
        >>> sorted(stats.keys())
        ['eps', 'feature_spec_hash', 'mean', 'std']
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

    Examples:
        >>> X = np.array([[1.0, 3.0]], dtype=np.float32)
        >>> stats = {"mean": [1.0, 1.0], "std": [1.0, 2.0]}
        >>> apply_xgb_feature_stats(X, stats).shape
        (1, 2)
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


def _weighted_feature_block(values_hwc: np.ndarray, weight: float) -> np.ndarray | None:
    """Return a weighted feature block without redundant casts.

    Examples:
        >>> _weighted_feature_block(np.ones((1, 1, 1), dtype=np.float32), 1.0).shape
        (1, 1, 1)
    """
    if values_hwc.size == 0 or values_hwc.shape[-1] == 0:
        return None
    values = (
        values_hwc
        if values_hwc.dtype == np.float32
        else values_hwc.astype(np.float32, copy=False)
    )
    if float(weight) == 1.0:
        return values
    return values * np.float32(weight)


def _fuse_patch_features_fast_xgb(
    dino_feats_hwc: np.ndarray,
    img_tile_hw3: np.ndarray | None,
    ps: int,
    *,
    xgb_feature_stats: dict | None = None,
) -> np.ndarray:
    """Fast XGB-only fusion path without layout bookkeeping.

    Examples:
        >>> feats = np.ones((1, 1, 2), dtype=np.float32)
        >>> _fuse_patch_features_fast_xgb(feats, None, 16).shape
        (1, 1, 2)
    """
    hf = cfg.model.hybrid_features
    hp, wp, _ = dino_feats_hwc.shape
    use_hybrid = bool(hf.enabled)
    feat_blocks: list[np.ndarray] = []

    if (not use_hybrid) or hf.blocks.dino.enabled:
        with perf_span(
            "fuse_patch_features", substage="dino_block", extra={"mode": "xgb"}
        ):
            dino_block = _weighted_feature_block(
                dino_feats_hwc,
                _mode_weight(hf.blocks.dino, "xgb") if use_hybrid else 1.0,
            )
        if dino_block is not None:
            feat_blocks.append(dino_block)

    if use_hybrid and img_tile_hw3 is not None:
        with perf_span(
            "fuse_patch_features",
            substage="prepare_image_blocks",
            extra={"mode": "xgb"},
        ):
            h_eff = hp * ps
            w_eff = wp * ps
            img_c = img_tile_hw3[:h_eff, :w_eff, :3]
            img_f = (
                img_c
                if img_c.dtype == np.float32
                else img_c.astype(np.float32, copy=False)
            )
            if img_f.max() > 1.5:
                img_f = img_f / np.float32(255.0)
            patch_rgb = img_f.reshape(hp, ps, wp, ps, 3)
            gray = (
                0.2989 * img_f[:, :, 0]
                + 0.5870 * img_f[:, :, 1]
                + 0.1140 * img_f[:, :, 2]
            ).astype(np.float32)
            gy, gx = np.gradient(gray)
            grad_mag = np.hypot(gx, gy).astype(np.float32)
            grad_ori = np.mod(np.arctan2(gy, gx), 2.0 * np.pi).astype(np.float32)
            rows = (np.arange(h_eff, dtype=np.int32) // ps).reshape(-1, 1)
            cols = (np.arange(w_eff, dtype=np.int32) // ps).reshape(1, -1)
            patch_idx_map = rows * wp + cols

        if hf.blocks.rgb_stats.enabled:
            with perf_span("fuse_patch_features", substage="rgb_stats_block"):
                rgb_mean = patch_rgb.mean(axis=(1, 3)).astype(np.float32)
                rgb_std = patch_rgb.std(axis=(1, 3)).astype(np.float32)
                block = _weighted_feature_block(
                    np.concatenate([rgb_mean, rgb_std], axis=-1),
                    _mode_weight(hf.blocks.rgb_stats, "xgb"),
                )
            if block is not None:
                feat_blocks.append(block)

        if hf.blocks.hsv_mean.enabled:
            with perf_span("fuse_patch_features", substage="hsv_mean_block"):
                hsv = _rgb_to_hsv_image(img_f)
                patch_hsv = hsv.reshape(hp, ps, wp, ps, 3)
                block = _weighted_feature_block(
                    patch_hsv.mean(axis=(1, 3)).astype(np.float32),
                    _mode_weight(hf.blocks.hsv_mean, "xgb"),
                )
            if block is not None:
                feat_blocks.append(block)

        if hf.blocks.grad_stats.enabled:
            with perf_span("fuse_patch_features", substage="grad_stats_block"):
                grad_blocks = grad_mag.reshape(hp, ps, wp, ps)
                grad_mean = grad_blocks.mean(axis=(1, 3))
                grad_std = grad_blocks.std(axis=(1, 3))
                grad_stats = np.stack([grad_mean, grad_std], axis=-1).astype(np.float32)
                block = _weighted_feature_block(
                    grad_stats,
                    _mode_weight(hf.blocks.grad_stats, "xgb"),
                )
            if block is not None:
                feat_blocks.append(block)

        if hf.blocks.grad_orient_hist.enabled:
            with perf_span("fuse_patch_features", substage="grad_orient_hist_block"):
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
                block = _weighted_feature_block(
                    hist.reshape(hp, wp, bins),
                    _mode_weight(hf.blocks.grad_orient_hist, "xgb"),
                )
            if block is not None:
                feat_blocks.append(block)

        if hf.blocks.lbp_hist.enabled:
            with perf_span("fuse_patch_features", substage="lbp_hist_block"):
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
                    lbp_hist = hist.reshape(hp, wp, lbp_bins)
                else:
                    lbp_hist = np.zeros((hp, wp, lbp_bins), dtype=np.float32)
                block = _weighted_feature_block(
                    lbp_hist,
                    _mode_weight(hf.blocks.lbp_hist, "xgb"),
                )
            if block is not None:
                feat_blocks.append(block)

    if not feat_blocks:
        fused = (
            dino_feats_hwc
            if dino_feats_hwc.dtype == np.float32
            else dino_feats_hwc.astype(np.float32, copy=False)
        )
    elif len(feat_blocks) == 1:
        fused = feat_blocks[0]
    else:
        with perf_span("fuse_patch_features", substage="concat_blocks"):
            fused = np.concatenate(feat_blocks, axis=-1)

    if use_hybrid and hf.xgb_zscore and xgb_feature_stats is not None:
        with perf_span("fuse_patch_features", substage="xgb_zscore"):
            fused = _apply_xgb_stats_hwc(fused, xgb_feature_stats)
    return fused


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

    Examples:
        >>> callable(fuse_patch_features)
        True
    """
    if mode not in {"knn", "xgb"}:
        raise ValueError(f"mode must be 'knn' or 'xgb', got {mode}")
    if dino_feats_hwc.ndim != 3:
        raise ValueError("dino_feats_hwc must be (Hp, Wp, C)")
    if mode == "xgb" and not return_layout:
        return (
            _fuse_patch_features_fast_xgb(
                dino_feats_hwc,
                img_tile_hw3,
                ps,
                xgb_feature_stats=xgb_feature_stats,
            ),
            None,
        )

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
        with perf_span(
            "fuse_patch_features", substage="dino_block", extra={"mode": mode}
        ):
            add_block(
                "dino",
                dino_feats_hwc,
                [f"dino_{i}" for i in range(dino_dim)] if return_layout else [],
                dino_weight,
            )

    if use_hybrid and img_tile_hw3 is not None:
        with perf_span(
            "fuse_patch_features",
            substage="prepare_image_blocks",
            extra={"mode": mode},
        ):
            h_eff = hp * ps
            w_eff = wp * ps
            img_c = img_tile_hw3[:h_eff, :w_eff, :3]
            img_f = img_c.astype(np.float32)
            if img_f.max() > 1.5:
                img_f /= 255.0
            patch_rgb = img_f.reshape(hp, ps, wp, ps, 3)
            gray = (
                0.2989 * img_f[:, :, 0]
                + 0.5870 * img_f[:, :, 1]
                + 0.1140 * img_f[:, :, 2]
            ).astype(np.float32)
            gy, gx = np.gradient(gray)
            grad_mag = np.hypot(gx, gy).astype(np.float32)
            grad_ori = np.mod(np.arctan2(gy, gx), 2.0 * np.pi).astype(np.float32)

            rows = (np.arange(h_eff, dtype=np.int32) // ps).reshape(-1, 1)
            cols = (np.arange(w_eff, dtype=np.int32) // ps).reshape(1, -1)
            patch_idx_map = rows * wp + cols

        if hf.blocks.rgb_stats.enabled:
            with perf_span("fuse_patch_features", substage="rgb_stats_block"):
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
            with perf_span("fuse_patch_features", substage="hsv_mean_block"):
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
            with perf_span("fuse_patch_features", substage="grad_stats_block"):
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
            with perf_span("fuse_patch_features", substage="grad_orient_hist_block"):
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
            with perf_span("fuse_patch_features", substage="lbp_hist_block"):
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
        with perf_span("fuse_patch_features", substage="concat_blocks"):
            fused = np.concatenate(feat_blocks, axis=-1).astype(np.float32)

    if use_hybrid and mode == "knn" and hf.knn_l2_normalize:
        with perf_span("fuse_patch_features", substage="knn_l2_normalize"):
            fused = l2_normalize(fused)
    if use_hybrid and mode == "xgb" and hf.xgb_zscore and xgb_feature_stats is not None:
        with perf_span("fuse_patch_features", substage="xgb_zscore"):
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
