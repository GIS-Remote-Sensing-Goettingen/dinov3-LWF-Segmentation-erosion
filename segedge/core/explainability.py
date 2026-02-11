"""Explainability utilities for SegEdge inference outputs."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

logger = logging.getLogger(__name__)


XAI_SUMMARY_FIELDS = [
    "timestamp_utc",
    "stage",
    "image_id",
    "tile_path",
    "xgb_top_dim_1",
    "xgb_top_dim_2",
    "xgb_top_dim_3",
    "xgb_top_dim_4",
    "xgb_top_dim_5",
    "xgb_gain_share_top5",
    "knn_score_mean",
    "knn_score_p95",
    "xgb_score_mean",
    "xgb_score_p95",
    "champion_fg_ratio",
    "xai_total_s",
]


def _parse_feature_index(feature_name: str) -> int | None:
    """Parse XGBoost feature key like ``f42`` into an integer index.

    Examples:
        >>> _parse_feature_index("f42")
        42
        >>> _parse_feature_index("gain") is None
        True
    """
    if feature_name.startswith("f") and feature_name[1:].isdigit():
        return int(feature_name[1:])
    return None


def distribution_stats(values: np.ndarray) -> dict[str, float]:
    """Return compact distribution statistics.

    Args:
        values (np.ndarray): Input array.

    Returns:
        dict[str, float]: Min/max/mean/median/p95 stats.

    Examples:
        >>> stats = distribution_stats(np.array([0.0, 1.0, 2.0]))
        >>> stats["mean"]
        1.0
    """
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
        }
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
    }


def get_xgb_importance_dict(bst, top_k: int = 20) -> dict:
    """Extract top latent dimensions from XGBoost feature importances.

    Args:
        bst: Trained XGBoost booster.
        top_k (int): Max number of dimensions to return.

    Returns:
        dict: Importance payload with top dimensions and normalized shares.

    Examples:
        >>> class _B:
        ...     def get_score(self, importance_type="gain"):
        ...         return {"f0": 2.0, "f2": 1.0} if importance_type == "gain" else {}
        >>> out = get_xgb_importance_dict(_B(), top_k=2)
        >>> out["top_dims"]
        [0, 2]
    """
    if bst is None or not hasattr(bst, "get_score"):
        return {"top_dims": [], "importance": [], "xgb_gain_share_top5": 0.0}

    gain = bst.get_score(importance_type="gain") or {}
    weight = bst.get_score(importance_type="weight") or {}
    cover = bst.get_score(importance_type="cover") or {}

    feat_names = set(gain) | set(weight) | set(cover)
    dim_rows = []
    for feat_name in feat_names:
        dim = _parse_feature_index(feat_name)
        if dim is None:
            continue
        dim_rows.append(
            {
                "dim": dim,
                "feature": feat_name,
                "gain": float(gain.get(feat_name, 0.0)),
                "weight": float(weight.get(feat_name, 0.0)),
                "cover": float(cover.get(feat_name, 0.0)),
            }
        )

    dim_rows.sort(key=lambda r: (-r["gain"], r["dim"]))
    if top_k > 0:
        dim_rows = dim_rows[:top_k]

    total_gain_all = float(sum(float(v) for v in gain.values()))
    if total_gain_all > 0.0:
        for row in dim_rows:
            row["gain_share"] = float(row["gain"] / total_gain_all)
    else:
        for row in dim_rows:
            row["gain_share"] = 0.0

    top_dims = [int(r["dim"]) for r in dim_rows]
    top5_gain_share = float(sum(r["gain_share"] for r in dim_rows[:5]))
    return {
        "top_dims": top_dims,
        "importance": dim_rows,
        "xgb_gain_share_top5": top5_gain_share,
    }


def summarize_tile_latent_dims(
    prefetched_tiles: dict | None,
    top_dims: list[int],
    top_patches: int = 50,
) -> dict:
    """Summarize selected latent dimensions and top patch hotspots.

    Args:
        prefetched_tiles (dict | None): In-memory tile feature cache.
        top_dims (list[int]): Selected feature dimensions.
        top_patches (int): Number of hotspot patches to keep.

    Returns:
        dict: Dimension stats and top patch list.

    Examples:
        >>> feats = np.ones((2, 2, 3), dtype=np.float32)
        >>> pref = {(0, 0): {"feats": feats, "hp": 2, "wp": 2, "h_eff": 32, "w_eff": 32}}
        >>> out = summarize_tile_latent_dims(pref, [0, 1], top_patches=2)
        >>> out["tile_count"]
        1
    """
    if not prefetched_tiles:
        return {
            "tile_count": 0,
            "selected_dims": [],
            "dim_stats": [],
            "top_patches": [],
        }

    first = next(iter(prefetched_tiles.values()))
    feat_dim = int(first["feats"].shape[-1])
    dims = sorted({int(d) for d in top_dims if 0 <= int(d) < feat_dim})

    vals_by_dim: dict[int, list[np.ndarray]] = {d: [] for d in dims}
    candidates = []
    top_patches = max(0, int(top_patches))

    for (y, x), info in sorted(prefetched_tiles.items()):
        feats = np.asarray(info["feats"], dtype=np.float32)
        hp = int(info["hp"])
        wp = int(info["wp"])
        h_eff = int(info["h_eff"])
        w_eff = int(info["w_eff"])
        if feats.size == 0 or hp <= 0 or wp <= 0:
            continue

        if dims:
            for d in dims:
                vals_by_dim[d].append(feats[..., d].reshape(-1))
            patch_scores = np.mean(np.abs(feats[..., dims]), axis=2)
        else:
            patch_scores = np.linalg.norm(feats, axis=2)

        if top_patches <= 0:
            continue
        flat = patch_scores.reshape(-1)
        local_k = min(top_patches, flat.size)
        idxs = np.argpartition(flat, -local_k)[-local_k:]
        for idx in idxs:
            py = int(idx // wp)
            px = int(idx % wp)
            scale_y = float(h_eff) / float(hp)
            scale_x = float(w_eff) / float(wp)
            candidates.append(
                {
                    "y_px": int(round(y + (py + 0.5) * scale_y)),
                    "x_px": int(round(x + (px + 0.5) * scale_x)),
                    "score": float(flat[idx]),
                    "tile_y": int(y),
                    "tile_x": int(x),
                    "patch_y": py,
                    "patch_x": px,
                }
            )

    dim_stats = []
    for d in dims:
        if not vals_by_dim[d]:
            continue
        arr = np.concatenate(vals_by_dim[d], axis=0).astype(np.float32, copy=False)
        dim_stats.append(
            {
                "dim": int(d),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p95": float(np.percentile(arr, 95)),
                "abs_mean": float(np.mean(np.abs(arr))),
            }
        )

    candidates.sort(key=lambda row: (-row["score"], row["y_px"], row["x_px"]))
    return {
        "tile_count": int(len(prefetched_tiles)),
        "selected_dims": dims,
        "dim_stats": dim_stats,
        "top_patches": candidates[:top_patches] if top_patches > 0 else [],
    }


def build_dim_activation_map(
    prefetched_tiles: dict | None,
    image_shape: tuple[int, int],
    dims: list[int],
) -> np.ndarray:
    """Build a pixel-level activation proxy map from selected latent dimensions.

    Args:
        prefetched_tiles (dict | None): In-memory tile feature cache.
        image_shape (tuple[int, int]): (H, W) target shape.
        dims (list[int]): Selected feature dimensions.

    Returns:
        np.ndarray: Activation map in image resolution.

    Examples:
        >>> feats = np.ones((1, 1, 2), dtype=np.float32)
        >>> pref = {(0, 0): {"feats": feats, "hp": 1, "wp": 1, "h_eff": 4, "w_eff": 4}}
        >>> build_dim_activation_map(pref, (4, 4), [0]).shape
        (4, 4)
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    if not prefetched_tiles or h <= 0 or w <= 0 or not dims:
        return np.zeros((h, w), dtype=np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for (y, x), info in prefetched_tiles.items():
        feats = np.asarray(info["feats"], dtype=np.float32)
        hp = int(info["hp"])
        wp = int(info["wp"])
        h_eff = int(info["h_eff"])
        w_eff = int(info["w_eff"])
        if feats.size == 0 or hp <= 0 or wp <= 0:
            continue
        map_patch = np.mean(np.abs(feats[..., dims]), axis=2).astype(np.float32)
        map_pix = resize(
            map_patch,
            (h_eff, w_eff),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)
        out[y : y + h_eff, x : x + w_eff] += map_pix
        weight[y : y + h_eff, x : x + w_eff] += 1.0
    nz = weight > 0
    out[nz] /= weight[nz]
    return out


def summarize_knn_signals(
    score_knn: np.ndarray,
    saliency_knn: np.ndarray | None,
    sh_buffer_mask: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """Summarize kNN score/saliency distributions for explainability.

    Examples:
        >>> s = np.array([[0.1, 0.9]], dtype=np.float32)
        >>> b = np.array([[True, True]])
        >>> out = summarize_knn_signals(s, None, b, threshold=0.5)
        >>> out["buffered_positive_ratio"]
        0.5
    """
    score_stats = distribution_stats(score_knn)
    saliency_stats = (
        distribution_stats(saliency_knn) if saliency_knn is not None else None
    )
    buffer = sh_buffer_mask.astype(bool)
    if buffer.any():
        buffered_mean = float(np.mean(score_knn[buffer]))
        if threshold is None:
            buffered_positive_ratio = None
        else:
            buffered_positive_ratio = float(
                np.mean(score_knn[buffer] >= float(threshold))
            )
    else:
        buffered_mean = 0.0
        buffered_positive_ratio = 0.0 if threshold is not None else None
    return {
        "score_distribution": score_stats,
        "saliency_distribution": saliency_stats,
        "buffered_score_mean": buffered_mean,
        "buffered_positive_ratio": buffered_positive_ratio,
    }


def _normalize_map(values: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1].

    Examples:
        >>> _normalize_map(np.array([1.0, 3.0])).tolist()
        [0.0, 1.0]
    """
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return vals
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax - vmin <= 1e-8:
        return np.zeros_like(vals)
    return (vals - vmin) / (vmax - vmin)


def save_xai_tile_plot(
    img_b: np.ndarray,
    score_xgb: np.ndarray,
    score_knn: np.ndarray,
    dim_activation_map: np.ndarray,
    champion_mask: np.ndarray,
    sh_buffer_mask: np.ndarray,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """Save a 2x3 XAI panel figure for one tile.

    Examples:
        >>> isinstance(save_xai_tile_plot.__name__, str)
        True
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = np.asarray(axs)

    axs[0, 0].imshow(img_b)
    axs[0, 0].set_title("RGB")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(_normalize_map(score_xgb), cmap="coolwarm")
    axs[0, 1].set_title("XGB score")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(_normalize_map(score_knn), cmap="coolwarm")
    axs[0, 2].set_title("kNN score")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(_normalize_map(dim_activation_map), cmap="magma")
    axs[1, 0].set_title("Top-dim activation proxy")
    axs[1, 0].axis("off")

    overlay = img_b.copy()
    champion = champion_mask.astype(bool)
    overlay[champion] = (
        0.5 * overlay[champion] + 0.5 * np.array([255, 0, 0], dtype=np.float32)
    ).astype(overlay.dtype)
    axs[1, 1].imshow(overlay)
    axs[1, 1].set_title("Champion mask overlay")
    axs[1, 1].axis("off")

    buffer_overlay = img_b.copy()
    buffer = sh_buffer_mask.astype(bool)
    buffer_overlay[buffer] = (
        0.6 * buffer_overlay[buffer] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(buffer_overlay.dtype)
    axs[1, 2].imshow(buffer_overlay)
    axs[1, 2].set_title("SH buffer" + (f" ({title_suffix})" if title_suffix else ""))
    axs[1, 2].axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("xai plot saved: %s", out_path)


def write_xai_tile_json(out_path: str, payload: dict) -> None:
    """Write one per-tile explainability JSON payload.

    Examples:
        >>> isinstance(write_xai_tile_json.__name__, str)
        True
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def append_xai_summary_csv(out_path: str, row: dict) -> None:
    """Append one row to run-level XAI summary CSV.

    Examples:
        >>> isinstance(append_xai_summary_csv.__name__, str)
        True
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    exists = os.path.exists(out_path)
    with open(out_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=XAI_SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_xai_summary_row(
    stage: str,
    image_id: str,
    tile_path: str,
    top_dims: list[int],
    xgb_gain_share_top5: float,
    knn_score_stats: dict,
    xgb_score_stats: dict,
    champion_mask: np.ndarray,
    xai_total_s: float,
) -> dict:
    """Build one standardized XAI summary row.

    Examples:
        >>> row = build_xai_summary_row(
        ...     stage="validation",
        ...     image_id="tile",
        ...     tile_path="tile.tif",
        ...     top_dims=[1, 2],
        ...     xgb_gain_share_top5=0.4,
        ...     knn_score_stats={"mean": 0.1, "p95": 0.2},
        ...     xgb_score_stats={"mean": 0.3, "p95": 0.4},
        ...     champion_mask=np.zeros((2, 2), dtype=bool),
        ...     xai_total_s=1.0,
        ... )
        >>> row["xgb_top_dim_1"]
        1
    """
    dims = list(top_dims[:5]) + [""] * max(0, 5 - len(top_dims))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "image_id": image_id,
        "tile_path": tile_path,
        "xgb_top_dim_1": dims[0],
        "xgb_top_dim_2": dims[1],
        "xgb_top_dim_3": dims[2],
        "xgb_top_dim_4": dims[3],
        "xgb_top_dim_5": dims[4],
        "xgb_gain_share_top5": float(xgb_gain_share_top5),
        "knn_score_mean": float(knn_score_stats.get("mean", 0.0)),
        "knn_score_p95": float(knn_score_stats.get("p95", 0.0)),
        "xgb_score_mean": float(xgb_score_stats.get("mean", 0.0)),
        "xgb_score_p95": float(xgb_score_stats.get("p95", 0.0)),
        "champion_fg_ratio": float(np.mean(champion_mask.astype(bool))),
        "xai_total_s": float(xai_total_s),
    }


def select_holdout_tiles_for_xai(
    holdout_tiles: list[str],
    cap_enabled: bool,
    cap: int | None,
    seed: int,
) -> set[str]:
    """Select holdout tiles that will get explainability artifacts.

    Examples:
        >>> out = select_holdout_tiles_for_xai(["a", "b", "c"], True, 2, 42)
        >>> len(out) == 2
        True
    """
    tiles = list(holdout_tiles)
    if not cap_enabled:
        return set(tiles)
    cap_n = int(cap or 0)
    if cap_n <= 0:
        return set()
    if cap_n >= len(tiles):
        return set(tiles)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(tiles), size=cap_n, replace=False)
    return {tiles[int(i)] for i in idx}
