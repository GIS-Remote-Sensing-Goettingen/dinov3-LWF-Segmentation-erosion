"""Runtime helpers extracted from the main pipeline entrypoint."""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import time
from datetime import datetime, timezone

import fiona
import numpy as np
import rasterio
import rasterio.features as rfeatures
import yaml
from pyproj import CRS, Transformer
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import label as ndi_label
from scipy.ndimage import median_filter
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from skimage.morphology import skeletonize
from skimage.transform import resize

from ..core.config_loader import cfg
from ..core.crf_utils import refine_with_densecrf
from ..core.features import (
    hybrid_feature_spec_hash,
    prefetch_features_single_scale_image,
)
from ..core.io_utils import (
    build_sh_buffer_mask,
    export_mask_to_shapefile,
    load_dop20_image,
    rasterize_vector_labels,
    reproject_labels_to_image,
)
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from ..core.metrics_utils import compute_metrics
from ..core.plotting import (
    save_core_qualitative_plot,
    save_disagreement_entropy_plot,
    save_proposal_overlay_plot,
    save_score_threshold_plot,
    save_unified_plot,
)
from ..core.timing_utils import time_end, time_start
from ..core.xdboost import xgb_score_image_b

logger = logging.getLogger(__name__)

_CRF_PARALLEL_CONTEXTS: list[dict] | None = None
_ROADS_MASK_CACHE: dict[tuple[str, int, tuple[int, int] | None], np.ndarray] = {}
_ROADS_INDEX_CACHE: dict[tuple[str, str], tuple[STRtree | None, list]] = {}


def compute_budget_deadline(start_ts: float, hours: float) -> float:
    """Return the deadline timestamp for a wall-clock budget.

    Examples:
        >>> round(compute_budget_deadline(100.0, 1.0), 1)
        3700.0
    """
    return float(start_ts) + float(hours) * 3600.0


def is_budget_exceeded(deadline_ts: float | None, now_ts: float | None = None) -> bool:
    """Return True when the current time has reached/exceeded the deadline.

    Examples:
        >>> is_budget_exceeded(10.0, now_ts=11.0)
        True
    """
    if deadline_ts is None:
        return False
    now = time.time() if now_ts is None else float(now_ts)
    return now >= float(deadline_ts)


def remaining_budget_s(
    deadline_ts: float | None,
    now_ts: float | None = None,
) -> float | None:
    """Return remaining seconds before deadline (clamped at >=0).

    Examples:
        >>> remaining_budget_s(10.0, now_ts=7.0)
        3.0
    """
    if deadline_ts is None:
        return None
    now = time.time() if now_ts is None else float(now_ts)
    return max(0.0, float(deadline_ts) - now)


def deadline_ts_to_utc_iso(deadline_ts: float | None) -> str | None:
    """Convert epoch seconds to UTC ISO timestamp string.

    Examples:
        >>> deadline_ts_to_utc_iso(0.0).startswith("1970-01-01T00:00:00")
        True
    """
    if deadline_ts is None:
        return None
    return datetime.fromtimestamp(float(deadline_ts), tz=timezone.utc).isoformat()


def parse_utc_iso_to_epoch(value: str | None) -> float | None:
    """Parse ISO UTC timestamp to epoch seconds.

    Examples:
        >>> parse_utc_iso_to_epoch(None) is None
        True
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def build_time_budget_status(
    *,
    enabled: bool,
    hours: float,
    scope: str,
    cutover_mode: str,
    deadline_ts: float | None,
    clock_start_ts: float | None,
    cutover_triggered: bool,
    cutover_stage: str,
) -> dict | None:
    """Build a serializable snapshot of time-budget status.

    Examples:
        >>> s = build_time_budget_status(
        ...     enabled=True,
        ...     hours=1.0,
        ...     scope="total_wall_clock",
        ...     cutover_mode="immediate_inference",
        ...     deadline_ts=100.0,
        ...     clock_start_ts=0.0,
        ...     cutover_triggered=False,
        ...     cutover_stage="none",
        ... )
        >>> isinstance(s, dict)
        True
    """
    if not enabled:
        return None
    now = time.time()
    rem = remaining_budget_s(deadline_ts, now_ts=now)
    elapsed = None
    if clock_start_ts is not None:
        elapsed = max(0.0, now - float(clock_start_ts))
    return {
        "enabled": bool(enabled),
        "hours": float(hours),
        "scope": str(scope),
        "cutover_mode": str(cutover_mode),
        "deadline_utc": deadline_ts_to_utc_iso(deadline_ts),
        "remaining_s": float(rem) if rem is not None else None,
        "elapsed_s": float(elapsed) if elapsed is not None else None,
        "cutover_triggered": bool(cutover_triggered),
        "cutover_stage": str(cutover_stage),
    }


def _roads_mask_disk_cache_path(
    tile_path: str,
    downsample_factor: int,
    out_shape: tuple[int, int],
    target_shape: tuple[int, int] | None,
) -> str:
    """Return deterministic disk-cache path for a rasterized roads mask.

    Examples:
        >>> p = _roads_mask_disk_cache_path("a.tif", 1, (10, 10), None)
        >>> p.endswith(".npy")
        True
    """
    roads_path = cfg.io.paths.roads_mask_path or "<none>"
    candidate_dirs = []
    if cfg.io.paths.feature_dir:
        candidate_dirs.append(
            os.path.join(cfg.io.paths.feature_dir, "roads_masks_cache")
        )
    candidate_dirs.append(os.path.join(cfg.io.paths.output_dir, "roads_masks_cache"))
    candidate_dirs.append("/tmp/segedge_roads_masks_cache")
    cache_dir = None
    for cand in candidate_dirs:
        try:
            os.makedirs(cand, exist_ok=True)
            cache_dir = cand
            break
        except OSError:
            continue
    if cache_dir is None:
        cache_dir = "/tmp/segedge_roads_masks_cache"
        os.makedirs(cache_dir, exist_ok=True)
    roads_mtime = 0
    tile_mtime = 0
    try:
        roads_mtime = int(os.path.getmtime(roads_path))
    except OSError:
        pass
    try:
        tile_mtime = int(os.path.getmtime(tile_path))
    except OSError:
        pass
    sig = "|".join(
        [
            os.path.abspath(tile_path),
            str(tile_mtime),
            os.path.abspath(roads_path),
            str(roads_mtime),
            str(int(downsample_factor)),
            str(tuple(int(v) for v in out_shape)),
            str(
                tuple(int(v) for v in target_shape)
                if target_shape is not None
                else None
            ),
        ]
    )
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:20]
    return os.path.join(cache_dir, f"roads_mask_{digest}.npy")


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
    key = (tile_path, downsample_factor, tuple(target_shape) if target_shape else None)
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

    disk_cache_path = _roads_mask_disk_cache_path(
        tile_path,
        downsample_factor,
        out_shape,
        target_shape,
    )
    if os.path.exists(disk_cache_path):
        try:
            disk_mask = np.load(disk_cache_path).astype(bool)
            if target_shape is not None and disk_mask.shape != target_shape:
                disk_mask = resize(
                    disk_mask.astype("uint8"),
                    target_shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(bool)
            _ROADS_MASK_CACHE[key] = disk_mask
            logger.info(
                "roads mask disk cache hit: %s", os.path.basename(disk_cache_path)
            )
            return disk_mask
        except Exception:
            pass

    tree, geoms = _get_roads_index(tile_crs)
    if tree is None or not geoms:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        try:
            np.save(disk_cache_path, mask_empty.astype(np.uint8))
        except Exception:
            pass
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
        try:
            np.save(disk_cache_path, mask_empty.astype(np.uint8))
        except Exception:
            pass
        return mask_empty

    if isinstance(hits[0], (int, np.integer)):
        candidates = [geoms[int(idx)] for idx in hits]
    else:
        candidates = list(hits)
    shapes = [mapping(g) for g in candidates if g.intersects(tile_box)]
    if not shapes:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        try:
            np.save(disk_cache_path, mask_empty.astype(np.uint8))
        except Exception:
            pass
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
    try:
        np.save(disk_cache_path, mask_bool.astype(np.uint8))
    except Exception:
        logger.debug("roads mask disk cache write failed: %s", disk_cache_path)
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


def _compute_component_shape_metrics(
    component_mask: np.ndarray,
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    pixel_size_m: float,
) -> dict[str, float]:
    """Compute shape/context metrics for a connected component."""
    area_px = int(component_mask.sum())
    if area_px <= 0:
        return {
            "area_px": 0.0,
            "length_px": 0.0,
            "length_m": 0.0,
            "mean_width_px": 0.0,
            "mean_width_m": 0.0,
            "skeleton_ratio": 0.0,
            "pca_ratio": 0.0,
            "circularity": 1.0,
            "mean_score": 0.0,
            "road_overlap": 0.0,
            "centroid_row": 0.0,
            "centroid_col": 0.0,
        }

    coords = np.argwhere(component_mask)
    centroid_row = float(coords[:, 0].mean())
    centroid_col = float(coords[:, 1].mean())
    pca_ratio = 1.0
    if coords.shape[0] >= 2:
        centered = coords.astype(np.float32) - coords.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 0:
            eigvals = np.array([float(cov), 0.0], dtype=np.float32)
        else:
            eigvals = np.linalg.eigvalsh(cov).astype(np.float32)
            if eigvals.shape[0] == 1:
                eigvals = np.array([eigvals[0], 0.0], dtype=np.float32)
        eigvals = np.sort(eigvals)[::-1]
        lam1 = max(float(eigvals[0]), 1e-8)
        lam2 = max(float(eigvals[1]), 1e-8)
        pca_ratio = lam1 / lam2

    boundary = np.logical_and(component_mask, ~binary_erosion(component_mask))
    perimeter_px = float(boundary.sum())
    circularity = float(4.0 * np.pi * area_px / (perimeter_px**2 + 1e-8))

    skel = skeletonize(component_mask)
    length_px = float(skel.sum())
    if length_px > 0:
        dt = distance_transform_edt(component_mask)
        widths = 2.0 * dt[skel]
        mean_width_px = float(widths.mean()) if widths.size > 0 else 0.0
    else:
        mean_width_px = 0.0
    skeleton_ratio = float(length_px / max(mean_width_px, 1e-6))

    mean_score = (
        float(score_map[component_mask].mean()) if score_map is not None else 0.0
    )
    road_overlap = (
        float(roads_mask[component_mask].mean()) if roads_mask is not None else 0.0
    )
    return {
        "area_px": float(area_px),
        "length_px": float(length_px),
        "length_m": float(length_px * pixel_size_m),
        "mean_width_px": float(mean_width_px),
        "mean_width_m": float(mean_width_px * pixel_size_m),
        "skeleton_ratio": float(skeleton_ratio),
        "pca_ratio": float(pca_ratio),
        "circularity": float(circularity),
        "mean_score": float(mean_score),
        "road_overlap": float(road_overlap),
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
    }


def filter_novel_proposals(
    champion_mask: np.ndarray,
    labels_sh: np.ndarray,
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    pixel_size_m: float,
    sh_buffer_mask: np.ndarray | None = None,
    proposal_source_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Filter candidate components using elongated-object heuristics.

    The candidate source can be supplied explicitly (for example thresholded
    champion score across the whole tile), and scope can be constrained by
    SH buffer or expanded to whole tile via config.

    Examples:
        >>> callable(filter_novel_proposals)
        True
    """
    p = cfg.postprocess.novel_proposals
    if proposal_source_mask is None:
        source_mask = champion_mask.astype(bool)
    else:
        source_mask = proposal_source_mask.astype(bool)
    if p.search_scope == "sh_buffer":
        if sh_buffer_mask is None:
            scope_mask = np.ones_like(source_mask, dtype=bool)
        else:
            scope_mask = sh_buffer_mask.astype(bool)
    else:
        scope_mask = np.ones_like(source_mask, dtype=bool)
    candidate_mask = np.logical_and(source_mask, scope_mask)
    candidate_mask = np.logical_and(candidate_mask, ~(labels_sh > 0))

    if sh_buffer_mask is None:
        buffer_mask = np.zeros_like(candidate_mask, dtype=bool)
    else:
        buffer_mask = sh_buffer_mask.astype(bool)
    candidate_inside = np.logical_and(candidate_mask, buffer_mask)
    candidate_outside = np.logical_and(candidate_mask, ~buffer_mask)
    if not bool(p.enabled):
        return {
            "candidate_mask": candidate_mask,
            "candidate_inside_mask": candidate_inside,
            "candidate_outside_mask": candidate_outside,
            "accepted_inside_mask": np.zeros_like(candidate_mask, dtype=bool),
            "accepted_outside_mask": np.zeros_like(candidate_mask, dtype=bool),
            "evaluated_outside_mask": np.zeros_like(candidate_mask, dtype=bool),
            "accepted_mask": np.zeros_like(candidate_mask, dtype=bool),
            "rejected_mask": np.zeros_like(candidate_mask, dtype=bool),
            "records": [],
        }

    if int(p.connectivity) <= 1:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        structure = np.ones((3, 3), dtype=np.uint8)

    accepted_inside_mask = np.zeros_like(candidate_mask, dtype=bool)
    accepted_outside_mask = np.zeros_like(candidate_mask, dtype=bool)
    rejected_mask = np.zeros_like(candidate_mask, dtype=bool)
    records = []
    component_id = 0

    # Inside-buffer components are accepted by policy.
    inside_labels, inside_count = ndi_label(
        candidate_inside.astype(np.uint8), structure=structure
    )
    for local_id in range(1, int(inside_count) + 1):
        component_id += 1
        comp = inside_labels == local_id
        metrics = _compute_component_shape_metrics(
            comp, score_map, roads_mask, pixel_size_m
        )
        accepted_inside_mask |= comp
        records.append(
            {
                "component_id": int(component_id),
                "accepted": True,
                "acceptance_score": 1.0,
                "reject_reasons": [],
                "zone": "inside_buffer_auto",
                **metrics,
            }
        )

    outside_labels, outside_count = ndi_label(
        candidate_outside.astype(np.uint8), structure=structure
    )
    total_checks = 8.0
    for local_id in range(1, int(outside_count) + 1):
        component_id += 1
        comp = outside_labels == local_id
        metrics = _compute_component_shape_metrics(
            comp, score_map, roads_mask, pixel_size_m
        )
        checks = {
            "min_area_px": metrics["area_px"] >= float(p.min_area_px),
            "min_length_m": metrics["length_m"] >= float(p.min_length_m),
            "max_width_m": metrics["mean_width_m"] <= float(p.max_width_m),
            "min_skeleton_ratio": metrics["skeleton_ratio"]
            >= float(p.min_skeleton_ratio),
            "min_pca_ratio": metrics["pca_ratio"] >= float(p.min_pca_ratio),
            "max_circularity": metrics["circularity"] <= float(p.max_circularity),
            "min_mean_score": metrics["mean_score"] >= float(p.min_mean_score),
            "max_road_overlap": metrics["road_overlap"] <= float(p.max_road_overlap),
        }
        passed_checks = float(sum(1 for ok in checks.values() if ok))
        acceptance_score = passed_checks / total_checks
        accepted = all(checks.values())
        if accepted:
            accepted_outside_mask |= comp
        else:
            rejected_mask |= comp
        records.append(
            {
                "component_id": int(component_id),
                "accepted": bool(accepted),
                "acceptance_score": float(acceptance_score),
                "reject_reasons": [k for k, ok in checks.items() if not ok],
                "zone": "outside_evaluated",
                **metrics,
            }
        )

    accepted_mask = np.logical_or(accepted_inside_mask, accepted_outside_mask)
    evaluated_outside_mask = candidate_outside
    logger.info(
        "novel proposals: candidates=%s inside_auto=%s outside_eval=%s accepted=%s rejected=%s preset=%s",
        int(component_id),
        int(inside_count),
        int(outside_count),
        int(sum(1 for r in records if r["accepted"])),
        int(sum(1 for r in records if not r["accepted"])),
        str(getattr(p, "heuristic_preset", "unknown")),
    )
    return {
        "candidate_mask": candidate_mask,
        "candidate_inside_mask": candidate_inside,
        "candidate_outside_mask": candidate_outside,
        "accepted_inside_mask": accepted_inside_mask,
        "accepted_outside_mask": accepted_outside_mask,
        "evaluated_outside_mask": evaluated_outside_mask,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
        "records": records,
    }


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
    time_budget: dict | None = None,
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
        time_budget (dict | None): Optional time-budget status payload.

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
        "feature_spec_hash": hybrid_feature_spec_hash(),
        "hybrid_features_enabled": bool(cfg.model.hybrid_features.enabled),
    }
    if best_fold is not None:
        payload["selected_fold"] = {
            "fold_index": int(best_fold["fold_index"]),
            "val_tile": best_fold["val_tile"],
            "val_champion_shadow_iou": float(best_fold["val_champion_shadow_iou"]),
        }
    if time_budget is not None:
        payload["time_budget"] = time_budget
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
    plot_dir: str,
    context_radius: int,
    plot_with_metrics: bool = True,
) -> dict[str, object]:
    """Run inference on a holdout tile using tuned settings.

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
    xgb_feature_stats = tuned.get("xgb_feature_stats")

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

    knn_enabled = bool(tuned.get("knn_enabled", cfg.search.knn.enabled))
    xgb_enabled = bool(tuned.get("xgb_enabled", cfg.search.xgb.enabled))
    crf_enabled = bool(tuned.get("crf_enabled", cfg.search.crf.enabled))
    if not (knn_enabled or xgb_enabled):
        raise ValueError("both kNN and XGB are disabled for inference")

    score_knn_raw = None
    score_knn = None
    score_xgb = None
    mask_knn = np.zeros_like(sh_buffer_mask, dtype=bool)
    mask_xgb = np.zeros_like(sh_buffer_mask, dtype=bool)
    knn_thr = None
    xgb_thr = None
    metrics_knn = None
    metrics_xgb = None

    if knn_enabled:
        k = int(tuned["best_raw_config"]["k"])
        knn_thr = float(tuned["best_raw_config"]["threshold"])
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
            use_fp16_matmul=cfg.search.knn.use_fp16_knn,
            context_radius=context_radius,
        )
        score_knn_raw = score_knn
        score_knn = _apply_roads_penalty(score_knn, roads_mask, roads_penalty)
        mask_knn = (score_knn >= knn_thr) & sh_buffer_mask
        mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0
        metrics_knn = compute_metrics(mask_knn, gt_mask_eval)

    bst = tuned.get("bst")
    if xgb_enabled:
        if bst is None:
            raise ValueError("XGB enabled but no trained booster is available")
        xgb_thr = float(tuned["best_xgb_config"]["threshold"])
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
            xgb_feature_stats=xgb_feature_stats,
        )
        score_xgb = _apply_roads_penalty(score_xgb, roads_mask, roads_penalty)
        mask_xgb = (score_xgb >= xgb_thr) & sh_buffer_mask
        mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
        metrics_xgb = compute_metrics(mask_xgb, gt_mask_eval)

    champion_source = str(tuned.get("champion_source", "xgb"))
    if knn_enabled and xgb_enabled:
        if champion_source not in {"raw", "xgb"}:
            champion_source = "xgb"
    elif xgb_enabled:
        champion_source = "xgb"
    else:
        champion_source = "raw"

    if champion_source == "raw":
        active_stream = "knn"
        active_label = "kNN"
        champion_score = score_knn
        champion_thr = float(knn_thr)
        active_raw_mask = mask_knn
    else:
        active_stream = "xgb"
        active_label = "XGB"
        champion_score = score_xgb
        champion_thr = float(xgb_thr)
        active_raw_mask = mask_xgb

    crf_cfg = dict(tuned.get("best_crf_config") or {})
    crf_cfg_enabled = bool(crf_cfg.get("enabled", True)) and crf_enabled
    mask_crf_knn = mask_knn
    mask_crf_xgb = mask_xgb
    if crf_cfg_enabled:
        if knn_enabled and score_knn is not None and knn_thr is not None:
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
        if xgb_enabled and score_xgb is not None and xgb_thr is not None:
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

    best_crf_mask = mask_crf_knn if champion_source == "raw" else mask_crf_xgb
    active_crf_mask = best_crf_mask

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
    shadow_mask_knn = (
        _apply_shadow_filter(
            img_b,
            mask_crf_knn,
            shadow_cfg["weights"],
            shadow_cfg["threshold"],
            score_knn,
            protect_score,
        )
        if knn_enabled and score_knn is not None
        else np.zeros_like(sh_buffer_mask, dtype=bool)
    )
    shadow_mask_xgb = (
        _apply_shadow_filter(
            img_b,
            mask_crf_xgb,
            shadow_cfg["weights"],
            shadow_cfg["threshold"],
            score_xgb,
            protect_score,
        )
        if xgb_enabled and score_xgb is not None
        else np.zeros_like(sh_buffer_mask, dtype=bool)
    )
    active_shadow_mask = shadow_mask

    metrics_champion_raw = compute_metrics(active_raw_mask, gt_mask_eval)
    metrics_champion_crf = compute_metrics(active_crf_mask, gt_mask_eval)
    shadow_metrics = compute_metrics(active_shadow_mask, gt_mask_eval)
    metrics_knn_crf = (
        compute_metrics(mask_crf_knn, gt_mask_eval) if knn_enabled else None
    )
    metrics_knn_shadow = (
        compute_metrics(shadow_mask_knn, gt_mask_eval) if knn_enabled else None
    )
    metrics_xgb_crf = (
        compute_metrics(mask_crf_xgb, gt_mask_eval) if xgb_enabled else None
    )
    metrics_xgb_shadow = (
        compute_metrics(shadow_mask_xgb, gt_mask_eval) if xgb_enabled else None
    )

    proposal_cfg = cfg.postprocess.novel_proposals
    if proposal_cfg.source == "champion_score":
        proposal_threshold = (
            float(proposal_cfg.score_threshold)
            if proposal_cfg.score_threshold is not None
            else champion_thr
        )
        proposal_source_mask = champion_score >= proposal_threshold
    else:
        proposal_source_mask = active_shadow_mask
    proposal_bundle = filter_novel_proposals(
        active_shadow_mask,
        labels_sh,
        champion_score,
        roads_mask,
        pixel_size_m,
        sh_buffer_mask=sh_buffer_mask,
        proposal_source_mask=proposal_source_mask,
    )
    proposal_candidate_mask = proposal_bundle["candidate_mask"]
    proposal_candidate_inside_mask = proposal_bundle.get(
        "candidate_inside_mask", np.zeros_like(proposal_candidate_mask, dtype=bool)
    )
    proposal_evaluated_outside_mask = proposal_bundle.get(
        "evaluated_outside_mask", np.zeros_like(proposal_candidate_mask, dtype=bool)
    )
    proposal_accepted_inside_mask = proposal_bundle.get(
        "accepted_inside_mask", np.zeros_like(proposal_candidate_mask, dtype=bool)
    )
    proposal_accepted_outside_mask = proposal_bundle.get(
        "accepted_outside_mask", np.zeros_like(proposal_candidate_mask, dtype=bool)
    )
    proposal_accepted_mask = proposal_bundle["accepted_mask"]
    proposal_rejected_mask = proposal_bundle["rejected_mask"]

    if cfg.postprocess.novel_proposals.enabled:
        proposal_dir = os.path.join(shape_dir, "proposals")
        os.makedirs(proposal_dir, exist_ok=True)
        if proposal_accepted_mask.any():
            export_mask_to_shapefile(
                proposal_accepted_mask,
                holdout_path,
                os.path.join(proposal_dir, f"{image_id_b}_accepted.shp"),
            )
        if proposal_rejected_mask.any():
            export_mask_to_shapefile(
                proposal_rejected_mask,
                holdout_path,
                os.path.join(proposal_dir, f"{image_id_b}_rejected.shp"),
            )
        records_path = os.path.join(proposal_dir, f"{image_id_b}_proposal_records.csv")
        records = list(proposal_bundle.get("records", []))
        with open(records_path, "w", newline="", encoding="utf-8") as fh:
            fieldnames = [
                "component_id",
                "zone",
                "accepted",
                "acceptance_score",
                "reject_reasons",
                "area_px",
                "length_m",
                "mean_width_m",
                "skeleton_ratio",
                "pca_ratio",
                "circularity",
                "mean_score",
                "road_overlap",
                "centroid_row",
                "centroid_col",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(
                    {
                        "component_id": rec.get("component_id"),
                        "zone": rec.get("zone"),
                        "accepted": rec.get("accepted"),
                        "acceptance_score": rec.get("acceptance_score"),
                        "reject_reasons": ";".join(rec.get("reject_reasons", [])),
                        "area_px": rec.get("area_px"),
                        "length_m": rec.get("length_m"),
                        "mean_width_m": rec.get("mean_width_m"),
                        "skeleton_ratio": rec.get("skeleton_ratio"),
                        "pca_ratio": rec.get("pca_ratio"),
                        "circularity": rec.get("circularity"),
                        "mean_score": rec.get("mean_score"),
                        "road_overlap": rec.get("road_overlap"),
                        "centroid_row": rec.get("centroid_row"),
                        "centroid_col": rec.get("centroid_col"),
                    }
                )

    if active_stream == "xgb":
        champion_prob = np.clip(champion_score.astype(np.float32), 1e-6, 1.0 - 1e-6)
    else:
        s = champion_score.astype(np.float32)
        smin = float(np.min(s))
        smax = float(np.max(s))
        champion_prob = (s - smin) / (smax - smin + 1e-8)
        champion_prob = np.clip(champion_prob, 1e-6, 1.0 - 1e-6)
    disagreement_map = None
    if score_xgb is not None and score_knn is not None:
        disagreement_map = np.abs(score_xgb - score_knn).astype(np.float32)
    entropy_map = (
        -champion_prob * np.log(champion_prob)
        - (1.0 - champion_prob) * np.log(1.0 - champion_prob)
    ).astype(np.float32)

    metrics_map: dict[str, dict[str, float]] = {
        "champion_raw": metrics_champion_raw,
        "champion_crf": metrics_champion_crf,
        "champion_shadow": shadow_metrics,
    }
    if knn_enabled:
        metrics_map["knn_raw"] = metrics_knn
        metrics_map["knn_crf"] = metrics_knn_crf
        metrics_map["knn_shadow"] = metrics_knn_shadow
    if xgb_enabled:
        metrics_map["xgb_raw"] = metrics_xgb
        metrics_map["xgb_crf"] = metrics_xgb_crf
        metrics_map["xgb_shadow"] = metrics_xgb_shadow
    metrics_map = {
        key: {**metrics, "_weight": gt_weight} for key, metrics in metrics_map.items()
    }

    plot_masks = {
        f"{active_stream}_raw": active_raw_mask,
        f"{active_stream}_crf": active_crf_mask,
        f"{active_stream}_shadow": active_shadow_mask,
    }
    plot_score_maps = {active_stream: champion_score}

    save_unified_plot(
        img_b=img_b,
        gt_mask=gt_mask_eval,
        labels_sh=labels_sh,
        masks=plot_masks,
        metrics=metrics_map,
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        show_metrics=plot_with_metrics and gt_available,
        gt_available=gt_available,
        similarity_map=score_knn_raw if active_stream == "knn" else None,
        score_maps=plot_score_maps,
        proposal_masks={
            "candidate": proposal_candidate_mask,
            "accepted": proposal_accepted_mask,
            "rejected": proposal_rejected_mask,
        },
    )
    save_core_qualitative_plot(
        img_b=img_b,
        gt_mask=gt_mask_eval,
        pred_mask=active_shadow_mask,
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        gt_available=gt_available,
        model_label=active_label,
        boundary_band_px=int(cfg.runtime.plotting.boundary_band_px),
    )
    save_score_threshold_plot(
        score_map=champion_score,
        threshold=champion_thr,
        sh_buffer_mask=sh_buffer_mask,
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        model_label=active_label,
    )
    save_disagreement_entropy_plot(
        disagreement_map=disagreement_map,
        entropy_map=entropy_map,
        candidate_mask=proposal_candidate_mask,
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        model_label=active_label,
    )
    save_proposal_overlay_plot(
        img_b=img_b,
        prediction_mask=active_shadow_mask,
        candidate_mask=proposal_candidate_mask,
        candidate_inside_mask=proposal_candidate_inside_mask,
        evaluated_outside_mask=proposal_evaluated_outside_mask,
        accepted_mask=proposal_accepted_mask,
        accepted_inside_mask=proposal_accepted_inside_mask,
        accepted_outside_mask=proposal_accepted_outside_mask,
        rejected_mask=proposal_rejected_mask,
        proposal_records=proposal_bundle.get("records"),
        plot_dir=plot_dir,
        image_id_b=image_id_b,
        model_label=active_label,
        accept_rgb=tuple(cfg.runtime.plotting.proposal_accept_rgb),
        reject_rgb=tuple(cfg.runtime.plotting.proposal_reject_rgb),
        candidate_rgb=tuple(cfg.runtime.plotting.proposal_candidate_rgb),
    )

    return_masks = {
        f"{active_stream}_raw": active_raw_mask,
        f"{active_stream}_crf": active_crf_mask,
        f"{active_stream}_shadow": active_shadow_mask,
    }
    return {
        "ref_path": holdout_path,
        "image_id": image_id_b,
        "active_stream": active_stream,
        "gt_available": gt_available,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
        "metrics": metrics_map,
        "masks": return_masks,
        "proposals": proposal_bundle,
    }
