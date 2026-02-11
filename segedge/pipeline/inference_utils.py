from __future__ import annotations

import logging
import os

import fiona
import numpy as np
import rasterio
import rasterio.features as rfeatures
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from skimage.transform import resize

import config as cfg

from ..core.io_utils import (
    build_sh_buffer_mask,
    load_dop20_image,
    rasterize_vector_labels,
    reproject_labels_to_image,
)
from ..core.timing_utils import time_end, time_start

logger = logging.getLogger(__name__)

_ROADS_MASK_CACHE: dict[tuple[str, int], np.ndarray] = {}
_ROADS_INDEX_CACHE: dict[tuple[str, str], tuple[STRtree | None, list]] = {}


def _get_roads_index(tile_crs) -> tuple[STRtree | None, list]:
    """Load and cache road geometries in a spatial index.

    Args:
        tile_crs: Target CRS for geometries.

    Returns:
        tuple[STRtree | None, list]: Index and geometry list.

    Examples:
        >>> isinstance(_get_roads_index.__name__, str)
        True
    """
    roads_path = getattr(cfg, "ROADS_MASK_PATH", None)
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
        >>> isinstance(_get_roads_mask.__name__, str)
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
    simplify_tol = float(getattr(cfg, "ROADS_SIMPLIFY_TOLERANCE_M", 0.2))
    shapes = []
    for geom in candidates:
        if not geom.intersects(tile_box):
            continue
        clipped = geom.intersection(tile_box)
        if clipped.is_empty:
            continue
        if simplify_tol > 0:
            clipped = clipped.simplify(simplify_tol, preserve_topology=True)
            if clipped.is_empty:
                continue
        shapes.append(mapping(clipped))
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


def _compute_top_p(
    buffer_density: float,
    a: float,
    b: float,
    p_min: float,
    p_max: float,
) -> float:
    """Compute adaptive top-p inside the buffer.

    Args:
        buffer_density (float): Buffer fraction in [0, 1].
        a (float): Linear coefficient.
        b (float): Linear bias.
        p_min (float): Minimum p.
        p_max (float): Maximum p.

    Returns:
        float: Clipped p in [p_min, p_max].

    Examples:
        >>> _compute_top_p(0.2, 0.5, 0.0, 0.02, 0.1)
        0.1
    """
    p = a * buffer_density + b
    return float(np.clip(p, p_min, p_max))


def _top_p_threshold(
    score_map: np.ndarray, buffer_mask: np.ndarray, p: float
) -> tuple[float, np.ndarray]:
    """Compute top-p threshold and mask inside a buffer.

    Args:
        score_map (np.ndarray): Score map.
        buffer_mask (np.ndarray): Candidate region mask.
        p (float): Fraction in [0, 1].

    Returns:
        tuple[float, np.ndarray]: Threshold and selected mask.

    Examples:
        >>> import numpy as np
        >>> s = np.array([[0.1, 0.9, 0.2]])
        >>> m = np.array([[True, True, True]])
        >>> thr, sel = _top_p_threshold(s, m, 0.33)
        >>> sel.tolist()
        [[False, True, False]]
    """
    if p <= 0:
        return float("inf"), np.zeros_like(buffer_mask, dtype=bool)
    if p >= 1:
        return float(np.min(score_map)), buffer_mask.astype(bool)
    vals = score_map[buffer_mask]
    if vals.size == 0:
        return float("inf"), np.zeros_like(buffer_mask, dtype=bool)
    thr = float(np.quantile(vals, 1.0 - p))
    mask = (score_map >= thr) & buffer_mask
    return thr, mask


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
        >>> isinstance(load_b_tile_context.__name__, str)
        True
    """
    logger.info("loading tile: %s", img_path)
    debug_reproject = bool(getattr(cfg, "DEBUG_REPROJECT", False))
    t0_data = time_start()
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    img_b = load_dop20_image(img_path, downsample_factor=ds)
    if debug_reproject:
        logger.info("tile image: shape=%s dtype=%s", img_b.shape, img_b.dtype)
    labels_sh = reproject_labels_to_image(
        img_path, cfg.SOURCE_LABEL_RASTER, downsample_factor=ds
    )
    gt_mask = (
        rasterize_vector_labels(gt_vector_paths, img_path, downsample_factor=ds)
        if gt_vector_paths
        else None
    )
    if debug_reproject:
        logger.info(
            "tile labels_sh: shape=%s dtype=%s",
            labels_sh.shape,
            labels_sh.dtype,
        )
        if gt_mask is not None:
            logger.info(
                "tile gt_mask: shape=%s dtype=%s",
                gt_mask.shape,
                gt_mask.dtype,
            )
    time_end("data_loading_and_reprojection", t0_data)
    target_shape = img_b.shape[:2]
    if labels_sh.shape != target_shape:
        logger.warning(
            "labels_sh shape %s != image shape %s; resizing to match",
            labels_sh.shape,
            target_shape,
        )
        if debug_reproject:
            logger.info(
                "labels_sh resize: from=%s to=%s",
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
        if debug_reproject:
            logger.info(
                "gt_mask resize: from=%s to=%s",
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
        logger.info("GT positives on B: %s", gt_mask.sum())
    logger.info("SH_2022 positives on B: %s", (labels_sh > 0).sum())
    if debug_reproject:
        label_mask = labels_sh > 0
        label_nonzero = int(label_mask.sum())
        label_mean = float(label_mask.mean())
        col_any = np.asarray(label_mask.any(axis=0))
        row_any = np.asarray(label_mask.any(axis=1))
        if col_any.any():
            col_min = int(np.argmax(col_any))
            col_max = int(col_any.size - 1 - np.argmax(col_any[::-1]))
        else:
            col_min = -1
            col_max = -1
        if row_any.any():
            row_min = int(np.argmax(row_any))
            row_max = int(row_any.size - 1 - np.argmax(row_any[::-1]))
        else:
            row_min = -1
            row_max = -1
        left_margin = col_min if col_min >= 0 else -1
        right_margin = (labels_sh.shape[1] - 1 - col_max) if col_max >= 0 else -1
        top_margin = row_min if row_min >= 0 else -1
        bottom_margin = (labels_sh.shape[0] - 1 - row_max) if row_max >= 0 else -1
        logger.info(
            "labels_sh coverage: nonzero=%s mean=%.6f cols=%s..%s rows=%s..%s "
            "margins(L/R/T/B)=%s/%s/%s/%s",
            label_nonzero,
            label_mean,
            col_min,
            col_max,
            row_min,
            row_max,
            left_margin,
            right_margin,
            top_margin,
            bottom_margin,
        )

    with rasterio.open(img_path) as src:
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
    if debug_reproject:
        logger.info(
            "sh_buffer_mask coverage: mean=%.6f nonzero=%s",
            float(sh_buffer_mask.mean()),
            int(sh_buffer_mask.sum()),
        )
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
