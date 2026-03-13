"""Road-mask caching and score-penalty helpers."""

from __future__ import annotations

import hashlib
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

from ...core.config_loader import cfg
from ...core.timing_utils import perf_span, time_end, time_start

logger = logging.getLogger(__name__)

_ROADS_MASK_CACHE: dict[tuple[str, int, tuple[int, int] | None], np.ndarray] = {}
_ROADS_INDEX_CACHE: dict[tuple[str, str], tuple[STRtree | None, list]] = {}


def _roads_mask_disk_cache_path(
    tile_path: str,
    downsample_factor: int,
    out_shape: tuple[int, int],
    target_shape: tuple[int, int] | None,
) -> str:
    """Return deterministic disk-cache path for a rasterized roads mask.

    Examples:
        >>> _roads_mask_disk_cache_path("/tmp/tile.tif", 1, (4, 4), None).endswith(".npy")
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

    Examples:
        >>> isinstance(_get_roads_index(None), tuple)
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

    Examples:
        >>> callable(_get_roads_mask)
        True
    """
    key = (tile_path, downsample_factor, tuple(target_shape) if target_shape else None)
    if key in _ROADS_MASK_CACHE:
        with perf_span(
            "roads_mask",
            substage="memory_cache_hit",
            extra={"tile_path": tile_path, "target_shape": list(target_shape or ())},
        ):
            _ = 0
        return _ROADS_MASK_CACHE[key]

    with perf_span(
        "roads_mask",
        substage="open_tile_metadata",
        extra={"tile_path": tile_path, "downsample_factor": downsample_factor},
    ):
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
    with perf_span(
        "roads_mask",
        substage="disk_cache_lookup",
        extra={
            "tile_path": tile_path,
            "disk_cache_path": disk_cache_path,
            "target_shape": list(target_shape or ()),
        },
    ):
        disk_cache_exists = os.path.exists(disk_cache_path)
    if disk_cache_exists:
        try:
            with perf_span(
                "roads_mask",
                substage="disk_cache_load",
                extra={
                    "tile_path": tile_path,
                    "disk_cache_path": disk_cache_path,
                    "target_shape": list(target_shape or ()),
                },
            ):
                disk_mask = np.load(disk_cache_path).astype(bool)
                if target_shape is not None and disk_mask.shape != target_shape:
                    with perf_span(
                        "roads_mask",
                        substage="resize_to_target",
                        extra={
                            "tile_path": tile_path,
                            "source_shape": list(disk_mask.shape),
                            "target_shape": list(target_shape),
                            "roads_cache_hit": True,
                        },
                    ):
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

    with perf_span(
        "roads_mask",
        substage="index_lookup",
        extra={"tile_path": tile_path},
    ):
        tree, geoms = _get_roads_index(tile_crs)
    if tree is None or not geoms:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        try:
            with perf_span(
                "roads_mask",
                substage="disk_cache_write",
                extra={
                    "tile_path": tile_path,
                    "disk_cache_path": disk_cache_path,
                    "candidate_geometry_count": 0,
                    "intersecting_geometry_count": 0,
                },
            ):
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
    with perf_span(
        "roads_mask",
        substage="tree_query",
        extra={
            "tile_path": tile_path,
            "tile_width_m": float(tile_bounds.right - tile_bounds.left),
            "tile_height_m": float(tile_bounds.top - tile_bounds.bottom),
        },
    ):
        hits = tree.query(tile_box)
    if len(hits) == 0:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        try:
            with perf_span(
                "roads_mask",
                substage="disk_cache_write",
                extra={
                    "tile_path": tile_path,
                    "disk_cache_path": disk_cache_path,
                    "candidate_geometry_count": 0,
                    "intersecting_geometry_count": 0,
                },
            ):
                np.save(disk_cache_path, mask_empty.astype(np.uint8))
        except Exception:
            pass
        return mask_empty

    if isinstance(hits[0], (int, np.integer)):
        candidates = [geoms[int(idx)] for idx in hits]
    else:
        candidates = list(hits)
    with perf_span(
        "roads_mask",
        substage="candidate_filter",
        extra={
            "tile_path": tile_path,
            "candidate_geometry_count": int(len(candidates)),
        },
    ):
        shapes = [mapping(g) for g in candidates if g.intersects(tile_box)]
    if not shapes:
        mask_empty = np.zeros(out_shape, dtype=bool)
        _ROADS_MASK_CACHE[key] = mask_empty
        try:
            with perf_span(
                "roads_mask",
                substage="disk_cache_write",
                extra={
                    "tile_path": tile_path,
                    "disk_cache_path": disk_cache_path,
                    "candidate_geometry_count": int(len(candidates)),
                    "intersecting_geometry_count": 0,
                },
            ):
                np.save(disk_cache_path, mask_empty.astype(np.uint8))
        except Exception:
            pass
        return mask_empty

    t0 = time_start()
    with perf_span(
        "roads_mask",
        substage="rasterize",
        extra={
            "tile_path": tile_path,
            "candidate_geometry_count": int(len(candidates)),
            "intersecting_geometry_count": int(len(shapes)),
            "out_shape": list(out_shape),
        },
    ):
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
        with perf_span(
            "roads_mask",
            substage="resize_to_target",
            extra={
                "tile_path": tile_path,
                "source_shape": list(mask.shape),
                "target_shape": list(target_shape),
                "roads_cache_hit": False,
            },
        ):
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
        with perf_span(
            "roads_mask",
            substage="disk_cache_write",
            extra={
                "tile_path": tile_path,
                "disk_cache_path": disk_cache_path,
                "candidate_geometry_count": int(len(candidates)),
                "intersecting_geometry_count": int(len(shapes)),
            },
        ):
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

    Examples:
        >>> float(_apply_roads_penalty(np.ones((1, 1), dtype=np.float32), np.array([[True]]), 0.5)[0, 0])
        0.5
    """
    if roads_mask is None or penalty >= 1.0:
        return score_map
    if roads_mask.shape != score_map.shape:
        raise ValueError("roads_mask must match score_map shape")
    penalty_map = np.where(roads_mask, penalty, 1.0).astype(score_map.dtype)
    return score_map * penalty_map
