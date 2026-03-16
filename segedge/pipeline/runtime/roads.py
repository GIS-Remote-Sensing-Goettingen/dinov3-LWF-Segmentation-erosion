"""Road-mask caching and score-penalty helpers.

This module keeps the roads workflow readable by separating the three main steps:
- resolve tile/grid metadata and cache paths
- reuse cached masks when possible
- rasterize and persist a tile-local roads mask when caches miss
"""

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
from ...core.timing_utils import (
    perf_call,
    perf_metadata,
    perf_span,
    time_end,
    time_start,
)

logger = logging.getLogger(__name__)

_ROADS_MASK_CACHE: dict[tuple[str, int, tuple[int, int] | None], np.ndarray] = {}
_ROADS_INDEX_CACHE: dict[tuple[str, str], tuple[STRtree | None, list]] = {}


def _roads_cache_extra(
    tile_path: str,
    *,
    disk_cache_path: str | None = None,
    target_shape: tuple[int, int] | None = None,
    candidate_geometry_count: int | None = None,
    intersecting_geometry_count: int | None = None,
    roads_cache_hit: bool | None = None,
    source_shape: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Build a consistent `roads_mask` extra payload for perf records.

    Examples:
        >>> _roads_cache_extra('tile.tif', candidate_geometry_count=1)['tile_path']
        'tile.tif'
    """
    extra: dict[str, object] = {"tile_path": tile_path}
    if disk_cache_path is not None:
        extra["disk_cache_path"] = disk_cache_path
    if target_shape is not None:
        extra["target_shape"] = list(target_shape)
    if candidate_geometry_count is not None:
        extra["candidate_geometry_count"] = int(candidate_geometry_count)
    if intersecting_geometry_count is not None:
        extra["intersecting_geometry_count"] = int(intersecting_geometry_count)
    if roads_cache_hit is not None:
        extra["roads_cache_hit"] = bool(roads_cache_hit)
    if source_shape is not None:
        extra["source_shape"] = list(source_shape)
    return extra


def _open_tile_metadata(
    tile_path: str,
    downsample_factor: int,
) -> tuple[tuple[int, int], object, object, object]:
    """Return the tile raster metadata needed for roads rasterization.

    Examples:
        >>> callable(_open_tile_metadata)
        True
    """
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
        return out_shape, transform, tile_src.bounds, tile_src.crs


def _resize_roads_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a roads mask to the requested output grid.

    Examples:
        >>> _resize_roads_mask(np.ones((2, 2), dtype=np.uint8), (2, 2)).shape
        (2, 2)
    """
    return resize(
        mask,
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )


def _write_roads_disk_cache(
    disk_cache_path: str,
    mask: np.ndarray,
) -> None:
    """Persist a roads mask to the disk cache.

    Examples:
        >>> callable(_write_roads_disk_cache)
        True
    """
    np.save(disk_cache_path, mask.astype(np.uint8))


def _empty_roads_mask(out_shape: tuple[int, int]) -> np.ndarray:
    """Return an empty roads mask with the requested shape.

    Examples:
        >>> _empty_roads_mask((1, 2)).shape
        (1, 2)
    """
    return np.zeros(out_shape, dtype=bool)


def _load_disk_cached_roads_mask(
    *,
    disk_cache_path: str,
    tile_path: str,
    target_shape: tuple[int, int] | None,
) -> np.ndarray | None:
    """Load and optionally resize a cached roads mask from disk.

    Examples:
        >>> _load_disk_cached_roads_mask(
        ...     disk_cache_path="/tmp/missing.npy",
        ...     tile_path="tile.tif",
        ...     target_shape=None,
        ... ) is None
        True
    """
    if not os.path.exists(disk_cache_path):
        return None
    try:
        disk_mask = perf_call(
            np.load,
            disk_cache_path,
            stage="roads_mask",
            substage="disk_cache_load",
            extra=_roads_cache_extra(
                tile_path,
                disk_cache_path=disk_cache_path,
                target_shape=target_shape,
            ),
        ).astype(bool)
        if target_shape is not None and disk_mask.shape != target_shape:
            disk_mask = perf_call(
                _resize_roads_mask,
                disk_mask.astype("uint8"),
                target_shape,
                stage="roads_mask",
                substage="resize_to_target",
                extra=_roads_cache_extra(
                    tile_path,
                    target_shape=target_shape,
                    source_shape=disk_mask.shape,
                    roads_cache_hit=True,
                ),
            ).astype(bool)
        logger.info("roads mask disk cache hit: %s", os.path.basename(disk_cache_path))
        return disk_mask
    except Exception:
        return None


def _persist_roads_mask(
    *,
    key: tuple[str, int, tuple[int, int] | None],
    disk_cache_path: str,
    mask: np.ndarray,
    tile_path: str,
    candidate_geometry_count: int,
    intersecting_geometry_count: int,
) -> np.ndarray:
    """Store a roads mask in memory and best-effort disk cache.

    Examples:
        >>> mask = _persist_roads_mask(
        ...     key=("tile", 1, None),
        ...     disk_cache_path="/tmp/demo.npy",
        ...     mask=np.zeros((1, 1), dtype=bool),
        ...     tile_path="tile",
        ...     candidate_geometry_count=0,
        ...     intersecting_geometry_count=0,
        ... )
        >>> mask.shape
        (1, 1)
    """
    _ROADS_MASK_CACHE[key] = mask
    try:
        perf_call(
            _write_roads_disk_cache,
            disk_cache_path,
            mask,
            stage="roads_mask",
            substage="disk_cache_write",
            extra=_roads_cache_extra(
                tile_path,
                disk_cache_path=disk_cache_path,
                candidate_geometry_count=candidate_geometry_count,
                intersecting_geometry_count=intersecting_geometry_count,
            ),
        )
    except Exception:
        logger.debug("roads mask disk cache write failed: %s", disk_cache_path)
    return mask


def _candidate_geometries(
    tree, geoms: list, tile_box, tile_bounds, tile_path: str
) -> list:
    """Return the road geometries whose bounds intersect the tile.

    Examples:
        >>> callable(_candidate_geometries)
        True
    """
    hits = perf_call(
        tree.query,
        tile_box,
        stage="roads_mask",
        substage="tree_query",
        extra={
            "tile_path": tile_path,
            "tile_width_m": float(tile_bounds.right - tile_bounds.left),
            "tile_height_m": float(tile_bounds.top - tile_bounds.bottom),
        },
    )
    if len(hits) == 0:
        return []
    if isinstance(hits[0], (int, np.integer)):
        return [geoms[int(idx)] for idx in hits]
    return list(hits)


def _clip_candidate_shapes(
    candidates: list,
    tile_box,
    tile_path: str,
) -> list[dict[str, object]]:
    """Clip intersecting road geometries to the tile bounds before rasterization.

    Examples:
        >>> callable(_clip_candidate_shapes)
        True
    """
    with perf_span(
        "roads_mask",
        substage="candidate_filter",
        extra=_roads_cache_extra(
            tile_path,
            candidate_geometry_count=len(candidates),
        ),
    ):
        shapes = []
        for geom in candidates:
            if not geom.intersects(tile_box):
                continue
            clipped_geom = geom.intersection(tile_box)
            if clipped_geom.is_empty:
                continue
            # Clip large geometries to the tile before rasterization so one
            # pathological feature does not force full-geometry burn cost.
            shapes.append(mapping(clipped_geom))
    return shapes


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
        perf_metadata(
            "roads_mask",
            substage="memory_cache_hit",
            extra=_roads_cache_extra(tile_path, target_shape=target_shape),
        )
        return _ROADS_MASK_CACHE[key]

    # Stage 1: resolve tile-local metadata and cache paths.
    out_shape, transform, tile_bounds, tile_crs = perf_call(
        _open_tile_metadata,
        tile_path,
        downsample_factor,
        stage="roads_mask",
        substage="open_tile_metadata",
        extra={"tile_path": tile_path, "downsample_factor": downsample_factor},
    )

    disk_cache_path = _roads_mask_disk_cache_path(
        tile_path,
        downsample_factor,
        out_shape,
        target_shape,
    )
    disk_mask = perf_call(
        _load_disk_cached_roads_mask,
        disk_cache_path=disk_cache_path,
        tile_path=tile_path,
        target_shape=target_shape,
        stage="roads_mask",
        substage="disk_cache_lookup",
        extra=_roads_cache_extra(
            tile_path,
            disk_cache_path=disk_cache_path,
            target_shape=target_shape,
        ),
    )
    if disk_mask is not None:
        _ROADS_MASK_CACHE[key] = disk_mask
        return disk_mask

    # Stage 2: query spatial candidates and short-circuit empty tiles early.
    tree, geoms = perf_call(
        _get_roads_index,
        tile_crs,
        stage="roads_mask",
        substage="index_lookup",
        extra={"tile_path": tile_path},
    )
    if tree is None or not geoms:
        return _persist_roads_mask(
            key=key,
            disk_cache_path=disk_cache_path,
            mask=_empty_roads_mask(out_shape),
            tile_path=tile_path,
            candidate_geometry_count=0,
            intersecting_geometry_count=0,
        )

    tile_box = box(
        tile_bounds.left,
        tile_bounds.bottom,
        tile_bounds.right,
        tile_bounds.top,
    )
    candidates = _candidate_geometries(tree, geoms, tile_box, tile_bounds, tile_path)
    if not candidates:
        return _persist_roads_mask(
            key=key,
            disk_cache_path=disk_cache_path,
            mask=_empty_roads_mask(out_shape),
            tile_path=tile_path,
            candidate_geometry_count=0,
            intersecting_geometry_count=0,
        )

    shapes = _clip_candidate_shapes(candidates, tile_box, tile_path)
    if not shapes:
        return _persist_roads_mask(
            key=key,
            disk_cache_path=disk_cache_path,
            mask=_empty_roads_mask(out_shape),
            tile_path=tile_path,
            candidate_geometry_count=len(candidates),
            intersecting_geometry_count=0,
        )

    # Stage 3: rasterize the tile-local geometries and persist the result.
    t0 = time_start()
    with perf_span(
        "roads_mask",
        substage="rasterize",
        extra=_roads_cache_extra(
            tile_path,
            candidate_geometry_count=len(candidates),
            intersecting_geometry_count=len(shapes),
        )
        | {"out_shape": list(out_shape)},
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
        mask = perf_call(
            _resize_roads_mask,
            mask,
            target_shape,
            stage="roads_mask",
            substage="resize_to_target",
            extra=_roads_cache_extra(
                tile_path,
                target_shape=target_shape,
                source_shape=mask.shape,
                roads_cache_hit=False,
            ),
        ).astype("uint8")
    mask_bool = mask.astype(bool)
    time_end("roads_mask_rasterize", t0)
    logger.info(
        "roads mask rasterized: shapes=%s coverage=%.4f",
        len(shapes),
        float(mask_bool.mean()),
    )
    return _persist_roads_mask(
        key=key,
        disk_cache_path=disk_cache_path,
        mask=mask_bool,
        tile_path=tile_path,
        candidate_geometry_count=len(candidates),
        intersecting_geometry_count=len(shapes),
    )


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
