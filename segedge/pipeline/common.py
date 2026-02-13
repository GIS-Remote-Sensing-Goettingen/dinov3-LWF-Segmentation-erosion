"""Shared pipeline helpers for SegEdge entrypoints."""

from __future__ import annotations

import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import fiona
import numpy as np
import torch
from pyproj import Transformer
from rasterio import open as rio_open
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
from transformers import AutoImageProcessor, AutoModel

import config as cfg

from ..core.banks import build_banks_single_scale
from ..core.io_utils import (
    build_sh_buffer_mask,
    load_dop20_image,
    rasterize_vector_labels,
    reproject_labels_to_image,
)
from ..core.timing_utils import time_end, time_start

logger = logging.getLogger(__name__)

_GT_INDEX_CACHE: dict[tuple[tuple[str, ...], str], tuple[STRtree | None, list]] = {}
_GT_CRS_LOGGED: set[str] = set()
AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT = "gt_to_val_cap_holdout"
AUTO_SPLIT_MODE_LEGACY = "legacy_gt_source_val_holdout"


def _tile_has_gt(
    tile_path: str, gt_vector_paths: list[str], downsample_factor: int
) -> bool:
    """Return True if GT vectors overlap the tile.

    Args:
        tile_path (str): Tile path to test.
        gt_vector_paths (list[str]): GT vector paths.
        downsample_factor (int): Downsample factor (unused for intersection).

    Returns:
        bool: True if GT positives exist for the tile.

    Examples:
        >>> isinstance(_tile_has_gt.__name__, str)
        True
    """
    with rio_open(tile_path) as src:
        tile_crs = src.crs
        tile_bounds = src.bounds
    tile_box = box(
        tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top
    )
    tree, geoms = _get_gt_index(gt_vector_paths, tile_crs)
    if tree is None or not geoms:
        return False
    hits = tree.query(tile_box)
    if len(hits) == 0:
        return False
    first = hits[0]
    if isinstance(first, (int, np.integer)):
        return any(tile_box.intersects(geoms[int(idx)]) for idx in hits)
    return any(tile_box.intersects(geom) for geom in hits)


def tile_has_gt_overlap(
    tile_path: str,
    gt_vector_paths: list[str],
    downsample_factor: int | None = None,
) -> bool:
    """Return True when GT vectors overlap a tile footprint.

    Args:
        tile_path (str): Tile path to test.
        gt_vector_paths (list[str]): GT vector paths.
        downsample_factor (int | None): Optional downsample factor.

    Returns:
        bool: True if any GT geometry intersects the tile.

    Examples:
        >>> tile_has_gt_overlap("tile.tif", [], downsample_factor=1)
        False
    """
    if not gt_vector_paths:
        return False
    ds = int(
        downsample_factor
        if downsample_factor is not None
        else (getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    )
    return _tile_has_gt(tile_path, gt_vector_paths, ds)


def _load_gt_geometries(
    gt_vector_paths: list[str],
    target_crs,
) -> list:
    """Load GT geometries and reproject into target CRS if needed.

    Args:
        gt_vector_paths (list[str]): GT vector paths.
        target_crs (CRS | None): Target CRS for reprojection.

    Returns:
        list: List of shapely geometries.

    Examples:
        >>> isinstance(_load_gt_geometries.__name__, str)
        True
    """
    geoms = []
    target_crs_dict = target_crs.to_dict() if target_crs is not None else None
    for vp in gt_vector_paths:
        with fiona.open(vp, "r") as shp:
            vec_crs = shp.crs
            if not vec_crs:
                logger.warning(
                    "vector CRS missing for %s; assuming EPSG:4326 (WGS84)", vp
                )
                vec_crs = CRS.from_epsg(4326).to_dict()
            transformer = None
            if target_crs_dict and vec_crs != target_crs_dict:
                transformer = Transformer.from_crs(
                    vec_crs, target_crs_dict, always_xy=True
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
    return geoms


def _get_gt_index(
    gt_vector_paths: list[str],
    target_crs,
) -> tuple[STRtree | None, list]:
    """Return cached STRtree and geometries for the target CRS.

    Args:
        gt_vector_paths (list[str]): GT vector paths.
        target_crs (CRS | None): Target CRS for reprojection.

    Returns:
        tuple[STRtree | None, list]: Spatial index and geometries.

    Examples:
        >>> isinstance(_get_gt_index.__name__, str)
        True
    """
    crs_key = target_crs.to_string() if target_crs is not None else "<none>"
    cache_key = (tuple(gt_vector_paths), crs_key)
    if cache_key in _GT_INDEX_CACHE:
        return _GT_INDEX_CACHE[cache_key]
    if crs_key not in _GT_CRS_LOGGED:
        logger.info("GT presence CRS key: %s", crs_key)
        _GT_CRS_LOGGED.add(crs_key)
    geoms = _load_gt_geometries(gt_vector_paths, target_crs)
    tree = STRtree(geoms) if geoms else None
    _GT_INDEX_CACHE[cache_key] = (tree, geoms)
    return tree, geoms


def _resolve_gt_workers(num_workers: int | None) -> int:
    """Resolve worker count for GT presence checks.

    Args:
        num_workers (int | None): Requested workers.

    Returns:
        int: Resolved worker count (>=1).

    Examples:
        >>> _resolve_gt_workers(2) >= 1
        True
    """
    if num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            try:
                num_workers = int(slurm_cpus)
            except ValueError:
                num_workers = None
        if num_workers is None:
            num_workers = os.cpu_count() or 1
    return max(1, int(num_workers))


def _chunk_tiles(tile_paths: list[str], chunk_size: int) -> list[list[str]]:
    """Chunk a list of tile paths into smaller lists.

    Args:
        tile_paths (list[str]): Tile paths.
        chunk_size (int): Chunk size.

    Returns:
        list[list[str]]: Chunked tiles.

    Examples:
        >>> _chunk_tiles(["a", "b", "c"], 2)
        [['a', 'b'], ['c']]
    """
    return [
        tile_paths[i : i + chunk_size] for i in range(0, len(tile_paths), chunk_size)
    ]


def _tiles_with_gt_chunk(
    tile_paths: list[str],
    gt_vector_paths: list[str],
    downsample_factor: int,
) -> tuple[list[str], list[str]]:
    """Split a chunk of tiles into GT vs holdout lists.

    Args:
        tile_paths (list[str]): Tile paths for the chunk.
        gt_vector_paths (list[str]): GT vector paths.
        downsample_factor (int): Downsample factor for rasterization.

    Returns:
        tuple[list[str], list[str]]: GT tiles and holdout tiles.

    Examples:
        >>> isinstance(_tiles_with_gt_chunk.__name__, str)
        True
    """
    gt_tiles = []
    holdout_tiles = []
    for tile_path in tile_paths:
        if _tile_has_gt(tile_path, gt_vector_paths, downsample_factor):
            gt_tiles.append(tile_path)
        else:
            holdout_tiles.append(tile_path)
    return gt_tiles, holdout_tiles


def _cap_inference_tiles(
    holdout_tiles: list[str],
    cap_enabled: bool,
    cap: int | None,
    seed: int,
) -> list[str]:
    """Return a deterministic capped subset of holdout tiles.

    Args:
        holdout_tiles (list[str]): Holdout tile paths.
        cap_enabled (bool): Whether capping is enabled.
        cap (int | None): Maximum number of tiles when enabled.
        seed (int): Seed for deterministic sampling.

    Returns:
        list[str]: Capped holdout tile paths.

    Raises:
        ValueError: If capping is enabled and cap is not positive.

    Examples:
        >>> _cap_inference_tiles(["b", "a"], False, 1, 42)
        ['a', 'b']
        >>> _cap_inference_tiles(["a", "b", "c"], True, 2, 42)
        ['a', 'c']
    """
    tiles_sorted = sorted(holdout_tiles)
    if not cap_enabled:
        return tiles_sorted
    if cap is None or int(cap) <= 0:
        raise ValueError("INFERENCE_TILE_CAP must be > 0 when cap is enabled")
    cap_int = int(cap)
    if len(tiles_sorted) <= cap_int:
        return tiles_sorted
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tiles_sorted), size=cap_int, replace=False)
    return sorted(tiles_sorted[int(i)] for i in idx)


def _split_tiles_from_gt_presence(
    gt_tiles: list[str],
    holdout_tiles: list[str],
    mode: str,
    val_fraction: float,
    seed: int,
    cap_enabled: bool,
    cap: int | None,
    cap_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve source/val/holdout lists from GT-presence partitions.

    Args:
        gt_tiles (list[str]): Tiles overlapping GT vectors.
        holdout_tiles (list[str]): Tiles without GT overlap.
        mode (str): Auto split mode.
        val_fraction (float): Validation fraction for legacy mode.
        seed (int): Legacy split seed.
        cap_enabled (bool): Enable holdout cap for GT->val mode.
        cap (int | None): Holdout cap value.
        cap_seed (int): Holdout cap seed.

    Returns:
        tuple[list[str], list[str], list[str]]: Source, validation, holdout tiles.

    Examples:
        >>> _split_tiles_from_gt_presence(
        ...     ["g2", "g1"], ["h2", "h1"], AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT,
        ...     0.5, 42, True, 1, 11
        ... )
        ([], ['g1', 'g2'], ['h1'])
        >>> src, val, hold = _split_tiles_from_gt_presence(
        ...     ["g1", "g2", "g3"], ["h1"], AUTO_SPLIT_MODE_LEGACY,
        ...     0.34, 42, False, None, 0
        ... )
        >>> len(src) + len(val), hold
        (3, ['h1'])
    """
    gt_sorted = sorted(gt_tiles)
    holdout_sorted = sorted(holdout_tiles)
    if mode == AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT:
        val_tiles = gt_sorted
        holdout_capped = _cap_inference_tiles(
            holdout_sorted,
            cap_enabled=cap_enabled,
            cap=cap,
            seed=cap_seed,
        )
        return [], val_tiles, holdout_capped
    if mode != AUTO_SPLIT_MODE_LEGACY:
        raise ValueError(
            "AUTO_SPLIT_MODE must be one of "
            f"{AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT!r} or "
            f"{AUTO_SPLIT_MODE_LEGACY!r}"
        )
    if len(gt_sorted) == 1:
        logger.warning(
            "only one GT tile found; using it for both source and validation"
        )
        return gt_sorted, gt_sorted, holdout_sorted

    rng = np.random.default_rng(seed)
    indices = np.arange(len(gt_sorted))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(gt_sorted) * val_fraction)))
    if val_count >= len(gt_sorted):
        val_count = len(gt_sorted) - 1
    val_idx = set(indices[:val_count].tolist())
    source_tiles = [p for i, p in enumerate(gt_sorted) if i not in val_idx]
    val_tiles = [p for i, p in enumerate(gt_sorted) if i in val_idx]
    return source_tiles, val_tiles, holdout_sorted


def resolve_tile_splits_from_gt(
    tiles_dir: str,
    tile_glob: str,
    gt_vector_paths: list[str],
    val_fraction: float,
    seed: int,
    downsample_factor: int | None = None,
    num_workers: int | None = None,
    mode: str | None = None,
    inference_tile_cap_enabled: bool | None = None,
    inference_tile_cap: int | None = None,
    inference_tile_cap_seed: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve source/val/holdout tiles using GT presence.

    Behavior depends on `mode`:
    - `gt_to_val_cap_holdout`: all GT tiles become validation and non-GT tiles can
      be capped for holdout.
    - `legacy_gt_source_val_holdout`: GT tiles are split into source/validation and
      non-GT tiles are holdout.

    Args:
        tiles_dir (str): Directory containing tiles.
        tile_glob (str): Glob pattern for tile files.
        gt_vector_paths (list[str]): Ground-truth vector paths.
        val_fraction (float): Fraction of GT tiles for validation.
        seed (int): RNG seed for split.
        downsample_factor (int | None): Downsample factor for GT presence checks.
        num_workers (int | None): Worker count for GT presence checks.
        mode (str | None): Auto split mode.
        inference_tile_cap_enabled (bool | None): Holdout cap toggle.
        inference_tile_cap (int | None): Holdout cap value.
        inference_tile_cap_seed (int | None): Holdout cap seed.

    Returns:
        tuple[list[str], list[str], list[str]]: Source, validation, holdout tiles.

    Examples:
        >>> isinstance(resolve_tile_splits_from_gt.__name__, str)
        True
    """
    if not gt_vector_paths:
        raise ValueError("EVAL_GT_VECTORS must be set for auto split")
    if not os.path.isdir(tiles_dir):
        raise ValueError(f"tiles directory not found: {tiles_dir}")
    tile_paths = sorted(glob.glob(os.path.join(tiles_dir, tile_glob)))
    if not tile_paths:
        raise ValueError(f"no tiles found in {tiles_dir} with {tile_glob}")
    if downsample_factor is None:
        downsample_factor = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    if mode is None:
        mode = str(getattr(cfg, "AUTO_SPLIT_MODE", AUTO_SPLIT_MODE_LEGACY)).strip()
    if inference_tile_cap_enabled is None:
        inference_tile_cap_enabled = bool(
            getattr(cfg, "INFERENCE_TILE_CAP_ENABLED", False)
        )
    if inference_tile_cap is None:
        inference_tile_cap = getattr(cfg, "INFERENCE_TILE_CAP", None)
    if inference_tile_cap_seed is None:
        inference_tile_cap_seed = int(getattr(cfg, "INFERENCE_TILE_CAP_SEED", seed))
    debug_reproject = bool(getattr(cfg, "DEBUG_REPROJECT", False))

    raster_path = getattr(cfg, "SOURCE_LABEL_RASTER", None)
    filtered_paths = []
    excluded_by_raster = 0
    if raster_path:
        with rio_open(raster_path) as src:
            raster_crs = src.crs
            raster_bounds = src.bounds
        for tile_path in tile_paths:
            with rio_open(tile_path) as tile_src:
                tile_bounds = tile_src.bounds
                tile_crs = tile_src.crs
            if raster_crs is not None and tile_crs is not None:
                if tile_crs != raster_crs:
                    tile_bounds = transform_bounds(
                        tile_crs, raster_crs, *tile_bounds, densify_pts=21
                    )
            tb_left, tb_bottom, tb_right, tb_top = tile_bounds
            rb_left, rb_bottom, rb_right, rb_top = raster_bounds
            intersects = not (
                tb_right <= rb_left
                or tb_left >= rb_right
                or tb_top <= rb_bottom
                or tb_bottom >= rb_top
            )
            if debug_reproject:
                inter_left = max(tb_left, rb_left)
                inter_right = min(tb_right, rb_right)
                inter_bottom = max(tb_bottom, rb_bottom)
                inter_top = min(tb_top, rb_top)
                inter_w = max(0.0, inter_right - inter_left)
                inter_h = max(0.0, inter_top - inter_bottom)
                tile_area = max(0.0, (tb_right - tb_left) * (tb_top - tb_bottom))
                coverage_ratio = (
                    (inter_w * inter_h / tile_area) if tile_area > 0 else 0.0
                )
                left_gap = max(0.0, rb_left - tb_left)
                right_gap = max(0.0, tb_right - rb_right)
                top_gap = max(0.0, tb_top - rb_top)
                bottom_gap = max(0.0, rb_bottom - tb_bottom)
                logger.info(
                    "auto split: label coverage tile=%s bounds=%s "
                    "raster_bounds=%s ratio=%.4f gaps(L/R/T/B)="
                    "(%.3f, %.3f, %.3f, %.3f) intersects=%s",
                    tile_path,
                    tuple(tile_bounds),
                    tuple(raster_bounds),
                    coverage_ratio,
                    left_gap,
                    right_gap,
                    top_gap,
                    bottom_gap,
                    intersects,
                )
            if intersects:
                filtered_paths.append(tile_path)
            else:
                excluded_by_raster += 1
    else:
        filtered_paths = tile_paths
    if excluded_by_raster:
        logger.info(
            "auto split: excluded %s tiles with no SOURCE_LABEL_RASTER overlap",
            excluded_by_raster,
        )
    tile_paths = filtered_paths
    if not tile_paths:
        raise ValueError("no tiles overlap SOURCE_LABEL_RASTER")

    num_workers = _resolve_gt_workers(num_workers)
    logger.info(
        "auto split: scanning %s tiles for GT using %s workers",
        len(tile_paths),
        num_workers,
    )
    gt_tiles = []
    holdout_tiles = []
    if num_workers <= 1:
        for tile_path in tile_paths:
            has_gt = _tile_has_gt(tile_path, gt_vector_paths, downsample_factor)
            if has_gt:
                gt_tiles.append(tile_path)
            else:
                holdout_tiles.append(tile_path)
    else:
        chunk_count = max(1, num_workers * 4)
        chunk_size = max(1, (len(tile_paths) + chunk_count - 1) // chunk_count)
        chunks = _chunk_tiles(tile_paths, chunk_size)
        logger.info(
            "auto split: chunk_size=%s, chunks=%s",
            chunk_size,
            len(chunks),
        )
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for gt_chunk, holdout_chunk in ex.map(
                _tiles_with_gt_chunk,
                chunks,
                repeat(gt_vector_paths),
                repeat(downsample_factor),
            ):
                gt_tiles.extend(gt_chunk)
                holdout_tiles.extend(holdout_chunk)

    if not gt_tiles:
        raise ValueError("no tiles overlap GT vectors; cannot build source/val")
    holdout_total = len(holdout_tiles)
    source_tiles, val_tiles, holdout_tiles = _split_tiles_from_gt_presence(
        gt_tiles=gt_tiles,
        holdout_tiles=holdout_tiles,
        mode=mode,
        val_fraction=val_fraction,
        seed=seed,
        cap_enabled=bool(inference_tile_cap_enabled),
        cap=inference_tile_cap,
        cap_seed=int(inference_tile_cap_seed),
    )
    if mode == AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT:
        logger.info(
            "auto split mode=%s: val(gt)=%s holdout_total=%s holdout_selected=%s",
            mode,
            len(val_tiles),
            holdout_total,
            len(holdout_tiles),
        )
    return source_tiles, val_tiles, holdout_tiles


def init_model(model_name: str):
    """Load a DINO backbone + processor on CPU/GPU with timing.

    Args:
        model_name (str): HuggingFace model name.

    Returns:
        tuple: (model, processor, device).

    Examples:
        >>> isinstance(init_model.__name__, str)
        True
    """
    t0 = time_start()
    processor = AutoImageProcessor.from_pretrained(model_name)
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        logger.info("CUDA_VISIBLE_DEVICES=%s", visible if visible else "<unset>")
        device = torch.device("cuda:0")
        try:
            gpu_name = torch.cuda.get_device_name(device)
            logger.info("Using GPU device: %s", gpu_name)
        except Exception:
            logger.info("Using GPU device: cuda:0")
    else:
        device = torch.device("cpu")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    time_end("init_model", t0)
    return model, processor, device


def build_banks_for_sources(
    model, processor, device, ps, tile_size, stride, feature_dir
):
    """Build positive/negative banks from configured source tiles.

    Args:
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str): Feature cache directory.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Positive and negative banks.

    Examples:
        >>> isinstance(build_banks_for_sources.__name__, str)
        True
    """
    img_a_paths = getattr(cfg, "SOURCE_TILES", None) or [cfg.SOURCE_TILE]
    lab_a_paths = [cfg.SOURCE_LABEL_RASTER] * len(img_a_paths)
    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)

    pos_banks = []
    neg_banks = []
    for img_a_path, lab_a_path in zip(img_a_paths, lab_a_paths, strict=True):
        image_id_a = os.path.splitext(os.path.basename(img_a_path))[0]
        logger.info("bank source A: %s (labels: %s)", img_a_path, lab_a_path)
        img_a = load_dop20_image(img_a_path, downsample_factor=ds)
        labels_a = reproject_labels_to_image(
            img_a_path, lab_a_path, downsample_factor=ds
        )
        pos_i, neg_i = build_banks_single_scale(
            img_a,
            labels_a,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            getattr(cfg, "POS_FRAC_THRESH", 0.1),
            None,
            feature_dir,
            image_id_a,
            cfg.BANK_CACHE_DIR,
            context_radius=context_radius,
        )
        if pos_i.size > 0:
            pos_banks.append(pos_i)
        if neg_i is not None and len(neg_i) > 0:
            neg_banks.append(neg_i)

    pos_bank = np.concatenate(pos_banks, axis=0)
    neg_bank = np.concatenate(neg_banks, axis=0) if neg_banks else None
    logger.info(
        "combined banks: pos=%s, neg=%s",
        len(pos_bank),
        0 if neg_bank is None else len(neg_bank),
    )
    return pos_bank, neg_bank


def build_xgb_training_data(ps, tile_size, stride, feature_dir):
    """Build XGB training data from configured source tiles.

    Args:
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str): Feature cache directory.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix and labels.

    Examples:
        >>> isinstance(build_xgb_training_data.__name__, str)
        True
    """
    from ..core.xdboost import build_xgb_dataset

    img_a_paths = getattr(cfg, "SOURCE_TILES", None) or [cfg.SOURCE_TILE]
    lab_a_paths = [cfg.SOURCE_LABEL_RASTER] * len(img_a_paths)
    context_radius = int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0)
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)

    X_list = []
    y_list = []
    for img_a_path, lab_a_path in zip(img_a_paths, lab_a_paths, strict=True):
        image_id_a = os.path.splitext(os.path.basename(img_a_path))[0]
        img_a = load_dop20_image(img_a_path, downsample_factor=ds)
        labels_a = reproject_labels_to_image(
            img_a_path, lab_a_path, downsample_factor=ds
        )
        X_i, y_i = build_xgb_dataset(
            img_a,
            labels_a,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id_a,
            pos_frac=cfg.POS_FRAC_THRESH,
            max_neg=getattr(cfg, "MAX_NEG_BANK", 8000),
            context_radius=context_radius,
        )
        if X_i.size > 0 and y_i.size > 0:
            X_list.append(X_i)
            y_list.append(y_i)
    X = np.vstack(X_list) if X_list else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=np.float32)
    return X, y


def prep_b_tile(img_path, gt_paths):
    """Prepare Image B context (image, labels, GT mask, buffer mask).

    Args:
        img_path (str): Image B path.
        gt_paths (list[str]): Vector GT paths.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            img_b, labels_sh, gt_mask_eval, sh_buffer_mask

    Examples:
        >>> isinstance(prep_b_tile.__name__, str)
        True
    """
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    img_b = load_dop20_image(img_path, downsample_factor=ds)
    labels_sh = reproject_labels_to_image(
        img_path, cfg.SOURCE_LABEL_RASTER, downsample_factor=ds
    )
    gt_mask_b = rasterize_vector_labels(gt_paths, img_path, downsample_factor=ds)

    with __import__("rasterio").open(img_path) as src:
        pixel_size_m = abs(src.transform.a)
    pixel_size_m = pixel_size_m * ds
    buffer_m = cfg.BUFFER_M
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    sh_buffer_mask = build_sh_buffer_mask(labels_sh, buffer_pixels)
    if getattr(cfg, "CLIP_GT_TO_BUFFER", False):
        gt_mask_eval = np.logical_and(gt_mask_b, sh_buffer_mask)
    else:
        gt_mask_eval = gt_mask_b

    return img_b, labels_sh, gt_mask_eval, sh_buffer_mask


def log_metrics(tag, metrics):
    """Log a metrics dictionary with a consistent message format.

    Args:
        tag (str): Prefix tag for the log line.
        metrics (dict): Metrics dictionary with iou/f1/precision/recall.

    Examples:
        >>> log_metrics("demo", {"iou": 0.1, "f1": 0.2, "precision": 0.3, "recall": 0.4})
    """
    logger.info(
        "%s IoU=%.3f, F1=%.3f, P=%.3f, R=%.3f",
        tag,
        metrics["iou"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    )
