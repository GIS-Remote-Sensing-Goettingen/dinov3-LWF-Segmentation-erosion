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
        >>> callable(_tile_has_gt)
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
        >>> callable(_load_gt_geometries)
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
        >>> callable(_get_gt_index)
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
        >>> callable(_tiles_with_gt_chunk)
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


def resolve_tile_splits_from_gt(
    tiles_dir: str,
    tile_glob: str,
    gt_vector_paths: list[str],
    val_fraction: float,
    seed: int,
    downsample_factor: int | None = None,
    num_workers: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve source/val/holdout tiles using GT presence.

    Tiles with any GT positives are split into source (train) and validation.
    Tiles without GT positives are assigned to holdout.

    Args:
        tiles_dir (str): Directory containing tiles.
        tile_glob (str): Glob pattern for tile files.
        gt_vector_paths (list[str]): Ground-truth vector paths.
        val_fraction (float): Fraction of GT tiles for validation.
        seed (int): RNG seed for split.
        downsample_factor (int | None): Downsample factor for GT presence checks.
        num_workers (int | None): Worker count for GT presence checks.

    Returns:
        tuple[list[str], list[str], list[str]]: Source, validation, holdout tiles.

    Examples:
        >>> callable(resolve_tile_splits_from_gt)
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
    if len(gt_tiles) == 1:
        logger.warning(
            "only one GT tile found; using it for both source and validation"
        )
        return gt_tiles, gt_tiles, holdout_tiles

    rng = np.random.default_rng(seed)
    indices = np.arange(len(gt_tiles))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(gt_tiles) * val_fraction)))
    if val_count >= len(gt_tiles):
        val_count = len(gt_tiles) - 1
    val_idx = set(indices[:val_count].tolist())
    source_tiles = [p for i, p in enumerate(gt_tiles) if i not in val_idx]
    val_tiles = [p for i, p in enumerate(gt_tiles) if i in val_idx]
    return source_tiles, val_tiles, holdout_tiles


def init_model(model_name: str):
    """Load a DINO backbone + processor on CPU/GPU with timing.

    Args:
        model_name (str): HuggingFace model name.

    Returns:
        tuple: (model, processor, device).

    Examples:
        >>> callable(init_model)
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
        >>> callable(build_banks_for_sources)
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
        >>> callable(build_xgb_training_data)
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
        >>> callable(prep_b_tile)
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
