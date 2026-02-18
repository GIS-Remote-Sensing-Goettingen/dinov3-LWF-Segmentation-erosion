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

from ..core.banks import build_banks_single_scale
from ..core.config_loader import cfg
from ..core.features import (
    apply_xgb_feature_stats,
    fit_xgb_feature_stats,
    prefetch_features_single_scale_image,
)
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
    gt_tiles, holdout_tiles = resolve_tiles_from_gt_presence(
        tiles_dir,
        tile_glob,
        gt_vector_paths,
        downsample_factor=downsample_factor,
        num_workers=num_workers,
    )

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


def resolve_tiles_from_gt_presence(
    tiles_dir: str,
    tile_glob: str,
    gt_vector_paths: list[str],
    downsample_factor: int | None = None,
    num_workers: int | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve GT-positive and inference tiles from a directory.

    Tiles with any GT overlap are returned as GT-positive tiles. Remaining tiles
    are returned as inference tiles.

    Args:
        tiles_dir (str): Directory containing tiles.
        tile_glob (str): Glob pattern for tile files.
        gt_vector_paths (list[str]): Ground-truth vector paths.
        downsample_factor (int | None): Downsample factor for GT presence checks.
        num_workers (int | None): Worker count for GT presence checks.

    Returns:
        tuple[list[str], list[str]]: GT-positive tiles, inference tiles.

    Examples:
        >>> callable(resolve_tiles_from_gt_presence)
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
        downsample_factor = int(cfg.model.backbone.resample_factor or 1)

    raster_path = cfg.io.paths.source_label_raster
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
    inference_tiles = []
    if num_workers <= 1:
        for tile_path in tile_paths:
            has_gt = _tile_has_gt(tile_path, gt_vector_paths, downsample_factor)
            if has_gt:
                gt_tiles.append(tile_path)
            else:
                inference_tiles.append(tile_path)
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
                inference_tiles.extend(holdout_chunk)

    return sorted(gt_tiles), sorted(inference_tiles)


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


def _source_augmentation_modes() -> list[str]:
    """Return enabled source augmentation modes from config.

    Examples:
        >>> "orig" in _source_augmentation_modes()
        True
    """
    aug = cfg.model.augmentation
    if not aug.enabled:
        return ["orig"]
    modes: list[str] = []
    if aug.include_identity:
        modes.append("orig")
    if aug.horizontal_flip:
        modes.append("flip_lr")
    if aug.vertical_flip:
        modes.append("flip_ud")
    for deg in aug.rotations_deg:
        d = int(deg) % 360
        if d == 0:
            if not aug.include_identity:
                modes.append("orig")
            continue
        if d not in {90, 180, 270}:
            logger.warning("skipping unsupported augmentation rotation: %s", deg)
            continue
        modes.append(f"rot{d}")
    if not modes:
        modes = ["orig"]
    deduped = []
    seen: set[str] = set()
    for mode in modes:
        if mode in seen:
            continue
        deduped.append(mode)
        seen.add(mode)
    return deduped


def _apply_source_augmentation(
    img: np.ndarray,
    labels: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a geometric augmentation to source image and labels.

    Examples:
        >>> import numpy as np
        >>> x = np.zeros((2, 2, 3), dtype=np.uint8)
        >>> y = np.zeros((2, 2), dtype=np.uint8)
        >>> _apply_source_augmentation(x, y, "orig")[0].shape
        (2, 2, 3)
    """
    if mode == "orig":
        return img, labels
    if mode == "flip_lr":
        return np.flip(img, axis=1).copy(), np.flip(labels, axis=1).copy()
    if mode == "flip_ud":
        return np.flip(img, axis=0).copy(), np.flip(labels, axis=0).copy()
    if mode.startswith("rot"):
        deg = int(mode.replace("rot", ""))
        if deg % 90 != 0:
            raise ValueError(f"rotation must be multiple of 90, got {deg}")
        k = (deg // 90) % 4
        return np.rot90(img, k).copy(), np.rot90(labels, k).copy()
    raise ValueError(f"unknown augmentation mode: {mode}")


def build_training_artifacts_for_tiles(
    source_tiles: list[str],
    source_label_raster: str,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_cache_mode: str,
    feature_dir: str | None,
    context_radius: int,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    dict | None,
    dict | None,
]:
    """Build banks and XGB training data from a set of source tiles.

    Args:
        source_tiles (list[str]): Source tile paths.
        source_label_raster (str): Source label raster path.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_cache_mode (str): Either ``disk`` or ``memory``.
        feature_dir (str | None): Feature cache directory.
        context_radius (int): Feature context radius.

    Returns:
        tuple[
            np.ndarray,
            np.ndarray | None,
            np.ndarray,
            np.ndarray,
            list[str],
            list[str],
            dict | None,
            dict | None,
        ]:
            Positive bank, negative bank, XGB feature matrix, XGB labels,
            base image ids, augmentation modes, XGB feature z-score stats,
            and feature layout metadata.

    Examples:
        >>> callable(build_training_artifacts_for_tiles)
        True
    """
    from ..core.xdboost import build_xgb_dataset

    if not source_tiles:
        raise ValueError("source tile list is empty")
    aug_modes = _source_augmentation_modes()
    logger.info("source augmentations: %s", ", ".join(aug_modes))

    pos_banks = []
    neg_banks = []
    x_list = []
    y_list = []
    feature_layout: dict | None = None
    base_image_ids = [os.path.splitext(os.path.basename(p))[0] for p in source_tiles]
    ds = int(cfg.model.backbone.resample_factor or 1)

    for img_a_path, image_id_a in zip(source_tiles, base_image_ids, strict=True):
        logger.info("source A base: %s (labels: %s)", img_a_path, source_label_raster)
        img_a_base = load_dop20_image(img_a_path, downsample_factor=ds)
        labels_a_base = reproject_labels_to_image(
            img_a_path, source_label_raster, downsample_factor=ds
        )
        for aug_mode in aug_modes:
            image_id_aug = (
                image_id_a if aug_mode == "orig" else f"{image_id_a}_{aug_mode}"
            )
            img_a, labels_a = _apply_source_augmentation(
                img_a_base, labels_a_base, aug_mode
            )
            logger.info(
                "source A augmented: %s mode=%s image_id=%s",
                img_a_path,
                aug_mode,
                image_id_aug,
            )
            prefetched_a = None
            if feature_cache_mode == "memory":
                logger.info("prefetch: Image A %s", image_id_aug)
                prefetched_a = prefetch_features_single_scale_image(
                    img_a,
                    model,
                    processor,
                    device,
                    ps,
                    tile_size,
                    stride,
                    None,
                    None,
                    image_id_aug,
                )

            pos_bank_i, neg_bank_i = build_banks_single_scale(
                img_a,
                labels_a,
                model,
                processor,
                device,
                ps,
                tile_size,
                stride,
                cfg.model.banks.pos_frac_thresh,
                None,
                feature_dir,
                image_id_aug,
                cfg.io.paths.bank_cache_dir,
                context_radius=context_radius,
                prefetched_tiles=prefetched_a,
            )
            if pos_bank_i.size > 0:
                pos_banks.append(pos_bank_i)
            if neg_bank_i is not None and len(neg_bank_i) > 0:
                neg_banks.append(neg_bank_i)

            x_i, y_i, layout_i = build_xgb_dataset(
                img_a,
                labels_a,
                ps,
                tile_size,
                stride,
                feature_dir,
                image_id_aug,
                pos_frac=cfg.model.banks.pos_frac_thresh,
                max_pos=cfg.model.banks.max_pos_bank,
                max_neg=cfg.model.banks.max_neg_bank,
                context_radius=context_radius,
                prefetched_tiles=prefetched_a,
                return_layout=feature_layout is None,
            )
            if x_i.size > 0 and y_i.size > 0:
                x_list.append(x_i)
                y_list.append(y_i)
            if feature_layout is None and layout_i is not None:
                feature_layout = layout_i

    if not pos_banks:
        raise ValueError("no positive banks were built; check source tiles and labels")
    pos_bank = np.concatenate(pos_banks, axis=0)
    neg_bank = np.concatenate(neg_banks, axis=0) if neg_banks else None
    logger.info(
        "combined banks: pos=%s, neg=%s",
        len(pos_bank),
        0 if neg_bank is None else len(neg_bank),
    )

    x = np.vstack(x_list) if x_list else np.empty((0, 0), dtype=np.float32)
    y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=np.float32)
    if x.size == 0 or y.size == 0:
        raise ValueError("XGBoost dataset is empty; check source tiles and labels")
    xgb_feature_stats = None
    if cfg.model.hybrid_features.enabled and cfg.model.hybrid_features.xgb_zscore:
        xgb_feature_stats = fit_xgb_feature_stats(
            x, eps=float(cfg.model.hybrid_features.zscore_eps)
        )
        x = apply_xgb_feature_stats(x, xgb_feature_stats)
    return (
        pos_bank,
        neg_bank,
        x,
        y,
        base_image_ids,
        aug_modes,
        xgb_feature_stats,
        feature_layout,
    )


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
    img_a_paths = cfg.io.paths.source_tiles or [cfg.io.paths.source_tile]
    lab_a_paths = [cfg.io.paths.source_label_raster] * len(img_a_paths)
    context_radius = int(cfg.model.banks.feat_context_radius or 0)
    ds = int(cfg.model.backbone.resample_factor or 1)

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
            cfg.model.banks.pos_frac_thresh,
            None,
            feature_dir,
            image_id_a,
            cfg.io.paths.bank_cache_dir,
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

    img_a_paths = cfg.io.paths.source_tiles or [cfg.io.paths.source_tile]
    lab_a_paths = [cfg.io.paths.source_label_raster] * len(img_a_paths)
    context_radius = int(cfg.model.banks.feat_context_radius or 0)
    ds = int(cfg.model.backbone.resample_factor or 1)

    X_list = []
    y_list = []
    for img_a_path, lab_a_path in zip(img_a_paths, lab_a_paths, strict=True):
        image_id_a = os.path.splitext(os.path.basename(img_a_path))[0]
        img_a = load_dop20_image(img_a_path, downsample_factor=ds)
        labels_a = reproject_labels_to_image(
            img_a_path, lab_a_path, downsample_factor=ds
        )
        X_i, y_i, _ = build_xgb_dataset(
            img_a,
            labels_a,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id_a,
            pos_frac=cfg.model.banks.pos_frac_thresh,
            max_pos=cfg.model.banks.max_pos_bank,
            max_neg=cfg.model.banks.max_neg_bank,
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
    ds = int(cfg.model.backbone.resample_factor or 1)
    img_b = load_dop20_image(img_path, downsample_factor=ds)
    labels_sh = reproject_labels_to_image(
        img_path, cfg.io.paths.source_label_raster, downsample_factor=ds
    )
    gt_mask_b = rasterize_vector_labels(gt_paths, img_path, downsample_factor=ds)

    with __import__("rasterio").open(img_path) as src:
        pixel_size_m = abs(src.transform.a)
    pixel_size_m = pixel_size_m * ds
    buffer_m = cfg.model.priors.buffer_m
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    sh_buffer_mask = build_sh_buffer_mask(labels_sh, buffer_pixels)
    if cfg.model.priors.clip_gt_to_buffer:
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
