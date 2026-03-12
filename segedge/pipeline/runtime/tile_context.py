"""Tile-loading helpers for validation and holdout inference."""

from __future__ import annotations

import logging

import numpy as np
from rasterio import open as rio_open
from skimage.transform import resize

from ...core.config_loader import cfg
from ...core.io_utils import (
    build_sh_buffer_mask,
    load_dop20_image,
    rasterize_vector_labels,
    reproject_labels_to_image,
)
from ...core.timing_utils import perf_span, time_end, time_start

logger = logging.getLogger(__name__)


def load_b_tile_context(img_path: str, gt_vector_paths: list[str] | None):
    """Load B tile, SH raster, GT (optional), and buffer mask.

    Examples:
        >>> callable(load_b_tile_context)
        True
    """
    logger.info("loading tile: %s", img_path)
    t0_data = time_start()
    ds = int(cfg.model.backbone.resample_factor or 1)
    with perf_span("load_b_tile_context", substage="load_image"):
        img_b = load_dop20_image(img_path, downsample_factor=ds)
    with perf_span("load_b_tile_context", substage="reproject_source_labels"):
        labels_sh = reproject_labels_to_image(
            img_path, cfg.io.paths.source_label_raster, downsample_factor=ds
        )
    gt_mask = None
    if gt_vector_paths:
        with perf_span("load_b_tile_context", substage="rasterize_gt_vectors"):
            gt_mask = rasterize_vector_labels(
                gt_vector_paths,
                img_path,
                downsample_factor=ds,
            )
    time_end("data_loading_and_reprojection", t0_data)
    target_shape = img_b.shape[:2]
    if labels_sh.shape != target_shape:
        with perf_span("load_b_tile_context", substage="resize_source_labels"):
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
        with perf_span("load_b_tile_context", substage="resize_gt_mask"):
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

    with perf_span("load_b_tile_context", substage="read_pixel_size"):
        with rio_open(img_path) as src:
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

    with perf_span("load_b_tile_context", substage="build_sh_buffer_mask"):
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
