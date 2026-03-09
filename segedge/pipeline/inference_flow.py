"""Shared inference flow helpers for run.py."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Callable

import numpy as np

from .common import filter_tiles_by_source_label_presence
from .runtime_utils import _update_phase_metrics, infer_on_holdout


def resolve_inference_tiles(
    *,
    infer_tiles_dir: str | None,
    infer_tile_glob: str | None,
    infer_tiles: list[str],
    legacy_inference_dir: str | None,
    legacy_inference_glob: str,
    legacy_holdout_tiles: list[str],
    logger,
) -> tuple[list[str], str | None, str]:
    """Resolve inference tiles from new io.inference settings with legacy fallback.

    Examples:
        >>> callable(resolve_inference_tiles)
        True
    """
    tiles_dir = infer_tiles_dir or legacy_inference_dir
    tile_glob = str(infer_tile_glob or legacy_inference_glob or "*.tif")
    explicit_tiles = list(infer_tiles or [])

    if tiles_dir:
        if not os.path.isdir(tiles_dir):
            raise ValueError(f"inference tiles_dir not found: {tiles_dir}")
        tiles = sorted(__import__("glob").glob(os.path.join(tiles_dir, tile_glob)))
        logger.info(
            "inference dir: %s (glob=%s) -> %s tiles",
            tiles_dir,
            tile_glob,
            len(tiles),
        )
        filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(tiles)
        if excluded_count:
            logger.info(
                "inference: excluded %s tiles with no SOURCE_LABEL_RASTER labels inside tile",
                excluded_count,
            )
        return filtered_tiles, tiles_dir, tile_glob
    if explicit_tiles:
        filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
            explicit_tiles
        )
        if excluded_count:
            logger.info(
                "inference: excluded %s explicit tiles with no SOURCE_LABEL_RASTER labels inside tile",
                excluded_count,
            )
        return filtered_tiles, None, tile_glob
    filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
        list(legacy_holdout_tiles)
    )
    if excluded_count:
        logger.info(
            "inference: excluded %s legacy holdout tiles with no SOURCE_LABEL_RASTER labels inside tile",
            excluded_count,
        )
    return filtered_tiles, None, tile_glob


def run_holdout_inference(
    *,
    holdout_tiles: list[str],
    processed_tiles: set[str],
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
    holdout_phase_metrics: dict[str, list[dict]],
    append_union: Callable[[str, str, np.ndarray, str, int], None],
    processed_log_path: str,
    write_checkpoint: Callable[[int], None],
    logger,
) -> int:
    """Run holdout inference tiles and update rolling checkpoints.

    Examples:
        >>> callable(run_holdout_inference)
        True
    """
    holdout_tiles_processed = len(processed_tiles)
    total_pending_tiles = sum(
        1 for tile_path in holdout_tiles if tile_path not in processed_tiles
    )
    pending_tile_index = 0
    for b_path in holdout_tiles:
        if b_path in processed_tiles:
            logger.info("holdout skip (already processed): %s", b_path)
            continue
        pending_tile_index += 1
        logger.info(
            "Processing tile %s, %s / %s",
            b_path,
            pending_tile_index,
            total_pending_tiles,
        )
        result = infer_on_holdout(
            b_path,
            gt_vector_paths,
            model,
            processor,
            device,
            pos_bank,
            neg_bank,
            tuned,
            ps,
            tile_size,
            stride,
            feature_dir,
            shape_dir,
            plot_dir,
            context_radius,
            plot_with_metrics=False,
        )
        if result["gt_available"]:
            _update_phase_metrics(holdout_phase_metrics, result["metrics"])
        holdout_tiles_processed += 1
        ref_path = result["ref_path"]
        masks = result["masks"]
        for mask_key, mask_val in masks.items():
            if "_" not in mask_key:
                continue
            stream, variant = mask_key.split("_", 1)
            append_union(stream, variant, mask_val, ref_path, holdout_tiles_processed)
        record = {
            "tile_path": b_path,
            "image_id": result["image_id"],
            "status": "done",
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(processed_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        write_checkpoint(holdout_tiles_processed)
    return holdout_tiles_processed
