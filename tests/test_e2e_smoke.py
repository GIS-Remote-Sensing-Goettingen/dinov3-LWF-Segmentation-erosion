"""End-to-end smoke test using configured tiles and labels."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

import config as cfg
from segedge.core.io_utils import (
    build_sh_buffer_mask,
    load_dop20_image,
    rasterize_vector_labels,
    reproject_labels_to_image,
)
from segedge.core.metrics_utils import compute_metrics


def _existing_paths(paths: list[str]) -> list[Path]:
    """Return existing paths from a list of strings.

    Examples:
        >>> _existing_paths(["/tmp/does_not_exist"]) == []
        True
    """
    return [Path(p) for p in paths if Path(p).exists()]


def test_e2e_smoke_single_tile():
    """Run a light E2E smoke test using one configured tile and all labels.

    Examples:
        >>> True
        True
    """
    tile_candidates = getattr(cfg, "SOURCE_TILES", None) or [cfg.SOURCE_TILE]
    existing_tiles = _existing_paths(tile_candidates)
    if not existing_tiles:
        pytest.skip("no configured tiles found on disk")

    labels_path = Path(cfg.SOURCE_LABEL_RASTER)
    if not labels_path.exists():
        pytest.skip("label raster not found on disk")

    gt_paths = _existing_paths(list(cfg.EVAL_GT_VECTORS))
    if not gt_paths:
        pytest.skip("no GT vector labels found on disk")

    rng = random.Random(42)
    tile_path = rng.choice(existing_tiles)

    img_b = load_dop20_image(str(tile_path))
    labels_sh = reproject_labels_to_image(str(tile_path), str(labels_path))
    gt_mask = rasterize_vector_labels([str(p) for p in gt_paths], str(tile_path))

    buffer_m = cfg.BUFFER_M
    with __import__("rasterio").open(str(tile_path)) as src:
        pixel_size_m = abs(src.transform.a)
    buffer_pixels = int(round(buffer_m / pixel_size_m))
    sh_buffer_mask = build_sh_buffer_mask(labels_sh, buffer_pixels)

    assert img_b.shape[:2] == labels_sh.shape
    assert gt_mask.shape == labels_sh.shape
    assert sh_buffer_mask.shape == labels_sh.shape

    pred_mask = np.zeros_like(gt_mask, dtype=bool)
    metrics = compute_metrics(pred_mask, gt_mask)
    assert set(metrics.keys()) >= {"iou", "f1", "precision", "recall"}
