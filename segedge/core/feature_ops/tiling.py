"""Tile and patch-grid helpers for feature extraction."""

from __future__ import annotations

import time

import numpy as np

from ..timing_utils import DEBUG_TIMING, DEBUG_TIMING_VERBOSE, time_end


def tile_iterator(
    image_hw3: np.ndarray,
    labels_hw: np.ndarray | None = None,
    tile_size: int = 1024,
    stride: int | None = None,
):
    """Yield (y, x, img_tile, label_tile) windows over an image.

    Examples:
        >>> tiles = list(tile_iterator(np.zeros((2, 2, 3), dtype=np.uint8), tile_size=2))
        >>> len(tiles)
        1
    """
    h, w = image_hw3.shape[:2]
    if stride is None:
        stride = tile_size
    y = 0
    while y < h:
        x = 0
        y_end = min(y + tile_size, h)
        while x < w:
            x_end = min(x + tile_size, w)
            img_tile = image_hw3[y:y_end, x:x_end]
            lab_tile = labels_hw[y:y_end, x:x_end] if labels_hw is not None else None
            yield y, x, img_tile, lab_tile
            x += stride
        y += stride


def crop_to_multiple_of_ps(
    img_tile_hw3: np.ndarray, labels_tile_hw: np.ndarray | None, ps: int
):
    """Crop a tile so height/width are multiples of patch size.

    Examples:
        >>> img = np.zeros((5, 5, 3), dtype=np.uint8)
        >>> crop_to_multiple_of_ps(img, None, 2)[2:]
        (4, 4)
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h, w = img_tile_hw3.shape[:2]
    h_eff = (h // ps) * ps
    w_eff = (w // ps) * ps
    img_c = img_tile_hw3[:h_eff, :w_eff]
    lab_c = labels_tile_hw[:h_eff, :w_eff] if labels_tile_hw is not None else None
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("crop_to_multiple_of_ps", t0)
    return img_c, lab_c, h_eff, w_eff


def labels_to_patch_masks(
    labels_tile: np.ndarray, hp: int, wp: int, pos_frac_thresh: float = 0.1
):
    """Convert pixel labels to patch-level positive/negative masks.

    Examples:
        >>> pos, neg = labels_to_patch_masks(np.ones((4, 4), dtype=np.uint8), 2, 2)
        >>> pos.shape, neg.shape
        ((2, 2), (2, 2))
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    h_eff, w_eff = labels_tile.shape
    patch_h = h_eff // hp
    patch_w = w_eff // wp
    labels_c = labels_tile[: hp * patch_h, : wp * patch_w]
    labels_bin = (labels_c > 0).astype(np.float32)
    blocks = labels_bin.reshape(hp, patch_h, wp, patch_w)
    frac_pos = blocks.mean(axis=(1, 3))
    pos_mask = frac_pos >= pos_frac_thresh
    neg_mask = frac_pos == 0.0
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("labels_to_patch_masks", t0)
    return pos_mask, neg_mask
