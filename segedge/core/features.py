"""Feature extraction and tiling utilities for SegEdge."""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from .config_loader import cfg
from .timing_utils import DEBUG_TIMING, DEBUG_TIMING_VERBOSE, time_end, time_start

logger = logging.getLogger(__name__)


def l2_normalize(feats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize feature vectors along the last dimension.

    Args:
        feats (np.ndarray): Feature array with last axis as channels.
        eps (float): Small epsilon for numerical stability.

    Returns:
        np.ndarray: L2-normalized feature array.

    Examples:
        >>> import numpy as np
        >>> feats = np.array([[3.0, 4.0]])
        >>> out = l2_normalize(feats)
        >>> np.allclose(out, np.array([[0.6, 0.8]]))
        True
    """
    t0 = time.perf_counter() if DEBUG_TIMING and DEBUG_TIMING_VERBOSE else None
    norms = np.linalg.norm(feats, axis=-1, keepdims=True) + eps
    out = feats / norms
    if DEBUG_TIMING and DEBUG_TIMING_VERBOSE:
        time_end("l2_normalize", t0)
    return out


def add_local_context_mean(feats_hwc: np.ndarray, radius: int) -> np.ndarray:
    """Add local spatial context to patch embeddings by averaging neighbors.

    Operates on patch-grid features (Hp x Wp x C) without mixing channels.

    Args:
        feats_hwc (np.ndarray): Patch-grid features.
        radius (int): Context radius in patch units.

    Returns:
        np.ndarray: Context-smoothed, L2-normalized features.

    Examples:
        >>> import numpy as np
        >>> feats = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        >>> out = add_local_context_mean(feats, radius=0)
        >>> np.array_equal(out, feats)
        True
    """
    if radius <= 0:
        return feats_hwc
    if feats_hwc.ndim != 3:
        raise ValueError(
            f"expected feats with shape (Hp, Wp, C), got {feats_hwc.shape}"
        )
    k = 2 * int(radius) + 1
    feats = feats_hwc.astype(np.float32, copy=False)
    feats_ctx = uniform_filter(feats, size=(k, k, 1), mode="reflect")
    return l2_normalize(feats_ctx)


def tile_iterator(
    image_hw3: np.ndarray,
    labels_hw: np.ndarray | None = None,
    tile_size: int = 1024,
    stride: int | None = None,
):
    """Yield (y, x, img_tile, label_tile) windows over an image.

    Args:
        image_hw3 (np.ndarray): HxWxC image array.
        labels_hw (np.ndarray | None): Optional label mask.
        tile_size (int): Tile size in pixels.
        stride (int | None): Stride in pixels; defaults to tile_size.

    Yields:
        tuple: (y, x, img_tile, label_tile) for each tile.

    Examples:
        >>> import numpy as np
        >>> tiles = list(tile_iterator(np.zeros((3, 4, 3)), tile_size=2, stride=2))
        >>> [(y, x, t.shape[:2]) for y, x, t, _ in tiles]
        [(0, 0, (2, 2)), (0, 2, (2, 2)), (2, 0, (1, 2)), (2, 2, (1, 2))]
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

    Args:
        img_tile_hw3 (np.ndarray): Image tile (H, W, C).
        labels_tile_hw (np.ndarray | None): Optional label tile.
        ps (int): Patch size in pixels.

    Returns:
        tuple: Cropped image, cropped labels, effective height, effective width.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((5, 7, 3))
        >>> labels = np.ones((5, 7))
        >>> img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(img, labels, 4)
        >>> img_c.shape, lab_c.shape, (h_eff, w_eff)
        ((4, 4, 3), (4, 4), (4, 4))
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

    Args:
        labels_tile (np.ndarray): Label mask at pixel resolution.
        hp (int): Patch grid height.
        wp (int): Patch grid width.
        pos_frac_thresh (float): Fraction of positive pixels to mark a patch as positive.

    Returns:
        tuple[np.ndarray, np.ndarray]: Boolean positive mask, boolean negative mask.

    Examples:
        >>> import numpy as np
        >>> labels = np.array([
        ...     [1, 1, 0, 0],
        ...     [1, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ...     [0, 0, 0, 0],
        ... ])
        >>> pos, neg = labels_to_patch_masks(labels, hp=2, wp=2, pos_frac_thresh=0.5)
        >>> pos.tolist(), neg.tolist()
        ([[True, False], [False, False]], [[False, True], [True, True]])
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


def tile_feature_path(feature_dir: str, image_id: str, y: int, x: int) -> str:
    """Return the canonical path for a tile's feature array.

    Args:
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.

    Returns:
        str: Absolute or relative path for the feature file.

    Examples:
        >>> tile_feature_path("feat", "img", 1, 2)
        'feat/img_y1_x2_features.npy'
    """
    fname = f"{image_id}_y{y}_x{x}_features.npy"
    return os.path.join(feature_dir, fname)


def tile_feature_meta_path(feature_dir: str, image_id: str, y: int, x: int) -> str:
    """Return the sidecar JSON path for feature metadata.

    Args:
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.

    Returns:
        str: Absolute or relative path for the metadata JSON.

    Examples:
        >>> tile_feature_meta_path("feat", "img", 1, 2)
        'feat/img_y1_x2_features.json'
    """
    fname = f"{image_id}_y{y}_x{x}_features.json"
    return os.path.join(feature_dir, fname)


def save_tile_features(
    feats_tile: np.ndarray,
    feature_dir: str,
    image_id: str,
    y: int,
    x: int,
    meta: dict | None = None,
):
    """Persist a tile's features to disk (and optional metadata).

    Args:
        feats_tile (np.ndarray): Feature tile array.
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.
        meta (dict | None): Optional metadata to serialize as JSON.

    Examples:
        >>> import numpy as np
        >>> import os
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     feats = np.zeros((2, 2, 3), dtype=np.float32)
        ...     save_tile_features(feats, d, "img", 0, 0, meta={"ps": 16, "resample_factor": 1})
        ...     os.path.exists(tile_feature_path(d, "img", 0, 0))
        True
    """
    os.makedirs(feature_dir, exist_ok=True)
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    np.save(fpath, feats_tile.astype(np.float32))
    if meta is not None:
        mpath = tile_feature_meta_path(feature_dir, image_id, y, x)
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f)


def load_tile_features_if_valid(
    feature_dir: str,
    image_id: str,
    y: int,
    x: int,
    expected_hp: int,
    expected_wp: int,
    ps: int,
    resample_factor: int,
) -> np.ndarray | None:
    """Load cached features if valid, otherwise return None.

    If metadata is missing or mismatched, the cache is removed.

    Args:
        feature_dir (str): Directory for cached features.
        image_id (str): Image identifier.
        y (int): Tile y offset.
        x (int): Tile x offset.
        expected_hp (int): Expected patch grid height.
        expected_wp (int): Expected patch grid width.
        ps (int): Patch size in pixels.
        resample_factor (int): Resample factor used for cached data.

    Returns:
        np.ndarray | None: Cached feature tile if valid, else None.

    Examples:
        >>> import numpy as np
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     feats = np.zeros((2, 2, 3), dtype=np.float32)
        ...     save_tile_features(feats, d, "img", 0, 0, meta={"ps": 16, "resample_factor": 1})
        ...     out = load_tile_features_if_valid(d, "img", 0, 0, 2, 2, ps=16, resample_factor=1)
        ...     out.shape
        (2, 2, 3)
    """
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    if not os.path.exists(fpath):
        return None
    mpath = tile_feature_meta_path(feature_dir, image_id, y, x)
    if not os.path.exists(mpath):
        try:
            os.remove(fpath)
        except OSError:
            pass
        return None
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        try:
            os.remove(fpath)
        except OSError:
            pass
        return None

    if meta.get("ps") != ps or meta.get("resample_factor") != resample_factor:
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None

    feats = np.load(fpath)
    if feats.shape[0] != expected_hp or feats.shape[1] != expected_wp:
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None
    return feats


def extract_patch_features_single_scale(
    image_hw3: np.ndarray, model, processor, device, ps: int = 16, aggregate_layers=None
):
    """Extract single-scale DINO patch features (Hp×Wp×C) from an RGB image.

    Args:
        image_hw3 (np.ndarray): RGB image array.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        aggregate_layers (list[int] | None): Optional layer indices to average.

    Returns:
        tuple[np.ndarray, int, int]: Feature grid, patch height, patch width.

    Examples:
        >>> callable(extract_patch_features_single_scale)
        True
    """
    t0 = time_start()
    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, h_proc, w_proc = pixel_values.shape
    with torch.no_grad():
        if aggregate_layers is None:
            out = model(**inputs)
            tokens = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states
            layers = [hidden_states[i] for i in aggregate_layers]
            tokens = torch.stack(layers, dim=0).mean(0)
    reg_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = tokens[:, 1 + reg_tokens :, :]
    num_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[2]
    hp = h_proc // ps
    wp = w_proc // ps
    assert hp * wp == num_tokens, f"patch-grid mismatch: {hp} * {wp} != {num_tokens}"
    feats = patch_tokens[0].cpu().numpy().reshape(hp, wp, dim)
    feats = l2_normalize(feats)
    time_end("extract_patch_features_single_scale", t0)
    return feats, hp, wp


def extract_patch_features_batch_single_scale(
    images_hw3: list[np.ndarray],
    model,
    processor,
    device,
    ps: int = 16,
    aggregate_layers=None,
):
    """Extract DINO patch features for a batch of same-sized RGB tiles.

    Args:
        images_hw3 (list[np.ndarray]): List of RGB tile arrays (same H/W).
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        aggregate_layers (list[int] | None): Optional layer indices to average.

    Returns:
        tuple[list[np.ndarray], int, int]: Feature grids, patch height, patch width.

    Examples:
        >>> callable(extract_patch_features_batch_single_scale)
        True
    """
    t0 = time_start()
    inputs = processor(
        images=images_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    pixel_values = inputs["pixel_values"]
    _, _, h_proc, w_proc = pixel_values.shape
    with torch.no_grad():
        if aggregate_layers is None:
            out = model(**inputs)
            tokens = out.last_hidden_state
        else:
            out = model(**inputs, output_hidden_states=True)
            hidden_states = out.hidden_states
            layers = [hidden_states[i] for i in aggregate_layers]
            tokens = torch.stack(layers, dim=0).mean(0)
    reg_tokens = getattr(model.config, "num_register_tokens", 0)
    patch_tokens = tokens[:, 1 + reg_tokens :, :]
    num_tokens, dim = patch_tokens.shape[1], patch_tokens.shape[2]
    hp = h_proc // ps
    wp = w_proc // ps
    assert hp * wp == num_tokens, f"patch-grid mismatch: {hp} * {wp} != {num_tokens}"
    feats_np = patch_tokens.cpu().numpy().reshape(len(images_hw3), hp, wp, dim)
    feats_list = [l2_normalize(feats_np[i]) for i in range(feats_np.shape[0])]
    time_end("extract_patch_features_batch_single_scale", t0)
    return feats_list, hp, wp


def prefetch_features_single_scale_image(
    img_hw3: np.ndarray,
    model,
    processor,
    device,
    ps: int = 16,
    tile_size: int = 1024,
    stride: int | None = None,
    aggregate_layers=None,
    feature_dir: str | None = None,
    image_id: str | None = None,
):
    """Precompute and cache all tile features for an image.

    Args:
        img_hw3 (np.ndarray): RGB image array.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int | None): Tile stride.
        aggregate_layers (list[int] | None): Optional layer indices to average.
        feature_dir (str | None): Optional cache directory.
        image_id (str | None): Optional image id for cache naming.

    Returns:
        dict: Cache keyed by (y, x) with feature arrays and tile shapes.

    Examples:
        >>> callable(prefetch_features_single_scale_image)
        True
    """
    t0 = time_start()
    cache = {}
    cached_tiles = computed_tiles = skipped_tiles = 0
    resample_factor = int(cfg.model.backbone.resample_factor or 1)
    batch_size = int(cfg.runtime.feature_batch_size or 1)
    batch_size = max(1, batch_size)
    pending: dict[tuple[int, int], list[tuple[int, int, np.ndarray, int, int]]] = {}

    def flush_pending(
        items: list[tuple[int, int, np.ndarray, int, int]],
    ) -> None:
        nonlocal computed_tiles
        if not items:
            return
        if batch_size <= 1:
            for y_i, x_i, img_i, h_i, w_i in items:
                feats_tile, hp_i, wp_i = extract_patch_features_single_scale(
                    img_i,
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=aggregate_layers,
                )
                computed_tiles += 1
                if feature_dir is not None and image_id is not None:
                    meta = {
                        "ps": ps,
                        "resample_factor": resample_factor,
                        "h_eff": h_i,
                        "w_eff": w_i,
                    }
                    save_tile_features(
                        feats_tile, feature_dir, image_id, y_i, x_i, meta=meta
                    )
                cache[(y_i, x_i)] = {
                    "feats": feats_tile,
                    "h_eff": h_i,
                    "w_eff": w_i,
                    "hp": hp_i,
                    "wp": wp_i,
                }
            return

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            imgs = [item[2] for item in chunk]
            if len(chunk) == 1:
                feats_tile, hp_i, wp_i = extract_patch_features_single_scale(
                    imgs[0],
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=aggregate_layers,
                )
                feats_list = [feats_tile]
            else:
                feats_list, hp_i, wp_i = extract_patch_features_batch_single_scale(
                    imgs,
                    model,
                    processor,
                    device,
                    ps=ps,
                    aggregate_layers=aggregate_layers,
                )
            for (y_i, x_i, _img_i, h_i, w_i), feats_tile in zip(
                chunk, feats_list, strict=True
            ):
                computed_tiles += 1
                if feature_dir is not None and image_id is not None:
                    meta = {
                        "ps": ps,
                        "resample_factor": resample_factor,
                        "h_eff": h_i,
                        "w_eff": w_i,
                    }
                    save_tile_features(
                        feats_tile, feature_dir, image_id, y_i, x_i, meta=meta
                    )
                cache[(y_i, x_i)] = {
                    "feats": feats_tile,
                    "h_eff": h_i,
                    "w_eff": w_i,
                    "hp": hp_i,
                    "wp": wp_i,
                }

    for y, x, img_tile, _ in tile_iterator(img_hw3, None, tile_size, stride):
        img_c, _, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, None, ps)
        if h_eff < ps or w_eff < ps:
            skipped_tiles += 1
            continue
        feats_tile = None
        hp = wp = None
        if feature_dir is not None and image_id is not None:
            hp = h_eff // ps
            wp = w_eff // ps
            feats_tile = load_tile_features_if_valid(
                feature_dir,
                image_id,
                y,
                x,
                expected_hp=hp,
                expected_wp=wp,
                ps=ps,
                resample_factor=resample_factor,
            )
            if feats_tile is not None:
                cached_tiles += 1
        if feats_tile is None:
            key = (h_eff, w_eff)
            pending.setdefault(key, []).append((y, x, img_c, h_eff, w_eff))
            if len(pending[key]) >= batch_size:
                flush_pending(pending.pop(key))
        else:
            cache[(y, x)] = {
                "feats": feats_tile,
                "h_eff": h_eff,
                "w_eff": w_eff,
                "hp": hp,
                "wp": wp,
            }

    for items in pending.values():
        flush_pending(items)
    time_end("prefetch_features_single_scale_image", t0)
    logger.info(
        "prefetch tiles=%s (cached=%s, computed=%s, skipped=%s)",
        len(cache),
        cached_tiles,
        computed_tiles,
        skipped_tiles,
    )
    return cache
