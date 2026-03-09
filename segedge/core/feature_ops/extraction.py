"""Patch-feature extraction and prefetch routines."""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..config_loader import cfg
from ..timing_utils import time_end, time_start
from .cache import load_tile_features_if_valid, save_tile_features
from .fusion import l2_normalize
from .spec import hybrid_feature_spec_hash
from .tiling import crop_to_multiple_of_ps, tile_iterator

logger = logging.getLogger(__name__)


def extract_patch_features_single_scale(
    image_hw3: np.ndarray, model, processor, device, ps: int = 16, aggregate_layers=None
):
    """Extract single-scale DINO patch features (Hp x Wp x C) from an RGB image.

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
                        "feature_spec_hash": hybrid_feature_spec_hash(),
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
                        "feature_spec_hash": hybrid_feature_spec_hash(),
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
