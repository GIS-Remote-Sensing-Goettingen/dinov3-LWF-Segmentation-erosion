"""Bank construction utilities for SegEdge."""

from __future__ import annotations

import logging
import os

import numpy as np
from skimage.morphology import disk, erosion

from .config_loader import cfg
from .features import (
    add_local_context_mean,
    crop_to_multiple_of_ps,
    extract_patch_features_single_scale,
    fuse_patch_features,
    hybrid_feature_spec_hash,
    labels_to_patch_masks,
    load_tile_features_if_valid,
    save_tile_features,
    tile_iterator,
)
from .timing_utils import time_end, time_start

logger = logging.getLogger(__name__)


def cleanup_bank_cache(
    bank_cache_dir: str,
    image_id: str,
    ps: int,
    context_radius: int,
    resample_factor: int,
    feature_spec_hash: str,
):
    """Remove stale bank cache files for this image_id with mismatched settings.

    Args:
        bank_cache_dir (str): Directory holding cached bank files.
        image_id (str): Image identifier for cache naming.
        ps (int): Patch size used to build banks.
        context_radius (int): Context radius used in feature smoothing.
        resample_factor (int): Resample factor used to build banks.

    Examples:
        >>> import os
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     keep = os.path.join(d, "img_ps16_ctx0_rs1_fhhash_pos_bank.npy")
        ...     drop = os.path.join(d, "img_ps8_ctx0_rs1_pos_bank.npy")
        ...     _ = open(keep, "w").close()
        ...     _ = open(drop, "w").close()
        ...     cleanup_bank_cache(d, "img", 16, 0, 1, "hash")
        ...     os.path.exists(keep), os.path.exists(drop)
        (True, False)
    """
    if not os.path.isdir(bank_cache_dir):
        return
    prefix = f"{image_id}_"
    keep_tag = (
        f"{image_id}_ps{ps}_ctx{int(context_radius)}_rs{int(resample_factor)}"
        f"_fh{feature_spec_hash}"
    )
    for fname in os.listdir(bank_cache_dir):
        if not fname.startswith(prefix):
            continue
        if keep_tag in fname:
            continue
        if fname.endswith("_pos_bank.npy") or fname.endswith("_neg_bank.npy"):
            try:
                os.remove(os.path.join(bank_cache_dir, fname))
            except OSError:
                pass


def build_banks_single_scale(
    img_a: np.ndarray,
    labels_a: np.ndarray,
    model,
    processor,
    device,
    ps: int = 16,
    tile_size: int = 1024,
    stride: int | None = None,
    pos_frac_thresh: float = 0.1,
    aggregate_layers=None,
    feature_dir: str | None = None,
    image_id: str | None = None,
    bank_cache_dir: str | None = None,
    context_radius: int = 0,
    prefetched_tiles: dict | None = None,
):
    """Build positive/negative patch banks from Image A using SH_2022 labels.

    Extracts (and optionally caches) DINO features per tile and aggregates
    pixel labels into patch labels using the provided threshold.

    Args:
        img_a (np.ndarray): Image A RGB array.
        labels_a (np.ndarray): Raster labels aligned to Image A.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int | None): Tile stride.
        pos_frac_thresh (float): Fraction threshold for positives.
        aggregate_layers (list[int] | None): Optional layer indices to average.
        feature_dir (str | None): Optional feature cache directory.
        image_id (str | None): Image identifier for caches.
        bank_cache_dir (str | None): Directory for bank caches.
        context_radius (int): Feature context radius.
        prefetched_tiles (dict | None): Optional in-memory tile feature cache.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: Positive and negative banks.

    Examples:
        >>> callable(build_banks_single_scale)
        True
    """
    t0 = time_start()

    if bank_cache_dir is not None and image_id is not None:
        os.makedirs(bank_cache_dir, exist_ok=True)
        resample_factor = int(cfg.model.backbone.resample_factor or 1)
        feat_hash = hybrid_feature_spec_hash()
        cleanup_bank_cache(
            bank_cache_dir,
            image_id,
            ps,
            context_radius,
            resample_factor,
            feat_hash,
        )
        cache_tag = (
            f"{image_id}_ps{ps}_ctx{int(context_radius)}_rs{resample_factor}"
            f"_fh{feat_hash}"
        )
        pos_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_pos_bank.npy")
        neg_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_neg_bank.npy")
        if os.path.exists(pos_cache_path):
            pos_bank = np.load(pos_cache_path)
            neg_bank = (
                np.load(neg_cache_path) if os.path.exists(neg_cache_path) else None
            )
            time_end("build_banks_single_scale(load_cache)", t0)
            logger.info("loaded banks from %s", bank_cache_dir)
            return pos_bank, neg_bank

    pos_list, neg_list = [], []
    cached_tiles = computed_tiles = 0

    erosion_radius = int(cfg.model.banks.bank_erosion_radius or 0)
    if erosion_radius > 0:
        labels_eroded = erosion((labels_a > 0).astype(bool), disk(erosion_radius))
    else:
        labels_eroded = (labels_a > 0).astype(bool)
    resample_factor = int(cfg.model.backbone.resample_factor or 1)

    for y, x, img_tile, lab_tile in tile_iterator(
        img_a, labels_eroded, tile_size, stride
    ):
        prefetched = prefetched_tiles.get((y, x)) if prefetched_tiles else None
        if prefetched is not None:
            h_eff = prefetched["h_eff"]
            w_eff = prefetched["w_eff"]
            if h_eff < ps or w_eff < ps:
                continue
            img_c = img_tile[:h_eff, :w_eff]
            lab_c = lab_tile[:h_eff, :w_eff] if lab_tile is not None else None
            feats_tile = prefetched["feats"]
            hp = prefetched["hp"]
            wp = prefetched["wp"]
            cached_tiles += 1
        else:
            img_c, lab_c, h_eff, w_eff = crop_to_multiple_of_ps(img_tile, lab_tile, ps)
            if h_eff < ps or w_eff < ps:
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
                feats_tile, hp, wp = extract_patch_features_single_scale(
                    img_c,
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
                        "h_eff": h_eff,
                        "w_eff": w_eff,
                        "feature_spec_hash": hybrid_feature_spec_hash(),
                    }
                    save_tile_features(
                        feats_tile, feature_dir, image_id, y, x, meta=meta
                    )

        if lab_c is None:
            logger.warning("missing labels for tile y=%s x=%s; skipping", y, x)
            continue
        if context_radius and context_radius > 0:
            feats_tile = add_local_context_mean(feats_tile, context_radius)
        feats_tile, _ = fuse_patch_features(
            feats_tile,
            img_c,
            ps,
            mode="knn",
            return_layout=False,
        )

        if hp is None or wp is None:
            logger.warning(
                "missing patch dimensions for tile y=%s x=%s; skipping", y, x
            )
            continue
        pos_mask, neg_mask = labels_to_patch_masks(
            lab_c, hp, wp, pos_frac_thresh=pos_frac_thresh
        )
        pos_feats_tile = feats_tile[pos_mask]
        neg_feats_tile = feats_tile[neg_mask]
        if pos_feats_tile.size > 0:
            pos_list.append(pos_feats_tile)
        if neg_feats_tile.size > 0:
            neg_list.append(neg_feats_tile)

    if not pos_list:
        logger.warning(
            "no positive patches found for image_id=%s; skipping this source tile",
            image_id,
        )
        return np.empty((0, 0), dtype=np.float32), None

    pos_bank = np.concatenate(pos_list, axis=0)
    neg_bank = np.concatenate(neg_list, axis=0) if neg_list else None

    logger.info("Positive bank size: %s patches", len(pos_bank))
    max_pos = int(cfg.model.banks.max_pos_bank)
    if len(pos_bank) > max_pos:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pos_bank), size=max_pos, replace=False)
        pos_bank = pos_bank[idx]
        logger.info(
            "subsampled positive bank to %s (MAX_POS_BANK=%s)",
            len(pos_bank),
            max_pos,
        )
    if neg_bank is not None:
        max_neg = cfg.model.banks.max_neg_bank
        logger.info("Negative bank size: %s patches", len(neg_bank))
        if len(neg_bank) > max_neg:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(neg_bank), size=max_neg, replace=False)
            neg_bank = neg_bank[idx]
            logger.info(
                "subsampled negative bank to %s (MAX_NEG_BANK=%s)",
                len(neg_bank),
                max_neg,
            )

    time_end("build_banks_single_scale", t0)
    logger.info("A: cached tiles=%s, computed tiles=%s", cached_tiles, computed_tiles)

    if bank_cache_dir is not None and image_id is not None:
        os.makedirs(bank_cache_dir, exist_ok=True)
        cache_tag = (
            f"{image_id}_ps{ps}_ctx{int(context_radius)}_rs{resample_factor}"
            f"_fh{hybrid_feature_spec_hash()}"
        )
        pos_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_pos_bank.npy")
        neg_cache_path = os.path.join(bank_cache_dir, f"{cache_tag}_neg_bank.npy")
        np.save(pos_cache_path, pos_bank.astype(np.float32))
        if neg_bank is not None:
            np.save(neg_cache_path, neg_bank.astype(np.float32))
        logger.info("saved banks to %s", bank_cache_dir)

    return pos_bank, neg_bank
