"""Patch-feature extraction and prefetch routines."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from ..config_loader import cfg
from ..timing_utils import perf_span, record_performance, time_end, time_start
from .cache import (
    image_feature_manifest_path,
    inspect_tile_features_if_valid,
    load_feature_manifest_if_valid,
    load_tile_features_if_valid,
    save_feature_manifest,
    save_tile_features,
    tile_feature_path,
)
from .fusion import l2_normalize
from .spec import hybrid_feature_spec_hash
from .tiling import crop_to_multiple_of_ps, tile_iterator

logger = logging.getLogger(__name__)


def _safe_file_size(path: str) -> int:
    """Return file size or zero when unavailable.

    Examples:
        >>> _safe_file_size("/tmp/missing") >= 0
        True
    """
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def _manifest_feature_path_if_loadable(
    feature_path: str,
    *,
    expected_hp: int,
    expected_wp: int,
) -> str | None:
    """Return the path when the cached feature array is readable and shaped correctly.

    Examples:
        >>> _manifest_feature_path_if_loadable("/tmp/missing.npy", expected_hp=1, expected_wp=1) is None
        True
    """
    try:
        feats = np.load(feature_path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if feats.shape[:2] != (expected_hp, expected_wp):
        return None
    return feature_path


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
    materialize_cached: bool = True,
):
    """Precompute and cache all tile features for an image.

    Examples:
        >>> callable(prefetch_features_single_scale_image)
        True
    """
    t0 = time_start()
    cache = {}
    cached_tiles = computed_tiles = skipped_tiles = 0
    manifest_cache_hits = 0
    cache_stats = {
        "feature_bytes_read": 0,
        "feature_bytes_written": 0,
        "feature_files_read": 0,
        "feature_files_written": 0,
        "manifest_bytes_read": 0,
        "manifest_bytes_written": 0,
        "manifest_files_read": 0,
        "manifest_files_written": 0,
    }
    resample_factor = int(cfg.model.backbone.resample_factor or 1)
    batch_size = int(cfg.runtime.feature_batch_size or 1)
    batch_size = max(1, batch_size)
    pending: dict[tuple[int, int], list[tuple[int, int, np.ndarray, int, int]]] = {}
    manifest_tiles: dict[tuple[int, int], dict[str, int]] | None = None
    manifest_entries: dict[tuple[int, int], dict[str, int]] = {}
    manifest_dirty = False
    if feature_dir is not None and image_id is not None:
        manifest_path = image_feature_manifest_path(feature_dir, image_id)
        with perf_span(
            "prefetch_features_single_scale_image",
            substage="load_feature_manifest",
            extra={"image_id": image_id},
        ):
            manifest_tiles = load_feature_manifest_if_valid(
                feature_dir,
                image_id,
                ps,
                resample_factor,
            )
        if manifest_tiles:
            manifest_entries.update(manifest_tiles)
        if os.path.exists(manifest_path):
            cache_stats["manifest_files_read"] += 1
            cache_stats["manifest_bytes_read"] += _safe_file_size(manifest_path)

    def flush_pending(
        items: list[tuple[int, int, np.ndarray, int, int]],
    ) -> None:
        nonlocal computed_tiles, manifest_dirty
        if not items:
            return
        if batch_size <= 1:
            for y_i, x_i, img_i, h_i, w_i in items:
                with perf_span(
                    "prefetch_features_single_scale_image",
                    substage="extract_single_tile_features",
                    extra={"image_id": image_id, "y": y_i, "x": x_i},
                ):
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
                    feature_path = tile_feature_path(feature_dir, image_id, y_i, x_i)
                    meta = {
                        "ps": ps,
                        "resample_factor": resample_factor,
                        "h_eff": h_i,
                        "w_eff": w_i,
                        "feature_spec_hash": hybrid_feature_spec_hash(),
                    }
                    with perf_span(
                        "prefetch_features_single_scale_image",
                        substage="save_computed_tile_features",
                        extra={"image_id": image_id, "y": y_i, "x": x_i},
                    ):
                        save_tile_features(
                            feats_tile, feature_dir, image_id, y_i, x_i, meta=meta
                        )
                    cache_stats["feature_files_written"] += 1
                    cache_stats["feature_bytes_written"] += _safe_file_size(
                        feature_path
                    )
                cache[(y_i, x_i)] = {
                    "feats": feats_tile,
                    "h_eff": h_i,
                    "w_eff": w_i,
                    "hp": hp_i,
                    "wp": wp_i,
                }
                manifest_entries[(y_i, x_i)] = {
                    "h_eff": int(h_i),
                    "w_eff": int(w_i),
                    "hp": int(hp_i),
                    "wp": int(wp_i),
                }
                manifest_dirty = True
            return

        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            imgs = [item[2] for item in chunk]
            if len(chunk) == 1:
                with perf_span(
                    "prefetch_features_single_scale_image",
                    substage="extract_single_tile_features",
                    extra={"image_id": image_id, "batch_size": 1},
                ):
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
                with perf_span(
                    "prefetch_features_single_scale_image",
                    substage="extract_batch_tile_features",
                    extra={"image_id": image_id, "batch_size": len(chunk)},
                ):
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
                    feature_path = tile_feature_path(feature_dir, image_id, y_i, x_i)
                    meta = {
                        "ps": ps,
                        "resample_factor": resample_factor,
                        "h_eff": h_i,
                        "w_eff": w_i,
                        "feature_spec_hash": hybrid_feature_spec_hash(),
                    }
                    with perf_span(
                        "prefetch_features_single_scale_image",
                        substage="save_computed_tile_features",
                        extra={"image_id": image_id, "y": y_i, "x": x_i},
                    ):
                        save_tile_features(
                            feats_tile, feature_dir, image_id, y_i, x_i, meta=meta
                        )
                    cache_stats["feature_files_written"] += 1
                    cache_stats["feature_bytes_written"] += _safe_file_size(
                        feature_path
                    )
                cache[(y_i, x_i)] = {
                    "feats": feats_tile,
                    "h_eff": h_i,
                    "w_eff": w_i,
                    "hp": hp_i,
                    "wp": wp_i,
                }
                manifest_entries[(y_i, x_i)] = {
                    "h_eff": int(h_i),
                    "w_eff": int(w_i),
                    "hp": int(hp_i),
                    "wp": int(wp_i),
                }
                manifest_dirty = True

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
            manifest_entry = (
                manifest_tiles.get((y, x)) if manifest_tiles is not None else None
            )
            if (
                manifest_entry is not None
                and manifest_entry["hp"] == hp
                and manifest_entry["wp"] == wp
                and manifest_entry["h_eff"] == h_eff
                and manifest_entry["w_eff"] == w_eff
            ):
                feature_path = tile_feature_path(feature_dir, image_id, y, x)
                if feature_dir is not None and not os.path.exists(feature_path):
                    manifest_dirty = True
                    manifest_entries.pop((y, x), None)
                else:
                    manifest_cache_hits += 1
                    with perf_span(
                        "prefetch_features_single_scale_image",
                        substage="manifest_cache_hit",
                        extra={"image_id": image_id, "y": y, "x": x},
                    ):
                        if materialize_cached:
                            checked_path = _manifest_feature_path_if_loadable(
                                feature_path,
                                expected_hp=hp,
                                expected_wp=wp,
                            )
                            if checked_path is None:
                                feats_tile = None
                                manifest_dirty = True
                                manifest_entries.pop((y, x), None)
                            else:
                                feats_tile = np.load(checked_path, mmap_mode="r")
                                feats_tile = np.asarray(feats_tile, dtype=np.float32)
                                cached_tiles += 1
                                cache_stats["feature_files_read"] += 1
                                cache_stats["feature_bytes_read"] += _safe_file_size(
                                    checked_path
                                )
                        else:
                            checked_path = _manifest_feature_path_if_loadable(
                                feature_path,
                                expected_hp=hp,
                                expected_wp=wp,
                            )
                            if checked_path is None:
                                manifest_dirty = True
                                manifest_entries.pop((y, x), None)
                            else:
                                cached_tiles += 1
                                cache_stats["feature_files_read"] += 1
                                cache_stats["feature_bytes_read"] += _safe_file_size(
                                    checked_path
                                )
                                cache[(y, x)] = {
                                    "feats": None,
                                    "feature_path": checked_path,
                                    "h_eff": h_eff,
                                    "w_eff": w_eff,
                                    "hp": hp,
                                    "wp": wp,
                                }
                                continue
            if feats_tile is None:
                with perf_span(
                    "prefetch_features_single_scale_image",
                    substage="validate_cached_tile",
                    extra={"image_id": image_id, "y": y, "x": x},
                ):
                    if materialize_cached:
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
                            cache_stats["feature_files_read"] += 1
                            cache_stats["feature_bytes_read"] += _safe_file_size(
                                tile_feature_path(feature_dir, image_id, y, x)
                            )
                    else:
                        feature_path = inspect_tile_features_if_valid(
                            feature_dir,
                            image_id,
                            y,
                            x,
                            expected_hp=hp,
                            expected_wp=wp,
                            ps=ps,
                            resample_factor=resample_factor,
                        )
                        if feature_path is not None:
                            cached_tiles += 1
                            cache_stats["feature_files_read"] += 1
                            cache_stats["feature_bytes_read"] += _safe_file_size(
                                feature_path
                            )
                            cache[(y, x)] = {
                                "feats": None,
                                "feature_path": feature_path,
                                "h_eff": h_eff,
                                "w_eff": w_eff,
                                "hp": hp,
                                "wp": wp,
                            }
                            manifest_entries[(y, x)] = {
                                "h_eff": int(h_eff),
                                "w_eff": int(w_eff),
                                "hp": int(hp),
                                "wp": int(wp),
                            }
                            manifest_dirty = True
                            continue
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
            if (
                feature_dir is not None
                and image_id is not None
                and hp is not None
                and wp is not None
            ):
                manifest_entries[(y, x)] = {
                    "h_eff": int(h_eff),
                    "w_eff": int(w_eff),
                    "hp": int(hp),
                    "wp": int(wp),
                }

    for items in pending.values():
        flush_pending(items)
    if feature_dir is not None and image_id is not None and manifest_entries:
        if (
            manifest_dirty
            or manifest_tiles is None
            or len(manifest_entries) != len(manifest_tiles)
        ):
            with perf_span(
                "prefetch_features_single_scale_image",
                substage="save_feature_manifest",
                extra={"image_id": image_id, "tile_count": len(manifest_entries)},
            ):
                save_feature_manifest(
                    feature_dir,
                    image_id,
                    ps,
                    resample_factor,
                    manifest_entries,
                )
            cache_stats["manifest_files_written"] += 1
            cache_stats["manifest_bytes_written"] += _safe_file_size(
                image_feature_manifest_path(feature_dir, image_id)
            )
    summary_extra = {
        "cached_tiles": int(cached_tiles),
        "computed_tiles": int(computed_tiles),
        "skipped_tiles": int(skipped_tiles),
        "manifest_cache_hits": int(manifest_cache_hits),
        "cache_hit_ratio": float(cached_tiles / max(1, cached_tiles + computed_tiles)),
        **cache_stats,
    }
    record_performance(
        "prefetch_features_single_scale_image",
        0.0,
        substage="cache_summary",
        extra={"image_id": image_id, **summary_extra},
    )
    time_end("prefetch_features_single_scale_image", t0)
    logger.info(
        "prefetch tiles=%s (cached=%s, computed=%s, skipped=%s, "
        "cache_hit_ratio=%.3f, manifest_hits=%s, read=%.1f MB/%s files, "
        "written=%.1f MB/%s files)",
        len(cache),
        cached_tiles,
        computed_tiles,
        skipped_tiles,
        float(cached_tiles / max(1, cached_tiles + computed_tiles)),
        manifest_cache_hits,
        cache_stats["feature_bytes_read"] / (1024 * 1024),
        cache_stats["feature_files_read"],
        cache_stats["feature_bytes_written"] / (1024 * 1024),
        cache_stats["feature_files_written"],
    )
    return cache
