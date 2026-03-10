"""Disk-cache helpers for patch features."""

from __future__ import annotations

import json
import os

import numpy as np

from .spec import hybrid_feature_spec_hash

_FEATURE_SPEC_HASH = hybrid_feature_spec_hash()
_MANIFEST_CACHE: dict[
    tuple[str, str, int, int, str], dict[tuple[int, int], dict[str, int]]
] = {}


def tile_feature_path(feature_dir: str, image_id: str, y: int, x: int) -> str:
    """Return the canonical path for a tile's feature array.

    Examples:
        >>> tile_feature_path("/tmp", "img", 1, 2).endswith("img_y1_x2_features.npy")
        True
    """
    fname = f"{image_id}_y{y}_x{x}_features.npy"
    return os.path.join(feature_dir, fname)


def tile_feature_meta_path(feature_dir: str, image_id: str, y: int, x: int) -> str:
    """Return the sidecar JSON path for feature metadata.

    Examples:
        >>> tile_feature_meta_path("/tmp", "img", 1, 2).endswith("img_y1_x2_features.json")
        True
    """
    fname = f"{image_id}_y{y}_x{x}_features.json"
    return os.path.join(feature_dir, fname)


def image_feature_manifest_path(feature_dir: str, image_id: str) -> str:
    """Return the per-image manifest path for cached feature tiles.

    Examples:
        >>> image_feature_manifest_path("/tmp", "img").endswith("img_feature_manifest.json")
        True
    """
    return os.path.join(feature_dir, f"{image_id}_feature_manifest.json")


def _manifest_cache_key(
    feature_dir: str,
    image_id: str,
    ps: int,
    resample_factor: int,
) -> tuple[str, str, int, int, str]:
    return (
        os.path.abspath(feature_dir),
        image_id,
        int(ps),
        int(resample_factor),
        _FEATURE_SPEC_HASH,
    )


def load_feature_manifest_if_valid(
    feature_dir: str,
    image_id: str,
    ps: int,
    resample_factor: int,
) -> dict[tuple[int, int], dict[str, int]] | None:
    """Load a per-image feature manifest when its global metadata matches.

    Examples:
        >>> load_feature_manifest_if_valid("/tmp", "missing", 16, 1) is None
        True
    """
    cache_key = _manifest_cache_key(feature_dir, image_id, ps, resample_factor)
    cached = _MANIFEST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    manifest_path = image_feature_manifest_path(feature_dir, image_id)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("image_id") != image_id:
        return None
    if payload.get("ps") != ps or payload.get("resample_factor") != resample_factor:
        return None
    if payload.get("feature_spec_hash") != _FEATURE_SPEC_HASH:
        return None
    tiles_raw = payload.get("tiles", {})
    if not isinstance(tiles_raw, dict):
        return None
    tiles: dict[tuple[int, int], dict[str, int]] = {}
    for key, entry in tiles_raw.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        try:
            y_str, x_str = key.split(",", 1)
            y = int(y_str)
            x = int(x_str)
            tiles[(y, x)] = {
                "h_eff": int(entry["h_eff"]),
                "w_eff": int(entry["w_eff"]),
                "hp": int(entry["hp"]),
                "wp": int(entry["wp"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    _MANIFEST_CACHE[cache_key] = tiles
    return tiles


def save_feature_manifest(
    feature_dir: str,
    image_id: str,
    ps: int,
    resample_factor: int,
    tiles: dict[tuple[int, int], dict[str, int]],
) -> None:
    """Persist a per-image cache manifest for fast validation on cache hits.

    Examples:
        >>> callable(save_feature_manifest)
        True
    """
    os.makedirs(feature_dir, exist_ok=True)
    payload = {
        "feature_spec_hash": _FEATURE_SPEC_HASH,
        "image_id": image_id,
        "ps": int(ps),
        "resample_factor": int(resample_factor),
        "tiles": {
            f"{int(y)},{int(x)}": {
                "h_eff": int(entry["h_eff"]),
                "w_eff": int(entry["w_eff"]),
                "hp": int(entry["hp"]),
                "wp": int(entry["wp"]),
            }
            for (y, x), entry in sorted(tiles.items())
        },
    }
    manifest_path = image_feature_manifest_path(feature_dir, image_id)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
    _MANIFEST_CACHE[_manifest_cache_key(feature_dir, image_id, ps, resample_factor)] = {
        (int(y), int(x)): {
            "h_eff": int(entry["h_eff"]),
            "w_eff": int(entry["w_eff"]),
            "hp": int(entry["hp"]),
            "wp": int(entry["wp"]),
        }
        for (y, x), entry in tiles.items()
    }


def save_tile_features(
    feats_tile: np.ndarray,
    feature_dir: str,
    image_id: str,
    y: int,
    x: int,
    meta: dict | None = None,
):
    """Persist a tile's features to disk (and optional metadata).

    Examples:
        >>> callable(save_tile_features)
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

    Examples:
        >>> load_tile_features_if_valid("/tmp", "missing", 0, 0, 1, 1, 16, 1) is None
        True
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
    if meta.get("feature_spec_hash") != _FEATURE_SPEC_HASH:
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None

    try:
        feats = np.load(fpath)
    except (OSError, ValueError):
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None
    if feats.shape[0] != expected_hp or feats.shape[1] != expected_wp:
        try:
            os.remove(fpath)
            os.remove(mpath)
        except OSError:
            pass
        return None
    return feats


def inspect_tile_features_if_valid(
    feature_dir: str,
    image_id: str,
    y: int,
    x: int,
    expected_hp: int,
    expected_wp: int,
    ps: int,
    resample_factor: int,
) -> str | None:
    """Return the feature path when cache metadata and shape are valid.

    Examples:
        >>> inspect_tile_features_if_valid("/tmp", "missing", 0, 0, 1, 1, 16, 1) is None
        True
    """
    fpath = tile_feature_path(feature_dir, image_id, y, x)
    if not os.path.exists(fpath):
        return None
    mpath = tile_feature_meta_path(feature_dir, image_id, y, x)
    if not os.path.exists(mpath):
        return None
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("ps") != ps or meta.get("resample_factor") != resample_factor:
        return None
    if meta.get("feature_spec_hash") != _FEATURE_SPEC_HASH:
        return None
    try:
        feats = np.load(fpath, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if feats.shape[0] != expected_hp or feats.shape[1] != expected_wp:
        return None
    return fpath
