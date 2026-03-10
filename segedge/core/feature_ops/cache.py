"""Disk-cache helpers for patch features."""

from __future__ import annotations

import json
import os

import numpy as np

from .spec import hybrid_feature_spec_hash

_FEATURE_SPEC_HASH = hybrid_feature_spec_hash()


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

    feats = np.load(fpath)
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
    except OSError:
        return None
    if feats.shape[0] != expected_hp or feats.shape[1] != expected_wp:
        return None
    return fpath
