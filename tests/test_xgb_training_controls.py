"""Tests for XGB data-quality and class-balance controls."""

from __future__ import annotations

import numpy as np

import segedge.core.xdboost as xdboost_module
from segedge.core.features import labels_to_patch_masks
from segedge.core.xdboost import (
    _auto_scale_pos_weight,
    _build_binary_sample_weights,
    _group_kfold_index_splits,
    _kfold_index_splits,
    build_xgb_dataset,
)


def test_labels_to_patch_masks_respects_ignore_band() -> None:
    """Patches in the ambiguity band should be excluded from both classes."""
    labels = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    pos, neg = labels_to_patch_masks(
        labels, hp=1, wp=1, pos_frac_thresh=0.6, neg_frac_thresh=0.2
    )
    assert bool(pos[0, 0]) is False
    assert bool(neg[0, 0]) is False


def test_auto_scale_pos_weight_clamps_to_config() -> None:
    """Auto class ratio should be clamped by XGB_CLASS_WEIGHT_MAX."""
    y = np.array([1] + [0] * 100, dtype=np.float32)
    original = xdboost_module.cfg.XGB_CLASS_WEIGHT_MAX
    try:
        xdboost_module.cfg.XGB_CLASS_WEIGHT_MAX = 10.0
        assert _auto_scale_pos_weight(y) == 10.0
    finally:
        xdboost_module.cfg.XGB_CLASS_WEIGHT_MAX = original


def test_build_binary_sample_weights_can_be_disabled() -> None:
    """Disabling sample weights should return None."""
    y = np.array([1, 0, 0], dtype=np.float32)
    original = xdboost_module.cfg.XGB_USE_SAMPLE_WEIGHTS
    try:
        xdboost_module.cfg.XGB_USE_SAMPLE_WEIGHTS = False
        assert _build_binary_sample_weights(y) is None
    finally:
        xdboost_module.cfg.XGB_USE_SAMPLE_WEIGHTS = original


def test_build_xgb_dataset_uses_neg_frac_max_with_prefetched_features() -> None:
    """Dataset builder should keep clean negatives and ignore uncertain patches."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    labels = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    feats = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    prefetched = {
        (0, 0): {
            "h_eff": 4,
            "w_eff": 4,
            "feats": feats,
            "hp": 2,
            "wp": 2,
        }
    }

    X, y = build_xgb_dataset(
        img,
        labels,
        ps=2,
        tile_size=4,
        stride=4,
        feature_dir=None,
        image_id="tile",
        pos_frac=0.5,
        neg_frac_max=0.1,
        max_neg=100,
        context_radius=0,
        prefetched_tiles=prefetched,
    )

    assert X.shape == (3, 3)
    assert y.shape == (3,)
    assert int(np.count_nonzero(y > 0.5)) == 1
    assert int(np.count_nonzero(y <= 0.5)) == 2


def test_build_xgb_dataset_applies_per_tile_negative_cap() -> None:
    """Per-tile negative caps should limit local class domination."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[:2, :2] = 1  # one positive patch, all others negative
    feats = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    prefetched = {
        (0, 0): {
            "h_eff": 8,
            "w_eff": 8,
            "feats": feats,
            "hp": 4,
            "wp": 4,
        }
    }

    original_cap = xdboost_module.cfg.XGB_MAX_NEG_PER_TILE
    try:
        xdboost_module.cfg.XGB_MAX_NEG_PER_TILE = 3
        X, y = build_xgb_dataset(
            img,
            labels,
            ps=2,
            tile_size=8,
            stride=8,
            feature_dir=None,
            image_id="tile",
            pos_frac=0.5,
            neg_frac_max=0.0,
            max_neg=100,
            context_radius=0,
            prefetched_tiles=prefetched,
        )
    finally:
        xdboost_module.cfg.XGB_MAX_NEG_PER_TILE = original_cap

    assert X.shape[0] == 4
    assert int(np.count_nonzero(y > 0.5)) == 1
    assert int(np.count_nonzero(y <= 0.5)) == 3


def test_kfold_index_splits_are_disjoint_and_cover_all_samples() -> None:
    """K-fold helper should produce disjoint folds that cover each sample once."""
    splits = _kfold_index_splits(n_samples=10, n_splits=3, seed=42)
    assert len(splits) == 3
    val_all = np.concatenate([va for _, va in splits])
    assert sorted(val_all.tolist()) == list(range(10))
    for train_idx, val_idx in splits:
        assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_group_kfold_index_splits_keep_groups_isolated() -> None:
    """Grouped folds should never split a group across train and validation."""
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    splits = _group_kfold_index_splits(groups, n_splits=3, seed=42, shuffle=True)
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        assert train_groups.isdisjoint(val_groups)


def test_group_kfold_index_splits_returns_empty_for_single_group() -> None:
    """Grouped k-fold requires at least two unique groups."""
    groups = np.zeros((6,), dtype=np.int32)
    splits = _group_kfold_index_splits(groups, n_splits=3, seed=42, shuffle=True)
    assert splits == []
