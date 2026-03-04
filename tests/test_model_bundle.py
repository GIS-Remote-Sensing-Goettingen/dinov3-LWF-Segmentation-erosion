"""Tests for persisted model-bundle save/load helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from segedge.pipeline.artifacts import (
    load_model_bundle,
    save_model_bundle,
    validate_bundle_compatibility,
)


def test_model_bundle_roundtrip_knn_only(tmp_path):
    """kNN-only bundle should roundtrip without XGB model artifact.

    Examples:
        >>> True
        True
    """
    bundle_dir = tmp_path / "bundle"
    pos_bank = np.arange(24, dtype=np.float32).reshape(6, 4)
    neg_bank = np.arange(12, dtype=np.float32).reshape(3, 4)
    tuned = {
        "bst": None,
        "best_raw_config": {"k": 15, "threshold": 0.2},
        "best_xgb_config": {"k": -1, "threshold": 0.2, "source": "xgb_disabled"},
        "best_crf_config": {"enabled": False, "k": 15},
        "shadow_cfg": {
            "weights": [1.0, 1.0, 1.0],
            "threshold": 120,
            "protect_score": 0.5,
        },
        "champion_source": "raw",
        "roads_penalty": 0.8,
        "xgb_feature_stats": None,
        "feature_layout": {"feature_names": ["f0", "f1", "f2", "f3"]},
        "knn_enabled": True,
        "xgb_enabled": False,
        "crf_enabled": False,
    }
    meta = save_model_bundle(
        str(bundle_dir),
        tuned,
        pos_bank,
        neg_bank,
        model_name="dummy-backbone",
        patch_size=16,
        resample_factor=1,
        tile_size=512,
        stride=256,
        context_radius=0,
    )
    assert os.path.exists(meta["manifest"])
    loaded = load_model_bundle(str(bundle_dir))
    assert loaded["tuned"]["xgb_enabled"] is False
    assert loaded["tuned"]["bst"] is None
    assert loaded["pos_bank"].shape == pos_bank.shape
    assert loaded["neg_bank"].shape == neg_bank.shape
    assert loaded["tuned"]["best_raw_config"]["k"] == 15


def test_model_bundle_compatibility_validation(tmp_path):
    """Compatibility checks should fail with actionable mismatch details.

    Examples:
        >>> True
        True
    """
    bundle_dir = tmp_path / "bundle"
    tuned = {
        "bst": None,
        "best_raw_config": {"k": 5, "threshold": 0.1},
        "best_xgb_config": {"k": -1, "threshold": 0.1, "source": "xgb_disabled"},
        "best_crf_config": {"enabled": False, "k": 5},
        "shadow_cfg": {
            "weights": [1.0, 1.0, 1.0],
            "threshold": 80,
            "protect_score": 0.4,
        },
        "champion_source": "raw",
        "roads_penalty": 1.0,
        "xgb_feature_stats": None,
        "feature_layout": None,
        "knn_enabled": True,
        "xgb_enabled": False,
        "crf_enabled": False,
    }
    save_model_bundle(
        str(bundle_dir),
        tuned,
        np.ones((4, 8), dtype=np.float32),
        None,
        model_name="bundle-backbone",
        patch_size=16,
        resample_factor=1,
        tile_size=512,
        stride=256,
        context_radius=1,
    )
    loaded = load_model_bundle(str(bundle_dir))
    validate_bundle_compatibility(
        loaded["manifest"],
        model_name="bundle-backbone",
        patch_size=16,
        resample_factor=1,
        tile_size=512,
        stride=256,
        context_radius=1,
    )
    with pytest.raises(ValueError, match="runtime.tile_size"):
        validate_bundle_compatibility(
            loaded["manifest"],
            model_name="bundle-backbone",
            patch_size=16,
            resample_factor=1,
            tile_size=1024,
            stride=256,
            context_radius=1,
        )
