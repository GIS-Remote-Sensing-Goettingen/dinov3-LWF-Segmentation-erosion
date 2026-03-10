"""Tests for persisted model-bundle save/load helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest
import xgboost as xgb

from segedge.pipeline.artifacts import (
    load_model_bundle,
    save_model_bundle,
    validate_bundle_compatibility,
)


def _tiny_xgb_booster():
    """Return a small trained XGB booster for artifact tests.

    Examples:
        >>> callable(_tiny_xgb_booster)
        True
    """
    x = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.float32)
    dtrain = xgb.DMatrix(x, label=y)
    return xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
        },
        dtrain,
        num_boost_round=3,
        verbose_eval=False,
    )


def test_model_bundle_roundtrip_xgb_only(tmp_path):
    """XGB-only bundle should roundtrip and avoid bank .npy artifacts.

    Examples:
        >>> True
        True
    """
    bundle_dir = tmp_path / "bundle"
    pos_bank = np.arange(24, dtype=np.float32).reshape(6, 4)  # ignored for persistence
    neg_bank = np.arange(12, dtype=np.float32).reshape(3, 4)  # ignored for persistence
    tuned = {
        "bst": _tiny_xgb_booster(),
        "best_raw_config": {"k": 15, "threshold": 0.2},
        "best_xgb_config": {"k": -1, "threshold": 0.2, "source": "xgb"},
        "best_crf_config": {"enabled": False, "k": 15},
        "shadow_cfg": {
            "weights": [1.0, 1.0, 1.0],
            "threshold": 120,
            "protect_score": 0.5,
        },
        "champion_source": "xgb",
        "roads_penalty": 0.8,
        "xgb_feature_stats": None,
        "feature_layout": {"feature_names": ["f0", "f1", "f2", "f3"]},
        "knn_enabled": False,
        "xgb_enabled": True,
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
    assert os.path.exists(bundle_dir / "xgb_model.json")
    assert not os.path.exists(bundle_dir / "pos_bank.npy")
    assert not os.path.exists(bundle_dir / "neg_bank.npy")
    loaded = load_model_bundle(str(bundle_dir))
    assert loaded["tuned"]["xgb_enabled"] is True
    assert loaded["tuned"]["knn_enabled"] is False
    assert loaded["tuned"]["bst"] is not None
    assert loaded["pos_bank"].size == 0
    assert loaded["neg_bank"] is None
    assert loaded["tuned"]["champion_source"] == "xgb"


def test_model_bundle_compatibility_validation(tmp_path):
    """Compatibility checks should fail with actionable mismatch details.

    Examples:
        >>> True
        True
    """
    bundle_dir = tmp_path / "bundle"
    tuned = {
        "bst": _tiny_xgb_booster(),
        "best_raw_config": {"k": 5, "threshold": 0.1},
        "best_xgb_config": {"k": -1, "threshold": 0.1, "source": "xgb"},
        "best_crf_config": {"enabled": False, "k": 5},
        "shadow_cfg": {
            "weights": [1.0, 1.0, 1.0],
            "threshold": 80,
            "protect_score": 0.4,
        },
        "champion_source": "xgb",
        "roads_penalty": 1.0,
        "xgb_feature_stats": None,
        "feature_layout": None,
        "knn_enabled": False,
        "xgb_enabled": True,
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


def test_model_bundle_requires_xgb_model_file(tmp_path):
    """Loading should fail when xgb_model artifact is missing.

    Examples:
        >>> True
        True
    """
    bundle_dir = tmp_path / "bundle"
    tuned = {
        "bst": _tiny_xgb_booster(),
        "best_raw_config": {"k": 5, "threshold": 0.1},
        "best_xgb_config": {"k": -1, "threshold": 0.1, "source": "xgb"},
        "best_crf_config": {"enabled": False, "k": 5},
        "shadow_cfg": {
            "weights": [1.0, 1.0, 1.0],
            "threshold": 80,
            "protect_score": 0.4,
        },
        "champion_source": "xgb",
        "roads_penalty": 1.0,
        "xgb_feature_stats": None,
        "feature_layout": None,
        "knn_enabled": False,
        "xgb_enabled": True,
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
    os.remove(bundle_dir / "xgb_model.json")
    with pytest.raises(ValueError, match="missing xgb_model artifact"):
        load_model_bundle(str(bundle_dir))
