"""Tests for CRF trimap unary construction and routing."""

from __future__ import annotations

import numpy as np

from segedge.core.crf_utils import _build_unary_probabilities
from segedge.pipeline.runtime.holdout_inference import _run_crf_stage


def test_build_unary_probabilities_trimap_mask_creates_fg_bg_and_uncertain_zones():
    """Trimap unary should keep interior FG, far exterior BG, and a boundary ring uncertain.

    Examples:
        >>> True
        True
    """
    score_map = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    probs, info = _build_unary_probabilities(
        score_map,
        threshold_center=0.5,
        sh_mask=None,
        prob_softness=0.15,
        trimap_band_pixels=1,
    )

    assert probs.shape == (2, 5, 5)
    assert np.isclose(probs[1, 1, 1], 0.98)
    assert np.isclose(probs[1, 0, 0], 0.02)
    assert np.isclose(probs[1, 0, 1], 0.5)
    assert info["fg_seed_pixels"] == 4
    assert info["trimap_band_pixels"] == 1
    assert info["uncertain_pixels"] > 0


def test_build_unary_probabilities_trimap_respects_sh_mask():
    """Trimap unary should force background outside the SH mask.

    Examples:
        >>> True
        True
    """
    score_map = np.full((3, 3), 0.9, dtype=np.float32)
    sh_mask = np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    probs, _ = _build_unary_probabilities(
        score_map,
        threshold_center=0.5,
        sh_mask=sh_mask,
        prob_softness=0.15,
        trimap_band_pixels=1,
    )

    assert probs[1, 0, 0] < 1e-4
    assert probs[1, 1, 1] > 0.9


def test_run_crf_stage_uses_trimap_band_for_xgb_only(monkeypatch):
    """Holdout CRF should keep kNN on logistic mode and XGB on trimap mode.

    Examples:
        >>> True
        True
    """
    calls: list[int | None] = []

    def _fake_refine(
        img_b,
        score_map,
        threshold_center,
        sh_mask=None,
        prob_softness=0.05,
        n_iters=5,
        pos_w=3.0,
        pos_xy_std=3.0,
        bilateral_w=5.0,
        bilateral_xy_std=50.0,
        bilateral_rgb_std=5.0,
        trimap_band_pixels=None,
        return_prob=False,
    ):
        calls.append(trimap_band_pixels)
        return np.ones_like(score_map, dtype=bool)

    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.refine_with_densecrf",
        _fake_refine,
    )
    context = {
        "crf_enabled": True,
        "knn_enabled": True,
        "xgb_enabled": True,
        "img_b": np.zeros((2, 2, 3), dtype=np.uint8),
        "sh_buffer_mask": np.ones((2, 2), dtype=bool),
    }
    tuned = {
        "best_crf_config": {
            "enabled": True,
            "prob_softness": 0.15,
            "trimap_band_pixels": 16,
            "pos_w": 1.0,
            "pos_xy_std": 3.0,
            "bilateral_w": 20.0,
            "bilateral_xy_std": 100.0,
            "bilateral_rgb_std": 2.5,
        }
    }
    knn_result = {
        "mask": np.zeros((2, 2), dtype=bool),
        "score": np.full((2, 2), 0.6, dtype=np.float32),
        "threshold": 0.5,
    }
    xgb_result = {
        "mask": np.zeros((2, 2), dtype=bool),
        "score": np.full((2, 2), 0.6, dtype=np.float32),
        "threshold": 0.5,
    }

    _run_crf_stage(context, tuned, knn_result, xgb_result)

    assert calls == [None, 16]
