"""Tests for explainability utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from segedge.core.explainability import (
    append_xai_summary_csv,
    build_dim_activation_map,
    get_xgb_importance_dict,
    select_holdout_tiles_for_xai,
    summarize_knn_signals,
)


class _FakeBooster:
    def __init__(self):
        """Create deterministic feature-importance maps.

        Examples:
            >>> isinstance(_FakeBooster(), _FakeBooster)
            True
        """
        self._maps = {
            "gain": {"f0": 3.0, "f3": 1.0, "f1": 2.0},
            "weight": {"f0": 10.0, "f1": 5.0},
            "cover": {"f0": 4.0, "f3": 2.0},
        }

    def get_score(self, importance_type="gain"):
        """Return map for a requested importance type.

        Examples:
            >>> _FakeBooster().get_score("gain")["f0"]
            3.0
        """
        return self._maps.get(importance_type, {})


def test_get_xgb_importance_dict_orders_dims_by_gain() -> None:
    """XGB importance rows should be sorted by descending gain.

    Examples:
        >>> True
        True
    """
    out = get_xgb_importance_dict(_FakeBooster(), top_k=3)
    assert out["top_dims"] == [0, 1, 3]
    assert out["xgb_gain_share_top5"] > 0.0
    assert len(out["importance"]) == 3


def test_build_dim_activation_map_shape_and_values() -> None:
    """Activation map should preserve target image shape.

    Examples:
        >>> True
        True
    """
    feats = np.ones((2, 2, 4), dtype=np.float32)
    prefetched = {(0, 0): {"feats": feats, "hp": 2, "wp": 2, "h_eff": 16, "w_eff": 16}}
    out = build_dim_activation_map(prefetched, image_shape=(16, 16), dims=[0, 1])
    assert out.shape == (16, 16)
    assert np.allclose(out, 1.0, atol=1e-6)


def test_select_holdout_tiles_for_xai_is_deterministic() -> None:
    """Holdout tile selection should be deterministic for a fixed seed.

    Examples:
        >>> True
        True
    """
    holdout = [f"tile_{i}.tif" for i in range(20)]
    s1 = select_holdout_tiles_for_xai(holdout, cap_enabled=True, cap=10, seed=42)
    s2 = select_holdout_tiles_for_xai(holdout, cap_enabled=True, cap=10, seed=42)
    assert s1 == s2
    assert len(s1) == 10
    assert s1.issubset(set(holdout))


def test_append_xai_summary_csv_writes_header_once(tmp_path: Path) -> None:
    """CSV append should emit a single header row.

    Examples:
        >>> True
        True
    """
    out = tmp_path / "xai_summary.csv"
    row = {
        "timestamp_utc": "2026-02-10T00:00:00",
        "stage": "validation",
        "image_id": "tile_a",
        "tile_path": "tile_a.tif",
        "xgb_top_dim_1": 1,
        "xgb_top_dim_2": 2,
        "xgb_top_dim_3": 3,
        "xgb_top_dim_4": 4,
        "xgb_top_dim_5": 5,
        "xgb_gain_share_top5": 0.8,
        "knn_score_mean": 0.3,
        "knn_score_p95": 0.9,
        "xgb_score_mean": 0.4,
        "xgb_score_p95": 0.95,
        "champion_fg_ratio": 0.2,
        "xai_total_s": 1.23,
    }
    append_xai_summary_csv(str(out), row)
    append_xai_summary_csv(str(out), row)

    with open(out, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 3
    assert rows[0][0] == "timestamp_utc"


def test_summarize_knn_signals_with_threshold() -> None:
    """kNN summary should include threshold-based buffered ratio.

    Examples:
        >>> True
        True
    """
    score = np.array([[0.1, 0.8], [0.7, 0.2]], dtype=np.float32)
    saliency = np.array([[0.2, 0.3], [0.4, 0.1]], dtype=np.float32)
    buffer_mask = np.array([[True, True], [False, False]])
    out = summarize_knn_signals(score, saliency, buffer_mask, threshold=0.5)
    assert out["buffered_positive_ratio"] == 0.5
    assert out["score_distribution"]["p95"] > 0.0
    assert out["saliency_distribution"] is not None
