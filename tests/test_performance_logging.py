"""Tests for structured performance logging and lazy feature prefetch."""

from __future__ import annotations

import json

import numpy as np

from segedge.core.feature_ops.cache import save_tile_features
from segedge.core.feature_ops.extraction import prefetch_features_single_scale_image
from segedge.core.feature_ops.spec import hybrid_feature_spec_hash
from segedge.core.timing_utils import (
    configure_performance_logging,
    emit_performance_summary,
    perf_span,
    performance_context,
)


def test_performance_log_writes_contextual_jsonl(tmp_path):
    """Structured performance spans should include tile and phase context.

    Examples:
        >>> True
        True
    """
    perf_path = tmp_path / "performance.jsonl"
    configure_performance_logging(str(perf_path), run_id="run_999")

    with performance_context(
        phase="holdout_inference",
        tile="tile_a.tif",
        image_id="tile_a",
    ):
        with perf_span("demo_stage", substage="unit_test", extra={"count": 3}):
            _ = 1 + 1
        emit_performance_summary("unit_test")

    lines = perf_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    span_record = json.loads(lines[0])
    summary_record = json.loads(lines[1])

    assert span_record["kind"] == "span"
    assert span_record["phase"] == "holdout_inference"
    assert span_record["tile"] == "tile_a.tif"
    assert span_record["image_id"] == "tile_a"
    assert span_record["stage"] == "demo_stage"
    assert span_record["substage"] == "unit_test"
    assert span_record["extra"] == {"count": 3}
    assert span_record["run_id"] == "run_999"
    assert summary_record["kind"] == "summary"
    assert summary_record["reason"] == "unit_test"


def test_prefetch_features_can_keep_cached_tiles_lazy(tmp_path):
    """XGB-only inference should reuse valid cached features lazily.

    Examples:
        >>> True
        True
    """
    feature_dir = tmp_path / "features"
    image_id = "tile_a"
    feats = np.ones((2, 2, 4), dtype=np.float32)
    save_tile_features(
        feats,
        str(feature_dir),
        image_id,
        0,
        0,
        meta={
            "ps": 16,
            "resample_factor": 1,
            "h_eff": 32,
            "w_eff": 32,
            "feature_spec_hash": hybrid_feature_spec_hash(),
        },
    )

    cache = prefetch_features_single_scale_image(
        np.zeros((32, 32, 3), dtype=np.uint8),
        model=None,
        processor=None,
        device=None,
        ps=16,
        tile_size=32,
        stride=32,
        aggregate_layers=None,
        feature_dir=str(feature_dir),
        image_id=image_id,
        materialize_cached=False,
    )

    assert (0, 0) in cache
    tile_info = cache[(0, 0)]
    assert tile_info["feats"] is None
    assert tile_info["feature_path"].endswith("tile_a_y0_x0_features.npy")
    assert tile_info["hp"] == 2
    assert tile_info["wp"] == 2
