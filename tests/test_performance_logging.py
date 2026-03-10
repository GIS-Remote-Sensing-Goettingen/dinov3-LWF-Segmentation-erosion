"""Tests for structured performance logging and lazy feature prefetch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from segedge.core.feature_ops.cache import (
    image_feature_manifest_path,
    load_feature_manifest_if_valid,
    save_tile_features,
    tile_feature_path,
)
from segedge.core.feature_ops.extraction import prefetch_features_single_scale_image
from segedge.core.feature_ops.spec import hybrid_feature_spec_hash
from segedge.core.timing_utils import (
    configure_performance_logging,
    emit_performance_summary,
    perf_span,
    performance_context,
)
from segedge.core.xdboost import (
    _iter_xgb_tile_payloads,
    train_xgb_classifier,
    xgb_score_image_b,
    xgb_score_image_b_legacy,
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

    manifest = load_feature_manifest_if_valid(str(feature_dir), image_id, 16, 1)
    assert manifest is not None
    assert manifest[(0, 0)] == {"h_eff": 32, "w_eff": 32, "hp": 2, "wp": 2}


def test_xgb_score_image_optimized_matches_legacy_on_cached_features(tmp_path):
    """Optimized XGB scoring should match the legacy scorer on cached tiles.

    Examples:
        >>> True
        True
    """
    rng = np.random.default_rng(0)
    feature_dir = tmp_path / "features"
    image_id = "tile_xgb"
    feats_a = rng.normal(size=(2, 2, 4)).astype(np.float32)
    feats_b = rng.normal(size=(2, 2, 4)).astype(np.float32)
    save_tile_features(
        feats_a,
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
    save_tile_features(
        feats_b,
        str(feature_dir),
        image_id,
        32,
        0,
        meta={
            "ps": 16,
            "resample_factor": 1,
            "h_eff": 32,
            "w_eff": 32,
            "feature_spec_hash": hybrid_feature_spec_hash(),
        },
    )
    prefetched = prefetch_features_single_scale_image(
        np.zeros((64, 32, 3), dtype=np.uint8),
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
    X = np.vstack(
        [
            feats_a.reshape(-1, feats_a.shape[-1]),
            feats_b.reshape(-1, feats_b.shape[-1]),
        ]
    ).astype(np.float32)
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    booster = train_xgb_classifier(
        X,
        y,
        use_gpu=False,
        num_boost_round=8,
        verbose_eval=False,
    )
    image = np.zeros((64, 32, 3), dtype=np.uint8)

    score_optimized = xgb_score_image_b(
        image,
        booster,
        16,
        32,
        32,
        str(feature_dir),
        image_id,
        prefetched_tiles=prefetched,
    )
    score_legacy = xgb_score_image_b_legacy(
        image,
        booster,
        16,
        32,
        32,
        str(feature_dir),
        image_id,
        prefetched_tiles=prefetched,
    )

    np.testing.assert_allclose(score_optimized, score_legacy, rtol=1e-6, atol=1e-6)
    assert np.array_equal(score_optimized >= 0.5, score_legacy >= 0.5)


def test_prefetch_manifest_hit_revalidates_lazy_cached_tile(tmp_path, monkeypatch):
    """Manifest hits should still reject unreadable lazy cached arrays.

    Examples:
        >>> True
        True
    """
    feature_dir = tmp_path / "features"
    image_id = "tile_bad_cache"
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
    _ = prefetch_features_single_scale_image(
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
    Path(tile_feature_path(str(feature_dir), image_id, 0, 0)).write_bytes(b"broken")

    monkeypatch.setattr(
        "segedge.core.feature_ops.extraction.extract_patch_features_single_scale",
        lambda *args, **kwargs: (np.full((2, 2, 4), 7.0, dtype=np.float32), 2, 2),
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

    tile_info = cache[(0, 0)]
    assert np.allclose(tile_info["feats"], np.full((2, 2, 4), 7.0, dtype=np.float32))
    assert "feature_path" not in tile_info
    assert image_feature_manifest_path(str(feature_dir), image_id).endswith(
        f"{image_id}_feature_manifest.json"
    )


def test_xgb_tile_payloads_are_streamed_lazily(monkeypatch):
    """Payload preparation should be lazy instead of buffering the whole image.

    Examples:
        >>> True
        True
    """
    prefetched_tiles = {
        (0, 0): {
            "feats": np.ones((1, 1, 2), dtype=np.float32),
            "h_eff": 16,
            "w_eff": 16,
            "hp": 1,
            "wp": 1,
        },
        (16, 0): {
            "feats": np.ones((1, 1, 2), dtype=np.float32),
            "h_eff": 16,
            "w_eff": 16,
            "hp": 1,
            "wp": 1,
        },
    }
    call_count = {"count": 0}

    def _fake_fuse(
        feats_tile, img_c, ps, *, mode, xgb_feature_stats=None, return_layout=False
    ):
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise RuntimeError("second tile reached")
        return feats_tile, None

    monkeypatch.setattr("segedge.core.xdboost.fuse_patch_features", _fake_fuse)

    payload_iter = _iter_xgb_tile_payloads(
        np.zeros((32, 16, 3), dtype=np.uint8),
        16,
        16,
        16,
        None,
        "img",
        prefetched_tiles=prefetched_tiles,
    )

    first_payload = next(payload_iter)
    assert first_payload["y"] == 0
    assert call_count["count"] == 1
    try:
        next(payload_iter)
    except RuntimeError as exc:
        assert "second tile reached" in str(exc)
    else:
        raise AssertionError("expected lazy second-tile evaluation to raise")
