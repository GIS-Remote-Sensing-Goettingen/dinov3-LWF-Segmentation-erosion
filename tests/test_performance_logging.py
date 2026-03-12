"""Tests for structured performance logging and lazy feature prefetch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from segedge.core.config_loader import cfg
from segedge.core.feature_ops.cache import (
    image_feature_manifest_path,
    load_feature_manifest_if_valid,
    save_tile_features,
    tile_feature_path,
)
from segedge.core.feature_ops.extraction import prefetch_features_single_scale_image
from segedge.core.feature_ops.fusion import fuse_patch_features
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
from segedge.pipeline.runtime.holdout_inference import infer_on_holdout


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


def test_prefetch_features_logs_cache_summary_metadata(tmp_path):
    """Prefetch should emit cache summary metadata for later cost analysis.

    Examples:
        >>> True
        True
    """
    perf_path = tmp_path / "performance.jsonl"
    configure_performance_logging(str(perf_path), run_id="run_999")
    feature_dir = tmp_path / "features"
    image_id = "tile_summary"
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

    rows = [
        json.loads(line)
        for line in perf_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary_rows = [
        row
        for row in rows
        if row["stage"] == "prefetch_features_single_scale_image"
        and row.get("substage") == "cache_summary"
    ]
    assert len(summary_rows) == 1
    extra = summary_rows[0]["extra"]
    assert extra["image_id"] == image_id
    assert extra["cached_tiles"] == 1
    assert extra["computed_tiles"] == 0
    assert extra["feature_files_read"] >= 1
    assert extra["feature_bytes_read"] > 0


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


def test_fuse_patch_features_fast_xgb_matches_generic_path():
    """The fast XGB fusion path should match the generic layout-producing path.

    Examples:
        >>> True
        True
    """
    rng = np.random.default_rng(1)
    dino = rng.normal(size=(2, 2, 4)).astype(np.float32)
    img = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)

    fused_fast, layout_fast = fuse_patch_features(
        dino,
        img,
        16,
        mode="xgb",
        return_layout=False,
    )
    fused_generic, layout_generic = fuse_patch_features(
        dino,
        img,
        16,
        mode="xgb",
        return_layout=True,
    )

    assert layout_fast is None
    assert layout_generic is not None
    np.testing.assert_allclose(fused_fast, fused_generic, rtol=1e-6, atol=1e-6)


def test_xgb_score_image_batches_multiple_small_tiles_into_one_predict(
    tmp_path,
    monkeypatch,
):
    """Small tiles should be combined into fewer XGB predict calls.

    Examples:
        >>> True
        True
    """
    feature_dir = tmp_path / "features"
    image_id = "tile_batch"
    image = np.zeros((64, 16, 3), dtype=np.uint8)
    prefetched = {}
    for y in range(0, 64, 16):
        prefetched[(y, 0)] = {
            "feats": np.ones((1, 1, 4), dtype=np.float32),
            "h_eff": 16,
            "w_eff": 16,
            "hp": 1,
            "wp": 1,
        }

    class _Booster:
        def __init__(self):
            self.calls = []

        def inplace_predict(self, feats):
            self.calls.append(int(feats.shape[0]))
            return np.full((feats.shape[0],), 0.25, dtype=np.float32)

    booster = _Booster()
    monkeypatch.setattr(cfg.runtime, "feature_batch_size", 1)

    score = xgb_score_image_b(
        image,
        booster,
        16,
        16,
        16,
        str(feature_dir),
        image_id,
        prefetched_tiles=prefetched,
    )

    assert score.shape == (64, 16)
    assert booster.calls == [4]


def test_infer_on_holdout_logs_prefetch_features_separately(tmp_path, monkeypatch):
    """Infer-on-holdout should log feature prefetch separately from tile context.

    Examples:
        >>> True
        True
    """
    perf_path = tmp_path / "performance.jsonl"
    configure_performance_logging(str(perf_path), run_id="run_999")

    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._load_holdout_tile_context",
        lambda *args, **kwargs: {
            "img_b": np.zeros((2, 2, 3), dtype=np.uint8),
            "labels_sh": np.zeros((2, 2), dtype=np.uint8),
            "gt_mask_eval": np.zeros((2, 2), dtype=bool),
            "gt_available": False,
            "gt_weight": 0.0,
            "sh_buffer_mask": np.ones((2, 2), dtype=bool),
            "buffer_m": 5.0,
            "pixel_size_m": 0.2,
            "image_id_b": "tile_a",
            "roads_mask": None,
            "roads_penalty": 1.0,
            "xgb_feature_stats": None,
            "knn_enabled": False,
            "xgb_enabled": True,
            "crf_enabled": True,
        },
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._prefetch_holdout_features",
        lambda *args, **kwargs: {
            (0, 0): {"feats": np.ones((1, 1, 2), dtype=np.float32)}
        },
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._compute_knn_stream",
        lambda *args, **kwargs: {
            "score": None,
            "score_raw": None,
            "threshold": None,
            "mask": np.zeros((2, 2), dtype=bool),
        },
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._compute_xgb_stream",
        lambda *args, **kwargs: {
            "score": np.full((2, 2), 0.8, dtype=np.float32),
            "score_raw": np.full((2, 2), 0.8, dtype=np.float32),
            "threshold": 0.5,
            "mask": np.ones((2, 2), dtype=bool),
        },
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._run_crf_stage",
        lambda *args, **kwargs: (
            np.zeros((2, 2), dtype=bool),
            np.ones((2, 2), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._apply_shadow_mask",
        lambda *args, **kwargs: np.ones((2, 2), dtype=bool),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._build_metrics_map",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.filter_novel_proposals",
        lambda *args, **kwargs: {
            "candidate_mask": np.zeros((2, 2), dtype=bool),
            "accepted_inside_mask": np.zeros((2, 2), dtype=bool),
            "accepted_outside_mask": np.zeros((2, 2), dtype=bool),
            "rejected_mask": np.zeros((2, 2), dtype=bool),
            "records": [],
        },
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._build_proposal_masks",
        lambda *args, **kwargs: {
            "candidate_mask": np.zeros((2, 2), dtype=bool),
            "candidate_inside_mask": np.zeros((2, 2), dtype=bool),
            "evaluated_outside_mask": np.zeros((2, 2), dtype=bool),
            "accepted_mask": np.zeros((2, 2), dtype=bool),
            "accepted_inside_mask": np.zeros((2, 2), dtype=bool),
            "accepted_outside_mask": np.zeros((2, 2), dtype=bool),
            "rejected_mask": np.zeros((2, 2), dtype=bool),
        },
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._export_proposal_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._compute_probability_and_diagnostics",
        lambda *args, **kwargs: (
            np.full((2, 2), 0.8, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._save_holdout_plots",
        lambda *args, **kwargs: None,
    )

    with performance_context(
        phase="holdout_inference",
        tile="tile_a.tif",
        image_id="tile_a",
    ):
        infer_on_holdout(
            holdout_path="tile_a.tif",
            gt_vector_paths=[],
            model=None,
            processor=None,
            device=None,
            pos_bank=np.zeros((0, 2), dtype=np.float32),
            neg_bank=np.zeros((0, 2), dtype=np.float32),
            tuned={
                "shadow_cfg": {"weights": [1.0, 1.0, 1.0], "threshold": 0.0},
                "xgb_enabled": True,
                "knn_enabled": False,
                "crf_enabled": True,
            },
            ps=16,
            tile_size=32,
            stride=32,
            feature_dir=None,
            shape_dir=str(tmp_path / "shapes"),
            plot_dir=str(tmp_path / "plots"),
            context_radius=0,
            save_plots=False,
        )

    rows = [
        json.loads(line) for line in perf_path.read_text(encoding="utf-8").splitlines()
    ]
    infer_rows = [
        row
        for row in rows
        if row.get("kind") == "span" and row.get("stage") == "infer_on_holdout"
    ]
    substages = {row.get("substage") for row in infer_rows}
    assert "load_context" in substages
    assert "prefetch_features" in substages
