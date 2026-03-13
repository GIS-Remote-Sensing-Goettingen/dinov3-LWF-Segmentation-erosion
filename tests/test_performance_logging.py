"""Tests for structured performance logging and lazy feature prefetch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

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
    xgb_score_image_b_streaming,
)
from segedge.pipeline.runtime import roads as roads_module
from segedge.pipeline.runtime.holdout_inference import (
    _load_holdout_tile_context,
    infer_on_holdout,
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


def test_xgb_score_image_streaming_matches_cached_prefetch_path(
    tmp_path,
    monkeypatch,
):
    """Streaming XGB scoring should match the cached-feature scorer.

    Examples:
        >>> True
        True
    """
    rng = np.random.default_rng(2)
    feature_dir = tmp_path / "features"
    image_id = "tile_stream"
    image = np.zeros((64, 32, 3), dtype=np.uint8)
    image[:32, :, :] = 32
    image[32:, :, :] = 96
    feats_top = rng.normal(size=(2, 2, 4)).astype(np.float32)
    feats_bottom = rng.normal(size=(2, 2, 4)).astype(np.float32)
    save_tile_features(
        feats_top,
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
        feats_bottom,
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
        image,
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
            feats_top.reshape(-1, feats_top.shape[-1]),
            feats_bottom.reshape(-1, feats_bottom.shape[-1]),
        ]
    ).astype(np.float32)
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.float32)
    booster = train_xgb_classifier(
        X,
        y,
        use_gpu=False,
        num_boost_round=8,
        verbose_eval=False,
    )

    def _fake_extract_single(img_hw3, *args, **kwargs):
        value = int(img_hw3[0, 0, 0])
        if value == 32:
            return feats_top.copy(), 2, 2
        if value == 96:
            return feats_bottom.copy(), 2, 2
        raise AssertionError(f"unexpected tile marker {value}")

    def _fake_extract_batch(images_hw3, *args, **kwargs):
        feats = []
        for img_hw3 in images_hw3:
            feats_tile, _, _ = _fake_extract_single(img_hw3)
            feats.append(feats_tile)
        return np.stack(feats, axis=0), 2, 2

    monkeypatch.setattr(
        "segedge.core.xdboost.extract_patch_features_single_scale",
        _fake_extract_single,
    )
    monkeypatch.setattr(
        "segedge.core.xdboost.extract_patch_features_batch_single_scale",
        _fake_extract_batch,
    )

    score_cached = xgb_score_image_b(
        image,
        booster,
        16,
        32,
        32,
        str(feature_dir),
        image_id,
        prefetched_tiles=prefetched,
    )
    score_streaming = xgb_score_image_b_streaming(
        image,
        booster,
        model=None,
        processor=None,
        device=None,
        ps=16,
        tile_size=32,
        stride=32,
    )

    np.testing.assert_allclose(score_streaming, score_cached, rtol=1e-6, atol=1e-6)
    assert np.array_equal(score_streaming >= 0.5, score_cached >= 0.5)


def test_xgb_score_image_streaming_respects_feature_batch_size(monkeypatch):
    """Streaming extraction should not exceed the configured feature batch size.

    Examples:
        >>> True
        True
    """
    image = np.zeros((128, 32, 3), dtype=np.uint8)
    for idx, y in enumerate(range(0, 128, 32), start=1):
        image[y : y + 32, :, :] = idx

    extract_calls: list[int] = []

    def _fake_extract_single(img_hw3, *args, **kwargs):
        value = float(img_hw3[0, 0, 0])
        feats_tile = np.full((2, 2, 4), value, dtype=np.float32)
        return feats_tile, 2, 2

    def _fake_extract_batch(images_hw3, *args, **kwargs):
        extract_calls.append(len(images_hw3))
        feats = [_fake_extract_single(img_hw3)[0] for img_hw3 in images_hw3]
        return np.stack(feats, axis=0), 2, 2

    def _fake_fuse(
        feats_tile, img_c, ps, *, mode, xgb_feature_stats=None, return_layout=False
    ):
        return feats_tile, None

    class _Booster:
        def inplace_predict(self, feats):
            return np.full((feats.shape[0],), 0.5, dtype=np.float32)

    monkeypatch.setattr(cfg.runtime, "feature_batch_size", 1)
    monkeypatch.setattr(
        "segedge.core.xdboost.extract_patch_features_single_scale",
        _fake_extract_single,
    )
    monkeypatch.setattr(
        "segedge.core.xdboost.extract_patch_features_batch_single_scale",
        _fake_extract_batch,
    )
    monkeypatch.setattr("segedge.core.xdboost.fuse_patch_features", _fake_fuse)

    score = xgb_score_image_b_streaming(
        image,
        _Booster(),
        model=None,
        processor=None,
        device=None,
        ps=16,
        tile_size=32,
        stride=32,
    )

    assert score.shape == (128, 32)
    assert extract_calls == []


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


def test_load_holdout_tile_context_logs_child_spans_and_metadata(
    tmp_path,
    monkeypatch,
):
    """Holdout context loading should expose child spans and workload metadata.

    Examples:
        >>> True
        True
    """
    perf_path = tmp_path / "performance.jsonl"
    configure_performance_logging(str(perf_path), run_id="run_999")

    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.load_b_tile_context",
        lambda *args, **kwargs: (
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            None,
            np.ones((4, 4), dtype=bool),
            np.array(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            5.0,
            0.2,
        ),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._get_roads_mask",
        lambda *args, **kwargs: np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        ),
    )

    with performance_context(
        phase="holdout_inference",
        tile="tile_a.tif",
        image_id="tile_a",
    ):
        ctx = _load_holdout_tile_context(
            "tile_a.tif",
            gt_vector_paths=[],
            tuned={"roads_penalty": 0.7, "xgb_enabled": True, "knn_enabled": False},
        )

    assert ctx["roads_penalty"] == 0.7
    rows = [
        json.loads(line)
        for line in perf_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    infer_rows = [
        row
        for row in rows
        if row.get("kind") == "span" and row.get("stage") == "infer_on_holdout"
    ]
    substage_map = {row.get("substage"): row for row in infer_rows}
    for substage in (
        "load_holdout_tile_context",
        "resolve_runtime_toggles",
        "load_roads_mask",
        "finalize_context",
    ):
        assert substage in substage_map

    finalize_extra = substage_map["finalize_context"]["extra"]
    assert finalize_extra["source_label_positive_pixels"] == 2
    assert finalize_extra["buffer_positive_pixels"] == 4
    assert finalize_extra["roads_positive_pixels"] == 2
    assert finalize_extra["source_label_coverage_ratio"] > 0.0
    assert finalize_extra["roads_coverage_ratio"] > 0.0


def test_roads_mask_logs_cache_and_rasterization_spans(tmp_path, monkeypatch):
    """Road-mask diagnostics should expose cache lookup and rasterization spans.

    Examples:
        >>> True
        True
    """
    perf_path = tmp_path / "performance.jsonl"
    configure_performance_logging(str(perf_path), run_id="run_999")
    roads_module._ROADS_MASK_CACHE.clear()
    roads_module._ROADS_INDEX_CACHE.clear()

    tile_path = tmp_path / "tile.tif"
    with rasterio.open(
        tile_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        dst.write(np.zeros((1, 4, 4), dtype=np.uint8))

    roads_path = tmp_path / "roads.shp"
    roads_path.write_bytes(b"")
    monkeypatch.setattr(cfg.io.paths, "roads_mask_path", str(roads_path))
    monkeypatch.setattr(cfg.io.paths, "feature_dir", str(tmp_path / "feature_cache"))
    monkeypatch.setattr(cfg.io.paths, "output_dir", str(tmp_path / "output"))

    geom = box(0.0, 0.0, 2.0, 4.0)

    class _Tree:
        def query(self, query_geom):
            assert query_geom is not None
            return [0]

    monkeypatch.setattr(
        roads_module,
        "_get_roads_index",
        lambda tile_crs: (_Tree(), [geom]),
    )

    with performance_context(
        phase="holdout_inference",
        tile=str(tile_path),
        image_id="tile",
    ):
        mask = roads_module._get_roads_mask(str(tile_path), 1, target_shape=(4, 4))

    assert mask is not None
    assert mask.shape == (4, 4)
    rows = [
        json.loads(line)
        for line in perf_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    roads_rows = [
        row
        for row in rows
        if row.get("kind") == "span" and row.get("stage") == "roads_mask"
    ]
    substages = {row.get("substage") for row in roads_rows}
    for substage in (
        "open_tile_metadata",
        "disk_cache_lookup",
        "index_lookup",
        "tree_query",
        "candidate_filter",
        "rasterize",
        "disk_cache_write",
    ):
        assert substage in substages
    raster_row = next(row for row in roads_rows if row.get("substage") == "rasterize")
    assert raster_row["extra"]["candidate_geometry_count"] == 1
    assert raster_row["extra"]["intersecting_geometry_count"] == 1


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
    metadata_row = next(
        row for row in infer_rows if row.get("substage") == "tile_workload_metadata"
    )
    assert metadata_row["extra"]["xgb_patch_rows"] == 0
