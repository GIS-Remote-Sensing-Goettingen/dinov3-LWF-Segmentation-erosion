"""Unit tests for auto-split mode and inference-tile capping."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.errors import GEOSException
from shapely.geometry import box

import segedge.pipeline.common as common_module
from segedge.pipeline.common import (
    AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT,
    AUTO_SPLIT_MODE_LEGACY,
    _bbox_overlap_ratio,
    _cap_inference_tiles,
    _split_tiles_from_gt_presence,
    resolve_source_train_gt_vectors,
    resolve_source_training_labels,
    run_source_validation_anti_leak_checks,
    tile_has_gt_overlap,
)


def test_cap_inference_tiles_disabled_returns_sorted() -> None:
    """Capping disabled should return all holdout tiles in sorted order.

    Examples:
        >>> True
        True
    """
    holdout = ["tile_c", "tile_a", "tile_b"]
    selected = _cap_inference_tiles(holdout, cap_enabled=False, cap=1, seed=42)
    assert selected == ["tile_a", "tile_b", "tile_c"]


def test_cap_inference_tiles_enabled_reproducible_subset() -> None:
    """Capped selection should be deterministic for a fixed seed.

    Examples:
        >>> True
        True
    """
    holdout = [f"tile_{i:03d}" for i in range(20)]
    selected_1 = _cap_inference_tiles(holdout, cap_enabled=True, cap=5, seed=7)
    selected_2 = _cap_inference_tiles(holdout, cap_enabled=True, cap=5, seed=7)
    assert selected_1 == selected_2
    assert len(selected_1) == 5
    assert selected_1 == sorted(selected_1)
    assert set(selected_1).issubset(set(holdout))


def test_cap_inference_tiles_enabled_requires_positive_cap() -> None:
    """Enabled capping rejects non-positive cap values.

    Examples:
        >>> True
        True
    """
    with pytest.raises(ValueError, match="INFERENCE_TILE_CAP must be > 0"):
        _cap_inference_tiles(["tile_a"], cap_enabled=True, cap=0, seed=42)


def test_split_tiles_mode_gt_to_val_cap_holdout() -> None:
    """GT->val mode puts all GT tiles into validation and caps holdout tiles.

    Examples:
        >>> True
        True
    """
    source, val, holdout = _split_tiles_from_gt_presence(
        gt_tiles=["g3", "g1", "g2"],
        holdout_tiles=["h4", "h2", "h1", "h3"],
        mode=AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT,
        val_fraction=0.5,
        seed=42,
        cap_enabled=True,
        cap=2,
        cap_seed=42,
    )
    assert source == []
    assert val == ["g1", "g2", "g3"]
    assert len(holdout) == 2
    assert holdout == sorted(holdout)
    assert set(holdout).issubset({"h1", "h2", "h3", "h4"})


def test_split_tiles_mode_legacy_preserves_source_plus_val_count() -> None:
    """Legacy mode still splits GT tiles into source/validation sets.

    Examples:
        >>> True
        True
    """
    source, val, holdout = _split_tiles_from_gt_presence(
        gt_tiles=["g1", "g2", "g3", "g4"],
        holdout_tiles=["h2", "h1"],
        mode=AUTO_SPLIT_MODE_LEGACY,
        val_fraction=0.5,
        seed=42,
        cap_enabled=True,
        cap=1,
        cap_seed=42,
    )
    assert len(source) + len(val) == 4
    assert set(source).isdisjoint(set(val))
    assert holdout == ["h1", "h2"]


def test_tile_has_gt_overlap_returns_false_without_vectors() -> None:
    """Public GT-overlap helper should short-circuit when vectors are missing.

    Examples:
        >>> True
        True
    """
    assert tile_has_gt_overlap("missing_tile.tif", [], downsample_factor=1) is False


def test_resolve_source_training_labels_prefers_gt_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GT-first mode should use GT labels when positives are present.

    Examples:
        >>> True
        True
    """
    src = np.zeros((4, 4), dtype=np.uint8)
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[1:3, 1:3] = 1
    monkeypatch.setattr(
        common_module,
        "reproject_labels_to_image",
        lambda *args, **kwargs: src,
    )
    monkeypatch.setattr(
        common_module,
        "rasterize_vector_labels",
        lambda *args, **kwargs: gt,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MODE",
        "gt_if_available",
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MIN_POS_PIXELS",
        1,
        raising=False,
    )

    labels, mode = resolve_source_training_labels(
        "tile.tif",
        "labels.tif",
        ["gt.shp"],
        downsample_factor=1,
    )
    assert mode == "gt_vectors"
    assert int((labels > 0).sum()) == 4


def test_resolve_source_training_labels_falls_back_when_gt_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GT-first mode should fallback to source raster under sparse GT.

    Examples:
        >>> True
        True
    """
    src = np.ones((4, 4), dtype=np.uint8)
    gt = np.zeros((4, 4), dtype=np.uint8)
    monkeypatch.setattr(
        common_module,
        "reproject_labels_to_image",
        lambda *args, **kwargs: src,
    )
    monkeypatch.setattr(
        common_module,
        "rasterize_vector_labels",
        lambda *args, **kwargs: gt,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MODE",
        "gt_if_available",
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MIN_POS_PIXELS",
        2,
        raising=False,
    )

    labels, mode = resolve_source_training_labels(
        "tile.tif",
        "labels.tif",
        ["gt.shp"],
        downsample_factor=1,
    )
    assert mode == "source_raster_fallback_low_gt_coverage"
    assert int((labels > 0).sum()) == 16


def test_resolve_source_training_labels_gt_only_requires_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GT-only mode should fail fast when GT vectors are missing.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(
        common_module,
        "reproject_labels_to_image",
        lambda *args, **kwargs: np.zeros((2, 2), dtype=np.uint8),
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MODE",
        "gt_only",
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_TRAIN_GT_VECTORS",
        None,
        raising=False,
    )
    with pytest.raises(ValueError, match="SOURCE_SUPERVISION_MODE=gt_only"):
        resolve_source_training_labels(
            "tile.tif",
            "labels.tif",
            [],
            downsample_factor=1,
        )


def test_bbox_overlap_ratio_uses_smaller_area() -> None:
    """Overlap ratio is normalized by smaller bbox area.

    Examples:
        >>> True
        True
    """
    a = type("B", (), {"left": 0, "right": 10, "bottom": 0, "top": 10})()
    b = type("B", (), {"left": 5, "right": 15, "bottom": 5, "top": 15})()
    assert _bbox_overlap_ratio(a, b) == pytest.approx(0.25)


def test_anti_leak_checks_detect_duplicate_source_validation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical source/validation tile paths should be flagged.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MODE",
        "source_raster",
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "ANTI_LEAK_FAIL_FAST",
        False,
        raising=False,
    )
    issues = run_source_validation_anti_leak_checks(
        source_tiles=["/tmp/tile_a.tif"],
        val_tiles=["/tmp/tile_a.tif"],
        eval_gt_vector_paths=[],
    )
    assert any("identical tile paths" in msg for msg in issues)


def test_anti_leak_checks_warn_on_gt_vector_reuse_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GT-based source supervision with unset SOURCE_TRAIN_GT_VECTORS is flagged.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MODE",
        "gt_if_available",
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_TRAIN_GT_VECTORS",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "ANTI_LEAK_FAIL_FAST",
        False,
        raising=False,
    )
    issues = run_source_validation_anti_leak_checks(
        source_tiles=[],
        val_tiles=[],
        eval_gt_vector_paths=["/tmp/eval_gt.shp"],
    )
    assert any("defaults to EVAL_GT_VECTORS" in msg for msg in issues)


def test_anti_leak_checks_use_resolved_source_train_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved source-train GT paths should suppress default-reuse warning."""
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_SUPERVISION_MODE",
        "gt_if_available",
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "SOURCE_TRAIN_GT_VECTORS",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        common_module.cfg,
        "ANTI_LEAK_FAIL_FAST",
        False,
        raising=False,
    )
    issues = run_source_validation_anti_leak_checks(
        source_tiles=[],
        val_tiles=[],
        eval_gt_vector_paths=["/tmp/eval_gt.shp"],
        source_train_gt_vector_paths=["/tmp/train_gt_auto.gpkg"],
    )
    assert all("defaults to EVAL_GT_VECTORS" not in msg for msg in issues)


def test_resolve_source_train_gt_vectors_auto_derive(monkeypatch, tmp_path) -> None:
    """Auto-derive should return generated train GT path when explicit paths are unset."""
    monkeypatch.setattr(
        common_module.cfg, "SOURCE_TRAIN_GT_VECTORS", None, raising=False
    )
    monkeypatch.setattr(
        common_module.cfg, "SOURCE_SUPERVISION_MODE", "gt_if_available", raising=False
    )
    monkeypatch.setattr(
        common_module.cfg, "AUTO_DERIVE_SOURCE_TRAIN_GT_VECTORS", True, raising=False
    )
    monkeypatch.setattr(
        common_module.cfg, "AUTO_GT_DERIVE_EXCLUSION_BUFFER_M", 1.0, raising=False
    )
    monkeypatch.setattr(
        common_module.cfg, "AUTO_GT_DERIVE_MIN_GEOM_AREA_M2", 0.0, raising=False
    )
    monkeypatch.setattr(
        common_module.cfg, "AUTO_GT_DERIVE_WRITE_EVAL_COPY", False, raising=False
    )
    monkeypatch.setattr(
        common_module,
        "_tile_union_geometry",
        lambda tiles, target_crs=None: (
            (box(0, 0, 10, 10), "EPSG:25832")
            if target_crs is None
            else (box(8, 8, 20, 20), target_crs)
        ),
    )
    monkeypatch.setattr(
        common_module,
        "_load_gt_geometries",
        lambda paths, target_crs: [box(0, 0, 9, 9), box(30, 30, 40, 40)],
    )
    captured = {}

    def _fake_write(path, geoms, target_crs, *, layer_name):
        captured["path"] = path
        captured["count"] = len(geoms)
        return path

    monkeypatch.setattr(common_module, "_write_derived_vectors_gpkg", _fake_write)
    out = resolve_source_train_gt_vectors(
        source_tiles=["s1.tif"],
        val_tiles=["v1.tif"],
        eval_gt_vector_paths=["eval.shp"],
        run_dir=str(tmp_path),
    )
    assert out is not None and len(out) == 1
    assert out[0].endswith("source_train_gt_auto.gpkg")
    assert captured["count"] == 1


def test_safe_intersection_falls_back_after_topology_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe intersection should recover from GEOS topology failures."""

    class _BadGeom:
        is_valid = False

        def intersection(self, other):  # pragma: no cover - exercised in helper
            raise GEOSException("boom")

    monkeypatch.setattr(
        common_module,
        "_safe_make_valid",
        lambda geom: box(0, 0, 2, 2) if isinstance(geom, _BadGeom) else geom,
    )
    out = common_module._safe_intersection(
        _BadGeom(),
        box(1, 1, 3, 3),
        context="unit-test",
    )
    assert out is not None
    assert out.area == pytest.approx(1.0)
