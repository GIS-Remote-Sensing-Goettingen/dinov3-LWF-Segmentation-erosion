"""Tests for inference tile resolution and per-tile holdout updates."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from segedge.core.config_loader import cfg
from segedge.pipeline.common import filter_tiles_by_source_label_raster_overlap
from segedge.pipeline.inference_flow import (
    resolve_inference_tiles,
    run_holdout_inference,
)
from segedge.pipeline.run import _maybe_run_holdout_inference


def _write_test_raster(
    path: Path,
    *,
    left: float,
    top: float,
    width: int = 10,
    height: int = 10,
    pixel_size: float = 1.0,
) -> None:
    """Write a small GeoTIFF for tile-overlap tests.

    Examples:
        >>> True
        True
    """
    transform = from_origin(left, top, pixel_size, pixel_size)
    data = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_resolve_inference_tiles_filters_directory_by_source_label_overlap(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Directory inference should skip tiles outside source-label raster bounds.

    Examples:
        >>> True
        True
    """
    labels_path = tmp_path / "labels.tif"
    infer_dir = tmp_path / "tiles"
    infer_dir.mkdir()
    tile_keep = infer_dir / "tile_keep.tif"
    tile_drop = infer_dir / "tile_drop.tif"
    _write_test_raster(labels_path, left=0, top=10)
    _write_test_raster(tile_keep, left=5, top=10)
    _write_test_raster(tile_drop, left=25, top=10)
    monkeypatch.setattr(cfg.io.paths, "source_label_raster", str(labels_path))

    with caplog.at_level(logging.INFO):
        tiles, tiles_dir, tile_glob = resolve_inference_tiles(
            infer_tiles_dir=str(infer_dir),
            infer_tile_glob="*.tif",
            infer_tiles=[],
            legacy_inference_dir=None,
            legacy_inference_glob="*.tif",
            legacy_holdout_tiles=[],
            logger=logging.getLogger("test_inference_flow"),
        )

    assert tiles == [str(tile_keep)]
    assert tiles_dir == str(infer_dir)
    assert tile_glob == "*.tif"
    assert "excluded 1 tiles with no SOURCE_LABEL_RASTER overlap" in caplog.text


def test_resolve_inference_tiles_filters_explicit_tiles_by_source_label_overlap(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Explicit inference tile lists should use the same overlap filter.

    Examples:
        >>> True
        True
    """
    labels_path = tmp_path / "labels.tif"
    tile_keep = tmp_path / "tile_keep.tif"
    tile_drop = tmp_path / "tile_drop.tif"
    _write_test_raster(labels_path, left=0, top=10)
    _write_test_raster(tile_keep, left=3, top=10)
    _write_test_raster(tile_drop, left=40, top=10)
    monkeypatch.setattr(cfg.io.paths, "source_label_raster", str(labels_path))

    with caplog.at_level(logging.INFO):
        tiles, tiles_dir, tile_glob = resolve_inference_tiles(
            infer_tiles_dir=None,
            infer_tile_glob="*.tif",
            infer_tiles=[str(tile_keep), str(tile_drop)],
            legacy_inference_dir=None,
            legacy_inference_glob="*.tif",
            legacy_holdout_tiles=[],
            logger=logging.getLogger("test_inference_flow"),
        )

    assert tiles == [str(tile_keep)]
    assert tiles_dir is None
    assert tile_glob == "*.tif"
    assert (
        "excluded 1 explicit tiles with no SOURCE_LABEL_RASTER overlap" in caplog.text
    )


def test_filter_tiles_by_source_label_raster_overlap_keeps_all_when_unset(
    tmp_path,
    monkeypatch,
):
    """Unconfigured source-label raster should leave inference tiles untouched.

    Examples:
        >>> True
        True
    """
    tile_a = tmp_path / "tile_a.tif"
    tile_b = tmp_path / "tile_b.tif"
    _write_test_raster(tile_a, left=0, top=10)
    _write_test_raster(tile_b, left=50, top=10)
    monkeypatch.setattr(cfg.io.paths, "source_label_raster", "")

    filtered_tiles, excluded_count = filter_tiles_by_source_label_raster_overlap(
        [str(tile_a), str(tile_b)]
    )

    assert filtered_tiles == [str(tile_a), str(tile_b)]
    assert excluded_count == 0


def test_maybe_run_holdout_inference_skips_empty_tiles(caplog):
    """Empty holdout sets should skip inference cleanly.

    Examples:
        >>> True
        True
    """
    runner_calls: list[str] = []
    with caplog.at_level(logging.WARNING):
        ran = _maybe_run_holdout_inference([], lambda: runner_calls.append("ran"))

    assert ran is False
    assert runner_calls == []
    assert "skipping holdout inference" in caplog.text


def test_run_holdout_inference_appends_and_checkpoints_per_tile(
    tmp_path,
    monkeypatch,
):
    """Each completed tile should append masks and write a checkpoint immediately.

    Examples:
        >>> True
        True
    """
    append_calls: list[tuple[str, str, int]] = []
    checkpoint_calls: list[int] = []
    call_order: list[tuple[str, str | int]] = []

    def _fake_infer_on_holdout(
        holdout_path,
        gt_vector_paths,
        model,
        processor,
        device,
        pos_bank,
        neg_bank,
        tuned,
        ps,
        tile_size,
        stride,
        feature_dir,
        shape_dir,
        plot_dir,
        context_radius,
        plot_with_metrics,
    ):
        mask = np.ones((2, 2), dtype=bool)
        return {
            "gt_available": False,
            "metrics": {},
            "image_id": Path(holdout_path).stem,
            "ref_path": holdout_path,
            "masks": {"champion_shadow": mask},
        }

    monkeypatch.setattr(
        "segedge.pipeline.inference_flow.infer_on_holdout",
        _fake_infer_on_holdout,
    )

    processed_log_path = tmp_path / "processed.jsonl"
    processed = run_holdout_inference(
        holdout_tiles=["tile_a.tif", "tile_b.tif"],
        processed_tiles=set(),
        gt_vector_paths=None,
        model=None,
        processor=None,
        device=None,
        pos_bank=np.zeros((0, 0), dtype=np.float32),
        neg_bank=None,
        tuned={},
        ps=16,
        tile_size=64,
        stride=32,
        feature_dir=None,
        shape_dir=str(tmp_path / "shapes"),
        plot_dir=str(tmp_path / "plots"),
        context_radius=0,
        holdout_phase_metrics={},
        append_union=lambda stream, variant, mask, ref_path, step: append_calls.append(
            (stream, ref_path, step)
        )
        or call_order.append(("append", ref_path)),
        processed_log_path=str(processed_log_path),
        write_checkpoint=lambda holdout_done: checkpoint_calls.append(holdout_done)
        or call_order.append(("checkpoint", holdout_done)),
        logger=logging.getLogger("test_inference_flow"),
    )

    assert processed == 2
    assert append_calls == [
        ("champion", "tile_a.tif", 1),
        ("champion", "tile_b.tif", 2),
    ]
    assert checkpoint_calls == [1, 2]
    assert call_order == [
        ("append", "tile_a.tif"),
        ("checkpoint", 1),
        ("append", "tile_b.tif"),
        ("checkpoint", 2),
    ]
    assert len(processed_log_path.read_text(encoding="utf-8").splitlines()) == 2
