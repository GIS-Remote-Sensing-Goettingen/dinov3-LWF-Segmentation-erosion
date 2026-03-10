"""Tests for inference tile resolution and per-tile holdout updates."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from segedge.core.config_loader import cfg
from segedge.pipeline.common import (
    filter_tiles_by_source_label_presence,
    resolve_tiles_from_gt_presence,
)
from segedge.pipeline.inference_flow import (
    resolve_inference_tiles,
    run_holdout_inference,
)
from segedge.pipeline.run import _maybe_run_holdout_inference
from segedge.pipeline.runtime.holdout_inference import _apply_inference_score_prior


def _write_test_raster(
    path: Path,
    *,
    left: float,
    top: float,
    width: int = 10,
    height: int = 10,
    pixel_size: float = 1.0,
    data: np.ndarray | None = None,
) -> None:
    """Write a small GeoTIFF for label-presence tests.

    Examples:
        >>> True
        True
    """
    transform = from_origin(left, top, pixel_size, pixel_size)
    raster_data = (
        np.zeros((height, width), dtype=np.uint8)
        if data is None
        else np.asarray(data, dtype=np.uint8)
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=raster_data.dtype,
        crs="EPSG:3857",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(raster_data, 1)


def test_resolve_inference_tiles_filters_directory_by_source_label_presence(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Directory inference should require positive source-label pixels.

    Examples:
        >>> True
        True
    """
    labels_path = tmp_path / "labels.tif"
    infer_dir = tmp_path / "tiles"
    infer_dir.mkdir()
    tile_keep = infer_dir / "tile_keep.tif"
    tile_empty = infer_dir / "tile_empty.tif"
    tile_drop = infer_dir / "tile_drop.tif"
    labels = np.zeros((10, 20), dtype=np.uint8)
    labels[:, :10] = 1
    _write_test_raster(labels_path, left=0, top=10, width=20, data=labels)
    _write_test_raster(tile_keep, left=0, top=10)
    _write_test_raster(tile_empty, left=10, top=10)
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
    assert (
        "excluded 2 tiles with no SOURCE_LABEL_RASTER labels inside tile" in caplog.text
    )


def test_resolve_inference_tiles_filters_explicit_tiles_by_source_label_presence(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Explicit inference tile lists should require positive labels too.

    Examples:
        >>> True
        True
    """
    labels_path = tmp_path / "labels.tif"
    tile_keep = tmp_path / "tile_keep.tif"
    tile_empty = tmp_path / "tile_empty.tif"
    tile_drop = tmp_path / "tile_drop.tif"
    labels = np.zeros((10, 20), dtype=np.uint8)
    labels[:, :10] = 1
    _write_test_raster(labels_path, left=0, top=10, width=20, data=labels)
    _write_test_raster(tile_keep, left=0, top=10)
    _write_test_raster(tile_empty, left=10, top=10)
    _write_test_raster(tile_drop, left=40, top=10)
    monkeypatch.setattr(cfg.io.paths, "source_label_raster", str(labels_path))

    with caplog.at_level(logging.INFO):
        tiles, tiles_dir, tile_glob = resolve_inference_tiles(
            infer_tiles_dir=None,
            infer_tile_glob="*.tif",
            infer_tiles=[str(tile_keep), str(tile_empty), str(tile_drop)],
            legacy_inference_dir=None,
            legacy_inference_glob="*.tif",
            legacy_holdout_tiles=[],
            logger=logging.getLogger("test_inference_flow"),
        )

    assert tiles == [str(tile_keep)]
    assert tiles_dir is None
    assert tile_glob == "*.tif"
    assert (
        "excluded 2 explicit tiles with no SOURCE_LABEL_RASTER labels inside tile"
        in caplog.text
    )


def test_filter_tiles_by_source_label_presence_keeps_all_when_unset(
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

    filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
        [str(tile_a), str(tile_b)]
    )

    assert filtered_tiles == [str(tile_a), str(tile_b)]
    assert excluded_count == 0


def test_resolve_tiles_from_gt_presence_prefilters_by_source_label_presence(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Auto split should exclude tiles with no source-label pixels.

    Examples:
        >>> True
        True
    """
    labels_path = tmp_path / "labels.tif"
    tiles_dir = tmp_path / "tiles"
    gt_path = tmp_path / "gt.shp"
    tiles_dir.mkdir()
    labels = np.zeros((10, 20), dtype=np.uint8)
    labels[:, :10] = 1
    _write_test_raster(labels_path, left=0, top=10, width=20, data=labels)
    tile_keep = tiles_dir / "tile_keep.tif"
    tile_empty = tiles_dir / "tile_empty.tif"
    _write_test_raster(tile_keep, left=0, top=10)
    _write_test_raster(tile_empty, left=10, top=10)
    monkeypatch.setattr(cfg.io.paths, "source_label_raster", str(labels_path))
    monkeypatch.setattr(
        "segedge.pipeline.common._tile_has_gt",
        lambda tile_path, gt_vector_paths, downsample_factor: True,
    )
    monkeypatch.setattr(
        "segedge.pipeline.common._tile_effective_gt_pixels",
        lambda tile_path, gt_vector_paths, downsample_factor: 1,
    )

    with caplog.at_level(logging.INFO):
        gt_tiles, inference_tiles = resolve_tiles_from_gt_presence(
            str(tiles_dir),
            "*.tif",
            [str(gt_path)],
            num_workers=1,
        )

    assert gt_tiles == [str(tile_keep)]
    assert inference_tiles == []
    assert (
        "auto split: excluded 1 tiles with no SOURCE_LABEL_RASTER labels inside tile"
        in caplog.text
    )


def test_apply_inference_score_prior_boosts_only_inside_source_labels(monkeypatch):
    """The manual score prior should affect only source-label pixels.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(cfg.io.inference.score_prior, "enabled", True)
    monkeypatch.setattr(cfg.io.inference.score_prior, "factor", 1.2)
    monkeypatch.setattr(cfg.io.inference.score_prior, "clip_max", 1.0)
    score_map = np.array([[0.4, 0.4]], dtype=np.float32)
    labels_sh = np.array([[1, 0]], dtype=np.uint8)

    boosted = _apply_inference_score_prior(
        score_map,
        labels_sh,
        "tile_a",
        final_inference_phase=True,
    )

    assert np.allclose(boosted, np.array([[0.48, 0.4]], dtype=np.float32))


def test_apply_inference_score_prior_is_disabled_outside_final_inference(
    monkeypatch,
):
    """Validation-phase inference should not receive the manual score prior.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(cfg.io.inference.score_prior, "enabled", True)
    monkeypatch.setattr(cfg.io.inference.score_prior, "factor", 1.2)
    score_map = np.array([[0.4, 0.4]], dtype=np.float32)
    labels_sh = np.array([[1, 0]], dtype=np.uint8)

    boosted = _apply_inference_score_prior(
        score_map,
        labels_sh,
        "tile_a",
        final_inference_phase=False,
    )

    assert np.allclose(boosted, score_map)


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
    caplog,
):
    """Each completed tile should append masks and write a checkpoint immediately.

    Examples:
        >>> True
        True
    """
    append_calls: list[tuple[str, str, int]] = []
    checkpoint_calls: list[int] = []
    call_order: list[tuple[str, str | int]] = []
    phase_flags: list[bool] = []

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
        final_inference_phase,
    ):
        """Return a deterministic fake inference payload.

        Examples:
            >>> True
            True
        """
        phase_flags.append(final_inference_phase)
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
    with caplog.at_level(logging.INFO):
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
    assert phase_flags == [True, True]
    assert len(processed_log_path.read_text(encoding="utf-8").splitlines()) == 2
    assert "Processing tile tile_a.tif, 1 / 2" in caplog.text
    assert "Processing tile tile_b.tif, 2 / 2" in caplog.text


def test_run_holdout_inference_progress_counts_only_pending_tiles(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Resume runs should count only pending tiles in progress logs.

    Examples:
        >>> True
        True
    """
    phase_flags: list[bool] = []

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
        final_inference_phase,
    ):
        """Return a deterministic fake inference payload.

        Examples:
            >>> True
            True
        """
        phase_flags.append(final_inference_phase)
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

    with caplog.at_level(logging.INFO):
        processed = run_holdout_inference(
            holdout_tiles=["tile_done.tif", "tile_pending.tif"],
            processed_tiles={"tile_done.tif"},
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
            append_union=lambda stream, variant, mask, ref_path, step: None,
            processed_log_path=str(tmp_path / "processed.jsonl"),
            write_checkpoint=lambda holdout_done: None,
            logger=logging.getLogger("test_inference_flow_resume"),
        )

    assert processed == 2
    assert phase_flags == [True]
    assert "holdout skip (already processed): tile_done.tif" in caplog.text
    assert "Processing tile tile_pending.tif, 1 / 1" in caplog.text
