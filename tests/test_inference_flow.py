"""Tests for inference tile resolution and per-tile holdout updates."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import from_origin

from segedge.core.config_loader import cfg
from segedge.pipeline import common as common_module
from segedge.pipeline.common import (
    filter_tiles_by_source_label_presence,
    resolve_tiles_from_gt_presence,
)
from segedge.pipeline.inference_flow import (
    resolve_inference_tiles,
    run_holdout_inference,
)
from segedge.pipeline.run import _maybe_run_holdout_inference
from segedge.pipeline.runtime.checkpointing import write_rolling_best_config
from segedge.pipeline.runtime.holdout_inference import (
    _apply_inference_score_prior,
    _build_xgb_plot_preview_masks,
    _compute_xgb_stream,
    _postprocess_stream_mask,
    _save_holdout_plots,
)


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


def test_filter_tiles_by_source_label_presence_reuses_cache_without_reopening_tiles(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Repeated runs should reuse cached tile decisions when inputs are unchanged.

    Examples:
        >>> True
        True
    """
    labels_path = tmp_path / "labels.tif"
    tile_keep = tmp_path / "tile_keep.tif"
    tile_drop = tmp_path / "tile_drop.tif"
    labels = np.zeros((10, 20), dtype=np.uint8)
    labels[:, :10] = 1
    _write_test_raster(labels_path, left=0, top=10, width=20, data=labels)
    _write_test_raster(tile_keep, left=0, top=10)
    _write_test_raster(tile_drop, left=10, top=10)
    monkeypatch.setattr(cfg.io.paths, "source_label_raster", str(labels_path))
    monkeypatch.setattr(cfg.io.paths, "output_dir", str(tmp_path / "output"))

    filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
        [str(tile_keep), str(tile_drop)]
    )

    assert filtered_tiles == [str(tile_keep)]
    assert excluded_count == 1

    original_rio_open = common_module.rio_open

    def _cached_only_open(path, *args, **kwargs):
        if Path(path) in {tile_keep, tile_drop}:
            raise AssertionError("tile raster should not be reopened on cache hit")
        return original_rio_open(path, *args, **kwargs)

    monkeypatch.setattr(common_module, "rio_open", _cached_only_open)
    with caplog.at_level(logging.INFO):
        cached_tiles, cached_excluded = filter_tiles_by_source_label_presence(
            [str(tile_keep), str(tile_drop)]
        )

    assert cached_tiles == [str(tile_keep)]
    assert cached_excluded == 1
    assert "source-label filter cache: reused 2/2 tile decisions" in caplog.text


def test_filter_tiles_by_source_label_presence_invalidates_cache_for_changed_tile(
    tmp_path,
    monkeypatch,
):
    """Changing a tile should invalidate its cached source-label decision.

    Examples:
        >>> True
        True
    """
    labels_path = tmp_path / "labels.tif"
    tile_path = tmp_path / "tile.tif"
    labels = np.zeros((10, 20), dtype=np.uint8)
    labels[:, :10] = 1
    _write_test_raster(labels_path, left=0, top=10, width=20, data=labels)
    _write_test_raster(tile_path, left=0, top=10)
    monkeypatch.setattr(cfg.io.paths, "source_label_raster", str(labels_path))
    monkeypatch.setattr(cfg.io.paths, "output_dir", str(tmp_path / "output"))

    filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
        [str(tile_path)]
    )
    assert filtered_tiles == [str(tile_path)]
    assert excluded_count == 0

    _write_test_raster(tile_path, left=10, top=10)

    filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
        [str(tile_path)]
    )

    assert filtered_tiles == []
    assert excluded_count == 1


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


def test_apply_inference_score_prior_applies_inside_and_outside_factors(monkeypatch):
    """The manual score prior should support separate inside/outside multipliers.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(cfg.io.inference.score_prior, "enabled", True)
    monkeypatch.setattr(cfg.io.inference.score_prior, "inside_factor", 1.2)
    monkeypatch.setattr(cfg.io.inference.score_prior, "outside_factor", 0.5)
    monkeypatch.setattr(cfg.io.inference.score_prior, "clip_max", 1.0)
    score_map = np.array([[0.4, 0.4]], dtype=np.float32)
    labels_sh = np.array([[1, 0]], dtype=np.uint8)

    boosted = _apply_inference_score_prior(
        score_map,
        labels_sh,
        "tile_a",
        final_inference_phase=True,
    )

    assert np.allclose(boosted, np.array([[0.48, 0.2]], dtype=np.float32))


def test_apply_inference_score_prior_is_disabled_outside_final_inference(
    monkeypatch,
):
    """Validation-phase inference should not receive the manual score prior.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(cfg.io.inference.score_prior, "enabled", True)
    monkeypatch.setattr(cfg.io.inference.score_prior, "inside_factor", 1.2)
    monkeypatch.setattr(cfg.io.inference.score_prior, "outside_factor", 0.5)
    score_map = np.array([[0.4, 0.4]], dtype=np.float32)
    labels_sh = np.array([[1, 0]], dtype=np.uint8)

    boosted = _apply_inference_score_prior(
        score_map,
        labels_sh,
        "tile_a",
        final_inference_phase=False,
    )

    assert np.allclose(boosted, score_map)


def test_postprocess_stream_mask_can_fill_enclosed_holes():
    """XGB raw-mask postprocessing should fill enclosed holes when enabled.

    Examples:
        >>> True
        True
    """
    base_mask = np.ones((7, 7), dtype=bool)
    base_mask[2:5, 2:5] = False
    sh_buffer_mask = np.ones((7, 7), dtype=bool)

    out = _postprocess_stream_mask(base_mask, sh_buffer_mask, fill_holes=True)

    assert out[3, 3]


def test_compute_xgb_stream_fills_holes_when_enabled(monkeypatch):
    """XGB raw masks should fill enclosed holes before CRF when configured.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(cfg.postprocess, "fill_holes_xgb", True)
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._score_xgb_with_guard",
        lambda **kwargs: np.array(
            [
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                [0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9],
                [0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9],
                [0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._apply_roads_penalty",
        lambda score_map, roads_mask, roads_penalty: score_map,
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._apply_inference_score_prior",
        lambda score_map, labels_sh, image_id_b, final_inference_phase: score_map,
    )
    context = {
        "sh_buffer_mask": np.ones((7, 7), dtype=bool),
        "gt_mask_eval": np.zeros((7, 7), dtype=bool),
        "xgb_enabled": True,
        "img_b": np.zeros((7, 7, 3), dtype=np.uint8),
        "image_id_b": "tile_a",
        "roads_mask": np.zeros((7, 7), dtype=bool),
        "roads_penalty": 1.0,
        "labels_sh": np.zeros((7, 7), dtype=np.uint8),
        "prefetched_b": {},
        "xgb_feature_stats": None,
    }
    tuned = {"bst": object(), "best_xgb_config": {"threshold": 0.5}}

    result = _compute_xgb_stream(
        context,
        tuned,
        ps=16,
        tile_size=16,
        stride=16,
        feature_dir=None,
        context_radius=0,
        final_inference_phase=False,
        xgb_guard_state=None,
    )

    assert result["mask"][3, 3]


def test_build_xgb_plot_preview_masks_show_unclipped_regions(monkeypatch):
    """Plot-only XGB previews should show regions outside the SH buffer.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(cfg.postprocess, "fill_holes_xgb", False)
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._apply_crf_to_stream",
        lambda img_b, score_map, threshold, sh_buffer_mask, crf_cfg, **kwargs: score_map
        >= threshold,
    )
    context = {
        "image_id_b": "tile_a",
        "img_b": np.zeros((3, 3, 3), dtype=np.uint8),
        "sh_buffer_mask": np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
    }
    champion_score = np.array(
        [
            [0.9, 0.9, 0.1],
            [0.9, 0.9, 0.1],
            [0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    out = _build_xgb_plot_preview_masks(
        context=context,
        champion_score=champion_score,
        champion_thr=0.5,
        active_crf_mask=np.zeros((3, 3), dtype=bool),
        tuned={
            "best_crf_config": {
                "enabled": True,
                "prob_softness": 0.1,
                "pos_w": 1.0,
                "pos_xy_std": 3.0,
                "bilateral_w": 12.0,
                "bilateral_xy_std": 40.0,
                "bilateral_rgb_std": 4.0,
                "trimap_band_pixels": 16,
            }
        },
    )

    assert out["xgb_raw_plot"][0, 1]
    assert out["xgb_crf_plot"][0, 1]


def test_write_rolling_best_config_exports_inside_and_outside_score_prior(
    tmp_path,
    monkeypatch,
):
    """Rolling checkpoints should export both score-prior multipliers.

    Examples:
        >>> True
        True
    """
    out_path = tmp_path / "rolling.yml"
    monkeypatch.setattr(cfg.io.inference.score_prior, "enabled", True)
    monkeypatch.setattr(cfg.io.inference.score_prior, "inside_factor", 1.4)
    monkeypatch.setattr(cfg.io.inference.score_prior, "outside_factor", 0.7)
    monkeypatch.setattr(cfg.io.inference.score_prior, "clip_max", 0.9)

    write_rolling_best_config(
        str(out_path),
        stage="holdout_inference",
        tuned={},
        fold_done=0,
        fold_total=0,
        holdout_done=1,
        holdout_total=2,
    )

    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert payload["inference_score_prior"]["inside_factor"] == 1.4
    assert payload["inference_score_prior"]["outside_factor"] == 0.7
    assert payload["inference_score_prior"]["clip_max"] == 0.9


def test_xgb_guard_falls_back_to_legacy_when_difference_is_meaningful(monkeypatch):
    """The scorer guard should switch to the legacy path after a meaningful diff.

    Examples:
        >>> True
        True
    """
    optimized_calls = {"count": 0}
    legacy_calls = {"count": 0}

    def _fake_optimized(*args, **kwargs):
        optimized_calls["count"] += 1
        return np.ones((2, 2), dtype=np.float32)

    def _fake_legacy(*args, **kwargs):
        legacy_calls["count"] += 1
        return np.zeros((2, 2), dtype=np.float32)

    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.xgb_score_image_b",
        _fake_optimized,
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.xgb_score_image_b_legacy",
        _fake_legacy,
    )
    context = {
        "sh_buffer_mask": np.ones((2, 2), dtype=bool),
        "gt_mask_eval": np.zeros((2, 2), dtype=bool),
        "xgb_enabled": True,
        "img_b": np.zeros((2, 2, 3), dtype=np.uint8),
        "image_id_b": "tile_guard",
        "prefetched_b": {},
        "roads_mask": None,
        "roads_penalty": 1.0,
        "xgb_feature_stats": None,
        "labels_sh": np.zeros((2, 2), dtype=np.uint8),
    }
    tuned = {
        "bst": object(),
        "best_xgb_config": {"threshold": 0.5},
    }
    guard_state = {
        "enabled": True,
        "checked_tiles": 0,
        "fallback_to_legacy": False,
        "guard_tiles": 3,
        "atol": 1e-5,
        "rtol": 1e-4,
    }

    result_a = _compute_xgb_stream(
        context,
        tuned,
        ps=16,
        tile_size=32,
        stride=32,
        feature_dir=None,
        context_radius=0,
        final_inference_phase=True,
        xgb_guard_state=guard_state,
    )
    result_b = _compute_xgb_stream(
        context,
        tuned,
        ps=16,
        tile_size=32,
        stride=32,
        feature_dir=None,
        context_radius=0,
        final_inference_phase=True,
        xgb_guard_state=guard_state,
    )

    assert guard_state["fallback_to_legacy"] is True
    assert np.array_equal(result_a["mask"], np.zeros((2, 2), dtype=bool))
    assert np.array_equal(result_b["mask"], np.zeros((2, 2), dtype=bool))
    assert optimized_calls["count"] == 1
    assert legacy_calls["count"] == 2


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
    save_plot_flags: list[bool] = []

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
        save_plots,
        xgb_guard_state,
    ):
        """Return a deterministic fake inference payload.

        Examples:
            >>> True
            True
        """
        phase_flags.append(final_inference_phase)
        save_plot_flags.append(save_plots)
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
    assert save_plot_flags == [True, True]
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
    save_plot_flags: list[bool] = []

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
        save_plots,
        xgb_guard_state,
    ):
        """Return a deterministic fake inference payload.

        Examples:
            >>> True
            True
        """
        phase_flags.append(final_inference_phase)
        save_plot_flags.append(save_plots)
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
    assert save_plot_flags == [True]
    assert "holdout skip (already processed): tile_done.tif" in caplog.text
    assert "Processing tile tile_pending.tif, 1 / 1" in caplog.text


def test_run_holdout_inference_plots_every_pending_n_tiles(
    tmp_path,
    monkeypatch,
):
    """Plot cadence should count only pending tiles in the current run.

    Examples:
        >>> True
        True
    """
    save_plot_flags: list[bool] = []

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
        save_plots,
        xgb_guard_state,
    ):
        save_plot_flags.append(save_plots)
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

    processed = run_holdout_inference(
        holdout_tiles=["tile_1.tif", "tile_2.tif", "tile_3.tif", "tile_4.tif"],
        processed_tiles={"tile_1.tif"},
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
        logger=logging.getLogger("test_inference_flow_plot_every"),
        plot_every=2,
    )

    assert processed == 4
    assert save_plot_flags == [True, False, True]


def test_save_holdout_plots_respects_individual_plot_toggles(monkeypatch, tmp_path):
    """Disabled inference plot types should not be rendered.

    Examples:
        >>> True
        True
    """
    calls = []

    def _record(name):
        def _inner(*args, **kwargs):
            calls.append(name)

        return _inner

    monkeypatch.setattr(cfg.io.inference.plots, "unified", False)
    monkeypatch.setattr(cfg.io.inference.plots, "qualitative_core", True)
    monkeypatch.setattr(cfg.io.inference.plots, "score_threshold", False)
    monkeypatch.setattr(cfg.io.inference.plots, "disagreement_entropy", True)
    monkeypatch.setattr(cfg.io.inference.plots, "proposal_overlay", False)
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.save_unified_plot",
        _record("unified"),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.save_core_qualitative_plot",
        _record("qualitative_core"),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.save_score_threshold_plot",
        _record("score_threshold"),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.save_disagreement_entropy_plot",
        _record("disagreement_entropy"),
    )
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.save_proposal_overlay_plot",
        _record("proposal_overlay"),
    )

    _save_holdout_plots(
        context={
            "image_id_b": "tile_plot",
            "img_b": np.zeros((2, 2, 3), dtype=np.uint8),
            "gt_mask_eval": np.zeros((2, 2), dtype=bool),
            "labels_sh": np.zeros((2, 2), dtype=np.uint8),
            "gt_available": False,
            "sh_buffer_mask": np.ones((2, 2), dtype=bool),
        },
        plot_dir=str(tmp_path),
        plot_with_metrics=False,
        active_stream="xgb",
        active_label="XGB",
        champion_score=np.zeros((2, 2), dtype=np.float32),
        champion_thr=0.5,
        active_raw_mask=np.zeros((2, 2), dtype=bool),
        active_crf_mask=np.zeros((2, 2), dtype=bool),
        active_shadow_mask=np.zeros((2, 2), dtype=bool),
        crf_cfg={},
        metrics_map={},
        proposal_masks={
            "candidate_mask": np.zeros((2, 2), dtype=bool),
            "candidate_inside_mask": np.zeros((2, 2), dtype=bool),
            "evaluated_outside_mask": np.zeros((2, 2), dtype=bool),
            "accepted_mask": np.zeros((2, 2), dtype=bool),
            "accepted_inside_mask": np.zeros((2, 2), dtype=bool),
            "accepted_outside_mask": np.zeros((2, 2), dtype=bool),
            "rejected_mask": np.zeros((2, 2), dtype=bool),
        },
        proposal_bundle={},
        score_knn_raw=None,
        disagreement_map=np.zeros((2, 2), dtype=np.float32),
        entropy_map=np.zeros((2, 2), dtype=np.float32),
    )

    assert calls == ["qualitative_core", "disagreement_entropy"]


def test_save_holdout_plots_passes_xgb_preview_masks_and_merged_proposals(
    monkeypatch,
    tmp_path,
):
    """Unified plot should receive outside-label XGB previews and merged proposal masks.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(cfg.io.inference.plots, "unified", True)
    monkeypatch.setattr(cfg.io.inference.plots, "qualitative_core", False)
    monkeypatch.setattr(cfg.io.inference.plots, "score_threshold", False)
    monkeypatch.setattr(cfg.io.inference.plots, "disagreement_entropy", False)
    monkeypatch.setattr(cfg.io.inference.plots, "proposal_overlay", False)
    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference._build_xgb_plot_preview_masks",
        lambda **kwargs: {
            "xgb_raw_plot": np.array([[1, 1], [0, 0]], dtype=bool),
            "xgb_crf_plot": np.array([[1, 0], [1, 0]], dtype=bool),
        },
    )
    captured = {}

    def _capture_unified(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "segedge.pipeline.runtime.holdout_inference.save_unified_plot",
        _capture_unified,
    )

    _save_holdout_plots(
        context={
            "image_id_b": "tile_plot",
            "img_b": np.zeros((2, 2, 3), dtype=np.uint8),
            "gt_mask_eval": np.zeros((2, 2), dtype=bool),
            "labels_sh": np.zeros((2, 2), dtype=np.uint8),
            "gt_available": False,
            "sh_buffer_mask": np.ones((2, 2), dtype=bool),
        },
        plot_dir=str(tmp_path),
        plot_with_metrics=False,
        active_stream="xgb",
        active_label="XGB",
        champion_score=np.zeros((2, 2), dtype=np.float32),
        champion_thr=0.5,
        active_raw_mask=np.zeros((2, 2), dtype=bool),
        active_crf_mask=np.zeros((2, 2), dtype=bool),
        active_shadow_mask=np.zeros((2, 2), dtype=bool),
        crf_cfg={"enabled": True, "trimap_band_pixels": 16},
        metrics_map={},
        proposal_masks={
            "candidate_mask": np.ones((2, 2), dtype=bool),
            "candidate_inside_mask": np.zeros((2, 2), dtype=bool),
            "evaluated_outside_mask": np.zeros((2, 2), dtype=bool),
            "accepted_mask": np.array([[1, 0], [0, 0]], dtype=bool),
            "accepted_inside_mask": np.zeros((2, 2), dtype=bool),
            "accepted_outside_mask": np.zeros((2, 2), dtype=bool),
            "rejected_mask": np.array([[0, 1], [0, 0]], dtype=bool),
        },
        proposal_bundle={},
        score_knn_raw=None,
        disagreement_map=np.zeros((2, 2), dtype=np.float32),
        entropy_map=np.zeros((2, 2), dtype=np.float32),
    )

    assert "xgb_raw_plot" in captured["masks"]
    assert "xgb_crf_plot" in captured["masks"]
    assert "candidate" not in captured["proposal_masks"]
    assert "accepted" in captured["proposal_masks"]
    assert "rejected" in captured["proposal_masks"]
