"""Primary pipeline entrypoint for SegEdge."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable

import yaml

from ..core.config_loader import cfg
from ..core.io_utils import (
    append_mask_to_union_shapefile,
    backup_union_shapefile,
    count_shapefile_features,
)
from ..core.logging_utils import setup_logging
from ..core.timing_utils import (
    configure_performance_logging,
    emit_performance_summary,
    time_end,
    time_start,
)
from .common import init_model, resolve_tiles_from_gt_presence
from .inference_flow import resolve_inference_tiles
from .runtime_utils import (
    _log_phase,
    build_time_budget_status,
    compute_budget_deadline,
    parse_utc_iso_to_epoch,
    remaining_budget_s,
)
from .workflows.inference_only import run_inference_only
from .workflows.loo_training import run_loo_training
from .workflows.manual_training import run_manual_training
from .workflows.shared import (
    _aggregate_fold_metrics,
    _build_loo_folds,
    _export_best_settings_dual,
    _maybe_run_holdout_inference,
    _novel_proposals_payload,
    consolidate_cached_features,
)

# Config-driven flags
KNN_ENABLED = bool(cfg.search.knn.enabled)
XGB_ENABLED = bool(cfg.search.xgb.enabled)
CRF_ENABLED = bool(cfg.search.crf.enabled)

logger = logging.getLogger(__name__)

__all__ = [
    "_aggregate_fold_metrics",
    "_build_loo_folds",
    "_export_best_settings_dual",
    "_maybe_run_holdout_inference",
    "_novel_proposals_payload",
    "_resolve_inference_model_bundle_dir",
    "main",
]


def _create_run_directories() -> dict[str, str]:
    """Create or resume the run directory tree and configure logging.

    Examples:
        >>> callable(_create_run_directories)
        True
    """
    output_root = cfg.io.paths.output_dir
    os.makedirs(output_root, exist_ok=True)

    resume_run = bool(cfg.runtime.resume_run)
    resume_dir = cfg.runtime.resume_run_dir
    if resume_run:
        if not resume_dir:
            raise ValueError("RESUME_RUN_DIR must be set when RESUME_RUN=True")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"RESUME_RUN_DIR not found: {resume_dir}")
        run_dir = resume_dir
        logger.info("resume run: %s", run_dir)
    else:
        existing = sorted(d for d in os.listdir(output_root) if d.startswith("run_"))
        next_idx = 1
        if existing:
            try:
                next_idx = (
                    max(
                        int(d.split("_")[1])
                        for d in existing
                        if d.split("_")[1].isdigit()
                    )
                    + 1
                )
            except ValueError:
                next_idx = len(existing) + 1
        run_dir = os.path.join(output_root, f"run_{next_idx:03d}")

    plot_dir = os.path.join(run_dir, "plots")
    validation_plot_dir = os.path.join(plot_dir, "validation")
    inference_plot_dir = os.path.join(plot_dir, "inference")
    shape_dir = os.path.join(run_dir, "shapes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(validation_plot_dir, exist_ok=True)
    os.makedirs(inference_plot_dir, exist_ok=True)
    os.makedirs(shape_dir, exist_ok=True)

    cfg.io.paths.plot_dir = plot_dir
    cfg.io.paths.best_settings_path = os.path.join(run_dir, "best_settings.yml")
    cfg.io.paths.log_path = os.path.join(run_dir, "run.log")
    setup_logging(cfg.io.paths.log_path)
    configure_performance_logging(
        os.path.join(run_dir, "performance.jsonl"),
        run_id=os.path.basename(run_dir),
    )
    return {
        "run_dir": run_dir,
        "plot_dir": plot_dir,
        "validation_plot_dir": validation_plot_dir,
        "inference_plot_dir": inference_plot_dir,
        "shape_dir": shape_dir,
    }


def _run_dir_sort_key(run_name: str) -> tuple[int, str]:
    """Return a stable descending-sort key for `run_*` directories.

    Examples:
        >>> _run_dir_sort_key("run_007")
        (7, 'run_007')
    """
    parts = run_name.split("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (int(parts[1]), run_name)
    return (-1, run_name)


def _resolve_inference_model_bundle_dir(
    model_bundle_dir: str | None,
    *,
    output_root: str,
    current_run_dir: str,
) -> str:
    """Resolve the bundle directory for inference-only runs.

    Examples:
        >>> _resolve_inference_model_bundle_dir(
        ...     "/tmp/bundle",
        ...     output_root="/tmp/out",
        ...     current_run_dir="/tmp/out/run_001",
        ... )
        '/tmp/bundle'
    """
    if model_bundle_dir:
        return model_bundle_dir

    current_run_dir_abs = os.path.abspath(current_run_dir)
    candidate_names = sorted(
        (
            name
            for name in os.listdir(output_root)
            if name.startswith("run_")
            and os.path.isdir(os.path.join(output_root, name))
        ),
        key=_run_dir_sort_key,
        reverse=True,
    )
    for run_name in candidate_names:
        run_dir = os.path.join(output_root, run_name)
        if os.path.abspath(run_dir) == current_run_dir_abs:
            continue
        bundle_dir = os.path.join(run_dir, "model_bundle")
        manifest_path = os.path.join(bundle_dir, "manifest.yml")
        if os.path.exists(manifest_path):
            logger.info(
                "inference-only: model_bundle_dir unset, using previous run bundle %s",
                bundle_dir,
            )
            return bundle_dir
    raise ValueError(
        "io.training=false and io.inference.model_bundle_dir is null, but no "
        f"previous run bundle was found under {output_root}"
    )


def _load_processed_tiles(processed_log_path: str, resume_run: bool) -> set[str]:
    """Load completed holdout tiles for resume mode.

    Examples:
        >>> _load_processed_tiles("/tmp/does-not-exist", False)
        set()
    """
    processed_tiles: set[str] = set()
    if not resume_run or not os.path.exists(processed_log_path):
        return processed_tiles

    with open(processed_log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "done" and record.get("tile_path"):
                processed_tiles.add(record["tile_path"])
    logger.info("resume: loaded %s processed tiles", len(processed_tiles))
    return processed_tiles


def _initialize_time_budget_state(
    *,
    resume_run: bool,
    rolling_best_settings_path: str,
    wall_clock_start_ts: float,
) -> dict[str, object]:
    """Build mutable time-budget state shared across workflows.

    Examples:
        >>> callable(_initialize_time_budget_state)
        True
    """
    time_budget_cfg = cfg.runtime.time_budget
    budget_state: dict[str, object] = {
        "enabled": bool(time_budget_cfg.enabled),
        "hours": float(time_budget_cfg.hours),
        "scope": str(time_budget_cfg.scope),
        "cutover_mode": str(time_budget_cfg.cutover_mode),
        "deadline_ts": None,
        "clock_start_ts": None,
        "cutover_triggered": False,
        "cutover_stage": "none",
    }

    if (
        budget_state["enabled"]
        and resume_run
        and os.path.exists(rolling_best_settings_path)
    ):
        try:
            with open(rolling_best_settings_path, "r", encoding="utf-8") as fh:
                rolling_payload = yaml.safe_load(fh) or {}
            budget_payload = rolling_payload.get("time_budget", {})
            parsed_deadline = parse_utc_iso_to_epoch(budget_payload.get("deadline_utc"))
            if parsed_deadline is not None:
                budget_state["deadline_ts"] = parsed_deadline
                budget_state["clock_start_ts"] = (
                    parsed_deadline - float(budget_state["hours"]) * 3600.0
                )
                logger.info(
                    "time budget resume: deadline_utc=%s remaining=%.1f s",
                    budget_payload.get("deadline_utc"),
                    float(remaining_budget_s(parsed_deadline) or 0.0),
                )
        except Exception:
            logger.warning("failed to read time budget from rolling_best_setting.yml")

    if (
        budget_state["enabled"]
        and budget_state["scope"] == "total_wall_clock"
        and budget_state["deadline_ts"] is None
    ):
        budget_state["clock_start_ts"] = wall_clock_start_ts
        budget_state["deadline_ts"] = compute_budget_deadline(
            wall_clock_start_ts,
            float(budget_state["hours"]),
        )

    return budget_state


def _current_time_budget_status(budget_state: dict[str, object]) -> dict | None:
    """Return the serialized time-budget status payload.

    Examples:
        >>> status = {
        ...     "enabled": False,
        ...     "hours": 0.0,
        ...     "scope": "x",
        ...     "cutover_mode": "y",
        ...     "deadline_ts": None,
        ...     "clock_start_ts": None,
        ...     "cutover_triggered": False,
        ...     "cutover_stage": "none",
        ... }
        >>> _current_time_budget_status(status) is None
        True
    """
    return build_time_budget_status(
        enabled=bool(budget_state["enabled"]),
        hours=float(budget_state["hours"]),
        scope=str(budget_state["scope"]),
        cutover_mode=str(budget_state["cutover_mode"]),
        deadline_ts=budget_state["deadline_ts"],
        clock_start_ts=budget_state["clock_start_ts"],
        cutover_triggered=bool(budget_state["cutover_triggered"]),
        cutover_stage=str(budget_state["cutover_stage"]),
    )


def _log_run_configuration(budget_state: dict[str, object]) -> None:
    """Emit top-level run configuration logs after logging is configured.

    Examples:
        >>> budget_state = {
        ...     "enabled": False,
        ...     "hours": 0.0,
        ...     "scope": "training_only",
        ...     "cutover_mode": "stop",
        ...     "deadline_ts": None,
        ...     "clock_start_ts": None,
        ...     "cutover_triggered": False,
        ...     "cutover_stage": "none",
        ... }
        >>> _log_run_configuration(budget_state)
    """
    logger.info(
        "model toggles: knn_enabled=%s xgb_enabled=%s crf_enabled=%s",
        KNN_ENABLED,
        XGB_ENABLED,
        CRF_ENABLED,
    )
    logger.info(
        (
            "proposal heuristics: preset=%s min_area_px=%s min_length_m=%.2f "
            "max_width_m=%.2f min_skeleton_ratio=%.2f min_pca_ratio=%.2f "
            "max_circularity=%.2f min_mean_score=%.2f max_road_overlap=%.2f"
        ),
        cfg.postprocess.novel_proposals.heuristic_preset,
        cfg.postprocess.novel_proposals.min_area_px,
        float(cfg.postprocess.novel_proposals.min_length_m),
        float(cfg.postprocess.novel_proposals.max_width_m),
        float(cfg.postprocess.novel_proposals.min_skeleton_ratio),
        float(cfg.postprocess.novel_proposals.min_pca_ratio),
        float(cfg.postprocess.novel_proposals.max_circularity),
        float(cfg.postprocess.novel_proposals.min_mean_score),
        float(cfg.postprocess.novel_proposals.max_road_overlap),
    )
    if budget_state["enabled"]:
        logger.info(
            "time budget: enabled=%s hours=%.2f scope=%s cutover_mode=%s deadline_utc=%s",
            budget_state["enabled"],
            float(budget_state["hours"]),
            budget_state["scope"],
            budget_state["cutover_mode"],
            (_current_time_budget_status(budget_state) or {}).get("deadline_utc"),
        )


def _initialize_union_state(
    *,
    shape_dir: str,
    resume_run: bool,
) -> tuple[dict[tuple[str, str], dict[str, str | int]], Callable[..., None]]:
    """Create rolling union targets and return an append helper.

    Examples:
        >>> callable(_initialize_union_state)
        True
    """
    union_backup_every = int(cfg.runtime.union_backup_every or 0)
    union_root = os.path.join(shape_dir, "unions")
    union_streams = ["knn", "xgb", "champion"]
    union_variants = ["raw", "crf", "shadow"]
    union_states: dict[tuple[str, str], dict[str, str | int]] = {}
    for stream in union_streams:
        for variant in union_variants:
            union_dir = os.path.join(union_root, stream, variant)
            os.makedirs(union_dir, exist_ok=True)
            union_path = os.path.join(union_dir, "union.shp")
            feature_id = count_shapefile_features(union_path) if resume_run else 0
            union_states[(stream, variant)] = {
                "path": union_path,
                "backup_dir": os.path.join(union_dir, "backups"),
                "feature_id": feature_id,
            }
            if resume_run and feature_id:
                logger.info(
                    "resume union: %s/%s features=%s",
                    stream,
                    variant,
                    feature_id,
                )
    logger.info(
        "rolling unions: root=%s backup_every=%s",
        union_root,
        union_backup_every,
    )

    def _append_union(
        stream: str,
        variant: str,
        mask,
        ref_path: str,
        step: int,
    ) -> None:
        state = union_states[(stream, variant)]
        union_path = str(state["path"])
        backup_dir = str(state["backup_dir"])
        feature_id = int(state["feature_id"])
        state["feature_id"] = append_mask_to_union_shapefile(
            mask,
            ref_path,
            union_path,
            start_id=feature_id,
        )
        if union_backup_every > 0 and step % union_backup_every == 0:
            backup_union_shapefile(union_path, backup_dir, step)

    return union_states, _append_union


def _resolve_tile_sets(
    *,
    training_enabled: bool,
    auto_split_tiles: bool,
    gt_vector_paths: list[str] | None,
) -> dict[str, object]:
    """Resolve source/validation/holdout tiles for the active mode.

    Examples:
        >>> callable(_resolve_tile_sets)
        True
    """
    inference_dir = None
    inference_glob = "*.tif"
    if not training_enabled:
        holdout_tiles, inference_dir, inference_glob = resolve_inference_tiles(
            infer_tiles_dir=cfg.io.inference.tiles_dir,
            infer_tile_glob=cfg.io.inference.tile_glob,
            infer_tiles=list(cfg.io.inference.tiles),
            legacy_inference_dir=cfg.io.paths.inference_dir,
            legacy_inference_glob=cfg.io.paths.inference_glob,
            legacy_holdout_tiles=list(cfg.io.paths.holdout_tiles),
            logger=logger,
        )
        logger.info(
            "inference-only mode: holdout=%s bundle_dir=%s",
            len(holdout_tiles),
            cfg.io.inference.model_bundle_dir,
        )
        return {
            "auto_split_tiles": False,
            "gt_tiles": [],
            "source_tiles": [],
            "val_tiles": [],
            "holdout_tiles": holdout_tiles,
            "inference_dir": inference_dir,
            "inference_glob": inference_glob,
        }

    if auto_split_tiles:
        gt_tiles, holdout_tiles = resolve_tiles_from_gt_presence(
            cfg.io.auto_split.tiles_dir,
            cfg.io.auto_split.tile_glob,
            gt_vector_paths,
            downsample_factor=cfg.io.auto_split.gt_presence_downsample,
            num_workers=cfg.io.auto_split.gt_presence_workers,
        )
        val_tiles = list(gt_tiles)
        logger.info(
            "auto split tiles: gt_tiles=%s inference_tiles=%s",
            len(val_tiles),
            len(holdout_tiles),
        )
        if not val_tiles:
            raise ValueError("no GT-positive tiles resolved for LOO training")
        if len(val_tiles) <= int(cfg.training.loo.val_tiles_per_fold or 1):
            raise ValueError(
                "training.loo.val_tiles_per_fold must be < number of GT-positive tiles"
            )
        return {
            "auto_split_tiles": True,
            "gt_tiles": gt_tiles,
            "source_tiles": [],
            "val_tiles": val_tiles,
            "holdout_tiles": holdout_tiles,
            "inference_dir": None,
            "inference_glob": "*.tif",
        }

    source_tiles = list(cfg.io.paths.source_tiles or [cfg.io.paths.source_tile])
    val_tiles = list(cfg.io.paths.val_tiles)
    holdout_tiles, inference_dir, inference_glob = resolve_inference_tiles(
        infer_tiles_dir=cfg.io.inference.tiles_dir,
        infer_tile_glob=cfg.io.inference.tile_glob,
        infer_tiles=list(cfg.io.inference.tiles),
        legacy_inference_dir=cfg.io.paths.inference_dir,
        legacy_inference_glob=cfg.io.paths.inference_glob,
        legacy_holdout_tiles=list(cfg.io.paths.holdout_tiles),
        logger=logger,
    )
    if not source_tiles:
        raise ValueError(
            "io.paths.source_tiles/source_tile must be set when io.auto_split.enabled=false"
        )
    if not val_tiles:
        raise ValueError(
            "io.paths.val_tiles must be set when io.auto_split.enabled=false"
        )
    logger.info(
        "manual tiles: source=%s val=%s holdout=%s",
        len(source_tiles),
        len(val_tiles),
        len(holdout_tiles),
    )
    return {
        "auto_split_tiles": False,
        "gt_tiles": [],
        "source_tiles": source_tiles,
        "val_tiles": val_tiles,
        "holdout_tiles": holdout_tiles,
        "inference_dir": inference_dir,
        "inference_glob": inference_glob,
    }


def _resolve_feature_cache(auto_split_tiles: bool) -> tuple[str, str | None]:
    """Resolve feature-cache mode and create the cache directory if needed.

    Examples:
        >>> callable(_resolve_feature_cache)
        True
    """
    feature_cache_mode = cfg.runtime.feature_cache_mode
    if feature_cache_mode not in {"disk", "memory"}:
        raise ValueError("FEATURE_CACHE_MODE must be 'disk' or 'memory'")
    if (
        feature_cache_mode == "memory"
        and auto_split_tiles
        and cfg.training.loo.enabled
        and cfg.model.augmentation.enabled
    ):
        logger.warning(
            "feature_cache_mode=memory with LOO+augmentation causes "
            "repeated DINO extraction; forcing disk cache"
        )
        feature_cache_mode = "disk"
    if feature_cache_mode == "disk":
        feature_dir = cfg.io.paths.feature_dir
        os.makedirs(feature_dir, exist_ok=True)
    else:
        feature_dir = None
    logger.info("feature cache mode: %s", feature_cache_mode)
    return feature_cache_mode, feature_dir


def main():
    """Run the full segmentation pipeline for configured tiles.

    Examples:
        >>> callable(main)
        True
    """
    t0_main = time_start()
    model_name = cfg.model.backbone.name

    run_paths = _create_run_directories()
    run_dir = run_paths["run_dir"]
    processed_log_path = os.path.join(run_dir, "processed_tiles.jsonl")
    rolling_best_settings_path = os.path.join(run_dir, "rolling_best_setting.yml")
    resume_run = bool(cfg.runtime.resume_run)
    processed_tiles = _load_processed_tiles(processed_log_path, resume_run)

    wall_clock_start_ts = time.time()
    budget_state = _initialize_time_budget_state(
        resume_run=resume_run,
        rolling_best_settings_path=rolling_best_settings_path,
        wall_clock_start_ts=wall_clock_start_ts,
    )
    _log_run_configuration(budget_state)

    _, append_union = _initialize_union_state(
        shape_dir=run_paths["shape_dir"],
        resume_run=resume_run,
    )

    _log_phase("START", "init_model")
    model, processor, device = init_model(model_name)
    _log_phase("END", "init_model")

    ps = cfg.model.backbone.patch_size
    tile_size = cfg.model.tiling.tile_size
    stride = cfg.model.tiling.stride
    training_enabled = bool(cfg.io.training)
    source_label_raster = cfg.io.paths.source_label_raster
    gt_vector_paths = cfg.io.paths.eval_gt_vectors
    context_radius = int(cfg.model.banks.feat_context_radius or 0)
    auto_split_tiles = bool(cfg.io.auto_split.enabled and training_enabled)

    tile_sets = _resolve_tile_sets(
        training_enabled=training_enabled,
        auto_split_tiles=auto_split_tiles,
        gt_vector_paths=gt_vector_paths,
    )
    feature_cache_mode, feature_dir = _resolve_feature_cache(
        bool(tile_sets["auto_split_tiles"])
    )

    if (
        training_enabled
        and budget_state["enabled"]
        and budget_state["scope"] == "training_only"
        and budget_state["deadline_ts"] is None
    ):
        budget_state["clock_start_ts"] = time.time()
        budget_state["deadline_ts"] = compute_budget_deadline(
            float(budget_state["clock_start_ts"]),
            float(budget_state["hours"]),
        )
    resolved_model_bundle_dir = cfg.io.inference.model_bundle_dir
    if not training_enabled:
        resolved_model_bundle_dir = _resolve_inference_model_bundle_dir(
            cfg.io.inference.model_bundle_dir,
            output_root=cfg.io.paths.output_dir,
            current_run_dir=run_dir,
        )

    common = {
        "append_union": append_union,
        "budget_hours": float(budget_state["hours"]),
        "budget_scope": str(budget_state["scope"]),
        "budget_state": budget_state,
        "bundle_output_dir": cfg.io.inference.model_bundle_dir
        or os.path.join(run_dir, "model_bundle"),
        "bundle_save_enabled": bool(cfg.io.inference.save_bundle),
        "context_radius": context_radius,
        "current_time_budget_status": lambda: _current_time_budget_status(budget_state),
        "device": device,
        "feature_cache_mode": feature_cache_mode,
        "feature_dir": feature_dir,
        "gt_vector_paths": gt_vector_paths,
        "holdout_phase_metrics": {},
        "inference_plot_dir": run_paths["inference_plot_dir"],
        "model": model,
        "model_bundle_dir": resolved_model_bundle_dir,
        "model_name": model_name,
        "plot_dir": run_paths["plot_dir"],
        "processed_log_path": processed_log_path,
        "processed_tiles": processed_tiles,
        "processor": processor,
        "ps": ps,
        "resample_factor": cfg.model.backbone.resample_factor,
        "roads_mask_path": cfg.io.paths.roads_mask_path,
        "rolling_best_settings_path": rolling_best_settings_path,
        "run_dir": run_dir,
        "shape_dir": run_paths["shape_dir"],
        "source_label_raster": source_label_raster,
        "stride": stride,
        "tile_size": tile_size,
        "train_image_ids": [],
        "consolidation_tiles": [],
        "should_consolidate": True,
        "val_phase_metrics": {},
        "validation_plot_dir": run_paths["validation_plot_dir"],
        "xgb_enabled": XGB_ENABLED,
    }

    if not training_enabled:
        run_inference_only(
            common,
            holdout_tiles=list(tile_sets["holdout_tiles"]),
            inference_dir=tile_sets["inference_dir"],
            inference_glob=str(tile_sets["inference_glob"]),
        )
    elif not bool(tile_sets["auto_split_tiles"]):
        run_manual_training(
            common,
            source_tiles=list(tile_sets["source_tiles"]),
            val_tiles=list(tile_sets["val_tiles"]),
            holdout_tiles=list(tile_sets["holdout_tiles"]),
            inference_dir=tile_sets["inference_dir"],
            inference_glob=str(tile_sets["inference_glob"]),
        )
    else:
        if not cfg.training.loo.enabled:
            raise ValueError("training.loo.enabled must be true for this pipeline")
        run_loo_training(
            common,
            gt_tiles=list(tile_sets["gt_tiles"]),
            val_tiles=list(tile_sets["val_tiles"]),
            holdout_tiles=list(tile_sets["holdout_tiles"]),
            min_train_tiles=int(cfg.training.loo.min_train_tiles or 1),
            val_tiles_per_fold=int(cfg.training.loo.val_tiles_per_fold or 1),
            min_gt_positive_pixels=int(cfg.training.loo.min_gt_positive_pixels or 0),
            low_gt_policy=str(cfg.training.loo.low_gt_policy),
            budget_enabled=bool(budget_state["enabled"]),
            budget_deadline_ts=budget_state["deadline_ts"],
            budget_cutover_mode=str(budget_state["cutover_mode"]),
        )

    if common["should_consolidate"]:
        consolidate_cached_features(
            feature_cache_mode=feature_cache_mode,
            feature_dir=feature_dir,
            train_image_ids=list(common["train_image_ids"]),
            inference_tiles=list(common["consolidation_tiles"]),
        )

    emit_performance_summary("run_complete")
    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()
