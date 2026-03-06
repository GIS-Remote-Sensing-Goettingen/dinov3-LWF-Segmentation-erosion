"""Primary pipeline entrypoint for SegEdge."""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import yaml

from ..core.config_loader import cfg
from ..core.features import hybrid_feature_spec_hash, serialize_xgb_feature_stats
from ..core.io_utils import (
    append_mask_to_union_shapefile,
    backup_union_shapefile,
    consolidate_features_for_image,
    count_shapefile_features,
    export_best_settings,
)
from ..core.logging_utils import setup_logging
from ..core.plotting import save_dino_channel_importance_plot
from ..core.timing_utils import time_end, time_start
from ..core.xdboost import train_xgb_classifier
from .artifacts import (
    load_model_bundle,
    save_model_bundle,
    validate_bundle_compatibility,
)
from .common import (
    build_training_artifacts_for_tiles,
    init_model,
    resolve_tiles_from_gt_presence,
)
from .inference_flow import resolve_inference_tiles, run_holdout_inference
from .runtime_utils import (
    _log_phase,
    _summarize_phase_metrics,
    _update_phase_metrics,
    _weighted_mean,
    build_time_budget_status,
    compute_budget_deadline,
    infer_on_holdout,
    is_budget_exceeded,
    load_b_tile_context,
    parse_utc_iso_to_epoch,
    remaining_budget_s,
    summarize_phase_metrics_mean_std,
    write_rolling_best_config,
)
from .tuning import TimeBudgetExceededError, tune_on_validation_multi

# Config-driven flags
KNN_ENABLED = bool(cfg.search.knn.enabled)
XGB_ENABLED = bool(cfg.search.xgb.enabled)
CRF_ENABLED = bool(cfg.search.crf.enabled)

logger = logging.getLogger(__name__)


def _build_loo_folds(
    gt_tiles: list[str],
    val_tiles_per_fold: int,
) -> list[dict[str, list[str]]]:
    """Build deterministic cyclic LOO-style folds with configurable val window.

    Examples:
        >>> folds = _build_loo_folds(["a", "b", "c", "d"], 2)
        >>> len(folds), folds[0]["val_paths"], folds[0]["train_paths"]
        (4, ['a', 'b'], ['c', 'd'])
    """
    n_tiles = len(gt_tiles)
    if n_tiles == 0:
        return []
    val_count = max(1, int(val_tiles_per_fold))
    if val_count >= n_tiles:
        raise ValueError("training.loo.val_tiles_per_fold must be < number of GT tiles")
    folds: list[dict[str, list[str]]] = []
    for i in range(n_tiles):
        val_paths = [gt_tiles[(i + j) % n_tiles] for j in range(val_count)]
        val_set = set(val_paths)
        train_paths = [p for p in gt_tiles if p not in val_set]
        folds.append({"train_paths": train_paths, "val_paths": val_paths})
    return folds


def _aggregate_fold_metrics(results: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate per-tile metrics in a fold using GT-weighted means.

    Examples:
        >>> rows = [{"metrics": {"champion_shadow": {"iou": 0.5, "_weight": 2.0}}}]
        >>> out = _aggregate_fold_metrics(rows)
        >>> round(out["champion_shadow"]["iou"], 3)
        0.5
    """
    if not results:
        return {}
    out: dict[str, dict[str, float]] = {}
    metric_keys = ["iou", "f1", "precision", "recall"]
    for phase in results[0]["metrics"].keys():
        rows = [r["metrics"][phase] for r in results]
        weights = [float(r.get("_weight", 0.0)) for r in rows]
        out[phase] = {
            k: _weighted_mean([r.get(k, 0.0) for r in rows], weights)
            for k in metric_keys
        }
        out[phase]["_weight"] = float(sum(weights))
    return out


def _novel_proposals_payload() -> dict[str, object]:
    """Return the active novel-proposal configuration as a serializable dict.

    Examples:
        >>> callable(_novel_proposals_payload)
        True
    """
    proposal_cfg = cfg.postprocess.novel_proposals
    return {
        "enabled": proposal_cfg.enabled,
        "heuristic_preset": proposal_cfg.heuristic_preset,
        "search_scope": proposal_cfg.search_scope,
        "source": proposal_cfg.source,
        "score_threshold": proposal_cfg.score_threshold,
        "min_area_px": proposal_cfg.min_area_px,
        "min_length_m": proposal_cfg.min_length_m,
        "max_width_m": proposal_cfg.max_width_m,
        "min_skeleton_ratio": proposal_cfg.min_skeleton_ratio,
        "min_pca_ratio": proposal_cfg.min_pca_ratio,
        "max_circularity": proposal_cfg.max_circularity,
        "min_mean_score": proposal_cfg.min_mean_score,
        "max_road_overlap": proposal_cfg.max_road_overlap,
        "connectivity": proposal_cfg.connectivity,
    }


def _export_best_settings_dual(
    best_raw_config,
    best_crf_config,
    model_name,
    img_path,
    img2_path,
    buffer_m,
    pixel_size_m,
    shadow_cfg=None,
    best_xgb_config: dict | None = None,
    champion_source: str | None = None,
    xgb_model_info: dict | None = None,
    model_info: dict | None = None,
    extra_settings: dict | None = None,
    inference_best_settings_path: str | None = None,
) -> tuple[str, str]:
    """Write both current and legacy best-settings YAML files."""
    if inference_best_settings_path is None:
        raise ValueError("inference_best_settings_path must be provided")
    export_best_settings(
        best_raw_config,
        best_crf_config,
        model_name,
        img_path,
        img2_path,
        buffer_m,
        pixel_size_m,
        shadow_cfg=shadow_cfg,
        best_xgb_config=best_xgb_config,
        champion_source=champion_source,
        xgb_model_info=xgb_model_info,
        model_info=model_info,
        extra_settings=extra_settings,
        best_settings_path=inference_best_settings_path,
    )
    legacy_best_settings_path = os.path.join(
        os.path.dirname(inference_best_settings_path), "best_setting.yml"
    )
    export_best_settings(
        best_raw_config,
        best_crf_config,
        model_name,
        img_path,
        img2_path,
        buffer_m,
        pixel_size_m,
        shadow_cfg=shadow_cfg,
        best_xgb_config=best_xgb_config,
        champion_source=champion_source,
        xgb_model_info=xgb_model_info,
        model_info=model_info,
        extra_settings=extra_settings,
        best_settings_path=legacy_best_settings_path,
    )
    return inference_best_settings_path, legacy_best_settings_path


def main():
    """Run the full segmentation pipeline for configured tiles.

    Examples:
        >>> callable(main)
        True
    """

    t0_main = time_start()
    model_name = cfg.model.backbone.name

    # ------------------------------------------------------------
    # Output organization (one folder per run)
    # ------------------------------------------------------------
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
    processed_log_path = os.path.join(run_dir, "processed_tiles.jsonl")
    processed_tiles: set[str] = set()
    rolling_best_settings_path = os.path.join(run_dir, "rolling_best_setting.yml")
    wall_clock_start_ts = time.time()
    if resume_run and os.path.exists(processed_log_path):
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

    time_budget_cfg = cfg.runtime.time_budget
    budget_enabled = bool(time_budget_cfg.enabled)
    budget_hours = float(time_budget_cfg.hours)
    budget_scope = str(time_budget_cfg.scope)
    budget_cutover_mode = str(time_budget_cfg.cutover_mode)
    budget_deadline_ts: float | None = None
    budget_clock_start_ts: float | None = None
    cutover_triggered = False
    cutover_stage = "none"

    if budget_enabled and resume_run and os.path.exists(rolling_best_settings_path):
        try:
            with open(rolling_best_settings_path, "r", encoding="utf-8") as fh:
                rolling_payload = yaml.safe_load(fh) or {}
            budget_payload = rolling_payload.get("time_budget", {})
            parsed_deadline = parse_utc_iso_to_epoch(budget_payload.get("deadline_utc"))
            if parsed_deadline is not None:
                budget_deadline_ts = parsed_deadline
                budget_clock_start_ts = budget_deadline_ts - budget_hours * 3600.0
                logger.info(
                    "time budget resume: deadline_utc=%s remaining=%.1f s",
                    budget_payload.get("deadline_utc"),
                    float(remaining_budget_s(budget_deadline_ts) or 0.0),
                )
        except Exception:
            logger.warning("failed to read time budget from rolling_best_setting.yml")

    if (
        budget_enabled
        and budget_scope == "total_wall_clock"
        and budget_deadline_ts is None
    ):
        budget_clock_start_ts = wall_clock_start_ts
        budget_deadline_ts = compute_budget_deadline(wall_clock_start_ts, budget_hours)
    if budget_enabled:
        logger.info(
            "time budget: enabled=%s hours=%.2f scope=%s cutover_mode=%s deadline_utc=%s",
            budget_enabled,
            budget_hours,
            budget_scope,
            budget_cutover_mode,
            (
                build_time_budget_status(
                    enabled=budget_enabled,
                    hours=budget_hours,
                    scope=budget_scope,
                    cutover_mode=budget_cutover_mode,
                    deadline_ts=budget_deadline_ts,
                    clock_start_ts=budget_clock_start_ts,
                    cutover_triggered=cutover_triggered,
                    cutover_stage=cutover_stage,
                )
                or {}
            ).get("deadline_utc"),
        )

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
        stream: str, variant: str, mask, ref_path: str, step: int
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

    def _current_time_budget_status() -> dict | None:
        return build_time_budget_status(
            enabled=budget_enabled,
            hours=budget_hours,
            scope=budget_scope,
            cutover_mode=budget_cutover_mode,
            deadline_ts=budget_deadline_ts,
            clock_start_ts=budget_clock_start_ts,
            cutover_triggered=cutover_triggered,
            cutover_stage=cutover_stage,
        )

    # ------------------------------------------------------------
    # Init DINOv3 model & processor
    # ------------------------------------------------------------
    _log_phase("START", "init_model")
    model, processor, device = init_model(model_name)
    _log_phase("END", "init_model")
    ps = cfg.model.backbone.patch_size
    tile_size = cfg.model.tiling.tile_size
    stride = cfg.model.tiling.stride

    # ------------------------------------------------------------
    # Resolve paths to imagery + SH_2022 + GT vector labels
    # ------------------------------------------------------------
    source_label_raster = cfg.io.paths.source_label_raster
    gt_vector_paths = cfg.io.paths.eval_gt_vectors
    auto_split_tiles = cfg.io.auto_split.enabled
    training_enabled = bool(cfg.io.training)
    context_radius = int(cfg.model.banks.feat_context_radius or 0)
    model_bundle_info: dict | None = None
    inference_dir = None
    inference_glob = "*.tif"

    # ------------------------------------------------------------
    # Resolve training/validation/holdout tiles
    # ------------------------------------------------------------
    if not training_enabled:
        gt_tiles = []
        source_tiles = []
        val_tiles = []
        holdout_tiles, inference_dir, inference_glob = resolve_inference_tiles(
            infer_tiles_dir=cfg.io.inference.tiles_dir,
            infer_tile_glob=cfg.io.inference.tile_glob,
            infer_tiles=list(cfg.io.inference.tiles),
            legacy_inference_dir=cfg.io.paths.inference_dir,
            legacy_inference_glob=cfg.io.paths.inference_glob,
            legacy_holdout_tiles=list(cfg.io.paths.holdout_tiles),
            logger=logger,
        )
        auto_split_tiles = False
        if not holdout_tiles:
            raise ValueError(
                "no inference tiles resolved; set io.inference.tiles_dir, "
                "io.inference.tiles, or io.paths.holdout_tiles"
            )
        logger.info(
            "inference-only mode: holdout=%s bundle_dir=%s",
            len(holdout_tiles),
            cfg.io.inference.model_bundle_dir,
        )
    elif auto_split_tiles:
        tiles_dir = cfg.io.auto_split.tiles_dir
        tile_glob = cfg.io.auto_split.tile_glob
        downsample_factor = cfg.io.auto_split.gt_presence_downsample
        num_workers = cfg.io.auto_split.gt_presence_workers
        gt_tiles, holdout_tiles = resolve_tiles_from_gt_presence(
            tiles_dir,
            tile_glob,
            gt_vector_paths,
            downsample_factor=downsample_factor,
            num_workers=num_workers,
        )
        val_tiles = list(gt_tiles)
        logger.info(
            "auto split tiles: gt_tiles=%s inference_tiles=%s",
            len(val_tiles),
            len(holdout_tiles),
        )
        if not val_tiles:
            raise ValueError("no GT-positive tiles resolved for LOO training")
        min_train_tiles = int(cfg.training.loo.min_train_tiles or 1)
        val_tiles_per_fold = int(cfg.training.loo.val_tiles_per_fold or 1)
        min_gt_positive_pixels = int(cfg.training.loo.min_gt_positive_pixels or 0)
        low_gt_policy = str(cfg.training.loo.low_gt_policy)
        if len(val_tiles) <= val_tiles_per_fold:
            raise ValueError(
                "training.loo.val_tiles_per_fold must be < number of GT-positive tiles"
            )
        if not holdout_tiles:
            logger.warning(
                "no inference-only tiles resolved; skipping holdout inference"
            )
    else:
        gt_tiles = []
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
        if not holdout_tiles:
            logger.warning(
                "no inference tiles resolved in manual mode; "
                "set io.inference.tiles_dir/io.inference.tiles "
                "or io.paths.inference_dir/io.paths.holdout_tiles"
            )
        logger.info(
            "manual tiles: source=%s val=%s holdout=%s",
            len(source_tiles),
            len(val_tiles),
            len(holdout_tiles),
        )

    # ------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------
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

    val_phase_metrics: dict[str, list[dict]] = {}
    holdout_phase_metrics: dict[str, list[dict]] = {}
    val_buffer_m = None
    val_pixel_size_m = None
    loo_fold_records = []
    skipped_fold_records = []
    best_fold_runtime_artifacts: dict | None = None
    best_fold_runtime_iou = -1.0
    tile_gt_positive_cache: dict[str, int] = {}
    bundle_save_enabled = bool(cfg.io.inference.save_bundle)
    bundle_output_dir = cfg.io.inference.model_bundle_dir or os.path.join(
        run_dir, "model_bundle"
    )

    if not training_enabled:
        model_bundle_dir = cfg.io.inference.model_bundle_dir
        if not model_bundle_dir:
            raise ValueError(
                "io.training=false requires io.inference.model_bundle_dir to be set"
            )
        _log_phase("START", "load_model_bundle")
        loaded_bundle = load_model_bundle(model_bundle_dir)
        validate_bundle_compatibility(
            loaded_bundle["manifest"],
            model_name=cfg.model.backbone.name,
            patch_size=ps,
            resample_factor=cfg.model.backbone.resample_factor,
            tile_size=tile_size,
            stride=stride,
            context_radius=context_radius,
        )
        tuned = loaded_bundle["tuned"]
        pos_bank = loaded_bundle["pos_bank"]
        neg_bank = loaded_bundle["neg_bank"]
        model_bundle_info = {
            "path": model_bundle_dir,
            "version": loaded_bundle["manifest"].get("bundle_version"),
            "manifest": os.path.join(model_bundle_dir, "manifest.yml"),
            "created_utc": loaded_bundle["manifest"].get("created_utc"),
        }
        _log_phase("END", "load_model_bundle")

        write_rolling_best_config(
            rolling_best_settings_path,
            stage="inference_only_bundle_loaded",
            tuned=tuned,
            fold_done=0,
            fold_total=0,
            holdout_done=len(processed_tiles),
            holdout_total=len(holdout_tiles),
            best_fold=None,
            time_budget=_current_time_budget_status(),
            model_bundle=model_bundle_info,
        )

        inference_best_settings_path = os.path.join(
            run_dir, "inference_best_setting.yml"
        )
        xgb_feature_stats_payload = serialize_xgb_feature_stats(
            tuned.get("xgb_feature_stats")
        )
        feature_layout_payload = tuned.get("feature_layout")
        _, legacy_best_settings_path = _export_best_settings_dual(
            tuned["best_raw_config"],
            tuned["best_crf_config"],
            cfg.model.backbone.name,
            [f"model_bundle={model_bundle_dir}"],
            f"inference_tiles={len(holdout_tiles)}",
            0.0,
            0.0,
            shadow_cfg=tuned["shadow_cfg"],
            best_xgb_config=tuned["best_xgb_config"],
            champion_source=tuned["champion_source"],
            extra_settings={
                "mode": "inference_only",
                "tile_size": tile_size,
                "stride": stride,
                "patch_size": ps,
                "feat_context_radius": context_radius,
                "roads_penalty": tuned.get("roads_penalty", 1.0),
                "roads_mask_path": cfg.io.paths.roads_mask_path,
                "model_toggles": {
                    "knn_enabled": tuned.get("knn_enabled"),
                    "xgb_enabled": tuned.get("xgb_enabled"),
                    "crf_enabled": tuned.get("crf_enabled"),
                },
                "novel_proposals": _novel_proposals_payload(),
                "feature_spec_hash": hybrid_feature_spec_hash(),
                "xgb_feature_stats": xgb_feature_stats_payload,
                "feature_layout": feature_layout_payload,
                "time_budget": _current_time_budget_status(),
                "holdout_tiles_count": len(holdout_tiles),
                "inference_dir": inference_dir,
                "inference_glob": inference_glob,
                "model_bundle": model_bundle_info,
            },
            inference_best_settings_path=inference_best_settings_path,
        )
        logger.info(
            "wrote best settings: %s and %s",
            inference_best_settings_path,
            legacy_best_settings_path,
        )

        _log_phase("START", "holdout_inference")
        run_holdout_inference(
            holdout_tiles=holdout_tiles,
            processed_tiles=processed_tiles,
            gt_vector_paths=gt_vector_paths,
            model=model,
            processor=processor,
            device=device,
            pos_bank=pos_bank,
            neg_bank=neg_bank,
            tuned=tuned,
            ps=ps,
            tile_size=tile_size,
            stride=stride,
            feature_dir=feature_dir,
            shape_dir=shape_dir,
            plot_dir=inference_plot_dir,
            context_radius=context_radius,
            holdout_phase_metrics=holdout_phase_metrics,
            append_union=lambda stream, variant, mask, ref_path, step: (
                _append_union(stream, variant, mask, ref_path, step)
                if (stream, variant) in union_states
                else None
            ),
            processed_log_path=processed_log_path,
            write_checkpoint=lambda holdout_done: write_rolling_best_config(
                rolling_best_settings_path,
                stage="holdout_inference",
                tuned=tuned,
                fold_done=0,
                fold_total=0,
                holdout_done=holdout_done,
                holdout_total=len(holdout_tiles),
                best_fold=None,
                time_budget=_current_time_budget_status(),
                model_bundle=model_bundle_info,
            ),
            logger=logger,
        )
        _log_phase("END", "holdout_inference")

        _summarize_phase_metrics(holdout_phase_metrics, "holdout")
        if feature_cache_mode == "disk":
            if feature_dir is None:
                raise ValueError("feature_dir must be set for disk cache mode")
            _log_phase("START", "feature_consolidation")
            for b_path in holdout_tiles:
                image_id_b = os.path.splitext(os.path.basename(b_path))[0]
                consolidate_features_for_image(feature_dir, image_id_b)
            _log_phase("END", "feature_consolidation")

        time_end("main (total)", t0_main)
        return

    if (
        budget_enabled
        and budget_scope == "training_only"
        and budget_deadline_ts is None
    ):
        budget_clock_start_ts = time.time()
        budget_deadline_ts = compute_budget_deadline(
            budget_clock_start_ts, budget_hours
        )

    if not auto_split_tiles:
        _log_phase("START", "manual_training_artifacts")
        (
            pos_bank,
            neg_bank,
            x,
            y,
            image_id_a_list,
            aug_modes,
            xgb_feature_stats,
            feature_layout,
        ) = build_training_artifacts_for_tiles(
            source_tiles,
            source_label_raster,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            feature_cache_mode,
            feature_dir,
            context_radius,
        )
        _log_phase("END", "manual_training_artifacts")

        _log_phase("START", "validation_tuning")
        tuned = tune_on_validation_multi(
            val_tiles,
            gt_vector_paths,
            model,
            processor,
            device,
            pos_bank,
            neg_bank,
            x,
            y,
            xgb_feature_stats,
            feature_layout,
            ps,
            tile_size,
            stride,
            feature_dir,
            context_radius,
        )
        _log_phase("END", "validation_tuning")
        tuned = {
            **tuned,
            "xgb_feature_stats": xgb_feature_stats,
            "feature_layout": feature_layout,
        }
        if bundle_save_enabled and tuned.get("bst") is not None:
            model_bundle_info = save_model_bundle(
                bundle_output_dir,
                tuned,
                pos_bank,
                neg_bank,
                model_name=cfg.model.backbone.name,
                patch_size=ps,
                resample_factor=cfg.model.backbone.resample_factor,
                tile_size=tile_size,
                stride=stride,
                context_radius=context_radius,
            )
            logger.info("saved model bundle: %s", model_bundle_info["path"])
        elif bundle_save_enabled:
            logger.warning(
                "bundle save requested but no XGB model is available; skipping save"
            )

        _log_phase("START", "validation_inference")
        for val_path in val_tiles:
            val_result = infer_on_holdout(
                val_path,
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
                validation_plot_dir,
                context_radius,
                plot_with_metrics=True,
            )
            if val_result["gt_available"]:
                _update_phase_metrics(val_phase_metrics, val_result["metrics"])
            if val_buffer_m is None:
                val_buffer_m = val_result["buffer_m"]
                val_pixel_size_m = val_result["pixel_size_m"]
        _log_phase("END", "validation_inference")

        weighted_phase_metrics: dict[str, dict[str, float]] = {}
        metric_keys = ["iou", "f1", "precision", "recall"]
        for phase, metrics_list in val_phase_metrics.items():
            weights = [float(m.get("_weight", 0.0)) for m in metrics_list]
            weighted_phase_metrics[phase] = {
                key: _weighted_mean([m.get(key, 0.0) for m in metrics_list], weights)
                for key in metric_keys
            }

        manual_best_fold = {
            "fold_index": 1,
            "val_tile": ",".join(val_tiles),
            "val_champion_shadow_iou": float(
                weighted_phase_metrics.get("champion_shadow", {}).get("iou", 0.0)
            ),
        }
        write_rolling_best_config(
            rolling_best_settings_path,
            stage="manual_validation_tuning",
            tuned=tuned,
            fold_done=1,
            fold_total=1,
            holdout_done=len(processed_tiles),
            holdout_total=len(holdout_tiles),
            best_fold=manual_best_fold,
            time_budget=_current_time_budget_status(),
            model_bundle=model_bundle_info,
        )

        inference_best_settings_path = os.path.join(
            run_dir, "inference_best_setting.yml"
        )
        xgb_feature_stats_payload = serialize_xgb_feature_stats(
            tuned.get("xgb_feature_stats")
        )
        feature_layout_payload = tuned.get("feature_layout")
        _, legacy_best_settings_path = _export_best_settings_dual(
            tuned["best_raw_config"],
            tuned["best_crf_config"],
            cfg.model.backbone.name,
            source_tiles,
            f"inference_tiles={len(holdout_tiles)}",
            float(val_buffer_m) if val_buffer_m is not None else 0.0,
            float(val_pixel_size_m) if val_pixel_size_m is not None else 0.0,
            shadow_cfg=tuned["shadow_cfg"],
            best_xgb_config=tuned["best_xgb_config"],
            champion_source=tuned["champion_source"],
            extra_settings={
                "mode": "manual",
                "tile_size": tile_size,
                "stride": stride,
                "patch_size": ps,
                "feat_context_radius": context_radius,
                "neg_alpha": cfg.model.banks.neg_alpha,
                "pos_frac_thresh": cfg.model.banks.pos_frac_thresh,
                "max_pos_bank": cfg.model.banks.max_pos_bank,
                "max_neg_bank": cfg.model.banks.max_neg_bank,
                "roads_penalty": tuned.get("roads_penalty", 1.0),
                "roads_mask_path": cfg.io.paths.roads_mask_path,
                "model_toggles": {
                    "knn_enabled": cfg.search.knn.enabled,
                    "xgb_enabled": cfg.search.xgb.enabled,
                    "crf_enabled": cfg.search.crf.enabled,
                },
                "novel_proposals": _novel_proposals_payload(),
                "feature_spec_hash": hybrid_feature_spec_hash(),
                "xgb_feature_stats": xgb_feature_stats_payload,
                "feature_layout": feature_layout_payload,
                "time_budget": _current_time_budget_status(),
                "cutover_triggered": cutover_triggered,
                "cutover_stage": cutover_stage,
                "source_tiles_count": len(source_tiles),
                "val_tiles_count": len(val_tiles),
                "holdout_tiles_count": len(holdout_tiles),
                "inference_dir": inference_dir,
                "inference_glob": inference_glob,
                "weighted_phase_metrics": weighted_phase_metrics,
                "model_bundle": model_bundle_info,
                "loo": {"enabled": False},
            },
            inference_best_settings_path=inference_best_settings_path,
        )
        logger.info(
            "wrote best settings: %s and %s",
            inference_best_settings_path,
            legacy_best_settings_path,
        )

        _log_phase("START", "holdout_inference")
        run_holdout_inference(
            holdout_tiles=holdout_tiles,
            processed_tiles=processed_tiles,
            gt_vector_paths=gt_vector_paths,
            model=model,
            processor=processor,
            device=device,
            pos_bank=pos_bank,
            neg_bank=neg_bank,
            tuned=tuned,
            ps=ps,
            tile_size=tile_size,
            stride=stride,
            feature_dir=feature_dir,
            shape_dir=shape_dir,
            plot_dir=inference_plot_dir,
            context_radius=context_radius,
            holdout_phase_metrics=holdout_phase_metrics,
            append_union=lambda stream, variant, mask, ref_path, step: (
                _append_union(stream, variant, mask, ref_path, step)
                if (stream, variant) in union_states
                else None
            ),
            processed_log_path=processed_log_path,
            write_checkpoint=lambda holdout_done: write_rolling_best_config(
                rolling_best_settings_path,
                stage="holdout_inference",
                tuned=tuned,
                fold_done=1,
                fold_total=1,
                holdout_done=holdout_done,
                holdout_total=len(holdout_tiles),
                best_fold=manual_best_fold,
                time_budget=_current_time_budget_status(),
                model_bundle=model_bundle_info,
            ),
            logger=logger,
        )
        _log_phase("END", "holdout_inference")

        _summarize_phase_metrics(val_phase_metrics, "validation")
        _summarize_phase_metrics(holdout_phase_metrics, "holdout")

        if feature_cache_mode == "disk":
            if feature_dir is None:
                raise ValueError("feature_dir must be set for disk cache mode")
            _log_phase("START", "feature_consolidation")
            for image_id_a in image_id_a_list:
                consolidate_features_for_image(feature_dir, image_id_a)
            for b_path in val_tiles + holdout_tiles:
                image_id_b = os.path.splitext(os.path.basename(b_path))[0]
                consolidate_features_for_image(feature_dir, image_id_b)
            _log_phase("END", "feature_consolidation")

        time_end("main (total)", t0_main)
        return

    # ------------------------------------------------------------
    # LOO tuning/search: train on N-1 GT tiles, validate on the left-out tile
    # ------------------------------------------------------------
    if not cfg.training.loo.enabled:
        raise ValueError("training.loo.enabled must be true for this pipeline")
    if (
        budget_enabled
        and budget_scope == "training_only"
        and budget_deadline_ts is None
    ):
        budget_clock_start_ts = time.time()
        budget_deadline_ts = compute_budget_deadline(
            budget_clock_start_ts, budget_hours
        )

    def _is_budget_exceeded_now() -> bool:
        return bool(budget_enabled and is_budget_exceeded(budget_deadline_ts))

    def _get_tile_gt_positive_pixels(tile_path: str) -> int:
        if tile_path in tile_gt_positive_cache:
            return tile_gt_positive_cache[tile_path]
        _, _, _, gt_mask_eval_tile, _, _, _ = load_b_tile_context(
            tile_path, gt_vector_paths
        )
        pixels = int(gt_mask_eval_tile.sum()) if gt_mask_eval_tile is not None else 0
        tile_gt_positive_cache[tile_path] = pixels
        return pixels

    loo_folds = _build_loo_folds(val_tiles, val_tiles_per_fold)
    fold_total = len(loo_folds)
    _log_phase("START", "loo_validation_tuning")
    for fold_idx, fold in enumerate(loo_folds, start=1):
        train_tiles = list(fold["train_paths"])
        val_paths = list(fold["val_paths"])
        if _is_budget_exceeded_now():
            cutover_triggered = True
            cutover_stage = "loo_validation_tuning"
            logger.warning(
                "time budget exceeded before LOO fold %s/%s; cutover_mode=%s",
                fold_idx,
                fold_total,
                budget_cutover_mode,
            )
            break

        fold_gt_positive_pixels = int(
            sum(_get_tile_gt_positive_pixels(p) for p in val_paths)
        )
        if (
            low_gt_policy == "skip_fold"
            and fold_gt_positive_pixels < min_gt_positive_pixels
        ):
            logger.warning(
                "LOO fold %s/%s skipped: val_tiles=%s gt_positive_pixels=%s < min_gt_positive_pixels=%s",
                fold_idx,
                fold_total,
                len(val_paths),
                fold_gt_positive_pixels,
                min_gt_positive_pixels,
            )
            skipped_fold_records.append(
                {
                    "fold_index": int(fold_idx),
                    "val_tiles": list(val_paths),
                    "reason": "low_gt",
                    "gt_positive_pixels": int(fold_gt_positive_pixels),
                }
            )
            continue

        if not train_tiles:
            logger.warning(
                "LOO fold %s has no train tiles; reusing validation tile as source",
                fold_idx,
            )
            train_tiles = [val_paths[0]]
        if len(train_tiles) < min_train_tiles:
            logger.warning(
                "LOO fold %s train tiles=%s < min_train_tiles=%s",
                fold_idx,
                len(train_tiles),
                min_train_tiles,
            )
        logger.info(
            "LOO fold %s/%s train_tiles=%s val_tiles=%s",
            fold_idx,
            fold_total,
            len(train_tiles),
            len(val_paths),
        )
        try:
            (
                pos_bank_fold,
                neg_bank_fold,
                x_fold,
                y_fold,
                image_id_a_fold,
                aug_modes_fold,
                xgb_feature_stats_fold,
                feature_layout_fold,
            ) = build_training_artifacts_for_tiles(
                train_tiles,
                source_label_raster,
                model,
                processor,
                device,
                ps,
                tile_size,
                stride,
                feature_cache_mode,
                feature_dir,
                context_radius,
            )
            tuned_fold = tune_on_validation_multi(
                val_paths,
                gt_vector_paths,
                model,
                processor,
                device,
                pos_bank_fold,
                neg_bank_fold,
                x_fold,
                y_fold,
                xgb_feature_stats_fold,
                feature_layout_fold,
                ps,
                tile_size,
                stride,
                feature_dir,
                context_radius,
                should_stop=_is_budget_exceeded_now,
            )
            fold_val_results: list[dict] = []
            fold_plot_dir = os.path.join(validation_plot_dir, f"fold_{fold_idx:02d}")
            for val_path in val_paths:
                if _is_budget_exceeded_now():
                    raise TimeBudgetExceededError("fold_validation_inference")
                fold_result = infer_on_holdout(
                    val_path,
                    gt_vector_paths,
                    model,
                    processor,
                    device,
                    pos_bank_fold,
                    neg_bank_fold,
                    tuned_fold,
                    ps,
                    tile_size,
                    stride,
                    feature_dir,
                    shape_dir,
                    fold_plot_dir,
                    context_radius,
                    plot_with_metrics=True,
                )
                fold_val_results.append(fold_result)
                if fold_result["gt_available"]:
                    _update_phase_metrics(val_phase_metrics, fold_result["metrics"])
                if val_buffer_m is None:
                    val_buffer_m = fold_result["buffer_m"]
                    val_pixel_size_m = fold_result["pixel_size_m"]
        except TimeBudgetExceededError as exc:
            cutover_triggered = True
            cutover_stage = "loo_validation_tuning"
            logger.warning(
                "time budget exceeded during LOO fold %s/%s at stage=%s; stopping training",
                fold_idx,
                fold_total,
                str(exc),
            )
            if loo_fold_records:
                current_best_fold = max(
                    loo_fold_records,
                    key=lambda r: float(r["val_champion_shadow_iou"]),
                )
                write_rolling_best_config(
                    rolling_best_settings_path,
                    stage="loo_validation_tuning_cutover",
                    tuned=current_best_fold["tuned"],
                    fold_done=len(loo_fold_records),
                    fold_total=fold_total,
                    holdout_done=len(processed_tiles),
                    holdout_total=len(holdout_tiles),
                    best_fold=current_best_fold,
                    time_budget=_current_time_budget_status(),
                )
            break

        fold_metrics = _aggregate_fold_metrics(fold_val_results)
        fold_iou = float(fold_metrics["champion_shadow"]["iou"])
        val_tile_summary = ",".join(val_paths)
        loo_fold_records.append(
            {
                "fold_index": fold_idx,
                "val_tile": val_tile_summary,
                "val_tiles": list(val_paths),
                "train_tiles_count": len(train_tiles),
                "val_champion_shadow_iou": fold_iou,
                "val_gt_positive_pixels": int(fold_gt_positive_pixels),
                "roads_penalty": float(tuned_fold.get("roads_penalty", 1.0)),
                "champion_source": tuned_fold["champion_source"],
                "best_raw_config": tuned_fold["best_raw_config"],
                "best_xgb_config": tuned_fold["best_xgb_config"],
                "best_crf_config": tuned_fold["best_crf_config"],
                "best_shadow_config": tuned_fold["shadow_cfg"],
                "phase_metrics": fold_metrics,
                "tuned": tuned_fold,
            }
        )
        if fold_iou > best_fold_runtime_iou:
            best_fold_runtime_iou = fold_iou
            best_fold_runtime_artifacts = {
                "pos_bank": pos_bank_fold,
                "neg_bank": neg_bank_fold,
                "image_id_a_list": image_id_a_fold,
                "aug_modes": aug_modes_fold,
                "xgb_feature_stats": xgb_feature_stats_fold,
                "feature_layout": feature_layout_fold,
            }
        current_best_fold = max(
            loo_fold_records,
            key=lambda r: float(r["val_champion_shadow_iou"]),
        )
        write_rolling_best_config(
            rolling_best_settings_path,
            stage="loo_validation_tuning",
            tuned=current_best_fold["tuned"],
            fold_done=len(loo_fold_records),
            fold_total=fold_total,
            holdout_done=len(processed_tiles),
            holdout_total=len(holdout_tiles),
            best_fold=current_best_fold,
            time_budget=_current_time_budget_status(),
        )
    _log_phase("END", "loo_validation_tuning")
    if not loo_fold_records:
        if cutover_triggered:
            raise ValueError(
                "time budget exhausted before any LOO fold completed; cannot continue to inference"
            )
        if skipped_fold_records:
            raise ValueError(
                "LOO tuning produced no fold records because all folds were skipped by low-GT policy"
            )
        raise ValueError("LOO tuning produced no fold records")

    best_fold = max(
        loo_fold_records,
        key=lambda r: float(r["val_champion_shadow_iou"]),
    )
    selected_tuned = best_fold["tuned"]
    logger.info(
        "LOO selected fold=%s val_tile=%s champion_shadow_iou=%.3f",
        best_fold["fold_index"],
        best_fold["val_tile"],
        float(best_fold["val_champion_shadow_iou"]),
    )

    # ------------------------------------------------------------
    # Final training on all GT tiles with selected LOO hyperparameters
    # ------------------------------------------------------------
    pos_bank: np.ndarray | None = None
    neg_bank: np.ndarray | None = None
    image_id_a_list: list[str] = []
    aug_modes: list[str] = []
    tuned = dict(selected_tuned)
    halt_before_inference = False
    final_stage_name = "final_model_ready"

    if budget_enabled and is_budget_exceeded(budget_deadline_ts):
        logger.warning(
            "time budget exceeded before final_all_gt_training; cutover_mode=%s",
            budget_cutover_mode,
        )
        if budget_cutover_mode in {"immediate_inference", "stop"}:
            cutover_triggered = True
            cutover_stage = "final_all_gt_training"
            if best_fold_runtime_artifacts is None:
                raise ValueError(
                    "time-budget cutover requested, but no fold artifacts are available"
                )
            pos_bank = best_fold_runtime_artifacts["pos_bank"]
            neg_bank = best_fold_runtime_artifacts["neg_bank"]
            image_id_a_list = list(best_fold_runtime_artifacts["image_id_a_list"])
            aug_modes = list(best_fold_runtime_artifacts["aug_modes"])
            if tuned.get("xgb_feature_stats") is None:
                tuned["xgb_feature_stats"] = best_fold_runtime_artifacts[
                    "xgb_feature_stats"
                ]
            if tuned.get("feature_layout") is None:
                tuned["feature_layout"] = best_fold_runtime_artifacts["feature_layout"]
            if budget_cutover_mode == "immediate_inference":
                final_stage_name = "final_model_ready_cutover"
                logger.warning(
                    "time-budget cutover: skipping final all-GT retraining and "
                    "starting inference from best completed fold"
                )
            else:
                final_stage_name = "time_budget_stop"
                halt_before_inference = True
                logger.warning(
                    "time-budget cutover: mode=stop, will write outputs and exit"
                )

    if pos_bank is None:
        _log_phase("START", "final_all_gt_training")
        (
            pos_bank,
            neg_bank,
            X,
            y,
            image_id_a_list,
            aug_modes,
            xgb_feature_stats,
            feature_layout,
        ) = build_training_artifacts_for_tiles(
            gt_tiles,
            source_label_raster,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            feature_cache_mode,
            feature_dir,
            context_radius,
        )
        final_bst = tuned.get("bst")
        if XGB_ENABLED:
            best_xgb_params = selected_tuned["best_xgb_config"].get("params")
            feature_names = (
                list(feature_layout.get("feature_names", []))
                if feature_layout is not None
                else None
            )
            if feature_names is not None and len(feature_names) != X.shape[1]:
                logger.warning(
                    "final training feature layout mismatch: names=%s X=%s; disabling names",
                    len(feature_names),
                    X.shape[1],
                )
                feature_names = None
            final_bst = train_xgb_classifier(
                X,
                y,
                use_gpu=cfg.search.xgb.use_gpu,
                num_boost_round=cfg.search.xgb.num_boost_round,
                verbose_eval=cfg.search.xgb.verbose_eval,
                param_overrides=best_xgb_params,
                feature_names=feature_names,
            )
        tuned = {
            **selected_tuned,
            "bst": final_bst,
            "xgb_feature_stats": xgb_feature_stats,
            "feature_layout": feature_layout,
        }
        _log_phase("END", "final_all_gt_training")
    if bundle_save_enabled and tuned.get("bst") is not None:
        model_bundle_info = save_model_bundle(
            bundle_output_dir,
            tuned,
            pos_bank,
            neg_bank,
            model_name=cfg.model.backbone.name,
            patch_size=ps,
            resample_factor=cfg.model.backbone.resample_factor,
            tile_size=tile_size,
            stride=stride,
            context_radius=context_radius,
        )
        logger.info("saved model bundle: %s", model_bundle_info["path"])
    elif bundle_save_enabled:
        logger.warning(
            "bundle save requested but no XGB model is available; skipping save"
        )
    write_rolling_best_config(
        rolling_best_settings_path,
        stage=final_stage_name,
        tuned=tuned,
        fold_done=len(loo_fold_records),
        fold_total=fold_total,
        holdout_done=len(processed_tiles),
        holdout_total=len(holdout_tiles),
        best_fold=best_fold,
        time_budget=_current_time_budget_status(),
        model_bundle=model_bundle_info,
    )

    loo_fold_export = []
    for fold in loo_fold_records:
        loo_fold_export.append(
            {
                "fold_index": int(fold["fold_index"]),
                "val_tile": fold["val_tile"],
                "val_tiles": list(fold.get("val_tiles", [])),
                "train_tiles_count": int(fold["train_tiles_count"]),
                "val_champion_shadow_iou": float(fold["val_champion_shadow_iou"]),
                "val_gt_positive_pixels": int(fold.get("val_gt_positive_pixels", 0)),
                "roads_penalty": float(fold["roads_penalty"]),
                "champion_source": fold["champion_source"],
                "best_raw_config": fold["best_raw_config"],
                "best_xgb_config": fold["best_xgb_config"],
                "best_crf_config": fold["best_crf_config"],
                "best_shadow_config": fold["best_shadow_config"],
            }
        )
    loo_phase_mean_std = summarize_phase_metrics_mean_std(val_phase_metrics)

    weighted_phase_metrics: dict[str, dict[str, float]] = {}
    metric_keys = ["iou", "f1", "precision", "recall"]
    for phase, metrics_list in val_phase_metrics.items():
        weights = [float(m.get("_weight", 0.0)) for m in metrics_list]
        weighted_phase_metrics[phase] = {
            key: _weighted_mean([m.get(key, 0.0) for m in metrics_list], weights)
            for key in metric_keys
        }

    inference_best_settings_path = os.path.join(run_dir, "inference_best_setting.yml")
    bst = tuned.get("bst")
    xgb_model_info: dict[str, object] = {}
    dino_importance_plot = save_dino_channel_importance_plot(
        bst,
        tuned.get("feature_layout"),
        cfg.io.paths.plot_dir,
        top_k=20,
    )
    if bst is not None:
        best_iter = getattr(bst, "best_iteration", None)
        best_score = getattr(bst, "best_score", None)
        xgb_model_info = {
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "best_score": float(best_score) if best_score is not None else None,
            "num_features": int(bst.num_features()),
            "attributes": bst.attributes(),
            "dino_importance_plot": dino_importance_plot,
        }
    xgb_feature_stats_payload = serialize_xgb_feature_stats(
        tuned.get("xgb_feature_stats")
    )
    feature_layout_payload = tuned.get("feature_layout")
    model_info = {
        "backbone": {
            "name": cfg.model.backbone.name,
            "patch_size": cfg.model.backbone.patch_size,
            "resample_factor": cfg.model.backbone.resample_factor,
        },
        "tiling": {"tile_size": tile_size, "stride": stride},
        "augmentation": {
            "enabled": cfg.model.augmentation.enabled,
            "modes": aug_modes,
        },
        "xgb_search": {
            "enabled": cfg.search.xgb.enabled,
            "use_gpu": cfg.search.xgb.use_gpu,
            "num_boost_round": cfg.search.xgb.num_boost_round,
            "early_stop": cfg.search.xgb.early_stop,
            "fixed_threshold": cfg.search.xgb.fixed_threshold,
            "param_grid_size": len(cfg.search.xgb.param_grid),
        },
        "knn_search": {
            "enabled": cfg.search.knn.enabled,
            "k_values": cfg.search.knn.k_values,
            "threshold_range": {
                "start": cfg.search.knn.thresholds.start,
                "stop": cfg.search.knn.thresholds.stop,
                "count": cfg.search.knn.thresholds.count,
            },
        },
        "crf_search": {
            "enabled": cfg.search.crf.enabled,
            "max_configs": cfg.search.crf.max_configs,
        },
        "hybrid_features": {
            "enabled": cfg.model.hybrid_features.enabled,
            "feature_spec_hash": hybrid_feature_spec_hash(),
            "feature_layout": feature_layout_payload,
            "xgb_feature_stats": xgb_feature_stats_payload,
        },
        "time_budget": {
            "enabled": budget_enabled,
            "hours": budget_hours,
            "scope": budget_scope,
            "cutover_mode": budget_cutover_mode,
            "cutover_triggered": cutover_triggered,
            "cutover_stage": cutover_stage,
        },
    }
    _, legacy_best_settings_path = _export_best_settings_dual(
        tuned["best_raw_config"],
        tuned["best_crf_config"],
        cfg.model.backbone.name,
        gt_tiles,
        f"inference_tiles={len(holdout_tiles)}",
        float(val_buffer_m) if val_buffer_m is not None else 0.0,
        float(val_pixel_size_m) if val_pixel_size_m is not None else 0.0,
        shadow_cfg=tuned["shadow_cfg"],
        best_xgb_config=tuned["best_xgb_config"],
        champion_source=tuned["champion_source"],
        xgb_model_info=xgb_model_info,
        model_info=model_info,
        extra_settings={
            "mode": "loo",
            "tile_size": tile_size,
            "stride": stride,
            "patch_size": ps,
            "feat_context_radius": context_radius,
            "neg_alpha": cfg.model.banks.neg_alpha,
            "pos_frac_thresh": cfg.model.banks.pos_frac_thresh,
            "max_pos_bank": cfg.model.banks.max_pos_bank,
            "max_neg_bank": cfg.model.banks.max_neg_bank,
            "roads_penalty": tuned.get("roads_penalty", 1.0),
            "roads_mask_path": cfg.io.paths.roads_mask_path,
            "model_toggles": {
                "knn_enabled": cfg.search.knn.enabled,
                "xgb_enabled": cfg.search.xgb.enabled,
                "crf_enabled": cfg.search.crf.enabled,
            },
            "novel_proposals": _novel_proposals_payload(),
            "feature_spec_hash": hybrid_feature_spec_hash(),
            "xgb_feature_stats": xgb_feature_stats_payload,
            "feature_layout": feature_layout_payload,
            "time_budget": _current_time_budget_status(),
            "cutover_triggered": cutover_triggered,
            "cutover_stage": cutover_stage,
            "model_bundle": model_bundle_info,
            "val_tiles_count": len(val_tiles),
            "holdout_tiles_count": len(holdout_tiles),
            "weighted_phase_metrics": weighted_phase_metrics,
            "loo": {
                "enabled": True,
                "fold_count": len(loo_fold_export),
                "fold_total": fold_total,
                "min_train_tiles": min_train_tiles,
                "val_tiles_per_fold": val_tiles_per_fold,
                "min_gt_positive_pixels": min_gt_positive_pixels,
                "low_gt_policy": low_gt_policy,
                "selected_fold_index": int(best_fold["fold_index"]),
                "selected_val_tile": best_fold["val_tile"],
                "phase_metrics_mean_std": loo_phase_mean_std,
                "skipped_folds": skipped_fold_records,
                "folds": loo_fold_export,
            },
        },
        inference_best_settings_path=inference_best_settings_path,
    )
    logger.info(
        "wrote best settings: %s and %s",
        inference_best_settings_path,
        legacy_best_settings_path,
    )

    if halt_before_inference:
        _summarize_phase_metrics(val_phase_metrics, "loo_validation")
        _summarize_phase_metrics(holdout_phase_metrics, "holdout")
        time_end("main (total)", t0_main)
        return

    _log_phase("START", "holdout_inference")
    run_holdout_inference(
        holdout_tiles=holdout_tiles,
        processed_tiles=processed_tiles,
        gt_vector_paths=gt_vector_paths,
        model=model,
        processor=processor,
        device=device,
        pos_bank=pos_bank,
        neg_bank=neg_bank,
        tuned=tuned,
        ps=ps,
        tile_size=tile_size,
        stride=stride,
        feature_dir=feature_dir,
        shape_dir=shape_dir,
        plot_dir=inference_plot_dir,
        context_radius=context_radius,
        holdout_phase_metrics=holdout_phase_metrics,
        append_union=lambda stream, variant, mask, ref_path, step: (
            _append_union(stream, variant, mask, ref_path, step)
            if (stream, variant) in union_states
            else None
        ),
        processed_log_path=processed_log_path,
        write_checkpoint=lambda holdout_done: write_rolling_best_config(
            rolling_best_settings_path,
            stage="holdout_inference",
            tuned=tuned,
            fold_done=len(loo_fold_records),
            fold_total=fold_total,
            holdout_done=holdout_done,
            holdout_total=len(holdout_tiles),
            best_fold=best_fold,
            time_budget=_current_time_budget_status(),
            model_bundle=model_bundle_info,
        ),
        logger=logger,
    )
    _log_phase("END", "holdout_inference")

    _summarize_phase_metrics(val_phase_metrics, "loo_validation")
    _summarize_phase_metrics(holdout_phase_metrics, "holdout")

    # ------------------------------------------------------------
    # Consolidate tile-level feature files (.npy) → one per image
    # ------------------------------------------------------------
    if feature_cache_mode == "disk":
        if feature_dir is None:
            raise ValueError("feature_dir must be set for disk cache mode")
        _log_phase("START", "feature_consolidation")
        for image_id_a in image_id_a_list:
            consolidate_features_for_image(feature_dir, image_id_a)
        for b_path in val_tiles + holdout_tiles:
            image_id_b = os.path.splitext(os.path.basename(b_path))[0]
            consolidate_features_for_image(feature_dir, image_id_b)
        _log_phase("END", "feature_consolidation")

    time_end("main (total)", t0_main)


if __name__ == "__main__":
    main()
