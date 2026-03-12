"""Manual training/validation/holdout workflow."""

from __future__ import annotations

import logging
import os

from ...core.config_loader import cfg
from ...core.features import hybrid_feature_spec_hash
from ..artifacts import save_model_bundle
from ..common import build_training_artifacts_for_tiles
from ..runtime_utils import (
    _log_phase,
    _summarize_phase_metrics,
    _update_phase_metrics,
    infer_on_holdout,
    write_rolling_best_config,
)
from ..tuning import tune_on_validation_multi
from .shared import (
    _export_best_settings_dual,
    _inference_score_prior_payload,
    _novel_proposals_payload,
    build_weighted_phase_metrics,
    export_shared_feature_payloads,
    run_holdout_with_checkpoint,
)

logger = logging.getLogger(__name__)


def run_manual_training(
    common: dict,
    *,
    source_tiles: list[str],
    val_tiles: list[str],
    holdout_tiles: list[str],
    inference_dir: str | None,
    inference_glob: str,
) -> None:
    """Run manual train/validate/infer workflow.

    Examples:
        >>> callable(run_manual_training)
        True
    """
    ps = common["ps"]
    tile_size = common["tile_size"]
    stride = common["stride"]
    context_radius = common["context_radius"]
    model = common["model"]
    processor = common["processor"]
    device = common["device"]
    source_label_raster = common["source_label_raster"]
    gt_vector_paths = common["gt_vector_paths"]
    feature_cache_mode = common["feature_cache_mode"]
    feature_dir = common["feature_dir"]
    inference_feature_dir = common["inference_feature_dir"]
    val_phase_metrics = common["val_phase_metrics"]
    holdout_phase_metrics = common["holdout_phase_metrics"]
    run_dir = common["run_dir"]
    current_time_budget_status = common["current_time_budget_status"]

    _log_phase("START", "manual_training_artifacts")
    (
        pos_bank,
        neg_bank,
        x,
        y,
        image_id_a_list,
        _aug_modes,
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
    model_bundle_info = None
    if common["bundle_save_enabled"] and tuned.get("bst") is not None:
        model_bundle_info = save_model_bundle(
            common["bundle_output_dir"],
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
    elif common["bundle_save_enabled"]:
        logger.warning(
            "bundle save requested but no XGB model is available; skipping save"
        )

    val_buffer_m = None
    val_pixel_size_m = None
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
            common["shape_dir"],
            common["validation_plot_dir"],
            context_radius,
            plot_with_metrics=True,
        )
        if val_result["gt_available"]:
            _update_phase_metrics(val_phase_metrics, val_result["metrics"])
        if val_buffer_m is None:
            val_buffer_m = val_result["buffer_m"]
            val_pixel_size_m = val_result["pixel_size_m"]
    _log_phase("END", "validation_inference")

    weighted_phase_metrics = build_weighted_phase_metrics(val_phase_metrics)
    manual_best_fold = {
        "fold_index": 1,
        "val_tile": ",".join(val_tiles),
        "val_champion_shadow_iou": float(
            weighted_phase_metrics.get("champion_shadow", {}).get("iou", 0.0)
        ),
    }
    write_rolling_best_config(
        common["rolling_best_settings_path"],
        stage="manual_validation_tuning",
        tuned=tuned,
        fold_done=1,
        fold_total=1,
        holdout_done=len(common["processed_tiles"]),
        holdout_total=len(holdout_tiles),
        best_fold=manual_best_fold,
        time_budget=current_time_budget_status(),
        model_bundle=model_bundle_info,
    )

    inference_best_settings_path = os.path.join(run_dir, "inference_best_setting.yml")
    xgb_feature_stats_payload, feature_layout_payload = export_shared_feature_payloads(
        tuned
    )
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
            "roads_mask_path": common["roads_mask_path"],
            "model_toggles": {
                "knn_enabled": cfg.search.knn.enabled,
                "xgb_enabled": cfg.search.xgb.enabled,
                "crf_enabled": cfg.search.crf.enabled,
            },
            "novel_proposals": _novel_proposals_payload(),
            "inference_score_prior": _inference_score_prior_payload(),
            "feature_spec_hash": hybrid_feature_spec_hash(),
            "xgb_feature_stats": xgb_feature_stats_payload,
            "feature_layout": feature_layout_payload,
            "time_budget": current_time_budget_status(),
            "cutover_triggered": False,
            "cutover_stage": "none",
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

    run_holdout_with_checkpoint(
        holdout_tiles=holdout_tiles,
        processed_tiles=common["processed_tiles"],
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
        feature_dir=inference_feature_dir,
        shape_dir=common["shape_dir"],
        plot_dir=common["inference_plot_dir"],
        context_radius=context_radius,
        holdout_phase_metrics=holdout_phase_metrics,
        append_union=common["append_union"],
        processed_log_path=common["processed_log_path"],
        write_checkpoint=lambda holdout_done: write_rolling_best_config(
            common["rolling_best_settings_path"],
            stage="holdout_inference",
            tuned=tuned,
            fold_done=1,
            fold_total=1,
            holdout_done=holdout_done,
            holdout_total=len(holdout_tiles),
            best_fold=manual_best_fold,
            time_budget=current_time_budget_status(),
            model_bundle=model_bundle_info,
        ),
    )

    _summarize_phase_metrics(val_phase_metrics, "validation")
    _summarize_phase_metrics(holdout_phase_metrics, "holdout")
    common["model_bundle_info"] = model_bundle_info
    common["train_image_ids"] = image_id_a_list
    common["consolidation_tiles"] = val_tiles + holdout_tiles
