"""Inference-only workflow."""

from __future__ import annotations

import logging
import os

from ...core.features import hybrid_feature_spec_hash
from ..artifacts import load_model_bundle, validate_bundle_compatibility
from ..runtime_utils import (
    _log_phase,
    _summarize_phase_metrics,
    write_rolling_best_config,
)
from .shared import (
    _export_best_settings_dual,
    _inference_score_prior_payload,
    _novel_proposals_payload,
    export_shared_feature_payloads,
    run_holdout_with_checkpoint,
)

logger = logging.getLogger(__name__)


def run_inference_only(
    common: dict,
    *,
    holdout_tiles: list[str],
    inference_dir: str | None,
    inference_glob: str,
) -> None:
    """Run the inference-only branch.

    Examples:
        >>> callable(run_inference_only)
        True
    """
    model_bundle_dir = common["model_bundle_dir"]
    ps = common["ps"]
    tile_size = common["tile_size"]
    stride = common["stride"]
    context_radius = common["context_radius"]
    model = common["model"]
    processor = common["processor"]
    device = common["device"]
    gt_vector_paths = common["gt_vector_paths"]
    run_dir = common["run_dir"]
    feature_dir = common["inference_feature_dir"]
    feature_cache_mode = common["inference_feature_cache_mode"]
    processed_tiles = common["processed_tiles"]
    rolling_best_settings_path = common["rolling_best_settings_path"]
    holdout_phase_metrics = common["holdout_phase_metrics"]
    current_time_budget_status = common["current_time_budget_status"]

    _log_phase("START", "load_model_bundle")
    loaded_bundle = load_model_bundle(model_bundle_dir)
    validate_bundle_compatibility(
        loaded_bundle["manifest"],
        model_name=common["model_name"],
        patch_size=ps,
        resample_factor=common["resample_factor"],
        tile_size=tile_size,
        stride=stride,
        context_radius=context_radius,
    )
    _log_phase("END", "load_model_bundle")
    tuned = loaded_bundle["tuned"]
    pos_bank = loaded_bundle["pos_bank"]
    neg_bank = loaded_bundle["neg_bank"]
    model_bundle_info = {
        "path": model_bundle_dir,
        "version": loaded_bundle["manifest"].get("bundle_version"),
        "manifest": os.path.join(model_bundle_dir, "manifest.yml"),
        "created_utc": loaded_bundle["manifest"].get("created_utc"),
    }

    write_rolling_best_config(
        rolling_best_settings_path,
        stage="inference_only_bundle_loaded",
        tuned=tuned,
        fold_done=0,
        fold_total=0,
        holdout_done=len(processed_tiles),
        holdout_total=len(holdout_tiles),
        best_fold=None,
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
        common["model_name"],
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
            "roads_mask_path": common["roads_mask_path"],
            "model_toggles": {
                "knn_enabled": tuned.get("knn_enabled"),
                "xgb_enabled": tuned.get("xgb_enabled"),
                "crf_enabled": tuned.get("crf_enabled"),
            },
            "novel_proposals": _novel_proposals_payload(),
            "inference_score_prior": _inference_score_prior_payload(),
            "feature_spec_hash": hybrid_feature_spec_hash(),
            "xgb_feature_stats": xgb_feature_stats_payload,
            "feature_layout": feature_layout_payload,
            "time_budget": current_time_budget_status(),
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

    run_holdout_with_checkpoint(
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
        shape_dir=common["shape_dir"],
        plot_dir=common["inference_plot_dir"],
        context_radius=context_radius,
        holdout_phase_metrics=holdout_phase_metrics,
        append_union=common["append_union"],
        processed_log_path=common["processed_log_path"],
        write_checkpoint=lambda holdout_done: write_rolling_best_config(
            rolling_best_settings_path,
            stage="holdout_inference",
            tuned=tuned,
            fold_done=0,
            fold_total=0,
            holdout_done=holdout_done,
            holdout_total=len(holdout_tiles),
            best_fold=None,
            time_budget=current_time_budget_status(),
            model_bundle=model_bundle_info,
        ),
    )

    _summarize_phase_metrics(holdout_phase_metrics, "holdout")
    common["model_bundle_info"] = model_bundle_info
    common["train_image_ids"] = []
    common["consolidation_tiles"] = holdout_tiles
    if feature_cache_mode == "disk":
        common["consolidation_tiles"] = holdout_tiles
