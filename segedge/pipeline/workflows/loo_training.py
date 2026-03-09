"""LOO training and final holdout inference workflow."""

from __future__ import annotations

import logging
import os

import numpy as np

from ...core.config_loader import cfg
from ...core.features import hybrid_feature_spec_hash
from ...core.plotting import save_dino_channel_importance_plot
from ...core.xdboost import train_xgb_classifier
from ..artifacts import save_model_bundle
from ..common import build_training_artifacts_for_tiles
from ..runtime_utils import (
    _log_phase,
    _summarize_phase_metrics,
    _update_phase_metrics,
    infer_on_holdout,
    is_budget_exceeded,
    load_b_tile_context,
    summarize_phase_metrics_mean_std,
    write_rolling_best_config,
)
from ..tuning import TimeBudgetExceededError, tune_on_validation_multi
from .shared import (
    _aggregate_fold_metrics,
    _build_loo_folds,
    _export_best_settings_dual,
    _novel_proposals_payload,
    build_weighted_phase_metrics,
    export_shared_feature_payloads,
    run_holdout_with_checkpoint,
)

logger = logging.getLogger(__name__)


def _set_budget_cutover(
    budget_state: dict[str, object],
    runtime_state: dict[str, object],
    stage: str,
) -> None:
    """Persist the active budget cutover stage.

    Examples:
        >>> budget_state = {"cutover_triggered": False, "cutover_stage": "none"}
        >>> runtime_state = {"cutover_triggered": False, "cutover_stage": "none"}
        >>> _set_budget_cutover(budget_state, runtime_state, "demo")
        >>> runtime_state["cutover_stage"]
        'demo'
    """
    runtime_state["cutover_triggered"] = True
    runtime_state["cutover_stage"] = stage
    budget_state["cutover_triggered"] = True
    budget_state["cutover_stage"] = stage


def _is_budget_exceeded_now(
    budget_enabled: bool,
    budget_deadline_ts: float | None,
) -> bool:
    """Return whether the time budget has been exceeded.

    Examples:
        >>> _is_budget_exceeded_now(False, None)
        False
    """
    return bool(budget_enabled and is_budget_exceeded(budget_deadline_ts))


def _get_tile_gt_positive_pixels(
    tile_path: str,
    gt_vector_paths: list[str],
    cache: dict[str, int],
) -> int:
    """Return cached GT-positive pixels for a tile.

    Examples:
        >>> callable(_get_tile_gt_positive_pixels)
        True
    """
    if tile_path in cache:
        return cache[tile_path]
    _, _, _, gt_mask_eval_tile, _, _, _ = load_b_tile_context(
        tile_path, gt_vector_paths
    )
    pixels = int(gt_mask_eval_tile.sum()) if gt_mask_eval_tile is not None else 0
    cache[tile_path] = pixels
    return pixels


def _normalize_fold_train_tiles(
    train_tiles: list[str],
    val_paths: list[str],
    fold_idx: int,
    min_train_tiles: int,
) -> list[str]:
    """Normalize the train tile list and emit warnings when needed.

    Examples:
        >>> _normalize_fold_train_tiles([], ["a"], 1, 2)
        ['a']
    """
    resolved_train_tiles = list(train_tiles)
    if not resolved_train_tiles:
        logger.warning(
            "LOO fold %s has no train tiles; reusing validation tile as source",
            fold_idx,
        )
        resolved_train_tiles = [val_paths[0]]
    if len(resolved_train_tiles) < min_train_tiles:
        logger.warning(
            "LOO fold %s train tiles=%s < min_train_tiles=%s",
            fold_idx,
            len(resolved_train_tiles),
            min_train_tiles,
        )
    return resolved_train_tiles


def _run_single_loo_fold(
    *,
    common: dict,
    fold_idx: int,
    fold_total: int,
    train_tiles: list[str],
    val_paths: list[str],
    gt_vector_paths: list[str],
    source_label_raster: str | None,
    feature_cache_mode: str,
    feature_dir: str | None,
    context_radius: int,
    should_stop,
    val_phase_metrics: dict[str, list[dict]],
) -> tuple[dict, dict, float | None, float | None]:
    """Run training, tuning, and validation inference for one LOO fold.

    Examples:
        >>> callable(_run_single_loo_fold)
        True
    """
    logger.info(
        "LOO fold %s/%s train_tiles=%s val_tiles=%s",
        fold_idx,
        fold_total,
        len(train_tiles),
        len(val_paths),
    )
    ps = common["ps"]
    tile_size = common["tile_size"]
    stride = common["stride"]
    model = common["model"]
    processor = common["processor"]
    device = common["device"]
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
        should_stop=should_stop,
    )
    fold_plot_dir = os.path.join(common["validation_plot_dir"], f"fold_{fold_idx:02d}")
    fold_val_results: list[dict] = []
    val_buffer_m = None
    val_pixel_size_m = None
    for val_path in val_paths:
        if should_stop():
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
            common["shape_dir"],
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
    fold_metrics = _aggregate_fold_metrics(fold_val_results)
    fold_record = {
        "fold_index": fold_idx,
        "val_tile": ",".join(val_paths),
        "val_tiles": list(val_paths),
        "train_tiles_count": len(train_tiles),
        "val_champion_shadow_iou": float(fold_metrics["champion_shadow"]["iou"]),
        "roads_penalty": float(tuned_fold.get("roads_penalty", 1.0)),
        "champion_source": tuned_fold["champion_source"],
        "best_raw_config": tuned_fold["best_raw_config"],
        "best_xgb_config": tuned_fold["best_xgb_config"],
        "best_crf_config": tuned_fold["best_crf_config"],
        "best_shadow_config": tuned_fold["shadow_cfg"],
        "phase_metrics": fold_metrics,
        "tuned": tuned_fold,
    }
    runtime_artifacts = {
        "pos_bank": pos_bank_fold,
        "neg_bank": neg_bank_fold,
        "image_id_a_list": image_id_a_fold,
        "aug_modes": aug_modes_fold,
        "xgb_feature_stats": xgb_feature_stats_fold,
        "feature_layout": feature_layout_fold,
    }
    return fold_record, runtime_artifacts, val_buffer_m, val_pixel_size_m


def _write_loo_checkpoint(
    *,
    common: dict,
    stage: str,
    tuned: dict,
    fold_done: int,
    fold_total: int,
    holdout_total: int,
    best_fold: dict,
    model_bundle: dict | None = None,
) -> None:
    """Write the rolling LOO checkpoint.

    Examples:
        >>> callable(_write_loo_checkpoint)
        True
    """
    write_rolling_best_config(
        common["rolling_best_settings_path"],
        stage=stage,
        tuned=tuned,
        fold_done=fold_done,
        fold_total=fold_total,
        holdout_done=len(common["processed_tiles"]),
        holdout_total=holdout_total,
        best_fold=best_fold,
        time_budget=common["current_time_budget_status"](),
        model_bundle=model_bundle,
    )


def _run_loo_validation_phase(
    *,
    common: dict,
    gt_vector_paths: list[str],
    val_tiles: list[str],
    holdout_tiles: list[str],
    min_train_tiles: int,
    val_tiles_per_fold: int,
    min_gt_positive_pixels: int,
    low_gt_policy: str,
    budget_enabled: bool,
    budget_deadline_ts: float | None,
    budget_cutover_mode: str,
    runtime_state: dict[str, object],
) -> tuple[list[dict], list[dict], dict | None, float | None, float | None]:
    """Run all LOO validation folds and collect their outputs.

    Examples:
        >>> callable(_run_loo_validation_phase)
        True
    """
    tile_gt_positive_cache: dict[str, int] = {}
    loo_fold_records: list[dict] = []
    skipped_fold_records: list[dict] = []
    best_fold_runtime_artifacts: dict | None = None
    best_fold_runtime_iou = -1.0
    val_buffer_m = None
    val_pixel_size_m = None
    loo_folds = _build_loo_folds(val_tiles, val_tiles_per_fold)
    fold_total = len(loo_folds)
    _log_phase("START", "loo_validation_tuning")
    for fold_idx, fold in enumerate(loo_folds, start=1):
        if _is_budget_exceeded_now(budget_enabled, budget_deadline_ts):
            _set_budget_cutover(
                common["budget_state"], runtime_state, "loo_validation_tuning"
            )
            logger.warning(
                "time budget exceeded before LOO fold %s/%s; cutover_mode=%s",
                fold_idx,
                fold_total,
                budget_cutover_mode,
            )
            break
        val_paths = list(fold["val_paths"])
        train_tiles = _normalize_fold_train_tiles(
            list(fold["train_paths"]),
            val_paths,
            fold_idx,
            min_train_tiles,
        )
        fold_gt_positive_pixels = int(
            sum(
                _get_tile_gt_positive_pixels(
                    path, gt_vector_paths, tile_gt_positive_cache
                )
                for path in val_paths
            )
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
        try:
            fold_record, runtime_artifacts, fold_buffer_m, fold_pixel_size_m = (
                _run_single_loo_fold(
                    common=common,
                    fold_idx=fold_idx,
                    fold_total=fold_total,
                    train_tiles=train_tiles,
                    val_paths=val_paths,
                    gt_vector_paths=gt_vector_paths,
                    source_label_raster=common["source_label_raster"],
                    feature_cache_mode=common["feature_cache_mode"],
                    feature_dir=common["feature_dir"],
                    context_radius=common["context_radius"],
                    should_stop=lambda: _is_budget_exceeded_now(
                        budget_enabled, budget_deadline_ts
                    ),
                    val_phase_metrics=common["val_phase_metrics"],
                )
            )
        except TimeBudgetExceededError as exc:
            _set_budget_cutover(
                common["budget_state"], runtime_state, "loo_validation_tuning"
            )
            logger.warning(
                "time budget exceeded during LOO fold %s/%s at stage=%s; stopping training",
                fold_idx,
                fold_total,
                str(exc),
            )
            if loo_fold_records:
                current_best_fold = max(
                    loo_fold_records,
                    key=lambda row: float(row["val_champion_shadow_iou"]),
                )
                _write_loo_checkpoint(
                    common=common,
                    stage="loo_validation_tuning_cutover",
                    tuned=current_best_fold["tuned"],
                    fold_done=len(loo_fold_records),
                    fold_total=fold_total,
                    holdout_total=len(holdout_tiles),
                    best_fold=current_best_fold,
                )
            break
        fold_record["val_gt_positive_pixels"] = int(fold_gt_positive_pixels)
        loo_fold_records.append(fold_record)
        if fold_buffer_m is not None and val_buffer_m is None:
            val_buffer_m = fold_buffer_m
            val_pixel_size_m = fold_pixel_size_m
        fold_iou = float(fold_record["val_champion_shadow_iou"])
        if fold_iou > best_fold_runtime_iou:
            best_fold_runtime_iou = fold_iou
            best_fold_runtime_artifacts = runtime_artifacts
        current_best_fold = max(
            loo_fold_records,
            key=lambda row: float(row["val_champion_shadow_iou"]),
        )
        _write_loo_checkpoint(
            common=common,
            stage="loo_validation_tuning",
            tuned=current_best_fold["tuned"],
            fold_done=len(loo_fold_records),
            fold_total=fold_total,
            holdout_total=len(holdout_tiles),
            best_fold=current_best_fold,
        )
    _log_phase("END", "loo_validation_tuning")
    return (
        loo_fold_records,
        skipped_fold_records,
        best_fold_runtime_artifacts,
        val_buffer_m,
        val_pixel_size_m,
    )


def _require_loo_records(
    loo_fold_records: list[dict],
    skipped_fold_records: list[dict],
    runtime_state: dict[str, object],
) -> None:
    """Raise a user-facing error when no LOO records were produced.

    Examples:
        >>> _require_loo_records([{"ok": True}], [], {"cutover_triggered": False})
    """
    if loo_fold_records:
        return
    if runtime_state["cutover_triggered"]:
        raise ValueError(
            "time budget exhausted before any LOO fold completed; cannot continue to inference"
        )
    if skipped_fold_records:
        raise ValueError(
            "LOO tuning produced no fold records because all folds were skipped by low-GT policy"
        )
    raise ValueError("LOO tuning produced no fold records")


def _prepare_final_training_state(
    *,
    common: dict,
    selected_tuned: dict,
    best_fold_runtime_artifacts: dict | None,
    budget_enabled: bool,
    budget_deadline_ts: float | None,
    budget_cutover_mode: str,
    runtime_state: dict[str, object],
) -> tuple[dict, np.ndarray | None, np.ndarray | None, list[str], list[str], bool, str]:
    """Resolve cutover behavior before final all-GT training.

    Examples:
        >>> callable(_prepare_final_training_state)
        True
    """
    tuned = dict(selected_tuned)
    pos_bank = None
    neg_bank = None
    image_id_a_list: list[str] = []
    aug_modes: list[str] = []
    halt_before_inference = False
    final_stage_name = "final_model_ready"
    if not _is_budget_exceeded_now(budget_enabled, budget_deadline_ts):
        return (
            tuned,
            pos_bank,
            neg_bank,
            image_id_a_list,
            aug_modes,
            halt_before_inference,
            final_stage_name,
        )
    logger.warning(
        "time budget exceeded before final_all_gt_training; cutover_mode=%s",
        budget_cutover_mode,
    )
    if budget_cutover_mode not in {"immediate_inference", "stop"}:
        return (
            tuned,
            pos_bank,
            neg_bank,
            image_id_a_list,
            aug_modes,
            halt_before_inference,
            final_stage_name,
        )
    _set_budget_cutover(common["budget_state"], runtime_state, "final_all_gt_training")
    if best_fold_runtime_artifacts is None:
        raise ValueError(
            "time-budget cutover requested, but no fold artifacts are available"
        )
    pos_bank = best_fold_runtime_artifacts["pos_bank"]
    neg_bank = best_fold_runtime_artifacts["neg_bank"]
    image_id_a_list = list(best_fold_runtime_artifacts["image_id_a_list"])
    aug_modes = list(best_fold_runtime_artifacts["aug_modes"])
    if tuned.get("xgb_feature_stats") is None:
        tuned["xgb_feature_stats"] = best_fold_runtime_artifacts["xgb_feature_stats"]
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
        logger.warning("time-budget cutover: mode=stop, will write outputs and exit")
    return (
        tuned,
        pos_bank,
        neg_bank,
        image_id_a_list,
        aug_modes,
        halt_before_inference,
        final_stage_name,
    )


def _train_final_all_gt_model(
    *,
    common: dict,
    gt_tiles: list[str],
    selected_tuned: dict,
    tuned: dict,
) -> tuple[np.ndarray, np.ndarray | None, dict, list[str], list[str]]:
    """Train the final all-GT model and return final artifacts.

    Examples:
        >>> callable(_train_final_all_gt_model)
        True
    """
    _log_phase("START", "final_all_gt_training")
    (
        pos_bank,
        neg_bank,
        x_matrix,
        y_labels,
        image_id_a_list,
        aug_modes,
        xgb_feature_stats,
        feature_layout,
    ) = build_training_artifacts_for_tiles(
        gt_tiles,
        common["source_label_raster"],
        common["model"],
        common["processor"],
        common["device"],
        common["ps"],
        common["tile_size"],
        common["stride"],
        common["feature_cache_mode"],
        common["feature_dir"],
        common["context_radius"],
    )
    final_bst = tuned.get("bst")
    if common["xgb_enabled"]:
        best_xgb_params = selected_tuned["best_xgb_config"].get("params")
        feature_names = (
            list(feature_layout.get("feature_names", []))
            if feature_layout is not None
            else None
        )
        if feature_names is not None and len(feature_names) != x_matrix.shape[1]:
            logger.warning(
                "final training feature layout mismatch: names=%s X=%s; disabling names",
                len(feature_names),
                x_matrix.shape[1],
            )
            feature_names = None
        final_bst = train_xgb_classifier(
            x_matrix,
            y_labels,
            use_gpu=cfg.search.xgb.use_gpu,
            num_boost_round=cfg.search.xgb.num_boost_round,
            verbose_eval=cfg.search.xgb.verbose_eval,
            param_overrides=best_xgb_params,
            feature_names=feature_names,
        )
    _log_phase("END", "final_all_gt_training")
    return (
        pos_bank,
        neg_bank,
        {
            **selected_tuned,
            "bst": final_bst,
            "xgb_feature_stats": xgb_feature_stats,
            "feature_layout": feature_layout,
        },
        image_id_a_list,
        aug_modes,
    )


def _save_model_bundle_if_enabled(
    common: dict,
    tuned: dict,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
) -> dict | None:
    """Save the model bundle when enabled and possible.

    Examples:
        >>> callable(_save_model_bundle_if_enabled)
        True
    """
    if not common["bundle_save_enabled"]:
        return None
    if tuned.get("bst") is None:
        logger.warning(
            "bundle save requested but no XGB model is available; skipping save"
        )
        return None
    model_bundle_info = save_model_bundle(
        common["bundle_output_dir"],
        tuned,
        pos_bank,
        neg_bank,
        model_name=cfg.model.backbone.name,
        patch_size=common["ps"],
        resample_factor=cfg.model.backbone.resample_factor,
        tile_size=common["tile_size"],
        stride=common["stride"],
        context_radius=common["context_radius"],
    )
    logger.info("saved model bundle: %s", model_bundle_info["path"])
    return model_bundle_info


def _build_xgb_model_info(tuned: dict) -> dict[str, object]:
    """Build the XGB model export payload.

    Examples:
        >>> _build_xgb_model_info({}) == {}
        True
    """
    bst = tuned.get("bst")
    if bst is None:
        return {}
    dino_importance_plot = save_dino_channel_importance_plot(
        bst,
        tuned.get("feature_layout"),
        cfg.io.paths.plot_dir,
        top_k=20,
    )
    best_iter = getattr(bst, "best_iteration", None)
    best_score = getattr(bst, "best_score", None)
    return {
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "best_score": float(best_score) if best_score is not None else None,
        "num_features": int(bst.num_features()),
        "attributes": bst.attributes(),
        "dino_importance_plot": dino_importance_plot,
    }


def _build_model_info(
    *,
    common: dict,
    aug_modes: list[str],
    xgb_feature_stats_payload: dict | None,
    feature_layout_payload: dict | None,
    budget_enabled: bool,
    budget_cutover_mode: str,
    runtime_state: dict[str, object],
) -> dict[str, object]:
    """Build the structured model metadata payload for exported settings.

    Examples:
        >>> callable(_build_model_info)
        True
    """
    return {
        "backbone": {
            "name": cfg.model.backbone.name,
            "patch_size": cfg.model.backbone.patch_size,
            "resample_factor": cfg.model.backbone.resample_factor,
        },
        "tiling": {"tile_size": common["tile_size"], "stride": common["stride"]},
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
            "hours": common["budget_hours"],
            "scope": common["budget_scope"],
            "cutover_mode": budget_cutover_mode,
            "cutover_triggered": runtime_state["cutover_triggered"],
            "cutover_stage": runtime_state["cutover_stage"],
        },
    }


def _export_loo_best_settings(
    *,
    common: dict,
    tuned: dict,
    gt_tiles: list[str],
    holdout_tiles: list[str],
    val_tiles: list[str],
    best_fold: dict,
    loo_fold_records: list[dict],
    skipped_fold_records: list[dict],
    val_buffer_m: float | None,
    val_pixel_size_m: float | None,
    aug_modes: list[str],
    min_train_tiles: int,
    val_tiles_per_fold: int,
    min_gt_positive_pixels: int,
    low_gt_policy: str,
    budget_enabled: bool,
    budget_cutover_mode: str,
    runtime_state: dict[str, object],
    model_bundle_info: dict | None,
) -> None:
    """Write the detailed LOO best-settings exports.

    Examples:
        >>> callable(_export_loo_best_settings)
        True
    """
    loo_fold_export = [
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
        for fold in loo_fold_records
    ]
    weighted_phase_metrics = build_weighted_phase_metrics(common["val_phase_metrics"])
    xgb_feature_stats_payload, feature_layout_payload = export_shared_feature_payloads(
        tuned
    )
    xgb_model_info = _build_xgb_model_info(tuned)
    model_info = _build_model_info(
        common=common,
        aug_modes=aug_modes,
        xgb_feature_stats_payload=xgb_feature_stats_payload,
        feature_layout_payload=feature_layout_payload,
        budget_enabled=budget_enabled,
        budget_cutover_mode=budget_cutover_mode,
        runtime_state=runtime_state,
    )
    inference_best_settings_path = os.path.join(
        common["run_dir"], "inference_best_setting.yml"
    )
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
            "tile_size": common["tile_size"],
            "stride": common["stride"],
            "patch_size": common["ps"],
            "feat_context_radius": common["context_radius"],
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
            "feature_spec_hash": hybrid_feature_spec_hash(),
            "xgb_feature_stats": xgb_feature_stats_payload,
            "feature_layout": feature_layout_payload,
            "time_budget": common["current_time_budget_status"](),
            "cutover_triggered": runtime_state["cutover_triggered"],
            "cutover_stage": runtime_state["cutover_stage"],
            "model_bundle": model_bundle_info,
            "val_tiles_count": len(val_tiles),
            "holdout_tiles_count": len(holdout_tiles),
            "weighted_phase_metrics": weighted_phase_metrics,
            "loo": {
                "enabled": True,
                "fold_count": len(loo_fold_export),
                "fold_total": len(_build_loo_folds(val_tiles, val_tiles_per_fold)),
                "min_train_tiles": min_train_tiles,
                "val_tiles_per_fold": val_tiles_per_fold,
                "min_gt_positive_pixels": min_gt_positive_pixels,
                "low_gt_policy": low_gt_policy,
                "selected_fold_index": int(best_fold["fold_index"]),
                "selected_val_tile": best_fold["val_tile"],
                "phase_metrics_mean_std": summarize_phase_metrics_mean_std(
                    common["val_phase_metrics"]
                ),
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


def _finalize_without_inference(
    common: dict,
    image_id_a_list: list[str],
) -> None:
    """Finalize workflow state when holdout inference is skipped.

    Examples:
        >>> common = {"val_phase_metrics": {}, "holdout_phase_metrics": {}}
        >>> _finalize_without_inference(common, ["a"])
        >>> common["should_consolidate"], common["consolidation_tiles"]
        (False, [])
    """
    _summarize_phase_metrics(common["val_phase_metrics"], "loo_validation")
    _summarize_phase_metrics(common["holdout_phase_metrics"], "holdout")
    common["should_consolidate"] = False
    common["train_image_ids"] = image_id_a_list
    common["consolidation_tiles"] = []


def _run_loo_holdout_phase(
    *,
    common: dict,
    tuned: dict,
    best_fold: dict,
    loo_fold_records: list[dict],
    holdout_tiles: list[str],
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    model_bundle_info: dict | None,
) -> None:
    """Run holdout inference with checkpoint updates for the LOO workflow.

    Examples:
        >>> callable(_run_loo_holdout_phase)
        True
    """
    run_holdout_with_checkpoint(
        holdout_tiles=holdout_tiles,
        processed_tiles=common["processed_tiles"],
        gt_vector_paths=common["gt_vector_paths"],
        model=common["model"],
        processor=common["processor"],
        device=common["device"],
        pos_bank=pos_bank,
        neg_bank=neg_bank,
        tuned=tuned,
        ps=common["ps"],
        tile_size=common["tile_size"],
        stride=common["stride"],
        feature_dir=common["feature_dir"],
        shape_dir=common["shape_dir"],
        plot_dir=common["inference_plot_dir"],
        context_radius=common["context_radius"],
        holdout_phase_metrics=common["holdout_phase_metrics"],
        append_union=common["append_union"],
        processed_log_path=common["processed_log_path"],
        write_checkpoint=lambda holdout_done: write_rolling_best_config(
            common["rolling_best_settings_path"],
            stage="holdout_inference",
            tuned=tuned,
            fold_done=len(loo_fold_records),
            fold_total=len(
                _build_loo_folds(common["val_tiles"], common["val_tiles_per_fold"])
            ),
            holdout_done=holdout_done,
            holdout_total=len(holdout_tiles),
            best_fold=best_fold,
            time_budget=common["current_time_budget_status"](),
            model_bundle=model_bundle_info,
        ),
    )


def run_loo_training(
    common: dict,
    *,
    gt_tiles: list[str],
    val_tiles: list[str],
    holdout_tiles: list[str],
    min_train_tiles: int,
    val_tiles_per_fold: int,
    min_gt_positive_pixels: int,
    low_gt_policy: str,
    budget_enabled: bool,
    budget_deadline_ts: float | None,
    budget_cutover_mode: str,
) -> None:
    """Run LOO tuning, optional final retraining, and holdout inference.

    Examples:
        >>> callable(run_loo_training)
        True
    """
    common["val_tiles"] = val_tiles
    common["val_tiles_per_fold"] = val_tiles_per_fold
    runtime_state = {
        "cutover_triggered": bool(common["budget_state"]["cutover_triggered"]),
        "cutover_stage": str(common["budget_state"]["cutover_stage"]),
    }
    (
        loo_fold_records,
        skipped_fold_records,
        best_fold_runtime_artifacts,
        val_buffer_m,
        val_pixel_size_m,
    ) = _run_loo_validation_phase(
        common=common,
        gt_vector_paths=common["gt_vector_paths"],
        val_tiles=val_tiles,
        holdout_tiles=holdout_tiles,
        min_train_tiles=min_train_tiles,
        val_tiles_per_fold=val_tiles_per_fold,
        min_gt_positive_pixels=min_gt_positive_pixels,
        low_gt_policy=low_gt_policy,
        budget_enabled=budget_enabled,
        budget_deadline_ts=budget_deadline_ts,
        budget_cutover_mode=budget_cutover_mode,
        runtime_state=runtime_state,
    )
    _require_loo_records(loo_fold_records, skipped_fold_records, runtime_state)
    best_fold = max(
        loo_fold_records,
        key=lambda row: float(row["val_champion_shadow_iou"]),
    )
    selected_tuned = best_fold["tuned"]
    logger.info(
        "LOO selected fold=%s val_tile=%s champion_shadow_iou=%.3f",
        best_fold["fold_index"],
        best_fold["val_tile"],
        float(best_fold["val_champion_shadow_iou"]),
    )
    (
        tuned,
        pos_bank,
        neg_bank,
        image_id_a_list,
        aug_modes,
        halt_before_inference,
        final_stage_name,
    ) = _prepare_final_training_state(
        common=common,
        selected_tuned=selected_tuned,
        best_fold_runtime_artifacts=best_fold_runtime_artifacts,
        budget_enabled=budget_enabled,
        budget_deadline_ts=budget_deadline_ts,
        budget_cutover_mode=budget_cutover_mode,
        runtime_state=runtime_state,
    )
    if pos_bank is None:
        pos_bank, neg_bank, tuned, image_id_a_list, aug_modes = (
            _train_final_all_gt_model(
                common=common,
                gt_tiles=gt_tiles,
                selected_tuned=selected_tuned,
                tuned=tuned,
            )
        )
    model_bundle_info = _save_model_bundle_if_enabled(common, tuned, pos_bank, neg_bank)
    _write_loo_checkpoint(
        common=common,
        stage=final_stage_name,
        tuned=tuned,
        fold_done=len(loo_fold_records),
        fold_total=len(_build_loo_folds(val_tiles, val_tiles_per_fold)),
        holdout_total=len(holdout_tiles),
        best_fold=best_fold,
        model_bundle=model_bundle_info,
    )
    _export_loo_best_settings(
        common=common,
        tuned=tuned,
        gt_tiles=gt_tiles,
        holdout_tiles=holdout_tiles,
        val_tiles=val_tiles,
        best_fold=best_fold,
        loo_fold_records=loo_fold_records,
        skipped_fold_records=skipped_fold_records,
        val_buffer_m=val_buffer_m,
        val_pixel_size_m=val_pixel_size_m,
        aug_modes=aug_modes,
        min_train_tiles=min_train_tiles,
        val_tiles_per_fold=val_tiles_per_fold,
        min_gt_positive_pixels=min_gt_positive_pixels,
        low_gt_policy=low_gt_policy,
        budget_enabled=budget_enabled,
        budget_cutover_mode=budget_cutover_mode,
        runtime_state=runtime_state,
        model_bundle_info=model_bundle_info,
    )
    if halt_before_inference:
        _finalize_without_inference(common, image_id_a_list)
        return
    _run_loo_holdout_phase(
        common=common,
        tuned=tuned,
        best_fold=best_fold,
        loo_fold_records=loo_fold_records,
        holdout_tiles=holdout_tiles,
        pos_bank=pos_bank,
        neg_bank=neg_bank,
        model_bundle_info=model_bundle_info,
    )
    _summarize_phase_metrics(common["val_phase_metrics"], "loo_validation")
    _summarize_phase_metrics(common["holdout_phase_metrics"], "holdout")
    common["model_bundle_info"] = model_bundle_info
    common["train_image_ids"] = image_id_a_list
    common["consolidation_tiles"] = val_tiles + holdout_tiles
