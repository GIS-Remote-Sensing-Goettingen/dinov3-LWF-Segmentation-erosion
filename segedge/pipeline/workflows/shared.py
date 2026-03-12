"""Shared workflow helpers for training and inference branches."""

from __future__ import annotations

import logging
import os
from typing import Callable

from ...core.config_loader import cfg
from ...core.features import serialize_xgb_feature_stats
from ...core.io_utils import consolidate_features_for_image, export_best_settings
from ..inference_flow import run_holdout_inference
from ..runtime_utils import _log_phase, _weighted_mean

logger = logging.getLogger(__name__)


def _build_loo_folds(
    gt_tiles: list[str],
    val_tiles_per_fold: int,
) -> list[dict[str, list[str]]]:
    """Build deterministic cyclic LOO-style folds with configurable val window.

    Examples:
        >>> folds = _build_loo_folds(["a", "b", "c"], 1)
        >>> len(folds)
        3
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
        >>> _aggregate_fold_metrics(rows)["champion_shadow"]["iou"]
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
        >>> isinstance(_novel_proposals_payload(), dict)
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


def _inference_score_prior_payload() -> dict[str, object]:
    """Return the active manual inference score-prior config as a dict.

    Examples:
        >>> isinstance(_inference_score_prior_payload(), dict)
        True
    """
    prior_cfg = cfg.io.inference.score_prior
    return {
        "enabled": prior_cfg.enabled,
        "apply_to": prior_cfg.apply_to,
        "target": prior_cfg.target,
        "mode": prior_cfg.mode,
        "inside_factor": prior_cfg.inside_factor,
        "outside_factor": prior_cfg.outside_factor,
        "clip_max": prior_cfg.clip_max,
    }


def _maybe_run_holdout_inference(
    holdout_tiles: list[str],
    runner: Callable[[], None],
) -> bool:
    """Run holdout inference only when tiles are available.

    Examples:
        >>> calls = []
        >>> _maybe_run_holdout_inference([], lambda: calls.append("ran"))
        False
    """
    if not holdout_tiles:
        logger.warning(
            "no inference tiles available after SOURCE_LABEL_RASTER filtering; "
            "skipping holdout inference"
        )
        return False
    _log_phase("START", "holdout_inference")
    runner()
    _log_phase("END", "holdout_inference")
    return True


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
    """Write both current and legacy best-settings YAML files.

    Examples:
        >>> callable(_export_best_settings_dual)
        True
    """
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


def build_weighted_phase_metrics(
    phase_metrics: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Aggregate weighted phase metrics.

    Examples:
        >>> build_weighted_phase_metrics({})
        {}
    """
    weighted_phase_metrics: dict[str, dict[str, float]] = {}
    metric_keys = ["iou", "f1", "precision", "recall"]
    for phase, metrics_list in phase_metrics.items():
        weights = [float(m.get("_weight", 0.0)) for m in metrics_list]
        weighted_phase_metrics[phase] = {
            key: _weighted_mean([m.get(key, 0.0) for m in metrics_list], weights)
            for key in metric_keys
        }
    return weighted_phase_metrics


def consolidate_cached_features(
    feature_dir: str | None,
    train_image_ids: list[str],
    inference_tiles: list[str],
    *,
    consolidate_training: bool,
    consolidate_inference: bool,
) -> None:
    """Consolidate per-tile feature caches after a run.

    Examples:
        >>> consolidate_cached_features(None, [], [], consolidate_training=False, consolidate_inference=False)
    """
    if not consolidate_training and not consolidate_inference:
        return
    if feature_dir is None:
        raise ValueError("feature_dir must be set for disk cache mode")
    _log_phase("START", "feature_consolidation")
    if consolidate_training:
        for image_id_a in train_image_ids:
            consolidate_features_for_image(feature_dir, image_id_a)
    if consolidate_inference:
        for b_path in inference_tiles:
            image_id_b = os.path.splitext(os.path.basename(b_path))[0]
            consolidate_features_for_image(feature_dir, image_id_b)
    _log_phase("END", "feature_consolidation")


def export_shared_feature_payloads(tuned: dict) -> tuple[dict | None, dict | None]:
    """Return serialized xgb feature stats and layout payloads.

    Examples:
        >>> export_shared_feature_payloads({})
        (None, None)
    """
    return (
        serialize_xgb_feature_stats(tuned.get("xgb_feature_stats")),
        tuned.get("feature_layout"),
    )


def run_holdout_with_checkpoint(
    *,
    holdout_tiles: list[str],
    processed_tiles: set[str],
    gt_vector_paths: list[str] | None,
    model,
    processor,
    device,
    pos_bank,
    neg_bank,
    tuned: dict,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    shape_dir: str,
    plot_dir: str,
    context_radius: int,
    holdout_phase_metrics: dict[str, list[dict]],
    append_union,
    processed_log_path: str,
    write_checkpoint,
) -> None:
    """Execute holdout inference with the standard wrapper.

    Examples:
        >>> callable(run_holdout_with_checkpoint)
        True
    """
    _maybe_run_holdout_inference(
        holdout_tiles,
        lambda: run_holdout_inference(
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
            plot_dir=plot_dir,
            context_radius=context_radius,
            holdout_phase_metrics=holdout_phase_metrics,
            append_union=append_union,
            processed_log_path=processed_log_path,
            write_checkpoint=write_checkpoint,
            logger=logger,
            final_inference_phase=True,
            plot_every=int(cfg.io.inference.plot_every),
        ),
    )
