"""Primary pipeline entrypoint for SegEdge."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Callable

import numpy as np
import torch
import yaml
from scipy.ndimage import median_filter

from ..core.config_loader import cfg
from ..core.crf_utils import refine_with_densecrf
from ..core.features import (
    hybrid_feature_spec_hash,
    prefetch_features_single_scale_image,
    serialize_xgb_feature_stats,
)
from ..core.io_utils import (
    append_mask_to_union_shapefile,
    backup_union_shapefile,
    consolidate_features_for_image,
    count_shapefile_features,
    export_best_settings,
    export_mask_to_shapefile,
)
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from ..core.logging_utils import setup_logging
from ..core.metrics_utils import (
    compute_metrics,
    compute_metrics_batch_cpu,
    compute_metrics_batch_gpu,
    compute_oracle_upper_bound,
)
from ..core.plotting import (
    save_core_qualitative_plot,
    save_dino_channel_importance_plot,
    save_disagreement_entropy_plot,
    save_proposal_overlay_plot,
    save_score_threshold_plot,
    save_unified_plot,
)
from ..core.timing_utils import time_end, time_start
from ..core.xdboost import (
    hyperparam_search_xgb_iou,
    train_xgb_classifier,
    xgb_score_image_b,
)
from .common import (
    build_training_artifacts_for_tiles,
    init_model,
    resolve_tiles_from_gt_presence,
)
from .runtime_utils import (
    _apply_roads_penalty,
    _apply_shadow_filter,
    _eval_crf_config,
    _get_roads_mask,
    _init_crf_parallel,
    _log_phase,
    _summarize_phase_metrics,
    _update_phase_metrics,
    _weighted_mean,
    build_time_budget_status,
    compute_budget_deadline,
    filter_novel_proposals,
    is_budget_exceeded,
    load_b_tile_context,
    parse_utc_iso_to_epoch,
    remaining_budget_s,
    summarize_phase_metrics_mean_std,
    write_rolling_best_config,
)

# Config-driven flags
USE_FP16_KNN = cfg.search.knn.use_fp16_knn
CRF_MAX_CONFIGS = cfg.search.crf.max_configs

logger = logging.getLogger(__name__)


class TimeBudgetExceededError(RuntimeError):
    """Raised when runtime budget is exceeded during tuning checkpoints."""


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


def tune_on_validation_multi(
    val_paths: list[str],
    gt_vector_paths: list[str],
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    X: np.ndarray,
    y: np.ndarray,
    xgb_feature_stats: dict | None,
    feature_layout: dict | None,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    should_stop: Callable[[], bool] | None = None,
):
    """Tune hyperparameters using weighted-mean IoU across validation tiles.

    Args:
        val_paths (list[str]): Validation tile paths.
        gt_vector_paths (list[str]): Vector GT paths.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        pos_bank (np.ndarray): Positive bank.
        neg_bank (np.ndarray | None): Negative bank.
        X (np.ndarray): XGB feature matrix.
        y (np.ndarray): XGB labels.
        xgb_feature_stats (dict | None): Fold-fitted XGB feature standardization stats.
        feature_layout (dict | None): Feature layout metadata.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str | None): Feature cache directory.
        context_radius (int): Feature context radius.
        should_stop (callable | None): Optional callback returning True when
            tuning should stop (for time-budget cutover).

    Returns:
        dict: Tuned configurations and models.

    Examples:
        >>> callable(tune_on_validation_multi)
        True
    """
    if not val_paths:
        raise ValueError("VAL_TILES is empty.")

    def _maybe_stop(stage: str) -> None:
        if should_stop is not None and should_stop():
            raise TimeBudgetExceededError(stage)

    _maybe_stop("validation_context_build")
    val_contexts = []
    ds = int(cfg.model.backbone.resample_factor or 1)
    for val_path in val_paths:
        _maybe_stop("validation_context_build")
        logger.info("tune: preparing validation tile %s", val_path)
        (
            img_b,
            labels_sh,
            gt_mask_B,
            gt_mask_eval,
            sh_buffer_mask,
            buffer_m,
            pixel_size_m,
        ) = load_b_tile_context(val_path, gt_vector_paths)
        if gt_mask_eval is None:
            raise ValueError("Validation requires GT vectors for metric-based tuning.")
        _ = compute_oracle_upper_bound(gt_mask_eval, sh_buffer_mask)
        gt_weight = float(gt_mask_eval.sum())
        image_id_b = os.path.splitext(os.path.basename(val_path))[0]
        prefetched_b = prefetch_features_single_scale_image(
            img_b,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            None,
            feature_dir,
            image_id_b,
        )
        roads_mask = _get_roads_mask(val_path, ds, target_shape=img_b.shape[:2])
        val_contexts.append(
            {
                "path": val_path,
                "image_id": image_id_b,
                "img_b": img_b,
                "labels_sh": labels_sh,
                "gt_mask_B": gt_mask_B,
                "gt_mask_eval": gt_mask_eval,
                "sh_buffer_mask": sh_buffer_mask,
                "gt_weight": gt_weight,
                "roads_mask": roads_mask,
                "prefetched_b": prefetched_b,
                "buffer_m": buffer_m,
                "pixel_size_m": pixel_size_m,
            }
        )

    # XGB training (shared across road penalties)
    use_gpu_xgb = cfg.search.xgb.use_gpu
    param_grid = cfg.search.xgb.param_grid
    num_boost_round = cfg.search.xgb.num_boost_round
    early_stop = cfg.search.xgb.early_stop
    verbose_eval = cfg.search.xgb.verbose_eval
    val_fraction = cfg.search.xgb.val_fraction
    fixed_xgb_threshold = cfg.search.xgb.fixed_threshold
    if param_grid is None:
        param_grid = [None]
    feature_names = (
        list(feature_layout.get("feature_names", []))
        if feature_layout is not None
        else None
    )
    if feature_names is not None and len(feature_names) != X.shape[1]:
        logger.warning(
            "feature layout dim mismatch: names=%s, X=%s; disabling XGB feature names",
            len(feature_names),
            X.shape[1],
        )
        feature_names = None

    xgb_candidates = []
    for overrides in param_grid:
        _maybe_stop("xgb_candidate_train")
        if overrides is None:
            bst = train_xgb_classifier(
                X,
                y,
                use_gpu=use_gpu_xgb,
                num_boost_round=num_boost_round,
                verbose_eval=verbose_eval,
                feature_names=feature_names,
            )
            params_used = None
        else:
            bst, params_used, _, _, _ = hyperparam_search_xgb_iou(
                X,
                y,
                (
                    [float(fixed_xgb_threshold)]
                    if fixed_xgb_threshold is not None
                    else [0.5]
                ),
                val_contexts[0]["sh_buffer_mask"],
                val_contexts[0]["gt_mask_eval"],
                val_contexts[0]["img_b"],
                ps,
                tile_size,
                stride,
                feature_dir,
                val_contexts[0]["image_id"],
                prefetched_tiles=val_contexts[0]["prefetched_b"],
                device=device,
                use_gpu=use_gpu_xgb,
                param_grid=[overrides],
                num_boost_round=num_boost_round,
                val_fraction=val_fraction,
                early_stopping_rounds=early_stop,
                verbose_eval=verbose_eval,
                seed=42,
                context_radius=context_radius,
                xgb_feature_stats=xgb_feature_stats,
                feature_names=feature_names,
            )
        xgb_candidates.append({"bst": bst, "params": params_used})

    roads_penalties = [float(p) for p in cfg.postprocess.roads.penalty_values]
    best_bundle = None
    best_champion_iou = None

    for penalty in roads_penalties:
        _maybe_stop("roads_penalty_search")
        logger.info("tune: roads penalty=%s", penalty)

        # kNN tuning (weighted-mean IoU across val tiles)
        best_raw_config = None
        for k in cfg.search.knn.k_values:
            _maybe_stop("knn_k_search")
            stats_by_thr = {
                thr: {
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "w": 0.0,
                    "n": 0,
                }
                for thr in cfg.search.knn.thresholds.values
            }
            for ctx in val_contexts:
                _maybe_stop("knn_tile_scoring")
                logger.info("tune: kNN scoring on %s (k=%s)", ctx["path"], k)
                score_full, _ = zero_shot_knn_single_scale_B_with_saliency(
                    img_b=ctx["img_b"],
                    pos_bank=pos_bank,
                    neg_bank=neg_bank,
                    model=model,
                    processor=processor,
                    device=device,
                    ps=ps,
                    tile_size=tile_size,
                    stride=stride,
                    k=k,
                    aggregate_layers=None,
                    feature_dir=feature_dir,
                    image_id=ctx["image_id"],
                    neg_alpha=cfg.model.banks.neg_alpha,
                    prefetched_tiles=ctx["prefetched_b"],
                    use_fp16_matmul=USE_FP16_KNN,
                    context_radius=context_radius,
                )
                score_full = _apply_roads_penalty(
                    score_full, ctx["roads_mask"], penalty
                )
                try:
                    metrics_list = compute_metrics_batch_gpu(
                        score_full,
                        cfg.search.knn.thresholds.values,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    metrics_list = compute_metrics_batch_cpu(
                        score_full,
                        cfg.search.knn.thresholds.values,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                    )
                weight = float(ctx["gt_weight"])
                for m in metrics_list:
                    stats = stats_by_thr[m["threshold"]]
                    stats["iou"] += float(m["iou"]) * weight
                    stats["f1"] += float(m["f1"]) * weight
                    stats["precision"] += float(m["precision"]) * weight
                    stats["recall"] += float(m["recall"]) * weight
                    stats["w"] += weight
                    stats["n"] += 1

            for thr, stats in stats_by_thr.items():
                if stats["w"] > 0:
                    weighted_iou = float(stats["iou"] / stats["w"])
                    weighted_f1 = float(stats["f1"] / stats["w"])
                    weighted_precision = float(stats["precision"] / stats["w"])
                    weighted_recall = float(stats["recall"] / stats["w"])
                else:
                    weighted_iou = 0.0
                    weighted_f1 = 0.0
                    weighted_precision = 0.0
                    weighted_recall = 0.0
                if best_raw_config is None or weighted_iou > best_raw_config["iou"]:
                    best_raw_config = {
                        "k": k,
                        "threshold": thr,
                        "source": "raw",
                        "iou": weighted_iou,
                        "f1": weighted_f1,
                        "precision": weighted_precision,
                        "recall": weighted_recall,
                    }
        if best_raw_config is None:
            raise ValueError("kNN tuning returned no results")

        # XGB tuning (weighted-mean IoU across val tiles)
        best_xgb_config = None
        best_bst = None
        xgb_thresholds = (
            [float(fixed_xgb_threshold)]
            if fixed_xgb_threshold is not None
            else list(cfg.search.knn.thresholds.values)
        )
        for candidate in xgb_candidates:
            _maybe_stop("xgb_candidate_eval")
            bst = candidate["bst"]
            params_used = candidate["params"]
            stats_by_thr = {
                thr: {
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "w": 0.0,
                    "n": 0,
                }
                for thr in xgb_thresholds
            }
            for ctx in val_contexts:
                _maybe_stop("xgb_tile_scoring")
                logger.info("tune: XGB scoring on %s", ctx["path"])
                score_full = xgb_score_image_b(
                    ctx["img_b"],
                    bst,
                    ps,
                    tile_size,
                    stride,
                    feature_dir,
                    ctx["image_id"],
                    prefetched_tiles=ctx["prefetched_b"],
                    context_radius=context_radius,
                    xgb_feature_stats=xgb_feature_stats,
                )
                score_full = _apply_roads_penalty(
                    score_full, ctx["roads_mask"], penalty
                )
                try:
                    metrics_list = compute_metrics_batch_gpu(
                        score_full,
                        xgb_thresholds,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    metrics_list = compute_metrics_batch_cpu(
                        score_full,
                        xgb_thresholds,
                        ctx["sh_buffer_mask"],
                        ctx["gt_mask_eval"],
                    )
                weight = float(ctx["gt_weight"])
                for m in metrics_list:
                    stats = stats_by_thr[m["threshold"]]
                    stats["iou"] += float(m["iou"]) * weight
                    stats["f1"] += float(m["f1"]) * weight
                    stats["precision"] += float(m["precision"]) * weight
                    stats["recall"] += float(m["recall"]) * weight
                    stats["w"] += weight
                    stats["n"] += 1

            for thr, stats in stats_by_thr.items():
                if stats["w"] > 0:
                    weighted_iou = float(stats["iou"] / stats["w"])
                    weighted_f1 = float(stats["f1"] / stats["w"])
                    weighted_precision = float(stats["precision"] / stats["w"])
                    weighted_recall = float(stats["recall"] / stats["w"])
                else:
                    weighted_iou = 0.0
                    weighted_f1 = 0.0
                    weighted_precision = 0.0
                    weighted_recall = 0.0
                cand = {
                    "k": -1,
                    "threshold": thr,
                    "source": "xgb",
                    "iou": weighted_iou,
                    "f1": weighted_f1,
                    "precision": weighted_precision,
                    "recall": weighted_recall,
                    "params": params_used,
                }
                if best_xgb_config is None or weighted_iou > best_xgb_config["iou"]:
                    best_xgb_config = cand
                    best_bst = bst
        if best_xgb_config is None or best_bst is None:
            raise ValueError("XGB tuning returned no results")

        champion_source = (
            "raw" if best_raw_config["iou"] >= best_xgb_config["iou"] else "xgb"
        )
        champion_iou = (
            best_raw_config["iou"]
            if champion_source == "raw"
            else best_xgb_config["iou"]
        )
        if best_champion_iou is None or champion_iou > best_champion_iou:
            best_champion_iou = champion_iou
            best_bundle = {
                "roads_penalty": penalty,
                "best_raw_config": best_raw_config,
                "best_xgb_config": best_xgb_config,
                "best_bst": best_bst,
                "champion_source": champion_source,
            }

    if best_bundle is None:
        raise ValueError("roads penalty tuning returned no results")

    roads_penalty = best_bundle["roads_penalty"]
    best_raw_config = best_bundle["best_raw_config"]
    best_xgb_config = best_bundle["best_xgb_config"]
    best_bst = best_bundle["best_bst"]
    champion_source = best_bundle["champion_source"]

    thr_center = (
        best_raw_config["threshold"]
        if champion_source == "raw"
        else best_xgb_config["threshold"]
    )
    for ctx in val_contexts:
        _maybe_stop("champion_score_rebuild")
        if champion_source == "raw":
            score_full, _ = zero_shot_knn_single_scale_B_with_saliency(
                img_b=ctx["img_b"],
                pos_bank=pos_bank,
                neg_bank=neg_bank,
                model=model,
                processor=processor,
                device=device,
                ps=ps,
                tile_size=tile_size,
                stride=stride,
                k=best_raw_config["k"],
                aggregate_layers=None,
                feature_dir=feature_dir,
                image_id=ctx["image_id"],
                neg_alpha=cfg.model.banks.neg_alpha,
                prefetched_tiles=ctx["prefetched_b"],
                use_fp16_matmul=USE_FP16_KNN,
                context_radius=context_radius,
            )
        else:
            score_full = xgb_score_image_b(
                ctx["img_b"],
                best_bst,
                ps,
                tile_size,
                stride,
                feature_dir,
                ctx["image_id"],
                prefetched_tiles=ctx["prefetched_b"],
                context_radius=context_radius,
                xgb_feature_stats=xgb_feature_stats,
            )
        score_full = _apply_roads_penalty(
            score_full, ctx["roads_mask"], float(roads_penalty)
        )
        ctx["score_full"] = score_full
        ctx["thr_center"] = thr_center

    # CRF tuning across val tiles
    crf_candidates = [
        (psf, pw, pxy, bw, bxy, brgb)
        for psf in cfg.search.crf.prob_softness_values
        for pw in cfg.search.crf.pos_w_values
        for pxy in cfg.search.crf.pos_xy_std_values
        for bw in cfg.search.crf.bilateral_w_values
        for bxy in cfg.search.crf.bilateral_xy_std_values
        for brgb in cfg.search.crf.bilateral_rgb_std_values
    ]
    _maybe_stop("crf_search")
    best_crf_cfg = None
    best_crf_iou = None
    crf_candidates = crf_candidates[:CRF_MAX_CONFIGS]
    num_workers = int(cfg.search.crf.num_workers or 1)
    logger.info(
        "tune: CRF grid search configs=%s, workers=%s",
        len(crf_candidates),
        num_workers,
    )
    _init_crf_parallel(val_contexts)
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for med_iou, cand in ex.map(_eval_crf_config, crf_candidates):
                _maybe_stop("crf_search")
                if best_crf_iou is None or med_iou > best_crf_iou:
                    best_crf_iou = med_iou
                    best_crf_cfg = {
                        "prob_softness": cand[0],
                        "pos_w": cand[1],
                        "pos_xy_std": cand[2],
                        "bilateral_w": cand[3],
                        "bilateral_xy_std": cand[4],
                        "bilateral_rgb_std": cand[5],
                    }
    else:
        for cand in crf_candidates:
            _maybe_stop("crf_search")
            med_iou, _ = _eval_crf_config(cand)
            if best_crf_iou is None or med_iou > best_crf_iou:
                best_crf_iou = med_iou
                best_crf_cfg = {
                    "prob_softness": cand[0],
                    "pos_w": cand[1],
                    "pos_xy_std": cand[2],
                    "bilateral_w": cand[3],
                    "bilateral_xy_std": cand[4],
                    "bilateral_rgb_std": cand[5],
                }
    if best_crf_cfg is None:
        raise ValueError("CRF tuning returned no results")

    # Shadow tuning across val tiles
    best_shadow_cfg = None
    best_shadow_iou = None
    protect_scores = cfg.postprocess.shadow.protect_scores
    for weights in cfg.postprocess.shadow.weight_sets:
        _maybe_stop("shadow_search")
        iou_by_key = {
            (thr, protect_score): {"sum": 0.0, "w": 0.0}
            for thr in cfg.postprocess.shadow.thresholds
            for protect_score in protect_scores
        }
        for ctx in val_contexts:
            _maybe_stop("shadow_tile_scoring")
            logger.info("tune: shadow scoring on %s", ctx["path"])
            score_full = ctx["score_full"]
            thr_center = ctx["thr_center"]

            mask_crf = refine_with_densecrf(
                ctx["img_b"],
                score_full,
                thr_center,
                ctx["sh_buffer_mask"],
                prob_softness=best_crf_cfg["prob_softness"],
                n_iters=5,
                pos_w=best_crf_cfg["pos_w"],
                pos_xy_std=best_crf_cfg["pos_xy_std"],
                bilateral_w=best_crf_cfg["bilateral_w"],
                bilateral_xy_std=best_crf_cfg["bilateral_xy_std"],
                bilateral_rgb_std=best_crf_cfg["bilateral_rgb_std"],
            )
            img_float = ctx["img_b"].astype(np.float32)
            w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
            wsum = (img_float * w).sum(axis=2)
            flat_base = mask_crf.reshape(-1)
            flat_gt = ctx["gt_mask_eval"].reshape(-1).astype(bool)
            vals = wsum.reshape(-1)[flat_base]
            gt_vals = flat_gt[flat_base]
            if vals.size == 0:
                continue
            thr_arr = np.array(
                cfg.postprocess.shadow.thresholds, dtype=np.float32
            ).reshape(-1, 1)
            mask_thr = vals[None, :] >= thr_arr
            gt_bool = gt_vals.astype(bool)
            score_vals = score_full.reshape(-1)[flat_base]
            weight = float(ctx["gt_weight"])
            for protect_score in protect_scores:
                protect_mask = score_vals >= protect_score
                mask_keep = mask_thr | protect_mask[None, :]
                tp = (
                    np.logical_and(mask_keep, gt_bool[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                fp = (
                    np.logical_and(mask_keep, ~gt_bool[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                fn = (
                    np.logical_and(~mask_keep, gt_bool[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                iou = tp / (tp + fp + fn + 1e-8)
                for i, thr in enumerate(cfg.postprocess.shadow.thresholds):
                    stats = iou_by_key[(thr, protect_score)]
                    stats["sum"] += float(iou[i]) * weight
                    stats["w"] += weight

        for (thr, protect_score), stats in iou_by_key.items():
            if stats["w"] <= 0:
                continue
            weighted_iou = float(stats["sum"] / stats["w"])
            if best_shadow_iou is None or weighted_iou > best_shadow_iou:
                best_shadow_iou = weighted_iou
                best_shadow_cfg = {
                    "weights": weights,
                    "threshold": thr,
                    "protect_score": protect_score,
                    "iou": weighted_iou,
                }
    if best_shadow_cfg is None:
        raise ValueError("shadow tuning returned no results")

    logger.info("tune: roads penalty selected=%s", roads_penalty)
    return {
        "bst": best_bst,
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "best_crf_config": {**best_crf_cfg, "k": best_raw_config["k"]},
        "shadow_cfg": best_shadow_cfg,
        "roads_penalty": float(roads_penalty),
        "xgb_feature_stats": xgb_feature_stats,
        "feature_layout": feature_layout,
    }


def infer_on_holdout(
    holdout_path: str,
    gt_vector_paths: list[str] | None,
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    tuned: dict,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    shape_dir: str,
    context_radius: int,
    plot_with_metrics: bool = True,
):
    """Run inference on a holdout tile using tuned settings.

    Args:
        holdout_path (str): Holdout tile path.
        gt_vector_paths (list[str] | None): Vector GT paths.
        model: DINO model.
        processor: DINO processor.
        device: Torch device.
        pos_bank (np.ndarray): Positive bank.
        neg_bank (np.ndarray | None): Negative bank.
        tuned (dict): Tuned configuration bundle.
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str | None): Feature cache directory.
        shape_dir (str): Output shapefile directory.
        context_radius (int): Feature context radius.
        plot_with_metrics (bool): Whether to show metrics on plots.

    Returns:
        dict: Masks, metrics, and metadata for the tile.

    Examples:
        >>> callable(infer_on_holdout)
        True
    """
    logger.info("inference: holdout tile %s", holdout_path)
    (
        img_b,
        labels_sh,
        _,
        gt_mask_eval,
        sh_buffer_mask,
        buffer_m,
        pixel_size_m,
    ) = load_b_tile_context(holdout_path, gt_vector_paths)
    gt_available = gt_mask_eval is not None
    if gt_mask_eval is None:
        logger.warning("Holdout has no GT; metrics will be reported as 0.0.")
        gt_mask_eval = np.zeros(img_b.shape[:2], dtype=bool)
    gt_weight = float(gt_mask_eval.sum())
    ds = int(cfg.model.backbone.resample_factor or 1)
    roads_mask = _get_roads_mask(holdout_path, ds, target_shape=img_b.shape[:2])
    roads_penalty = float(tuned.get("roads_penalty", 1.0))
    xgb_feature_stats = tuned.get("xgb_feature_stats")

    image_id_b = os.path.splitext(os.path.basename(holdout_path))[0]
    prefetched_b = prefetch_features_single_scale_image(
        img_b,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        None,
        feature_dir,
        image_id_b,
    )

    k = tuned["best_raw_config"]["k"]
    knn_thr = tuned["best_raw_config"]["threshold"]
    score_knn, _ = zero_shot_knn_single_scale_B_with_saliency(
        img_b,
        pos_bank,
        neg_bank,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        k=k,
        aggregate_layers=None,
        feature_dir=feature_dir,
        image_id=image_id_b,
        neg_alpha=cfg.model.banks.neg_alpha,
        prefetched_tiles=prefetched_b,
        use_fp16_matmul=USE_FP16_KNN,
        context_radius=context_radius,
    )
    score_knn_raw = score_knn
    score_knn = _apply_roads_penalty(score_knn, roads_mask, roads_penalty)
    mask_knn = (score_knn >= knn_thr) & sh_buffer_mask
    mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0
    metrics_knn = compute_metrics(mask_knn, gt_mask_eval)

    bst = tuned["bst"]
    xgb_thr = tuned["best_xgb_config"]["threshold"]
    score_xgb = xgb_score_image_b(
        img_b,
        bst,
        ps,
        tile_size,
        stride,
        feature_dir,
        image_id_b,
        prefetched_tiles=prefetched_b,
        context_radius=context_radius,
        xgb_feature_stats=xgb_feature_stats,
    )
    score_xgb = _apply_roads_penalty(score_xgb, roads_mask, roads_penalty)
    mask_xgb = (score_xgb >= xgb_thr) & sh_buffer_mask
    mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
    metrics_xgb = compute_metrics(mask_xgb, gt_mask_eval)

    champion_source = tuned["champion_source"]
    if champion_source == "raw":
        champion_score = score_knn
        champion_thr = float(knn_thr)
    else:
        champion_score = score_xgb
        champion_thr = float(xgb_thr)

    crf_cfg = tuned["best_crf_config"]
    mask_crf_knn = refine_with_densecrf(
        img_b,
        score_knn,
        knn_thr,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
    )
    mask_crf_xgb = refine_with_densecrf(
        img_b,
        score_xgb,
        xgb_thr,
        sh_buffer_mask,
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
    )
    if champion_source == "raw":
        best_crf_mask = mask_crf_knn
    else:
        best_crf_mask = mask_crf_xgb

    shadow_cfg = tuned["shadow_cfg"]
    protect_score = shadow_cfg.get("protect_score")
    shadow_mask = _apply_shadow_filter(
        img_b,
        best_crf_mask,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        champion_score,
        protect_score,
    )
    shadow_mask_knn = _apply_shadow_filter(
        img_b,
        mask_crf_knn,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        score_knn,
        protect_score,
    )
    shadow_mask_xgb = _apply_shadow_filter(
        img_b,
        mask_crf_xgb,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        score_xgb,
        protect_score,
    )
    metrics_knn_crf = compute_metrics(mask_crf_knn, gt_mask_eval)
    metrics_knn_shadow = compute_metrics(shadow_mask_knn, gt_mask_eval)
    metrics_xgb_crf = compute_metrics(mask_crf_xgb, gt_mask_eval)
    metrics_xgb_shadow = compute_metrics(shadow_mask_xgb, gt_mask_eval)
    champ_raw_mask = mask_knn if champion_source == "raw" else mask_xgb
    metrics_champion_raw = compute_metrics(champ_raw_mask, gt_mask_eval)
    metrics_champion_crf = compute_metrics(best_crf_mask, gt_mask_eval)
    shadow_metrics = compute_metrics(shadow_mask, gt_mask_eval)
    proposal_cfg = cfg.postprocess.novel_proposals
    if proposal_cfg.source == "champion_score":
        proposal_threshold = (
            float(proposal_cfg.score_threshold)
            if proposal_cfg.score_threshold is not None
            else champion_thr
        )
        proposal_source_mask = champion_score >= proposal_threshold
    else:
        proposal_source_mask = shadow_mask
    proposal_bundle = filter_novel_proposals(
        shadow_mask,
        labels_sh,
        champion_score,
        roads_mask,
        pixel_size_m,
        sh_buffer_mask=sh_buffer_mask,
        proposal_source_mask=proposal_source_mask,
    )
    proposal_candidate_mask = proposal_bundle["candidate_mask"]
    proposal_accepted_mask = proposal_bundle["accepted_mask"]
    proposal_rejected_mask = proposal_bundle["rejected_mask"]

    if cfg.postprocess.novel_proposals.enabled:
        proposal_dir = os.path.join(shape_dir, "proposals")
        os.makedirs(proposal_dir, exist_ok=True)
        if proposal_accepted_mask.any():
            export_mask_to_shapefile(
                proposal_accepted_mask,
                holdout_path,
                os.path.join(proposal_dir, f"{image_id_b}_accepted.shp"),
            )
        if proposal_rejected_mask.any():
            export_mask_to_shapefile(
                proposal_rejected_mask,
                holdout_path,
                os.path.join(proposal_dir, f"{image_id_b}_rejected.shp"),
            )

    if champion_source == "xgb":
        champion_prob = np.clip(champion_score.astype(np.float32), 1e-6, 1.0 - 1e-6)
    else:
        s = champion_score.astype(np.float32)
        smin = float(np.min(s))
        smax = float(np.max(s))
        champion_prob = (s - smin) / (smax - smin + 1e-8)
        champion_prob = np.clip(champion_prob, 1e-6, 1.0 - 1e-6)
    disagreement_map = np.abs(score_xgb - score_knn).astype(np.float32)
    entropy_map = (
        -champion_prob * np.log(champion_prob)
        - (1.0 - champion_prob) * np.log(1.0 - champion_prob)
    ).astype(np.float32)

    metrics_map = {
        "knn_raw": metrics_knn,
        "knn_crf": metrics_knn_crf,
        "knn_shadow": metrics_knn_shadow,
        "xgb_raw": metrics_xgb,
        "xgb_crf": metrics_xgb_crf,
        "xgb_shadow": metrics_xgb_shadow,
        "champion_raw": metrics_champion_raw,
        "champion_crf": metrics_champion_crf,
        "champion_shadow": shadow_metrics,
    }
    metrics_map = {
        key: {**metrics, "_weight": gt_weight} for key, metrics in metrics_map.items()
    }
    masks_map = {
        "knn_raw": mask_knn,
        "knn_crf": mask_crf_knn,
        "knn_shadow": shadow_mask_knn,
        "xgb_raw": mask_xgb,
        "xgb_crf": mask_crf_xgb,
        "xgb_shadow": shadow_mask_xgb,
        "champion_raw": champ_raw_mask,
        "champion_crf": best_crf_mask,
        "champion_shadow": shadow_mask,
    }
    save_unified_plot(
        img_b=img_b,
        gt_mask=gt_mask_eval,
        labels_sh=labels_sh,
        masks=masks_map,
        metrics=metrics_map,
        plot_dir=cfg.io.paths.plot_dir,
        image_id_b=image_id_b,
        show_metrics=plot_with_metrics and gt_available,
        gt_available=gt_available,
        similarity_map=score_knn_raw,
        score_maps={"knn": score_knn, "xgb": score_xgb},
        proposal_masks={
            "candidate": proposal_candidate_mask,
            "accepted": proposal_accepted_mask,
            "rejected": proposal_rejected_mask,
        },
    )
    save_core_qualitative_plot(
        img_b=img_b,
        gt_mask=gt_mask_eval,
        pred_mask=shadow_mask,
        plot_dir=cfg.io.paths.plot_dir,
        image_id_b=image_id_b,
        gt_available=gt_available,
        boundary_band_px=int(cfg.runtime.plotting.boundary_band_px),
    )
    save_score_threshold_plot(
        score_map=champion_score,
        threshold=champion_thr,
        sh_buffer_mask=sh_buffer_mask,
        plot_dir=cfg.io.paths.plot_dir,
        image_id_b=image_id_b,
    )
    save_disagreement_entropy_plot(
        disagreement_map=disagreement_map,
        entropy_map=entropy_map,
        candidate_mask=proposal_candidate_mask,
        plot_dir=cfg.io.paths.plot_dir,
        image_id_b=image_id_b,
    )
    save_proposal_overlay_plot(
        img_b=img_b,
        prediction_mask=shadow_mask,
        candidate_mask=proposal_candidate_mask,
        accepted_mask=proposal_accepted_mask,
        rejected_mask=proposal_rejected_mask,
        plot_dir=cfg.io.paths.plot_dir,
        image_id_b=image_id_b,
        accept_rgb=tuple(cfg.runtime.plotting.proposal_accept_rgb),
        reject_rgb=tuple(cfg.runtime.plotting.proposal_reject_rgb),
        candidate_rgb=tuple(cfg.runtime.plotting.proposal_candidate_rgb),
    )

    champ_raw_mask = mask_knn if champion_source == "raw" else mask_xgb
    return {
        "ref_path": holdout_path,
        "image_id": image_id_b,
        "gt_available": gt_available,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
        "metrics": metrics_map,
        "masks": {
            "knn_raw": mask_knn,
            "knn_crf": mask_crf_knn,
            "knn_shadow": shadow_mask_knn,
            "xgb_raw": mask_xgb,
            "xgb_crf": mask_crf_xgb,
            "xgb_shadow": shadow_mask_xgb,
            "champion_raw": champ_raw_mask,
            "champion_crf": best_crf_mask,
            "champion_shadow": shadow_mask,
        },
        "proposals": proposal_bundle,
    }


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
    shape_dir = os.path.join(run_dir, "shapes")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(shape_dir, exist_ok=True)
    cfg.io.paths.plot_dir = plot_dir
    cfg.io.paths.best_settings_path = os.path.join(run_dir, "best_settings.yml")
    cfg.io.paths.log_path = os.path.join(run_dir, "run.log")
    setup_logging(cfg.io.paths.log_path)
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

    # ------------------------------------------------------------
    # Resolve GT-positive tiles (LOO set) and inference-only tiles
    # ------------------------------------------------------------
    if not auto_split_tiles:
        raise ValueError(
            "io.auto_split.enabled must be true: directory-driven LOO training is required"
        )
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

    context_radius = int(cfg.model.banks.feat_context_radius or 0)
    min_train_tiles = int(cfg.training.loo.min_train_tiles or 1)
    val_tiles_per_fold = int(cfg.training.loo.val_tiles_per_fold or 1)
    min_gt_positive_pixels = int(cfg.training.loo.min_gt_positive_pixels or 0)
    low_gt_policy = str(cfg.training.loo.low_gt_policy)

    # Resolve required tile sets
    # ------------------------------------------------------------
    if not val_tiles:
        raise ValueError("no GT-positive tiles resolved for LOO training")
    if len(val_tiles) <= val_tiles_per_fold:
        raise ValueError(
            "training.loo.val_tiles_per_fold must be < number of GT-positive tiles"
        )
    if not holdout_tiles:
        logger.warning("no inference-only tiles resolved; skipping holdout inference")

    # ------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------
    feature_cache_mode = cfg.runtime.feature_cache_mode
    if feature_cache_mode not in {"disk", "memory"}:
        raise ValueError("FEATURE_CACHE_MODE must be 'disk' or 'memory'")
    if (
        feature_cache_mode == "memory"
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
    bst = tuned["bst"]
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
            "use_gpu": cfg.search.xgb.use_gpu,
            "num_boost_round": cfg.search.xgb.num_boost_round,
            "early_stop": cfg.search.xgb.early_stop,
            "fixed_threshold": cfg.search.xgb.fixed_threshold,
            "param_grid_size": len(cfg.search.xgb.param_grid),
        },
        "knn_search": {
            "k_values": cfg.search.knn.k_values,
            "threshold_range": {
                "start": cfg.search.knn.thresholds.start,
                "stop": cfg.search.knn.thresholds.stop,
                "count": cfg.search.knn.thresholds.count,
            },
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
    export_best_settings(
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
            "novel_proposals": {
                "enabled": cfg.postprocess.novel_proposals.enabled,
                "search_scope": cfg.postprocess.novel_proposals.search_scope,
                "source": cfg.postprocess.novel_proposals.source,
                "score_threshold": cfg.postprocess.novel_proposals.score_threshold,
                "min_area_px": cfg.postprocess.novel_proposals.min_area_px,
                "min_length_m": cfg.postprocess.novel_proposals.min_length_m,
                "max_width_m": cfg.postprocess.novel_proposals.max_width_m,
                "min_skeleton_ratio": cfg.postprocess.novel_proposals.min_skeleton_ratio,
                "min_pca_ratio": cfg.postprocess.novel_proposals.min_pca_ratio,
                "max_circularity": cfg.postprocess.novel_proposals.max_circularity,
                "min_mean_score": cfg.postprocess.novel_proposals.min_mean_score,
                "max_road_overlap": cfg.postprocess.novel_proposals.max_road_overlap,
                "connectivity": cfg.postprocess.novel_proposals.connectivity,
            },
            "feature_spec_hash": hybrid_feature_spec_hash(),
            "xgb_feature_stats": xgb_feature_stats_payload,
            "feature_layout": feature_layout_payload,
            "time_budget": _current_time_budget_status(),
            "cutover_triggered": cutover_triggered,
            "cutover_stage": cutover_stage,
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
        best_settings_path=inference_best_settings_path,
    )
    logger.info("wrote inference best settings: %s", inference_best_settings_path)

    if halt_before_inference:
        _summarize_phase_metrics(val_phase_metrics, "loo_validation")
        _summarize_phase_metrics(holdout_phase_metrics, "holdout")
        time_end("main (total)", t0_main)
        return

    _log_phase("START", "holdout_inference")
    holdout_tiles_processed = len(processed_tiles)
    for b_path in holdout_tiles:
        if b_path in processed_tiles:
            logger.info("holdout skip (already processed): %s", b_path)
            continue
        result = infer_on_holdout(
            b_path,
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
            context_radius,
            plot_with_metrics=False,
        )
        if result["gt_available"]:
            _update_phase_metrics(holdout_phase_metrics, result["metrics"])
        holdout_tiles_processed += 1
        ref_path = result["ref_path"]
        masks = result["masks"]
        for stream in ("knn", "xgb", "champion"):
            for variant in ("raw", "crf", "shadow"):
                mask_key = f"{stream}_{variant}"
                _append_union(
                    stream,
                    variant,
                    masks[mask_key],
                    ref_path,
                    holdout_tiles_processed,
                )
        record = {
            "tile_path": b_path,
            "image_id": result["image_id"],
            "status": "done",
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(processed_log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        write_rolling_best_config(
            rolling_best_settings_path,
            stage="holdout_inference",
            tuned=tuned,
            fold_done=len(loo_fold_records),
            fold_total=fold_total,
            holdout_done=holdout_tiles_processed,
            holdout_total=len(holdout_tiles),
            best_fold=best_fold,
            time_budget=_current_time_budget_status(),
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
