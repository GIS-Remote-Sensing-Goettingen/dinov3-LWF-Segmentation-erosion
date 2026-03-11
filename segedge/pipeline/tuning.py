"""Validation tuning utilities extracted from run.py."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Callable

import numpy as np
import torch

from ..core.config_loader import cfg
from ..core.crf_utils import refine_with_densecrf
from ..core.features import prefetch_features_single_scale_image
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from ..core.metrics_utils import (
    compute_metrics_batch_cpu,
    compute_metrics_batch_gpu,
    compute_oracle_upper_bound,
)
from ..core.xdboost import (
    hyperparam_search_xgb_iou,
    train_xgb_classifier,
    xgb_score_image_b,
)
from .runtime_utils import (
    _apply_roads_penalty,
    _eval_crf_config,
    _get_roads_mask,
    _init_crf_parallel,
    load_b_tile_context,
)

# Config-driven flags
USE_FP16_KNN = cfg.search.knn.use_fp16_knn
CRF_MAX_CONFIGS = cfg.search.crf.max_configs
KNN_ENABLED = bool(cfg.search.knn.enabled)
XGB_ENABLED = bool(cfg.search.xgb.enabled)
CRF_ENABLED = bool(cfg.search.crf.enabled)

logger = logging.getLogger(__name__)


class TimeBudgetExceededError(RuntimeError):
    """Raised when runtime budget is exceeded during tuning checkpoints."""


def _resolve_crf_workers(
    requested_workers: int,
    candidate_count: int,
    device,
) -> int:
    """Return a safe CRF worker count for the current runtime.

    Avoid process-based CRF search after CUDA initialization because forking a
    GPU-touched parent can terminate child workers abruptly. Also avoid
    oversubscribing workers beyond the number of CRF configs.

    Args:
        requested_workers (int): Configured CRF worker count.
        candidate_count (int): Number of CRF configs to evaluate.
        device: Torch device used by the pipeline.

    Returns:
        int: Effective CRF worker count.

    Examples:
        >>> _resolve_crf_workers(16, 1, torch.device("cpu"))
        1
        >>> _resolve_crf_workers(4, 4, torch.device("cuda:0"))
        1
    """
    safe_workers = max(1, min(int(requested_workers), int(candidate_count)))
    if getattr(device, "type", "cpu") == "cuda":
        return 1
    return safe_workers


def _maybe_stop(should_stop: Callable[[], bool] | None, stage: str) -> None:
    """Raise a budget exception when the optional stop callback fires.

    Examples:
        >>> _maybe_stop(None, "demo")
    """
    if should_stop is not None and should_stop():
        raise TimeBudgetExceededError(stage)


def _build_validation_contexts(
    val_paths: list[str],
    gt_vector_paths: list[str],
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    should_stop: Callable[[], bool] | None,
) -> list[dict]:
    """Build per-tile validation contexts used by tuning.

    Examples:
        >>> callable(_build_validation_contexts)
        True
    """
    val_contexts = []
    ds = int(cfg.model.backbone.resample_factor or 1)
    for val_path in val_paths:
        _maybe_stop(should_stop, "validation_context_build")
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
        val_contexts.append(
            {
                "path": val_path,
                "image_id": image_id_b,
                "img_b": img_b,
                "labels_sh": labels_sh,
                "gt_mask_B": gt_mask_B,
                "gt_mask_eval": gt_mask_eval,
                "sh_buffer_mask": sh_buffer_mask,
                "gt_weight": float(gt_mask_eval.sum()),
                "roads_mask": _get_roads_mask(
                    val_path, ds, target_shape=img_b.shape[:2]
                ),
                "prefetched_b": prefetched_b,
                "buffer_m": buffer_m,
                "pixel_size_m": pixel_size_m,
            }
        )
    return val_contexts


def _build_xgb_candidates(
    X: np.ndarray,
    y: np.ndarray,
    feature_layout: dict | None,
    xgb_feature_stats: dict | None,
    val_contexts: list[dict],
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    should_stop: Callable[[], bool] | None,
) -> list[dict]:
    """Train shared XGB candidates before roads-penalty evaluation.

    Examples:
        >>> callable(_build_xgb_candidates)
        True
    """
    if not XGB_ENABLED:
        return []
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
    param_grid = cfg.search.xgb.param_grid or [None]
    fixed_xgb_threshold = cfg.search.xgb.fixed_threshold
    candidates = []
    for overrides in param_grid:
        _maybe_stop(should_stop, "xgb_candidate_train")
        if overrides is None:
            bst = train_xgb_classifier(
                X,
                y,
                use_gpu=cfg.search.xgb.use_gpu,
                num_boost_round=cfg.search.xgb.num_boost_round,
                verbose_eval=cfg.search.xgb.verbose_eval,
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
                use_gpu=cfg.search.xgb.use_gpu,
                param_grid=[overrides],
                num_boost_round=cfg.search.xgb.num_boost_round,
                val_fraction=cfg.search.xgb.val_fraction,
                early_stopping_rounds=cfg.search.xgb.early_stop,
                verbose_eval=cfg.search.xgb.verbose_eval,
                seed=42,
                context_radius=context_radius,
                xgb_feature_stats=xgb_feature_stats,
                feature_names=feature_names,
            )
        candidates.append({"bst": bst, "params": params_used})
    return candidates


def _weighted_threshold_search(
    score_builder: Callable[[dict], np.ndarray],
    thresholds: list[float],
    val_contexts: list[dict],
    device,
) -> dict[float, dict[str, float]]:
    """Aggregate threshold metrics across validation contexts.

    Examples:
        >>> callable(_weighted_threshold_search)
        True
    """
    stats_by_thr = {
        thr: {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "w": 0.0}
        for thr in thresholds
    }
    for ctx in val_contexts:
        score_full = score_builder(ctx)
        try:
            metrics_list = compute_metrics_batch_gpu(
                score_full,
                thresholds,
                ctx["sh_buffer_mask"],
                ctx["gt_mask_eval"],
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            metrics_list = compute_metrics_batch_cpu(
                score_full,
                thresholds,
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
    return stats_by_thr


def _best_config_from_threshold_stats(
    stats_by_thr: dict[float, dict[str, float]],
    *,
    source: str,
    extra: dict | None = None,
) -> dict | None:
    """Convert aggregated threshold stats into the best config.

    Examples:
        >>> _best_config_from_threshold_stats(
        ...     {
        ...         0.5: {
        ...             "iou": 2.0,
        ...             "f1": 2.0,
        ...             "precision": 2.0,
        ...             "recall": 2.0,
        ...             "w": 4.0,
        ...         }
        ...     },
        ...     source="raw",
        ... )["threshold"]
        0.5
    """
    best = None
    for thr, stats in stats_by_thr.items():
        if stats["w"] > 0:
            weighted = {
                "iou": float(stats["iou"] / stats["w"]),
                "f1": float(stats["f1"] / stats["w"]),
                "precision": float(stats["precision"] / stats["w"]),
                "recall": float(stats["recall"] / stats["w"]),
            }
        else:
            weighted = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        candidate = {"threshold": thr, "source": source, **weighted}
        if extra is not None:
            candidate.update(extra)
        if best is None or candidate["iou"] > best["iou"]:
            best = candidate
    return best


def _tune_knn_for_penalty(
    penalty: float,
    val_contexts: list[dict],
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    should_stop: Callable[[], bool] | None,
) -> dict | None:
    """Tune kNN thresholds for one roads penalty.

    Examples:
        >>> callable(_tune_knn_for_penalty)
        True
    """
    if not KNN_ENABLED:
        return None
    best = None
    for k in cfg.search.knn.k_values:
        _maybe_stop(should_stop, "knn_k_search")
        stats_by_thr = _weighted_threshold_search(
            lambda ctx: _apply_roads_penalty(
                zero_shot_knn_single_scale_B_with_saliency(
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
                )[0],
                ctx["roads_mask"],
                penalty,
            ),
            list(cfg.search.knn.thresholds.values),
            val_contexts,
            device,
        )
        candidate = _best_config_from_threshold_stats(
            stats_by_thr,
            source="raw",
            extra={"k": k},
        )
        if candidate is not None and (best is None or candidate["iou"] > best["iou"]):
            best = candidate
    return best


def _tune_xgb_for_penalty(
    penalty: float,
    val_contexts: list[dict],
    xgb_candidates: list[dict],
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    xgb_feature_stats: dict | None,
    should_stop: Callable[[], bool] | None,
) -> tuple[dict | None, object | None]:
    """Tune XGB thresholds for one roads penalty.

    Examples:
        >>> callable(_tune_xgb_for_penalty)
        True
    """
    if not XGB_ENABLED:
        return None, None
    best = None
    best_bst = None
    thresholds = (
        [float(cfg.search.xgb.fixed_threshold)]
        if cfg.search.xgb.fixed_threshold is not None
        else list(cfg.search.knn.thresholds.values)
    )
    for candidate in xgb_candidates:
        _maybe_stop(should_stop, "xgb_candidate_eval")
        bst = candidate["bst"]
        stats_by_thr = _weighted_threshold_search(
            lambda ctx: _apply_roads_penalty(
                xgb_score_image_b(
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
                ),
                ctx["roads_mask"],
                penalty,
            ),
            thresholds,
            val_contexts,
            device,
        )
        tuned = _best_config_from_threshold_stats(
            stats_by_thr,
            source="xgb",
            extra={"k": -1, "params": candidate["params"]},
        )
        if tuned is not None and (best is None or tuned["iou"] > best["iou"]):
            best = tuned
            best_bst = bst
    return best, best_bst


def _select_best_penalty_bundle(
    roads_penalties: list[float],
    val_contexts: list[dict],
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    xgb_candidates: list[dict],
    xgb_feature_stats: dict | None,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    should_stop: Callable[[], bool] | None,
) -> dict:
    """Evaluate roads penalties and keep the best champion bundle.

    Examples:
        >>> callable(_select_best_penalty_bundle)
        True
    """
    best_bundle = None
    best_champion_iou = None
    for penalty in roads_penalties:
        _maybe_stop(should_stop, "roads_penalty_search")
        logger.info("tune: roads penalty=%s", penalty)
        best_raw_config = _tune_knn_for_penalty(
            penalty,
            val_contexts,
            model,
            processor,
            device,
            pos_bank,
            neg_bank,
            ps,
            tile_size,
            stride,
            feature_dir,
            context_radius,
            should_stop,
        )
        best_xgb_config, best_bst = _tune_xgb_for_penalty(
            penalty,
            val_contexts,
            xgb_candidates,
            device,
            ps,
            tile_size,
            stride,
            feature_dir,
            context_radius,
            xgb_feature_stats,
            should_stop,
        )
        if KNN_ENABLED and best_raw_config is None:
            raise ValueError("kNN tuning returned no results")
        if XGB_ENABLED and (best_xgb_config is None or best_bst is None):
            raise ValueError("XGB tuning returned no results")
        if KNN_ENABLED and XGB_ENABLED:
            champion_source = (
                "raw" if best_raw_config["iou"] >= best_xgb_config["iou"] else "xgb"
            )
            champion_iou = (
                best_raw_config["iou"]
                if champion_source == "raw"
                else best_xgb_config["iou"]
            )
        elif XGB_ENABLED:
            champion_source = "xgb"
            champion_iou = best_xgb_config["iou"]
        else:
            champion_source = "raw"
            champion_iou = best_raw_config["iou"]
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
    return best_bundle


def _rebuild_champion_scores(
    val_contexts: list[dict],
    champion_source: str,
    thr_center: float,
    roads_penalty: float,
    best_raw_config: dict | None,
    best_bst,
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    xgb_feature_stats: dict | None,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    should_stop: Callable[[], bool] | None,
) -> None:
    """Populate champion score maps back into validation contexts.

    Examples:
        >>> callable(_rebuild_champion_scores)
        True
    """
    for ctx in val_contexts:
        _maybe_stop(should_stop, "champion_score_rebuild")
        ctx["crf_use_trimap"] = champion_source == "xgb"
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
            thr_center = float(best_raw_config["threshold"])
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
        ctx["score_full"] = _apply_roads_penalty(
            score_full, ctx["roads_mask"], roads_penalty
        )
        ctx["thr_center"] = float(thr_center)


def _tune_crf_config(
    val_contexts: list[dict],
    device,
    should_stop: Callable[[], bool] | None,
) -> dict:
    """Tune CRF settings across validation contexts.

    Examples:
        >>> callable(_tune_crf_config)
        True
    """
    best_crf_cfg = {"enabled": bool(CRF_ENABLED), "trimap_band_pixels": 0}
    if not CRF_ENABLED:
        return best_crf_cfg

    use_trimap = any(bool(ctx.get("crf_use_trimap", False)) for ctx in val_contexts)
    crf_candidates = [
        (psf, trimap_band, pw, pxy, bw, bxy, brgb)
        for psf in cfg.search.crf.prob_softness_values
        for trimap_band in (
            cfg.search.crf.trimap_band_pixels_values if use_trimap else [0]
        )
        for pw in cfg.search.crf.pos_w_values
        for pxy in cfg.search.crf.pos_xy_std_values
        for bw in cfg.search.crf.bilateral_w_values
        for bxy in cfg.search.crf.bilateral_xy_std_values
        for brgb in cfg.search.crf.bilateral_rgb_std_values
    ][:CRF_MAX_CONFIGS]
    requested_workers = int(cfg.search.crf.num_workers or 1)
    num_workers = _resolve_crf_workers(requested_workers, len(crf_candidates), device)
    logger.info(
        "tune: CRF grid search configs=%s, workers=%s",
        len(crf_candidates),
        num_workers,
    )
    _init_crf_parallel(val_contexts)
    best_crf_iou = None

    def _update_best(cand, med_iou):
        nonlocal best_crf_iou, best_crf_cfg
        if best_crf_iou is None or med_iou > best_crf_iou:
            best_crf_iou = med_iou
            best_crf_cfg = {
                "enabled": True,
                "prob_softness": cand[0],
                "trimap_band_pixels": int(cand[1]),
                "pos_w": cand[2],
                "pos_xy_std": cand[3],
                "bilateral_w": cand[4],
                "bilateral_xy_std": cand[5],
                "bilateral_rgb_std": cand[6],
            }

    try:
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for med_iou, cand in ex.map(_eval_crf_config, crf_candidates):
                    _maybe_stop(should_stop, "crf_search")
                    _update_best(cand, med_iou)
        else:
            for cand in crf_candidates:
                _maybe_stop(should_stop, "crf_search")
                med_iou, _ = _eval_crf_config(cand)
                _update_best(cand, med_iou)
    except BrokenProcessPool:
        logger.exception("tune: CRF worker pool crashed; retrying CRF search serially")
        for cand in crf_candidates:
            _maybe_stop(should_stop, "crf_search")
            med_iou, _ = _eval_crf_config(cand)
            _update_best(cand, med_iou)

    if best_crf_iou is None:
        raise ValueError("CRF tuning returned no results")
    return best_crf_cfg


def _tune_shadow_config(
    val_contexts: list[dict],
    best_crf_cfg: dict,
    should_stop: Callable[[], bool] | None,
) -> dict:
    """Tune shadow settings across validation contexts.

    Examples:
        >>> callable(_tune_shadow_config)
        True
    """
    best_shadow_cfg = None
    best_shadow_iou = None
    for weights in cfg.postprocess.shadow.weight_sets:
        _maybe_stop(should_stop, "shadow_search")
        iou_by_key = {
            (thr, protect_score): {"sum": 0.0, "w": 0.0}
            for thr in cfg.postprocess.shadow.thresholds
            for protect_score in cfg.postprocess.shadow.protect_scores
        }
        for ctx in val_contexts:
            _maybe_stop(should_stop, "shadow_tile_scoring")
            score_full = ctx["score_full"]
            if CRF_ENABLED:
                mask_base = refine_with_densecrf(
                    ctx["img_b"],
                    score_full,
                    ctx["thr_center"],
                    ctx["sh_buffer_mask"],
                    prob_softness=best_crf_cfg["prob_softness"],
                    n_iters=5,
                    pos_w=best_crf_cfg["pos_w"],
                    pos_xy_std=best_crf_cfg["pos_xy_std"],
                    bilateral_w=best_crf_cfg["bilateral_w"],
                    bilateral_xy_std=best_crf_cfg["bilateral_xy_std"],
                    bilateral_rgb_std=best_crf_cfg["bilateral_rgb_std"],
                    trimap_band_pixels=(
                        int(best_crf_cfg.get("trimap_band_pixels", 0))
                        if bool(ctx.get("crf_use_trimap", False))
                        else None
                    ),
                )
            else:
                mask_base = (score_full >= ctx["thr_center"]) & ctx["sh_buffer_mask"]
            img_float = ctx["img_b"].astype(np.float32)
            vals = (
                (img_float * np.array(weights, dtype=np.float32).reshape(1, 1, 3))
                .sum(axis=2)
                .reshape(-1)[mask_base.reshape(-1)]
            )
            gt_vals = (
                ctx["gt_mask_eval"].reshape(-1).astype(bool)[mask_base.reshape(-1)]
            )
            score_vals = score_full.reshape(-1)[mask_base.reshape(-1)]
            if vals.size == 0:
                continue
            thr_arr = np.array(
                cfg.postprocess.shadow.thresholds, dtype=np.float32
            ).reshape(-1, 1)
            mask_thr = vals[None, :] >= thr_arr
            for protect_score in cfg.postprocess.shadow.protect_scores:
                protect_mask = score_vals >= protect_score
                mask_keep = mask_thr | protect_mask[None, :]
                tp = (
                    np.logical_and(mask_keep, gt_vals[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                fp = (
                    np.logical_and(mask_keep, ~gt_vals[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                fn = (
                    np.logical_and(~mask_keep, gt_vals[None, :])
                    .sum(axis=1)
                    .astype(np.float64)
                )
                iou = tp / (tp + fp + fn + 1e-8)
                for i, thr in enumerate(cfg.postprocess.shadow.thresholds):
                    stats = iou_by_key[(thr, protect_score)]
                    stats["sum"] += float(iou[i]) * float(ctx["gt_weight"])
                    stats["w"] += float(ctx["gt_weight"])
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
    return best_shadow_cfg


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
    if not (KNN_ENABLED or XGB_ENABLED):
        raise ValueError("both kNN and XGB are disabled; enable at least one model")
    val_contexts = _build_validation_contexts(
        val_paths,
        gt_vector_paths,
        model,
        processor,
        device,
        ps,
        tile_size,
        stride,
        feature_dir,
        should_stop,
    )
    xgb_candidates = _build_xgb_candidates(
        X,
        y,
        feature_layout,
        xgb_feature_stats,
        val_contexts,
        device,
        ps,
        tile_size,
        stride,
        feature_dir,
        context_radius,
        should_stop,
    )
    best_bundle = _select_best_penalty_bundle(
        [float(p) for p in cfg.postprocess.roads.penalty_values],
        val_contexts,
        model,
        processor,
        device,
        pos_bank,
        neg_bank,
        xgb_candidates,
        xgb_feature_stats,
        ps,
        tile_size,
        stride,
        feature_dir,
        context_radius,
        should_stop,
    )
    roads_penalty = float(best_bundle["roads_penalty"])
    best_raw_config = best_bundle["best_raw_config"]
    best_xgb_config = best_bundle["best_xgb_config"]
    best_bst = best_bundle["best_bst"]
    champion_source = best_bundle["champion_source"]
    thr_center = (
        float(best_raw_config["threshold"])
        if champion_source == "raw"
        else float(best_xgb_config["threshold"])
    )
    _rebuild_champion_scores(
        val_contexts,
        champion_source,
        thr_center,
        roads_penalty,
        best_raw_config,
        best_bst,
        model,
        processor,
        device,
        pos_bank,
        neg_bank,
        xgb_feature_stats,
        ps,
        tile_size,
        stride,
        feature_dir,
        context_radius,
        should_stop,
    )
    best_crf_cfg = _tune_crf_config(val_contexts, device, should_stop)
    best_shadow_cfg = _tune_shadow_config(val_contexts, best_crf_cfg, should_stop)
    logger.info("tune: roads penalty selected=%s", roads_penalty)
    if best_raw_config is None:
        best_raw_config = {
            "k": -1,
            "threshold": float(best_xgb_config["threshold"]),
            "source": "raw_disabled",
            "iou": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    if best_xgb_config is None:
        best_xgb_config = {
            "k": -1,
            "threshold": float(best_raw_config["threshold"]),
            "source": "xgb_disabled",
            "iou": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "params": None,
        }
    return {
        "bst": best_bst,
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "best_crf_config": {**best_crf_cfg, "k": best_raw_config.get("k", -1)},
        "shadow_cfg": best_shadow_cfg,
        "roads_penalty": float(roads_penalty),
        "xgb_feature_stats": xgb_feature_stats,
        "feature_layout": feature_layout,
        "knn_enabled": bool(KNN_ENABLED),
        "xgb_enabled": bool(XGB_ENABLED),
        "crf_enabled": bool(CRF_ENABLED),
    }
