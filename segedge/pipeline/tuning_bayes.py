from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import binary_dilation, disk

import config as cfg

from ..core.continuity import bridge_skeleton_gaps
from ..core.crf_utils import refine_with_densecrf
from ..core.features import prefetch_features_single_scale_image
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from ..core.metrics_utils import compute_metrics
from ..core.optuna_feedback import (
    build_optuna_callbacks_with_feedback,
    objective_weights,
)
from ..core.summary_utils import weighted_mean
from ..core.xdboost import xgb_score_image_b
from .inference_utils import (
    _apply_roads_penalty,
    _apply_shadow_filter,
    _compute_top_p,
    _top_p_threshold,
)

USE_FP16_KNN = getattr(cfg, "USE_FP16_KNN", True)


def get_optuna_module():
    try:
        import optuna  # type: ignore

        return optuna
    except Exception:
        return None


def mask_iou(
    pred_mask: np.ndarray,
    ref_mask: np.ndarray,
    region_mask: np.ndarray | None = None,
) -> float:
    """Compute IoU between binary masks inside an optional region.

    Examples:
        >>> isinstance(mask_iou.__name__, str)
        True
    """
    p = np.asarray(pred_mask).astype(bool)
    r = np.asarray(ref_mask).astype(bool)
    if region_mask is not None:
        reg = np.asarray(region_mask).astype(bool)
        p = np.logical_and(p, reg)
        r = np.logical_and(r, reg)
    tp = float(np.logical_and(p, r).sum())
    fp = float(np.logical_and(p, ~r).sum())
    fn = float(np.logical_and(~p, r).sum())
    return tp / (tp + fp + fn + 1e-8)


def robust_objective(iou_gt: float, iou_sh: float, w_gt: float, w_sh: float) -> float:
    return float(w_gt * iou_gt + w_sh * iou_sh)


def _f1_optimal_threshold(
    score_map: np.ndarray,
    gt_mask: np.ndarray,
    region_mask: np.ndarray,
    bins: int = 64,
) -> float:
    """Approximate F1-optimal threshold on a score map.

    Examples:
        >>> isinstance(_f1_optimal_threshold.__name__, str)
        True
    """
    reg = np.asarray(region_mask).astype(bool)
    if reg.sum() == 0:
        return 0.5
    y_true = np.asarray(gt_mask).astype(bool)[reg]
    y_score = np.asarray(score_map, dtype=np.float32)[reg]
    if y_score.size == 0:
        return 0.5
    lo = float(np.min(y_score))
    hi = float(np.max(y_score))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return float(lo if np.isfinite(lo) else 0.5)
    thresholds = np.linspace(lo, hi, int(max(8, bins)))
    best_thr = thresholds[0]
    best_f1 = -1.0
    for thr in thresholds:
        pred = y_score >= thr
        tp = float(np.logical_and(pred, y_true).sum())
        fp = float(np.logical_and(pred, ~y_true).sum())
        fn = float(np.logical_and(~pred, y_true).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def light_deterministic_perturbations(
    img_rgb: np.ndarray,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    """Generate deterministic light perturbations for robustness scoring.

    Examples:
        >>> isinstance(light_deterministic_perturbations.__name__, str)
        True
    """
    if count <= 0:
        return []
    base = img_rgb.astype(np.float32) / 255.0
    rng = np.random.default_rng(int(seed))
    outs: list[np.ndarray] = []
    for _ in range(int(count)):
        contrast = float(rng.uniform(0.95, 1.05))
        brightness = float(rng.uniform(0.95, 1.05))
        noise_sigma = float(rng.uniform(0.0, 0.01))
        blur_sigma = float(rng.uniform(0.0, 0.6))
        arr = (base - 0.5) * contrast + 0.5
        arr = arr * brightness
        if noise_sigma > 0.0:
            arr = arr + rng.normal(0.0, noise_sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        if blur_sigma > 1e-6:
            arr = gaussian_filter(arr, sigma=(blur_sigma, blur_sigma, 0.0))
        outs.append(np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8))
    return outs


def attach_perturbations_to_contexts(
    val_contexts: list[dict],
    count: int,
    seed: int,
) -> None:
    """Attach deterministic perturbed images to validation contexts.

    Examples:
        >>> isinstance(attach_perturbations_to_contexts.__name__, str)
        True
    """
    for idx, ctx in enumerate(val_contexts):
        ctx["perturbed_imgs"] = light_deterministic_perturbations(
            ctx["img_b"],
            count=count,
            seed=seed + idx * 1009,
        )


def score_maps_for_image(
    *,
    img_b: np.ndarray,
    ctx: dict,
    k: int,
    bst,
    roads_penalty: float,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    neg_alpha: float,
    use_prefetched: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute roads-penalized kNN/XGB score maps for one image.

    Examples:
        >>> isinstance(score_maps_for_image.__name__, str)
        True
    """
    prefetched = ctx.get("prefetched_b") if use_prefetched else None
    if prefetched is None:
        prefetched = prefetch_features_single_scale_image(
            img_b,
            model,
            processor,
            device,
            ps,
            tile_size,
            stride,
            None,
            None,
            f"{ctx['image_id']}_bo",
        )
    score_knn, _ = zero_shot_knn_single_scale_B_with_saliency(
        img_b=img_b,
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
        neg_alpha=float(neg_alpha),
        prefetched_tiles=prefetched,
        use_fp16_matmul=USE_FP16_KNN,
        context_radius=context_radius,
    )
    score_xgb = xgb_score_image_b(
        img_b,
        bst,
        ps,
        tile_size,
        stride,
        feature_dir,
        ctx["image_id"],
        prefetched_tiles=prefetched,
        context_radius=context_radius,
    )
    score_knn = _apply_roads_penalty(score_knn, ctx["roads_mask"], roads_penalty)
    score_xgb = _apply_roads_penalty(score_xgb, ctx["roads_mask"], roads_penalty)
    return score_knn, score_xgb


def threshold_knn_xgb(
    score_knn: np.ndarray,
    score_xgb: np.ndarray,
    *,
    buffer_density: float,
    sh_buffer_mask: np.ndarray,
    top_p_a: float,
    top_p_b: float,
    top_p_min: float,
    top_p_max: float,
) -> dict:
    """Threshold kNN/XGB score maps using adaptive top-p inside SH buffer.

    Examples:
        >>> isinstance(threshold_knn_xgb.__name__, str)
        True
    """
    p_val = _compute_top_p(buffer_density, top_p_a, top_p_b, top_p_min, top_p_max)
    knn_thr, mask_knn = _top_p_threshold(score_knn, sh_buffer_mask, p_val)
    xgb_thr, mask_xgb = _top_p_threshold(score_xgb, sh_buffer_mask, p_val)
    mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0
    mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0
    return {
        "mask_knn": mask_knn,
        "mask_xgb": mask_xgb,
        "knn_thr": float(knn_thr),
        "xgb_thr": float(xgb_thr),
    }


def make_optuna_study(optuna_mod, stage_name: str, seed: int):
    """Create an Optuna study with deterministic sampler/pruner settings.

    Examples:
        >>> isinstance(make_optuna_study.__name__, str)
        True
    """
    sampler_name = str(getattr(cfg, "BO_SAMPLER", "tpe")).strip().lower()
    if sampler_name == "cmaes":
        sampler = optuna_mod.samplers.CmaEsSampler(
            seed=int(seed),
            n_startup_trials=int(getattr(cfg, "BO_N_STARTUP_TRIALS", 12) or 12),
            warn_independent_sampling=False,
        )
    else:
        sampler = optuna_mod.samplers.TPESampler(
            seed=int(seed),
            n_startup_trials=int(getattr(cfg, "BO_N_STARTUP_TRIALS", 12) or 12),
            multivariate=bool(getattr(cfg, "BO_TPE_MULTIVARIATE", True)),
            group=bool(getattr(cfg, "BO_TPE_GROUP", True)),
        )
    pruner = optuna_mod.pruners.SuccessiveHalvingPruner(
        min_resource=int(getattr(cfg, "BO_PRUNER_MIN_RESOURCE", 1) or 1),
        reduction_factor=int(getattr(cfg, "BO_PRUNER_REDUCTION_FACTOR", 2) or 2),
        min_early_stopping_rate=int(getattr(cfg, "BO_PRUNER_WARMUP_STEPS", 1) or 1),
    )
    pruner_obj = pruner if bool(getattr(cfg, "BO_ENABLE_PRUNING", True)) else None
    storage_path = getattr(cfg, "BO_STORAGE_PATH", None)
    tag = str(getattr(cfg, "BO_STUDY_TAG", "") or "").strip()
    base = f"{getattr(cfg, 'BO_STUDY_NAME', 'segedge_tuning')}_{stage_name}"
    study_name = f"{base}_{tag}" if tag else base
    if storage_path:
        return optuna_mod.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner_obj,
            storage=f"sqlite:///{storage_path}",
            study_name=study_name,
            load_if_exists=True,
        )
    return optuna_mod.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner_obj,
    )


def _suggest_from_values(trial, name: str, values: list[float | int]):
    """Suggest a value from a finite list. >>> isinstance(_suggest_from_values.__name__, str)"""
    if len(values) == 1:
        return values[0]
    return trial.suggest_categorical(name, list(values))


def _range_spec(key: str) -> tuple | None:
    """Return a normalized range tuple from config. >>> isinstance(_range_spec.__name__, str)"""
    value = getattr(cfg, key, None)
    if isinstance(value, (list, tuple)) and len(value) in (2, 3):
        return tuple(value)
    return None


def _suggest_int_param(
    trial,
    *,
    name: str,
    range_key: str,
    values_key: str,
    default_values: list[int],
    override: tuple | list | None = None,
) -> int:
    """Suggest an integer parameter with range-first precedence.

    Examples:
        >>> isinstance(_suggest_int_param.__name__, str)
        True
    """
    if override is not None:
        if isinstance(override, tuple):
            if len(override) == 3:
                return int(
                    trial.suggest_int(
                        name,
                        int(override[0]),
                        int(override[1]),
                        step=int(override[2]),
                    )
                )
            return int(trial.suggest_int(name, int(override[0]), int(override[1])))
        return int(_suggest_from_values(trial, name, [int(v) for v in override]))
    spec = _range_spec(range_key)
    if spec is not None:
        if len(spec) == 3:
            return int(
                trial.suggest_int(
                    name,
                    int(spec[0]),
                    int(spec[1]),
                    step=int(spec[2]),
                )
            )
        return int(trial.suggest_int(name, int(spec[0]), int(spec[1])))
    values = [int(v) for v in (getattr(cfg, values_key, None) or default_values)]
    return int(_suggest_from_values(trial, name, values))


def _suggest_float_param(
    trial,
    *,
    name: str,
    range_key: str,
    values_key: str,
    default_values: list[float],
    log: bool = False,
    override: tuple | list | None = None,
) -> float:
    """Suggest a float parameter with range-first precedence.

    Examples:
        >>> isinstance(_suggest_float_param.__name__, str)
        True
    """
    if override is not None:
        if isinstance(override, tuple):
            return float(
                trial.suggest_float(
                    name,
                    float(override[0]),
                    float(override[1]),
                    log=bool(log),
                )
            )
        return float(_suggest_from_values(trial, name, [float(v) for v in override]))
    spec = _range_spec(range_key)
    if spec is not None:
        return float(
            trial.suggest_float(
                name,
                float(spec[0]),
                float(spec[1]),
                log=bool(log),
            )
        )
    values = [float(v) for v in (getattr(cfg, values_key, None) or default_values)]
    return float(_suggest_from_values(trial, name, values))


def _extract_study_importances(optuna_mod, study) -> dict[str, float]:
    """Extract Optuna parameter importances when available."""
    try:
        raw = optuna_mod.importance.get_param_importances(study)
        out: dict[str, float] = {}
        for key, val in raw.items():
            clean_key = key[:-4] if key.endswith("_idx") else key
            out[clean_key] = float(val)
        return out
    except Exception:
        return {}


def write_bo_importances_file(output_path: str, payload: dict) -> None:
    """Write a JSON artifact with stage-wise hyperparameter importances."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _predict_champion_shadow(
    *,
    img_b: np.ndarray,
    ctx: dict,
    k: int,
    bst,
    roads_penalty: float,
    top_p_a: float,
    top_p_b: float,
    top_p_min: float,
    top_p_max: float,
    champion_source: str,
    crf_cfg: dict,
    shadow_cfg: dict,
    bridge_cfg: dict | None,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    neg_alpha: float,
    use_prefetched: bool,
    use_dynamic_f1_threshold: bool = False,
) -> np.ndarray:
    score_knn, score_xgb = score_maps_for_image(
        img_b=img_b,
        ctx=ctx,
        k=k,
        bst=bst,
        roads_penalty=roads_penalty,
        model=model,
        processor=processor,
        device=device,
        ps=ps,
        tile_size=tile_size,
        stride=stride,
        feature_dir=feature_dir,
        context_radius=context_radius,
        pos_bank=pos_bank,
        neg_bank=neg_bank,
        neg_alpha=neg_alpha,
        use_prefetched=use_prefetched,
    )
    thr = threshold_knn_xgb(
        score_knn,
        score_xgb,
        buffer_density=float(ctx["buffer_density"]),
        sh_buffer_mask=ctx["sh_buffer_mask"],
        top_p_a=top_p_a,
        top_p_b=top_p_b,
        top_p_min=top_p_min,
        top_p_max=top_p_max,
    )
    champion_score = score_knn if champion_source == "raw" else score_xgb
    champion_thr = thr["knn_thr"] if champion_source == "raw" else thr["xgb_thr"]
    if use_dynamic_f1_threshold:
        champion_thr = _f1_optimal_threshold(
            champion_score,
            ctx["gt_mask_eval"],
            ctx["sh_buffer_mask"],
            bins=int(getattr(cfg, "BO_DYNAMIC_THRESHOLD_BINS", 64) or 64),
        )
    champion_crf, champion_prob = refine_with_densecrf(
        img_b,
        champion_score,
        champion_thr,
        ctx["sh_buffer_mask"],
        prob_softness=crf_cfg["prob_softness"],
        n_iters=5,
        pos_w=crf_cfg["pos_w"],
        pos_xy_std=crf_cfg["pos_xy_std"],
        bilateral_w=crf_cfg["bilateral_w"],
        bilateral_xy_std=crf_cfg["bilateral_xy_std"],
        bilateral_rgb_std=crf_cfg["bilateral_rgb_std"],
        return_prob=True,
    )
    bridged = champion_crf
    if bridge_cfg is not None:
        bridged = bridge_skeleton_gaps(
            champion_crf,
            champion_prob,
            max_gap_px=int(bridge_cfg["bridge_max_gap_px"]),
            max_pairs_per_endpoint=int(bridge_cfg["bridge_max_pairs"]),
            max_avg_cost=float(bridge_cfg["bridge_max_avg_cost"]),
            bridge_width_px=int(bridge_cfg["bridge_width_px"]),
            min_component_area_px=int(bridge_cfg["bridge_min_component_px"]),
            spur_prune_iters=int(bridge_cfg["bridge_spur_prune_iters"]),
        )
    return _apply_shadow_filter(
        img_b,
        bridged,
        shadow_cfg["weights"],
        shadow_cfg["threshold"],
        champion_score,
        shadow_cfg.get("protect_score"),
    )


def run_stage1_bayes(
    *,
    optuna_mod,
    val_contexts: list[dict],
    xgb_candidates: list[dict],
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
) -> dict:
    w_gt, w_sh = objective_weights(cfg)

    def objective(trial):
        roads_penalty = _suggest_float_param(
            trial,
            name="roads_penalty",
            range_key="BO_ROADS_PENALTY_RANGE",
            values_key="ROADS_PENALTY_VALUES",
            default_values=[1.0],
        )
        top_p_a = _suggest_float_param(
            trial,
            name="top_p_a",
            range_key="BO_TOP_P_A_RANGE",
            values_key="TOP_P_A_VALUES",
            default_values=[getattr(cfg, "TOP_P_A", 0.0)],
        )
        top_p_b = _suggest_float_param(
            trial,
            name="top_p_b",
            range_key="BO_TOP_P_B_RANGE",
            values_key="TOP_P_B_VALUES",
            default_values=[getattr(cfg, "TOP_P_B", 0.05)],
        )
        top_p_min = _suggest_float_param(
            trial,
            name="top_p_min",
            range_key="BO_TOP_P_MIN_RANGE",
            values_key="TOP_P_MIN_VALUES",
            default_values=[getattr(cfg, "TOP_P_MIN", 0.02)],
        )
        top_p_max = _suggest_float_param(
            trial,
            name="top_p_max",
            range_key="BO_TOP_P_MAX_RANGE",
            values_key="TOP_P_MAX_VALUES",
            default_values=[getattr(cfg, "TOP_P_MAX", 0.08)],
        )
        if top_p_min > top_p_max:
            raise optuna_mod.TrialPruned("invalid top-p bounds")
        k = _suggest_int_param(
            trial,
            name="k",
            range_key="BO_K_RANGE",
            values_key="K_VALUES",
            default_values=[200],
        )
        neg_alpha = _suggest_float_param(
            trial,
            name="neg_alpha",
            range_key="BO_NEG_ALPHA_RANGE",
            values_key="NEG_ALPHA_VALUES",
            default_values=[getattr(cfg, "NEG_ALPHA", 1.0)],
        )
        dilate_px = _suggest_int_param(
            trial,
            name="silver_core_dilate_px",
            range_key="BO_SILVER_CORE_DILATE_PX_RANGE",
            values_key="SILVER_CORE_DILATE_PX_VALUES",
            default_values=[getattr(cfg, "SILVER_CORE_DILATE_PX", 1)],
        )
        xgb_idx = int(
            trial.suggest_int("xgb_candidate_idx", 0, len(xgb_candidates) - 1)
        )
        bst = xgb_candidates[xgb_idx]["bst"]

        core_vals = []
        core_gt_vals = []
        core_sh_vals = []
        knn_stats = {"iou": [], "f1": [], "precision": [], "recall": []}
        xgb_stats = {"iou": [], "f1": [], "precision": [], "recall": []}
        weights = []

        for step, ctx in enumerate(val_contexts):
            score_knn, score_xgb = score_maps_for_image(
                img_b=ctx["img_b"],
                ctx=ctx,
                k=k,
                bst=bst,
                roads_penalty=roads_penalty,
                model=model,
                processor=processor,
                device=device,
                ps=ps,
                tile_size=tile_size,
                stride=stride,
                feature_dir=feature_dir,
                context_radius=context_radius,
                pos_bank=pos_bank,
                neg_bank=neg_bank,
                neg_alpha=neg_alpha,
                use_prefetched=True,
            )
            thr = threshold_knn_xgb(
                score_knn,
                score_xgb,
                buffer_density=float(ctx["buffer_density"]),
                sh_buffer_mask=ctx["sh_buffer_mask"],
                top_p_a=top_p_a,
                top_p_b=top_p_b,
                top_p_min=top_p_min,
                top_p_max=top_p_max,
            )
            metrics_knn = compute_metrics(thr["mask_knn"], ctx["gt_mask_eval"])
            metrics_xgb = compute_metrics(thr["mask_xgb"], ctx["gt_mask_eval"])
            for key in knn_stats:
                knn_stats[key].append(float(metrics_knn[key]))
                xgb_stats[key].append(float(metrics_xgb[key]))

            core_mask = np.logical_and(thr["mask_knn"], thr["mask_xgb"])
            if dilate_px > 0:
                core_mask = binary_dilation(core_mask, disk(dilate_px))
            iou_gt = float(compute_metrics(core_mask, ctx["gt_mask_eval"])["iou"])

            sh_ious = []
            for p_img in ctx.get("perturbed_imgs", []):
                score_knn_p, score_xgb_p = score_maps_for_image(
                    img_b=p_img,
                    ctx=ctx,
                    k=k,
                    bst=bst,
                    roads_penalty=roads_penalty,
                    model=model,
                    processor=processor,
                    device=device,
                    ps=ps,
                    tile_size=tile_size,
                    stride=stride,
                    feature_dir=feature_dir,
                    context_radius=context_radius,
                    pos_bank=pos_bank,
                    neg_bank=neg_bank,
                    neg_alpha=neg_alpha,
                    use_prefetched=False,
                )
                thr_p = threshold_knn_xgb(
                    score_knn_p,
                    score_xgb_p,
                    buffer_density=float(ctx["buffer_density"]),
                    sh_buffer_mask=ctx["sh_buffer_mask"],
                    top_p_a=top_p_a,
                    top_p_b=top_p_b,
                    top_p_min=top_p_min,
                    top_p_max=top_p_max,
                )
                core_mask_p = np.logical_and(thr_p["mask_knn"], thr_p["mask_xgb"])
                if dilate_px > 0:
                    core_mask_p = binary_dilation(core_mask_p, disk(dilate_px))
                sh_ious.append(
                    mask_iou(core_mask_p, ctx["labels_sh"] > 0, ctx["sh_buffer_mask"])
                )
            iou_sh = (
                float(np.mean(sh_ious))
                if sh_ious
                else mask_iou(
                    core_mask,
                    ctx["labels_sh"] > 0,
                    ctx["sh_buffer_mask"],
                )
            )
            core_vals.append(robust_objective(iou_gt, iou_sh, w_gt, w_sh))
            core_gt_vals.append(iou_gt)
            core_sh_vals.append(iou_sh)
            weights.append(float(ctx["gt_weight"]))
            trial.report(float(weighted_mean(core_vals, weights)), step=step)
            if bool(getattr(cfg, "BO_ENABLE_PRUNING", True)) and trial.should_prune():
                raise optuna_mod.TrialPruned()

        trial.set_user_attr(
            "weighted_iou_gt_core", float(weighted_mean(core_gt_vals, weights))
        )
        trial.set_user_attr(
            "weighted_iou_sh_core", float(weighted_mean(core_sh_vals, weights))
        )
        trial.set_user_attr("roads_penalty", float(roads_penalty))
        trial.set_user_attr("top_p_a", float(top_p_a))
        trial.set_user_attr("top_p_b", float(top_p_b))
        trial.set_user_attr("top_p_min", float(top_p_min))
        trial.set_user_attr("top_p_max", float(top_p_max))
        trial.set_user_attr("k", int(k))
        trial.set_user_attr("neg_alpha", float(neg_alpha))
        trial.set_user_attr("silver_core_dilate_px", int(dilate_px))
        trial.set_user_attr("xgb_candidate_idx", int(xgb_idx))
        for key in knn_stats:
            trial.set_user_attr(
                f"knn_{key}", float(weighted_mean(knn_stats[key], weights))
            )
            trial.set_user_attr(
                f"xgb_{key}", float(weighted_mean(xgb_stats[key], weights))
            )
        return float(weighted_mean(core_vals, weights))

    study = make_optuna_study(
        optuna_mod,
        "stage1_raw",
        seed=int(getattr(cfg, "BO_SEED", 42) or 42),
    )
    study.optimize(
        objective,
        n_trials=int(getattr(cfg, "BO_STAGE1_TRIALS", 50) or 50),
        timeout=getattr(cfg, "BO_TIMEOUT_S", None),
        callbacks=build_optuna_callbacks_with_feedback(
            cfg,
            stage_name="stage1_raw",
            default_patience=20,
        ),
    )
    best = study.best_trial
    xgb_idx = int(
        best.user_attrs.get("xgb_candidate_idx", best.params["xgb_candidate_idx"])
    )
    best_bst = xgb_candidates[xgb_idx]["bst"]
    best_params = xgb_candidates[xgb_idx]["params"]

    best_raw_config = {
        "k": int(best.user_attrs.get("k", getattr(cfg, "K_VALUES", [200])[0])),
        "source": "raw",
        "iou": float(best.user_attrs.get("knn_iou", 0.0)),
        "f1": float(best.user_attrs.get("knn_f1", 0.0)),
        "precision": float(best.user_attrs.get("knn_precision", 0.0)),
        "recall": float(best.user_attrs.get("knn_recall", 0.0)),
    }
    best_xgb_config = {
        "k": -1,
        "source": "xgb",
        "iou": float(best.user_attrs.get("xgb_iou", 0.0)),
        "f1": float(best.user_attrs.get("xgb_f1", 0.0)),
        "precision": float(best.user_attrs.get("xgb_precision", 0.0)),
        "recall": float(best.user_attrs.get("xgb_recall", 0.0)),
        "params": best_params,
    }
    champion_source = (
        "raw" if best_raw_config["iou"] >= best_xgb_config["iou"] else "xgb"
    )
    return {
        "roads_penalty": float(best.user_attrs.get("roads_penalty", 1.0)),
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "best_bst": best_bst,
        "champion_source": champion_source,
        "top_p_a": float(best.user_attrs.get("top_p_a", getattr(cfg, "TOP_P_A", 0.0))),
        "top_p_b": float(best.user_attrs.get("top_p_b", getattr(cfg, "TOP_P_B", 0.05))),
        "top_p_min": float(
            best.user_attrs.get("top_p_min", getattr(cfg, "TOP_P_MIN", 0.02))
        ),
        "top_p_max": float(
            best.user_attrs.get("top_p_max", getattr(cfg, "TOP_P_MAX", 0.08))
        ),
        "silver_core_iou": float(
            best.user_attrs.get("weighted_iou_gt_core", best.value)
        ),
        "silver_core_iou_sh": float(best.user_attrs.get("weighted_iou_sh_core", 0.0)),
        "silver_core_dilate_px": int(
            best.user_attrs.get(
                "silver_core_dilate_px",
                getattr(cfg, "SILVER_CORE_DILATE_PX", 1),
            )
        ),
        "neg_alpha": float(
            best.user_attrs.get("neg_alpha", getattr(cfg, "NEG_ALPHA", 1.0))
        ),
        "optuna_importances": _extract_study_importances(optuna_mod, study),
        "optuna_best_value": float(best.value),
    }


def run_stage2_bayes(
    *,
    optuna_mod,
    val_contexts: list[dict],
    stage1_bundle: dict,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
) -> tuple[dict, dict, dict]:
    w_gt, w_sh = objective_weights(cfg)
    champion_source = str(stage1_bundle["champion_source"])
    k = int(stage1_bundle["best_raw_config"]["k"])
    bst = stage1_bundle["best_bst"]
    roads_penalty = float(stage1_bundle["roads_penalty"])
    top_p_a = float(stage1_bundle["top_p_a"])
    top_p_b = float(stage1_bundle["top_p_b"])
    top_p_min = float(stage1_bundle["top_p_min"])
    top_p_max = float(stage1_bundle["top_p_max"])
    neg_alpha = float(stage1_bundle.get("neg_alpha", getattr(cfg, "NEG_ALPHA", 1.0)))
    weights_sets = list(getattr(cfg, "SHADOW_WEIGHT_SETS", [(1.0, 1.0, 1.0)]))
    shadow_thr_vals = [int(v) for v in getattr(cfg, "SHADOW_THRESHOLDS", [40])]
    protect_vals = [float(v) for v in getattr(cfg, "SHADOW_PROTECT_SCORES", [0.5])]
    top_n = int(getattr(cfg, "BO_STAGE2_TOP_N", 10) or 10)
    broad_frac = float(getattr(cfg, "BO_STAGE2_BROAD_FRACTION", 0.6) or 0.6)
    total_trials = int(getattr(cfg, "BO_STAGE2_TRIALS", 40) or 40)
    broad_trials = max(1, int(total_trials * broad_frac))
    refine_trials = max(0, total_trials - broad_trials)

    def _suggest_stage2(trial, refine: dict | None = None):
        def _override(key: str):
            if refine is None:
                return None
            return refine.get(key)

        crf_cfg = {
            "prob_softness": _suggest_float_param(
                trial,
                name="prob_softness",
                range_key="BO_CRF_PROB_SOFTNESS_RANGE",
                values_key="PROB_SOFTNESS_VALUES",
                default_values=[0.1],
                override=_override("prob_softness"),
            ),
            "pos_w": _suggest_float_param(
                trial,
                name="pos_w",
                range_key="BO_CRF_POS_W_RANGE",
                values_key="POS_W_VALUES",
                default_values=[1.0],
                override=_override("pos_w"),
            ),
            "pos_xy_std": _suggest_float_param(
                trial,
                name="pos_xy_std",
                range_key="BO_CRF_POS_XY_STD_RANGE",
                values_key="POS_XY_STD_VALUES",
                default_values=[3.0],
                override=_override("pos_xy_std"),
            ),
            "bilateral_w": _suggest_float_param(
                trial,
                name="bilateral_w",
                range_key="BO_CRF_BILATERAL_W_RANGE",
                values_key="BILATERAL_W_VALUES",
                default_values=[3.0],
                override=_override("bilateral_w"),
            ),
            "bilateral_xy_std": _suggest_float_param(
                trial,
                name="bilateral_xy_std",
                range_key="BO_CRF_BILATERAL_XY_STD_RANGE",
                values_key="BILATERAL_XY_STD_VALUES",
                default_values=[25.0],
                override=_override("bilateral_xy_std"),
            ),
            "bilateral_rgb_std": _suggest_float_param(
                trial,
                name="bilateral_rgb_std",
                range_key="BO_CRF_BILATERAL_RGB_STD_RANGE",
                values_key="BILATERAL_RGB_STD_VALUES",
                default_values=[3.0],
                override=_override("bilateral_rgb_std"),
            ),
        }
        weight_choices = _override("shadow_weight_idx")
        if isinstance(weight_choices, list):
            shadow_weight_idx = int(
                _suggest_from_values(trial, "shadow_weight_idx", weight_choices)
            )
        else:
            shadow_weight_idx = int(
                trial.suggest_int("shadow_weight_idx", 0, len(weights_sets) - 1)
            )
        shadow_cfg = {
            "weights": tuple(weights_sets[shadow_weight_idx]),
            "threshold": _suggest_int_param(
                trial,
                name="shadow_threshold",
                range_key="BO_SHADOW_THRESHOLD_RANGE",
                values_key="SHADOW_THRESHOLDS",
                default_values=shadow_thr_vals,
                override=_override("shadow_threshold"),
            ),
            "protect_score": _suggest_float_param(
                trial,
                name="shadow_protect_score",
                range_key="BO_SHADOW_PROTECT_SCORE_RANGE",
                values_key="SHADOW_PROTECT_SCORES",
                default_values=protect_vals,
                override=_override("shadow_protect_score"),
            ),
        }
        return crf_cfg, shadow_cfg, shadow_weight_idx

    def _objective_with_refine(trial, refine: dict | None = None):
        crf_cfg, shadow_cfg, shadow_weight_idx = _suggest_stage2(trial, refine=refine)
        vals = []
        gt_vals = []
        sh_vals = []
        weights = []
        for step, ctx in enumerate(val_contexts):
            shadow = _predict_champion_shadow(
                img_b=ctx["img_b"],
                ctx=ctx,
                k=k,
                bst=bst,
                roads_penalty=roads_penalty,
                top_p_a=top_p_a,
                top_p_b=top_p_b,
                top_p_min=top_p_min,
                top_p_max=top_p_max,
                champion_source=champion_source,
                crf_cfg=crf_cfg,
                shadow_cfg=shadow_cfg,
                bridge_cfg=None,
                model=model,
                processor=processor,
                device=device,
                ps=ps,
                tile_size=tile_size,
                stride=stride,
                feature_dir=feature_dir,
                context_radius=context_radius,
                pos_bank=pos_bank,
                neg_bank=neg_bank,
                neg_alpha=neg_alpha,
                use_prefetched=True,
                use_dynamic_f1_threshold=bool(
                    getattr(cfg, "BO_USE_DYNAMIC_F1_THRESHOLD", False)
                ),
            )
            iou_gt = float(compute_metrics(shadow, ctx["gt_mask_eval"])["iou"])
            sh_ious = []
            for p_img in ctx.get("perturbed_imgs", []):
                shadow_p = _predict_champion_shadow(
                    img_b=p_img,
                    ctx=ctx,
                    k=k,
                    bst=bst,
                    roads_penalty=roads_penalty,
                    top_p_a=top_p_a,
                    top_p_b=top_p_b,
                    top_p_min=top_p_min,
                    top_p_max=top_p_max,
                    champion_source=champion_source,
                    crf_cfg=crf_cfg,
                    shadow_cfg=shadow_cfg,
                    bridge_cfg=None,
                    model=model,
                    processor=processor,
                    device=device,
                    ps=ps,
                    tile_size=tile_size,
                    stride=stride,
                    feature_dir=feature_dir,
                    context_radius=context_radius,
                    pos_bank=pos_bank,
                    neg_bank=neg_bank,
                    neg_alpha=neg_alpha,
                    use_prefetched=False,
                    use_dynamic_f1_threshold=bool(
                        getattr(cfg, "BO_USE_DYNAMIC_F1_THRESHOLD", False)
                    ),
                )
                sh_ious.append(
                    mask_iou(shadow_p, ctx["labels_sh"] > 0, ctx["sh_buffer_mask"])
                )
            iou_sh = (
                float(np.mean(sh_ious))
                if sh_ious
                else mask_iou(
                    shadow,
                    ctx["labels_sh"] > 0,
                    ctx["sh_buffer_mask"],
                )
            )
            vals.append(robust_objective(iou_gt, iou_sh, w_gt, w_sh))
            gt_vals.append(iou_gt)
            sh_vals.append(iou_sh)
            weights.append(float(ctx["gt_weight"]))
            trial.report(float(weighted_mean(vals, weights)), step=step)
            if bool(getattr(cfg, "BO_ENABLE_PRUNING", True)) and trial.should_prune():
                raise optuna_mod.TrialPruned()

        trial.set_user_attr("weighted_iou_gt", float(weighted_mean(gt_vals, weights)))
        trial.set_user_attr("weighted_iou_sh", float(weighted_mean(sh_vals, weights)))
        trial.set_user_attr("prob_softness", float(crf_cfg["prob_softness"]))
        trial.set_user_attr("pos_w", float(crf_cfg["pos_w"]))
        trial.set_user_attr("pos_xy_std", float(crf_cfg["pos_xy_std"]))
        trial.set_user_attr("bilateral_w", float(crf_cfg["bilateral_w"]))
        trial.set_user_attr("bilateral_xy_std", float(crf_cfg["bilateral_xy_std"]))
        trial.set_user_attr("bilateral_rgb_std", float(crf_cfg["bilateral_rgb_std"]))
        trial.set_user_attr("shadow_weight_idx", int(shadow_weight_idx))
        trial.set_user_attr("shadow_threshold", int(shadow_cfg["threshold"]))
        trial.set_user_attr("shadow_protect_score", float(shadow_cfg["protect_score"]))
        return float(weighted_mean(vals, weights))

    stage2_broad = make_optuna_study(
        optuna_mod,
        "stage2_crf_shadow_broad",
        seed=int(getattr(cfg, "BO_SEED", 42) or 42) + 101,
    )
    stage2_broad.optimize(
        lambda tr: _objective_with_refine(tr, refine=None),
        n_trials=broad_trials,
        timeout=getattr(cfg, "BO_TIMEOUT_S", None),
        callbacks=build_optuna_callbacks_with_feedback(
            cfg,
            stage_name="stage2_broad",
            default_patience=20,
        ),
    )

    completed = [t for t in stage2_broad.trials if t.value is not None]
    completed.sort(key=lambda t: float(t.value), reverse=True)
    if not completed:
        raise ValueError("stage2 broad optimization produced no completed trials")
    top_trials = completed[: max(1, min(top_n, len(completed)))]

    refine_spec: dict[str, tuple | list] = {}
    if top_trials and refine_trials > 0:
        float_keys = [
            "prob_softness",
            "pos_w",
            "pos_xy_std",
            "bilateral_w",
            "bilateral_xy_std",
            "bilateral_rgb_std",
            "shadow_protect_score",
        ]
        for key in float_keys:
            vals = [float(t.user_attrs[key]) for t in top_trials if key in t.user_attrs]
            if vals:
                lo = min(vals)
                hi = max(vals)
                if abs(hi - lo) < 1e-12:
                    hi = lo + max(0.01 * max(1.0, abs(lo)), 1e-4)
                refine_spec[key] = (lo, hi)
        int_keys = ["shadow_threshold"]
        for key in int_keys:
            vals = [int(t.user_attrs[key]) for t in top_trials if key in t.user_attrs]
            if vals:
                refine_spec[key] = sorted(set(vals))
        w_vals = [
            int(t.user_attrs["shadow_weight_idx"])
            for t in top_trials
            if "shadow_weight_idx" in t.user_attrs
        ]
        if w_vals:
            refine_spec["shadow_weight_idx"] = sorted(set(w_vals))

    if refine_trials > 0:
        stage2_refine = make_optuna_study(
            optuna_mod,
            "stage2_crf_shadow_refine",
            seed=int(getattr(cfg, "BO_SEED", 42) or 42) + 151,
        )
        enqueued = 0
        for t in top_trials:
            enqueue = {}
            for key in [
                "prob_softness",
                "pos_w",
                "pos_xy_std",
                "bilateral_w",
                "bilateral_xy_std",
                "bilateral_rgb_std",
                "shadow_weight_idx",
                "shadow_threshold",
                "shadow_protect_score",
            ]:
                if key in t.user_attrs:
                    enqueue[key] = t.user_attrs[key]
            if enqueue:
                stage2_refine.enqueue_trial(enqueue)
                enqueued += 1
        stage2_refine.optimize(
            lambda tr: _objective_with_refine(tr, refine=refine_spec),
            n_trials=refine_trials,
            timeout=getattr(cfg, "BO_TIMEOUT_S", None),
            callbacks=build_optuna_callbacks_with_feedback(
                cfg,
                stage_name="stage2_refine",
                default_patience=20,
            ),
        )
        final_study = stage2_refine
    else:
        final_study = stage2_broad
        enqueued = 0

    best = final_study.best_trial
    best_crf_cfg = {
        "prob_softness": float(best.user_attrs.get("prob_softness", 0.1)),
        "pos_w": float(best.user_attrs.get("pos_w", 1.0)),
        "pos_xy_std": float(best.user_attrs.get("pos_xy_std", 3.0)),
        "bilateral_w": float(best.user_attrs.get("bilateral_w", 3.0)),
        "bilateral_xy_std": float(best.user_attrs.get("bilateral_xy_std", 25.0)),
        "bilateral_rgb_std": float(best.user_attrs.get("bilateral_rgb_std", 3.0)),
    }
    best_shadow_cfg = {
        "weights": tuple(
            weights_sets[int(best.user_attrs.get("shadow_weight_idx", 0))]
        ),
        "threshold": int(best.user_attrs.get("shadow_threshold", shadow_thr_vals[0])),
        "protect_score": float(
            best.user_attrs.get("shadow_protect_score", protect_vals[0])
        ),
        "iou": float(best.user_attrs.get("weighted_iou_gt", best.value)),
    }
    stage2_info = {
        "optuna_importances": _extract_study_importances(optuna_mod, final_study),
        "optuna_best_value": float(best.value),
        "refinement": {
            "enabled": bool(refine_trials > 0),
            "top_n": int(top_n),
            "broad_trials": int(broad_trials),
            "refine_trials": int(refine_trials),
            "enqueued_trials": int(enqueued),
            "narrowed": refine_spec,
        },
    }
    return best_crf_cfg, best_shadow_cfg, stage2_info


def run_stage3_bayes(
    *,
    optuna_mod,
    val_contexts: list[dict],
    stage1_bundle: dict,
    best_crf_cfg: dict,
    best_shadow_cfg: dict,
    model,
    processor,
    device,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
) -> dict:
    if not bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False)) or not bool(
        getattr(cfg, "BO_TUNE_BRIDGE", True)
    ):
        return {
            "bridge_max_gap_px": int(getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)),
            "bridge_max_pairs": int(getattr(cfg, "BRIDGE_MAX_PAIRS", 3)),
            "bridge_max_avg_cost": float(getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)),
            "bridge_width_px": int(getattr(cfg, "BRIDGE_WIDTH_PX", 2)),
            "bridge_min_component_px": int(
                getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300)
            ),
            "bridge_spur_prune_iters": int(getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15)),
            "iou": 0.0,
            "optuna_importances": {},
            "optuna_best_value": 0.0,
        }

    w_gt, w_sh = objective_weights(cfg)
    champion_source = str(stage1_bundle["champion_source"])
    k = int(stage1_bundle["best_raw_config"]["k"])
    bst = stage1_bundle["best_bst"]
    roads_penalty = float(stage1_bundle["roads_penalty"])
    top_p_a = float(stage1_bundle["top_p_a"])
    top_p_b = float(stage1_bundle["top_p_b"])
    top_p_min = float(stage1_bundle["top_p_min"])
    top_p_max = float(stage1_bundle["top_p_max"])

    neg_alpha = float(stage1_bundle.get("neg_alpha", getattr(cfg, "NEG_ALPHA", 1.0)))

    cached_entries: list[dict] = []
    for ctx in val_contexts:
        score_knn, score_xgb = score_maps_for_image(
            img_b=ctx["img_b"],
            ctx=ctx,
            k=k,
            bst=bst,
            roads_penalty=roads_penalty,
            model=model,
            processor=processor,
            device=device,
            ps=ps,
            tile_size=tile_size,
            stride=stride,
            feature_dir=feature_dir,
            context_radius=context_radius,
            pos_bank=pos_bank,
            neg_bank=neg_bank,
            neg_alpha=neg_alpha,
            use_prefetched=True,
        )
        thr = threshold_knn_xgb(
            score_knn,
            score_xgb,
            buffer_density=float(ctx["buffer_density"]),
            sh_buffer_mask=ctx["sh_buffer_mask"],
            top_p_a=top_p_a,
            top_p_b=top_p_b,
            top_p_min=top_p_min,
            top_p_max=top_p_max,
        )
        champion_score = score_knn if champion_source == "raw" else score_xgb
        champion_thr = thr["knn_thr"] if champion_source == "raw" else thr["xgb_thr"]
        champion_crf, champion_prob = refine_with_densecrf(
            ctx["img_b"],
            champion_score,
            champion_thr,
            ctx["sh_buffer_mask"],
            prob_softness=best_crf_cfg["prob_softness"],
            n_iters=5,
            pos_w=best_crf_cfg["pos_w"],
            pos_xy_std=best_crf_cfg["pos_xy_std"],
            bilateral_w=best_crf_cfg["bilateral_w"],
            bilateral_xy_std=best_crf_cfg["bilateral_xy_std"],
            bilateral_rgb_std=best_crf_cfg["bilateral_rgb_std"],
            return_prob=True,
        )
        perturbed = []
        for p_img in ctx.get("perturbed_imgs", []):
            s_knn_p, s_xgb_p = score_maps_for_image(
                img_b=p_img,
                ctx=ctx,
                k=k,
                bst=bst,
                roads_penalty=roads_penalty,
                model=model,
                processor=processor,
                device=device,
                ps=ps,
                tile_size=tile_size,
                stride=stride,
                feature_dir=feature_dir,
                context_radius=context_radius,
                pos_bank=pos_bank,
                neg_bank=neg_bank,
                neg_alpha=neg_alpha,
                use_prefetched=False,
            )
            thr_p = threshold_knn_xgb(
                s_knn_p,
                s_xgb_p,
                buffer_density=float(ctx["buffer_density"]),
                sh_buffer_mask=ctx["sh_buffer_mask"],
                top_p_a=top_p_a,
                top_p_b=top_p_b,
                top_p_min=top_p_min,
                top_p_max=top_p_max,
            )
            champion_score_p = s_knn_p if champion_source == "raw" else s_xgb_p
            champion_thr_p = (
                thr_p["knn_thr"] if champion_source == "raw" else thr_p["xgb_thr"]
            )
            champion_crf_p, champion_prob_p = refine_with_densecrf(
                p_img,
                champion_score_p,
                champion_thr_p,
                ctx["sh_buffer_mask"],
                prob_softness=best_crf_cfg["prob_softness"],
                n_iters=5,
                pos_w=best_crf_cfg["pos_w"],
                pos_xy_std=best_crf_cfg["pos_xy_std"],
                bilateral_w=best_crf_cfg["bilateral_w"],
                bilateral_xy_std=best_crf_cfg["bilateral_xy_std"],
                bilateral_rgb_std=best_crf_cfg["bilateral_rgb_std"],
                return_prob=True,
            )
            perturbed.append(
                {
                    "img": p_img,
                    "score": champion_score_p,
                    "crf": champion_crf_p,
                    "prob": champion_prob_p,
                }
            )
        cached_entries.append(
            {
                "ctx": ctx,
                "img": ctx["img_b"],
                "score": champion_score,
                "crf": champion_crf,
                "prob": champion_prob,
                "perturbed": perturbed,
            }
        )

    def objective(trial):
        bridge_cfg = {
            "bridge_max_gap_px": _suggest_int_param(
                trial,
                name="bridge_max_gap_px",
                range_key="BO_BRIDGE_MAX_GAP_PX_RANGE",
                values_key="BRIDGE_MAX_GAP_PX_VALUES",
                default_values=[getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)],
            ),
            "bridge_max_pairs": _suggest_int_param(
                trial,
                name="bridge_max_pairs",
                range_key="BO_BRIDGE_MAX_PAIRS_RANGE",
                values_key="BRIDGE_MAX_PAIRS_VALUES",
                default_values=[getattr(cfg, "BRIDGE_MAX_PAIRS", 3)],
            ),
            "bridge_max_avg_cost": _suggest_float_param(
                trial,
                name="bridge_max_avg_cost",
                range_key="BO_BRIDGE_MAX_AVG_COST_RANGE",
                values_key="BRIDGE_MAX_AVG_COST_VALUES",
                default_values=[getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)],
            ),
            "bridge_width_px": _suggest_int_param(
                trial,
                name="bridge_width_px",
                range_key="BO_BRIDGE_WIDTH_PX_RANGE",
                values_key="BRIDGE_WIDTH_PX_VALUES",
                default_values=[getattr(cfg, "BRIDGE_WIDTH_PX", 2)],
            ),
            "bridge_min_component_px": _suggest_int_param(
                trial,
                name="bridge_min_component_px",
                range_key="BO_BRIDGE_MIN_COMPONENT_PX_RANGE",
                values_key="BRIDGE_MIN_COMPONENT_PX_VALUES",
                default_values=[getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300)],
            ),
            "bridge_spur_prune_iters": _suggest_int_param(
                trial,
                name="bridge_spur_prune_iters",
                range_key="BO_BRIDGE_SPUR_PRUNE_ITERS_RANGE",
                values_key="BRIDGE_SPUR_PRUNE_ITERS_VALUES",
                default_values=[getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15)],
            ),
        }
        vals = []
        gt_vals = []
        sh_vals = []
        weights = []
        for step, entry in enumerate(cached_entries):
            ctx = entry["ctx"]
            bridged = bridge_skeleton_gaps(
                entry["crf"],
                entry["prob"],
                max_gap_px=int(bridge_cfg["bridge_max_gap_px"]),
                max_pairs_per_endpoint=int(bridge_cfg["bridge_max_pairs"]),
                max_avg_cost=float(bridge_cfg["bridge_max_avg_cost"]),
                bridge_width_px=int(bridge_cfg["bridge_width_px"]),
                min_component_area_px=int(bridge_cfg["bridge_min_component_px"]),
                spur_prune_iters=int(bridge_cfg["bridge_spur_prune_iters"]),
            )
            shadow = _apply_shadow_filter(
                entry["img"],
                bridged,
                best_shadow_cfg["weights"],
                best_shadow_cfg["threshold"],
                entry["score"],
                best_shadow_cfg.get("protect_score"),
            )
            iou_gt = float(compute_metrics(shadow, ctx["gt_mask_eval"])["iou"])
            sh_ious = []
            for p_entry in entry["perturbed"]:
                bridged_p = bridge_skeleton_gaps(
                    p_entry["crf"],
                    p_entry["prob"],
                    max_gap_px=int(bridge_cfg["bridge_max_gap_px"]),
                    max_pairs_per_endpoint=int(bridge_cfg["bridge_max_pairs"]),
                    max_avg_cost=float(bridge_cfg["bridge_max_avg_cost"]),
                    bridge_width_px=int(bridge_cfg["bridge_width_px"]),
                    min_component_area_px=int(bridge_cfg["bridge_min_component_px"]),
                    spur_prune_iters=int(bridge_cfg["bridge_spur_prune_iters"]),
                )
                shadow_p = _apply_shadow_filter(
                    p_entry["img"],
                    bridged_p,
                    best_shadow_cfg["weights"],
                    best_shadow_cfg["threshold"],
                    p_entry["score"],
                    best_shadow_cfg.get("protect_score"),
                )
                sh_ious.append(
                    mask_iou(shadow_p, ctx["labels_sh"] > 0, ctx["sh_buffer_mask"])
                )
            iou_sh = (
                float(np.mean(sh_ious))
                if sh_ious
                else mask_iou(
                    shadow,
                    ctx["labels_sh"] > 0,
                    ctx["sh_buffer_mask"],
                )
            )
            vals.append(robust_objective(iou_gt, iou_sh, w_gt, w_sh))
            gt_vals.append(iou_gt)
            sh_vals.append(iou_sh)
            weights.append(float(ctx["gt_weight"]))
            trial.report(float(weighted_mean(vals, weights)), step=step)
            if bool(getattr(cfg, "BO_ENABLE_PRUNING", True)) and trial.should_prune():
                raise optuna_mod.TrialPruned()

        trial.set_user_attr("weighted_iou_gt", float(weighted_mean(gt_vals, weights)))
        trial.set_user_attr("weighted_iou_sh", float(weighted_mean(sh_vals, weights)))
        for key, value in bridge_cfg.items():
            trial.set_user_attr(key, value)
        return float(weighted_mean(vals, weights))

    study = make_optuna_study(
        optuna_mod,
        "stage3_bridge",
        seed=int(getattr(cfg, "BO_SEED", 42) or 42) + 202,
    )
    study.optimize(
        objective,
        n_trials=int(getattr(cfg, "BO_STAGE3_TRIALS", 30) or 30),
        timeout=getattr(cfg, "BO_TIMEOUT_S", None),
        callbacks=build_optuna_callbacks_with_feedback(
            cfg,
            stage_name="stage3_bridge",
            default_patience=20,
        ),
    )
    best = study.best_trial
    return {
        "bridge_max_gap_px": int(
            best.user_attrs.get(
                "bridge_max_gap_px", getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)
            )
        ),
        "bridge_max_pairs": int(
            best.user_attrs.get("bridge_max_pairs", getattr(cfg, "BRIDGE_MAX_PAIRS", 3))
        ),
        "bridge_max_avg_cost": float(
            best.user_attrs.get(
                "bridge_max_avg_cost", getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)
            )
        ),
        "bridge_width_px": int(
            best.user_attrs.get("bridge_width_px", getattr(cfg, "BRIDGE_WIDTH_PX", 2))
        ),
        "bridge_min_component_px": int(
            best.user_attrs.get(
                "bridge_min_component_px",
                getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300),
            )
        ),
        "bridge_spur_prune_iters": int(
            best.user_attrs.get(
                "bridge_spur_prune_iters",
                getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15),
            )
        ),
        "iou": float(best.user_attrs.get("weighted_iou_gt", best.value)),
        "optuna_importances": _extract_study_importances(optuna_mod, study),
        "optuna_best_value": float(best.value),
        "frozen_upstream_cache_entries": len(cached_entries),
    }
