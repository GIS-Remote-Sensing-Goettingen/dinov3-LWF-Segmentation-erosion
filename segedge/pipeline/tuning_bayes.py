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
from ..core.metrics_utils import compute_metrics
from ..core.summary_utils import weighted_mean
from ..core.xdboost import xgb_score_image_b
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from .inference_utils import (
    _apply_roads_penalty,
    _apply_shadow_filter,
    _compute_top_p,
    _top_p_threshold,
)

USE_FP16_KNN = getattr(cfg, "USE_FP16_KNN", True)


def get_optuna_module():
    """Return the optuna module when available, else None.

    Examples:
        >>> mod = get_optuna_module()
        >>> mod is None or hasattr(mod, "create_study")
        True
    """
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
        >>> p = np.array([[1, 0], [0, 1]], dtype=bool)
        >>> r = np.array([[1, 0], [1, 0]], dtype=bool)
        >>> round(mask_iou(p, r), 3)
        0.333
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
    """Combine GT and SH IoU into one robustness-aware objective.

    Examples:
        >>> robust_objective(0.5, 0.25, 0.8, 0.2)
        0.45
    """
    return float(w_gt * iou_gt + w_sh * iou_sh)


def light_deterministic_perturbations(
    img_rgb: np.ndarray,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    """Generate deterministic light perturbations for robustness scoring.

    Examples:
        >>> x = np.zeros((4, 4, 3), dtype=np.uint8)
        >>> len(light_deterministic_perturbations(x, count=2, seed=7))
        2
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
        >>> c = [{"img_b": np.zeros((2, 2, 3), dtype=np.uint8)}]
        >>> attach_perturbations_to_contexts(c, count=1, seed=1)
        >>> len(c[0]["perturbed_imgs"])
        1
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
    use_prefetched: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute roads-penalized kNN/XGB score maps for one image.

    Examples:
        >>> callable(score_maps_for_image)
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
        neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
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
        >>> s = np.zeros((2, 2), dtype=np.float32)
        >>> m = np.ones((2, 2), dtype=bool)
        >>> out = threshold_knn_xgb(
        ...     s, s, buffer_density=1.0, sh_buffer_mask=m,
        ...     top_p_a=0.0, top_p_b=0.05, top_p_min=0.02, top_p_max=0.08
        ... )
        >>> sorted(out.keys())
        ['knn_thr', 'mask_knn', 'mask_xgb', 'xgb_thr']
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
        >>> callable(make_optuna_study)
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
    study_name = f"{getattr(cfg, 'BO_STUDY_NAME', 'segedge_tuning')}_{stage_name}"
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
    idx = int(trial.suggest_int(f"{name}_idx", 0, len(values) - 1))
    return values[idx]


def _extract_study_importances(optuna_mod, study) -> dict[str, float]:
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
    """Write a JSON artifact with stage-wise hyperparameter importances.

    Examples:
        >>> callable(write_bo_importances_file)
        True
    """
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
    use_prefetched: bool,
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


def _objective_weights() -> tuple[float, float]:
    w_gt = float(getattr(cfg, "BO_OBJECTIVE_W_GT", 0.8))
    w_sh = float(getattr(cfg, "BO_OBJECTIVE_W_SH", 0.2))
    w_sum = w_gt + w_sh
    if w_sum <= 0:
        return 1.0, 0.0
    return w_gt / w_sum, w_sh / w_sum


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
    """Run Bayesian optimization for raw-stage parameters.

    Examples:
        >>> callable(run_stage1_bayes)
        True
    """
    w_gt, w_sh = _objective_weights()
    roads_penalties = [float(p) for p in getattr(cfg, "ROADS_PENALTY_VALUES", [1.0])]
    a_values = [float(v) for v in (getattr(cfg, "TOP_P_A_VALUES", None) or [getattr(cfg, "TOP_P_A", 0.0)])]
    b_values = [float(v) for v in (getattr(cfg, "TOP_P_B_VALUES", None) or [getattr(cfg, "TOP_P_B", 0.05)])]
    pmin_values = [float(v) for v in (getattr(cfg, "TOP_P_MIN_VALUES", None) or [getattr(cfg, "TOP_P_MIN", 0.02)])]
    pmax_values = [float(v) for v in (getattr(cfg, "TOP_P_MAX_VALUES", None) or [getattr(cfg, "TOP_P_MAX", 0.08)])]
    k_values = [int(k) for k in getattr(cfg, "K_VALUES", [200])]
    dilate_values = [
        int(v)
        for v in getattr(
            cfg,
            "SILVER_CORE_DILATE_PX_VALUES",
            [getattr(cfg, "SILVER_CORE_DILATE_PX", 1)],
        )
    ]

    def objective(trial):
        roads_penalty = float(_suggest_from_values(trial, "roads_penalty", roads_penalties))
        top_p_a = float(_suggest_from_values(trial, "top_p_a", a_values))
        top_p_b = float(_suggest_from_values(trial, "top_p_b", b_values))
        top_p_min = float(_suggest_from_values(trial, "top_p_min", pmin_values))
        top_p_max = float(_suggest_from_values(trial, "top_p_max", pmax_values))
        if top_p_min > top_p_max:
            raise optuna_mod.TrialPruned("invalid top-p bounds")
        k = int(_suggest_from_values(trial, "k", k_values))
        dilate_px = int(_suggest_from_values(trial, "silver_core_dilate_px", dilate_values))
        xgb_idx = int(trial.suggest_int("xgb_candidate_idx", 0, len(xgb_candidates) - 1))
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
                sh_ious.append(mask_iou(core_mask_p, ctx["labels_sh"] > 0, ctx["sh_buffer_mask"]))
            iou_sh = float(np.mean(sh_ious)) if sh_ious else mask_iou(
                core_mask,
                ctx["labels_sh"] > 0,
                ctx["sh_buffer_mask"],
            )
            core_vals.append(robust_objective(iou_gt, iou_sh, w_gt, w_sh))
            core_gt_vals.append(iou_gt)
            core_sh_vals.append(iou_sh)
            weights.append(float(ctx["gt_weight"]))
            trial.report(float(weighted_mean(core_vals, weights)), step=step)
            if bool(getattr(cfg, "BO_ENABLE_PRUNING", True)) and trial.should_prune():
                raise optuna_mod.TrialPruned()

        trial.set_user_attr("weighted_iou_gt_core", float(weighted_mean(core_gt_vals, weights)))
        trial.set_user_attr("weighted_iou_sh_core", float(weighted_mean(core_sh_vals, weights)))
        for key in knn_stats:
            trial.set_user_attr(f"knn_{key}", float(weighted_mean(knn_stats[key], weights)))
            trial.set_user_attr(f"xgb_{key}", float(weighted_mean(xgb_stats[key], weights)))
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
    )
    best = study.best_trial
    xgb_idx = int(best.params["xgb_candidate_idx"])
    best_bst = xgb_candidates[xgb_idx]["bst"]
    best_params = xgb_candidates[xgb_idx]["params"]

    best_raw_config = {
        "k": int(k_values[int(best.params["k_idx"])]),
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
    champion_source = "raw" if best_raw_config["iou"] >= best_xgb_config["iou"] else "xgb"
    return {
        "roads_penalty": float(roads_penalties[int(best.params["roads_penalty_idx"])]),
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "best_bst": best_bst,
        "champion_source": champion_source,
        "top_p_a": float(a_values[int(best.params["top_p_a_idx"])]),
        "top_p_b": float(b_values[int(best.params["top_p_b_idx"])]),
        "top_p_min": float(pmin_values[int(best.params["top_p_min_idx"])]),
        "top_p_max": float(pmax_values[int(best.params["top_p_max_idx"])]),
        "silver_core_iou": float(best.user_attrs.get("weighted_iou_gt_core", best.value)),
        "silver_core_iou_sh": float(best.user_attrs.get("weighted_iou_sh_core", 0.0)),
        "silver_core_dilate_px": int(
            dilate_values[int(best.params["silver_core_dilate_px_idx"])]
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
    """Run Bayesian optimization for CRF + shadow parameters.

    Examples:
        >>> callable(run_stage2_bayes)
        True
    """
    w_gt, w_sh = _objective_weights()
    champion_source = str(stage1_bundle["champion_source"])
    k = int(stage1_bundle["best_raw_config"]["k"])
    bst = stage1_bundle["best_bst"]
    roads_penalty = float(stage1_bundle["roads_penalty"])
    top_p_a = float(stage1_bundle["top_p_a"])
    top_p_b = float(stage1_bundle["top_p_b"])
    top_p_min = float(stage1_bundle["top_p_min"])
    top_p_max = float(stage1_bundle["top_p_max"])
    weights_sets = list(getattr(cfg, "SHADOW_WEIGHT_SETS", [(1.0, 1.0, 1.0)]))
    shadow_thr_vals = [int(v) for v in getattr(cfg, "SHADOW_THRESHOLDS", [40])]
    protect_vals = [float(v) for v in getattr(cfg, "SHADOW_PROTECT_SCORES", [0.5])]

    def objective(trial):
        crf_cfg = {
            "prob_softness": float(
                _suggest_from_values(trial, "prob_softness", cfg.PROB_SOFTNESS_VALUES)
            ),
            "pos_w": float(_suggest_from_values(trial, "pos_w", cfg.POS_W_VALUES)),
            "pos_xy_std": float(
                _suggest_from_values(trial, "pos_xy_std", cfg.POS_XY_STD_VALUES)
            ),
            "bilateral_w": float(
                _suggest_from_values(trial, "bilateral_w", cfg.BILATERAL_W_VALUES)
            ),
            "bilateral_xy_std": float(
                _suggest_from_values(
                    trial,
                    "bilateral_xy_std",
                    cfg.BILATERAL_XY_STD_VALUES,
                )
            ),
            "bilateral_rgb_std": float(
                _suggest_from_values(
                    trial,
                    "bilateral_rgb_std",
                    cfg.BILATERAL_RGB_STD_VALUES,
                )
            ),
        }
        w_idx = int(trial.suggest_int("shadow_weight_idx", 0, len(weights_sets) - 1))
        shadow_cfg = {
            "weights": tuple(weights_sets[w_idx]),
            "threshold": int(
                _suggest_from_values(trial, "shadow_threshold", shadow_thr_vals)
            ),
            "protect_score": float(
                _suggest_from_values(trial, "shadow_protect_score", protect_vals)
            ),
        }

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
                use_prefetched=True,
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
                    use_prefetched=False,
                )
                sh_ious.append(mask_iou(shadow_p, ctx["labels_sh"] > 0, ctx["sh_buffer_mask"]))
            iou_sh = float(np.mean(sh_ious)) if sh_ious else mask_iou(
                shadow,
                ctx["labels_sh"] > 0,
                ctx["sh_buffer_mask"],
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
        return float(weighted_mean(vals, weights))

    study = make_optuna_study(
        optuna_mod,
        "stage2_crf_shadow",
        seed=int(getattr(cfg, "BO_SEED", 42) or 42) + 101,
    )
    study.optimize(
        objective,
        n_trials=int(getattr(cfg, "BO_STAGE2_TRIALS", 40) or 40),
        timeout=getattr(cfg, "BO_TIMEOUT_S", None),
    )
    best = study.best_trial
    best_crf_cfg = {
        "prob_softness": float(
            cfg.PROB_SOFTNESS_VALUES[int(best.params["prob_softness_idx"])]
        ),
        "pos_w": float(cfg.POS_W_VALUES[int(best.params["pos_w_idx"])]),
        "pos_xy_std": float(cfg.POS_XY_STD_VALUES[int(best.params["pos_xy_std_idx"])]),
        "bilateral_w": float(
            cfg.BILATERAL_W_VALUES[int(best.params["bilateral_w_idx"])]
        ),
        "bilateral_xy_std": float(
            cfg.BILATERAL_XY_STD_VALUES[int(best.params["bilateral_xy_std_idx"])]
        ),
        "bilateral_rgb_std": float(
            cfg.BILATERAL_RGB_STD_VALUES[int(best.params["bilateral_rgb_std_idx"])]
        ),
    }
    best_shadow_cfg = {
        "weights": tuple(weights_sets[int(best.params["shadow_weight_idx"])]),
        "threshold": int(shadow_thr_vals[int(best.params["shadow_threshold_idx"])]),
        "protect_score": float(
            protect_vals[int(best.params["shadow_protect_score_idx"])]
        ),
        "iou": float(best.user_attrs.get("weighted_iou_gt", best.value)),
    }
    stage2_info = {
        "optuna_importances": _extract_study_importances(optuna_mod, study),
        "optuna_best_value": float(best.value),
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
    """Run Bayesian optimization for bridge/skeleton parameters.

    Examples:
        >>> callable(run_stage3_bayes)
        True
    """
    if not bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False)):
        return {
            "bridge_max_gap_px": int(getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)),
            "bridge_max_pairs": int(getattr(cfg, "BRIDGE_MAX_PAIRS", 3)),
            "bridge_max_avg_cost": float(getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)),
            "bridge_width_px": int(getattr(cfg, "BRIDGE_WIDTH_PX", 2)),
            "bridge_min_component_px": int(getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300)),
            "bridge_spur_prune_iters": int(getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15)),
            "iou": 0.0,
            "optuna_importances": {},
            "optuna_best_value": 0.0,
        }

    w_gt, w_sh = _objective_weights()
    champion_source = str(stage1_bundle["champion_source"])
    k = int(stage1_bundle["best_raw_config"]["k"])
    bst = stage1_bundle["best_bst"]
    roads_penalty = float(stage1_bundle["roads_penalty"])
    top_p_a = float(stage1_bundle["top_p_a"])
    top_p_b = float(stage1_bundle["top_p_b"])
    top_p_min = float(stage1_bundle["top_p_min"])
    top_p_max = float(stage1_bundle["top_p_max"])

    gap_vals = [int(v) for v in getattr(cfg, "BRIDGE_MAX_GAP_PX_VALUES", [getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)])]
    pairs_vals = [int(v) for v in getattr(cfg, "BRIDGE_MAX_PAIRS_VALUES", [getattr(cfg, "BRIDGE_MAX_PAIRS", 3)])]
    avg_vals = [float(v) for v in getattr(cfg, "BRIDGE_MAX_AVG_COST_VALUES", [getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)])]
    width_vals = [int(v) for v in getattr(cfg, "BRIDGE_WIDTH_PX_VALUES", [getattr(cfg, "BRIDGE_WIDTH_PX", 2)])]
    comp_vals = [int(v) for v in getattr(cfg, "BRIDGE_MIN_COMPONENT_PX_VALUES", [getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300)])]
    spur_vals = [int(v) for v in getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS_VALUES", [getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15)])]

    def objective(trial):
        bridge_cfg = {
            "bridge_max_gap_px": int(
                _suggest_from_values(trial, "bridge_max_gap_px", gap_vals)
            ),
            "bridge_max_pairs": int(
                _suggest_from_values(trial, "bridge_max_pairs", pairs_vals)
            ),
            "bridge_max_avg_cost": float(
                _suggest_from_values(trial, "bridge_max_avg_cost", avg_vals)
            ),
            "bridge_width_px": int(
                _suggest_from_values(trial, "bridge_width_px", width_vals)
            ),
            "bridge_min_component_px": int(
                _suggest_from_values(trial, "bridge_min_component_px", comp_vals)
            ),
            "bridge_spur_prune_iters": int(
                _suggest_from_values(trial, "bridge_spur_prune_iters", spur_vals)
            ),
        }
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
                crf_cfg=best_crf_cfg,
                shadow_cfg=best_shadow_cfg,
                bridge_cfg=bridge_cfg,
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
                use_prefetched=True,
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
                    crf_cfg=best_crf_cfg,
                    shadow_cfg=best_shadow_cfg,
                    bridge_cfg=bridge_cfg,
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
                    use_prefetched=False,
                )
                sh_ious.append(mask_iou(shadow_p, ctx["labels_sh"] > 0, ctx["sh_buffer_mask"]))
            iou_sh = float(np.mean(sh_ious)) if sh_ious else mask_iou(
                shadow,
                ctx["labels_sh"] > 0,
                ctx["sh_buffer_mask"],
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
    )
    best = study.best_trial
    return {
        "bridge_max_gap_px": int(
            gap_vals[int(best.params["bridge_max_gap_px_idx"])]
        ),
        "bridge_max_pairs": int(pairs_vals[int(best.params["bridge_max_pairs_idx"])]),
        "bridge_max_avg_cost": float(
            avg_vals[int(best.params["bridge_max_avg_cost_idx"])]
        ),
        "bridge_width_px": int(width_vals[int(best.params["bridge_width_px_idx"])]),
        "bridge_min_component_px": int(
            comp_vals[int(best.params["bridge_min_component_px_idx"])]
        ),
        "bridge_spur_prune_iters": int(
            spur_vals[int(best.params["bridge_spur_prune_iters_idx"])]
        ),
        "iou": float(best.user_attrs.get("weighted_iou_gt", best.value)),
        "optuna_importances": _extract_study_importances(optuna_mod, study),
        "optuna_best_value": float(best.value),
    }
