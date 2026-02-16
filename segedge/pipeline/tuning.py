from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import median_filter
from skimage.morphology import binary_dilation, disk

import config as cfg

from ..core.continuity import bridge_skeleton_gaps
from ..core.crf_utils import refine_with_densecrf
from ..core.features import prefetch_features_single_scale_image
from ..core.knn import zero_shot_knn_single_scale_B_with_saliency
from ..core.metrics_utils import compute_metrics, compute_oracle_upper_bound
from ..core.optuna_csv import (
    collect_optuna_trials_from_storage,
    write_bayes_trial_phase_timing_csv,
    write_optuna_importance_csv,
    write_optuna_trials_csv,
)
from ..core.plotting import save_unified_plot
from ..core.summary_utils import weighted_mean
from ..core.xdboost import (
    hyperparam_search_xgb_iou,
    train_xgb_classifier,
    xgb_score_image_b,
)
from .inference_utils import (
    _apply_roads_penalty,
    _apply_shadow_filter,
    _compute_top_p,
    _get_roads_mask,
    _top_p_threshold,
    load_b_tile_context,
)
from .tuning_bayes import (
    get_optuna_module,
    run_stage1_bayes,
    run_stage2_bayes,
    run_stage3_bayes,
    write_bo_importances_file,
)
from .tuning_bayes_utils import attach_perturbations_to_contexts

logger = logging.getLogger(__name__)

USE_FP16_KNN = getattr(cfg, "USE_FP16_KNN", True)
CRF_MAX_CONFIGS = getattr(cfg, "CRF_MAX_CONFIGS", 64)

_CRF_PARALLEL_CONTEXTS: list[dict] | None = None


def _init_crf_parallel(contexts: list[dict]) -> None:
    global _CRF_PARALLEL_CONTEXTS
    _CRF_PARALLEL_CONTEXTS = contexts


def _eval_crf_config(cfg, n_iters: int = 5) -> tuple[float, tuple[float, ...]]:
    if _CRF_PARALLEL_CONTEXTS is None:
        raise RuntimeError("CRF contexts not initialized")
    prob_soft, pos_w, pos_xy, bi_w, bi_xy, bi_rgb = cfg
    ious = []
    weights = []
    for ctx in _CRF_PARALLEL_CONTEXTS:
        mask_crf_local = refine_with_densecrf(
            ctx["img_b"],
            ctx["score_full"],
            ctx["thr_center"],
            ctx["sh_buffer_mask"],
            prob_softness=prob_soft,
            n_iters=n_iters,
            pos_w=pos_w,
            pos_xy_std=pos_xy,
            bilateral_w=bi_w,
            bilateral_xy_std=bi_xy,
            bilateral_rgb_std=bi_rgb,
        )
        ious.append(compute_metrics(mask_crf_local, ctx["gt_mask_eval"])["iou"])
        weights.append(float(ctx["gt_weight"]))
    return weighted_mean(ious, weights), cfg


def _render_tuning_plots(
    val_contexts: list[dict],
    model,
    processor,
    device,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    best_raw_config: dict,
    best_bst,
    best_crf_cfg: dict,
    best_shadow_cfg: dict,
    champion_source: str,
    roads_penalty: float,
    top_p_a: float,
    top_p_b: float,
    top_p_min: float,
    top_p_max: float,
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
    best_bridge_cfg: dict | None = None,
) -> None:
    max_tiles = int(getattr(cfg, "TUNING_PLOT_MAX_TILES", 10) or 0)
    if max_tiles <= 0:
        return

    plot_dir = os.path.join(cfg.PLOT_DIR, "tuning")
    os.makedirs(plot_dir, exist_ok=True)
    logger.info(
        "tune: writing validation preview plots to %s (max=%s)",
        plot_dir,
        max_tiles,
    )
    bridge_enabled = bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False))
    protect_score = best_shadow_cfg.get("protect_score")

    for idx, ctx in enumerate(val_contexts[:max_tiles], start=1):
        image_id = ctx["image_id"]
        logger.info(
            "tune: preview plot %s/%s for %s",
            idx,
            max_tiles,
            ctx["path"],
        )
        p_val = _compute_top_p(
            ctx["buffer_density"], top_p_a, top_p_b, top_p_min, top_p_max
        )
        score_knn, _ = zero_shot_knn_single_scale_B_with_saliency(
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
            image_id=image_id,
            neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
            prefetched_tiles=ctx["prefetched_b"],
            use_fp16_matmul=USE_FP16_KNN,
            context_radius=context_radius,
        )
        score_knn = _apply_roads_penalty(score_knn, ctx["roads_mask"], roads_penalty)
        knn_thr, mask_knn = _top_p_threshold(score_knn, ctx["sh_buffer_mask"], p_val)
        mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0

        score_xgb = xgb_score_image_b(
            ctx["img_b"],
            best_bst,
            ps,
            tile_size,
            stride,
            feature_dir,
            image_id,
            prefetched_tiles=ctx["prefetched_b"],
            context_radius=context_radius,
        )
        score_xgb = _apply_roads_penalty(score_xgb, ctx["roads_mask"], roads_penalty)
        xgb_thr, mask_xgb = _top_p_threshold(score_xgb, ctx["sh_buffer_mask"], p_val)
        mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0

        mask_crf_knn, prob_crf_knn = refine_with_densecrf(
            ctx["img_b"],
            score_knn,
            knn_thr,
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
        mask_crf_xgb, prob_crf_xgb = refine_with_densecrf(
            ctx["img_b"],
            score_xgb,
            xgb_thr,
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
        mask_shadow_knn = _apply_shadow_filter(
            ctx["img_b"],
            mask_crf_knn,
            best_shadow_cfg["weights"],
            best_shadow_cfg["threshold"],
            score_knn,
            protect_score,
        )
        mask_shadow_xgb = _apply_shadow_filter(
            ctx["img_b"],
            mask_crf_xgb,
            best_shadow_cfg["weights"],
            best_shadow_cfg["threshold"],
            score_xgb,
            protect_score,
        )

        if champion_source == "raw":
            champion_raw = mask_knn
            champion_crf = mask_crf_knn
            champion_score = score_knn
            champion_prob = prob_crf_knn
        else:
            champion_raw = mask_xgb
            champion_crf = mask_crf_xgb
            champion_score = score_xgb
            champion_prob = prob_crf_xgb

        champion_bridge = champion_crf
        if bridge_enabled:
            bridge_cfg = best_bridge_cfg or {}
            champion_bridge = bridge_skeleton_gaps(
                champion_crf,
                champion_prob,
                max_gap_px=int(
                    bridge_cfg.get(
                        "bridge_max_gap_px", getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)
                    )
                ),
                max_pairs_per_endpoint=int(
                    bridge_cfg.get(
                        "bridge_max_pairs",
                        getattr(cfg, "BRIDGE_MAX_PAIRS", 3),
                    )
                ),
                max_avg_cost=float(
                    bridge_cfg.get(
                        "bridge_max_avg_cost",
                        getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0),
                    )
                ),
                bridge_width_px=int(
                    bridge_cfg.get(
                        "bridge_width_px", getattr(cfg, "BRIDGE_WIDTH_PX", 2)
                    )
                ),
                min_component_area_px=int(
                    bridge_cfg.get(
                        "bridge_min_component_px",
                        getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300),
                    )
                ),
                spur_prune_iters=int(
                    bridge_cfg.get(
                        "bridge_spur_prune_iters",
                        getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15),
                    )
                ),
            )
        champion_shadow = _apply_shadow_filter(
            ctx["img_b"],
            champion_bridge,
            best_shadow_cfg["weights"],
            best_shadow_cfg["threshold"],
            champion_score,
            protect_score,
        )

        masks_map = {
            "knn_raw": mask_knn,
            "knn_crf": mask_crf_knn,
            "knn_shadow": mask_shadow_knn,
            "xgb_raw": mask_xgb,
            "xgb_crf": mask_crf_xgb,
            "xgb_shadow": mask_shadow_xgb,
            "champion_raw": champion_raw,
            "champion_crf": champion_crf,
            "champion_shadow": champion_shadow,
        }
        if bridge_enabled:
            masks_map["champion_bridge"] = champion_bridge

        metrics_map = {
            key: compute_metrics(mask, ctx["gt_mask_eval"])
            for key, mask in masks_map.items()
        }
        save_unified_plot(
            img_b=ctx["img_b"],
            gt_mask=ctx["gt_mask_eval"],
            labels_sh=ctx["labels_sh"],
            masks=masks_map,
            metrics=metrics_map,
            plot_dir=plot_dir,
            image_id_b=f"{image_id}_tuning",
            show_metrics=True,
            gt_available=True,
            similarity_map=None,
            score_maps={"knn": score_knn, "xgb": score_xgb},
            skeleton=None,
            endpoints=None,
            bridge_enabled=bridge_enabled,
        )


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
    ps: int,
    tile_size: int,
    stride: int,
    feature_dir: str | None,
    context_radius: int,
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
        ps (int): Patch size.
        tile_size (int): Tile size in pixels.
        stride (int): Tile stride.
        feature_dir (str | None): Feature cache directory.
        context_radius (int): Feature context radius.

    Returns:
        dict: Tuned configurations and models.

    Examples:
        >>> isinstance(tune_on_validation_multi.__name__, str)
        True
    """
    if not val_paths:
        raise ValueError("VAL_TILES is empty.")

    val_contexts = []
    ds = int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1)
    for val_path in val_paths:
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
        buffer_density = float(sh_buffer_mask.mean())
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
                "buffer_density": buffer_density,
                "roads_mask": roads_mask,
                "prefetched_b": prefetched_b,
                "buffer_m": buffer_m,
                "pixel_size_m": pixel_size_m,
            }
        )

    # XGB training (shared across road penalties)
    use_gpu_xgb = getattr(cfg, "XGB_USE_GPU", True)
    param_grid = getattr(cfg, "XGB_PARAM_GRID", None)
    num_boost_round = getattr(cfg, "XGB_NUM_BOOST_ROUND", 300)
    early_stop = getattr(cfg, "XGB_EARLY_STOP", 40)
    verbose_eval = getattr(cfg, "XGB_VERBOSE_EVAL", 50)
    val_fraction = getattr(cfg, "XGB_VAL_FRACTION", 0.2)
    use_kfold_xgb = bool(getattr(cfg, "XGB_USE_KFOLD", False))
    kfold_splits_xgb = int(getattr(cfg, "XGB_KFOLD_SPLITS", 3) or 3)
    if param_grid is None:
        param_grid = [None]

    xgb_candidates = []
    for overrides in param_grid:
        if overrides is None and not use_kfold_xgb:
            bst = train_xgb_classifier(
                X,
                y,
                use_gpu=use_gpu_xgb,
                num_boost_round=num_boost_round,
                verbose_eval=verbose_eval,
            )
            params_used = None
        else:
            search_grid = [overrides] if overrides is not None else [{}]
            bst, params_used, _, _, _ = hyperparam_search_xgb_iou(
                X,
                y,
                [0.5],
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
                param_grid=search_grid,
                num_boost_round=num_boost_round,
                val_fraction=val_fraction,
                early_stopping_rounds=early_stop,
                verbose_eval=verbose_eval,
                seed=42,
                context_radius=context_radius,
                use_kfold=use_kfold_xgb,
                kfold_splits=kfold_splits_xgb,
            )
        xgb_candidates.append({"bst": bst, "params": params_used})

    tuning_mode = str(getattr(cfg, "TUNING_MODE", "grid")).strip().lower()
    if tuning_mode == "bayes":
        optuna_mod = get_optuna_module()
        if optuna_mod is None:
            logger.warning(
                "tune: TUNING_MODE='bayes' but optuna is unavailable; falling back to grid"
            )
        else:
            perturb_count = int(getattr(cfg, "BO_PERTURBATIONS_PER_TILE", 0) or 0)
            perturb_seed = int(getattr(cfg, "BO_PERTURB_SEED", 42) or 42)
            if perturb_count > 0:
                attach_perturbations_to_contexts(
                    val_contexts,
                    count=perturb_count,
                    seed=perturb_seed,
                )
            logger.info(
                "tune: Bayesian optimization enabled (stage trials: %s/%s/%s, perturbations=%s)",
                int(getattr(cfg, "BO_STAGE1_TRIALS", 50) or 50),
                int(getattr(cfg, "BO_STAGE2_TRIALS", 40) or 40),
                int(getattr(cfg, "BO_STAGE3_TRIALS", 30) or 30),
                perturb_count,
            )
            stage1_bundle = run_stage1_bayes(
                optuna_mod=optuna_mod,
                val_contexts=val_contexts,
                xgb_candidates=xgb_candidates,
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
            )
            best_crf_cfg, best_shadow_cfg, stage2_info = run_stage2_bayes(
                optuna_mod=optuna_mod,
                val_contexts=val_contexts,
                stage1_bundle=stage1_bundle,
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
            )
            best_bridge_cfg = run_stage3_bayes(
                optuna_mod=optuna_mod,
                val_contexts=val_contexts,
                stage1_bundle=stage1_bundle,
                best_crf_cfg=best_crf_cfg,
                best_shadow_cfg=best_shadow_cfg,
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
            )
            bo_importances_path = os.path.join(
                os.path.dirname(cfg.PLOT_DIR),
                str(
                    getattr(
                        cfg,
                        "BO_IMPORTANCE_FILENAME",
                        "bayes_hyperparam_importances.json",
                    )
                ),
            )
            bo_payload = {
                "mode": "bayes",
                "sampler": str(getattr(cfg, "BO_SAMPLER", "tpe")),
                "objective_weights": {
                    "w_gt": float(getattr(cfg, "BO_OBJECTIVE_W_GT", 0.8)),
                    "w_sh": float(getattr(cfg, "BO_OBJECTIVE_W_SH", 0.2)),
                },
                "stage1": {
                    "best_value": float(stage1_bundle.get("optuna_best_value", 0.0)),
                    "importances": stage1_bundle.get("optuna_importances", {}),
                    "best_params": {
                        "k": int(stage1_bundle["best_raw_config"]["k"]),
                        "neg_alpha": float(
                            stage1_bundle.get(
                                "neg_alpha",
                                getattr(cfg, "NEG_ALPHA", 1.0),
                            )
                        ),
                        "roads_penalty": float(stage1_bundle["roads_penalty"]),
                        "top_p_a": float(stage1_bundle["top_p_a"]),
                        "top_p_b": float(stage1_bundle["top_p_b"]),
                        "top_p_min": float(stage1_bundle["top_p_min"]),
                        "top_p_max": float(stage1_bundle["top_p_max"]),
                    },
                },
                "stage2": {
                    "best_value": float(stage2_info.get("optuna_best_value", 0.0)),
                    "importances": stage2_info.get("optuna_importances", {}),
                    "refinement": stage2_info.get("refinement", {}),
                },
                "stage3": {
                    "best_value": float(best_bridge_cfg.get("optuna_best_value", 0.0)),
                    "importances": best_bridge_cfg.get("optuna_importances", {}),
                    "frozen_upstream_cache_entries": int(
                        best_bridge_cfg.get("frozen_upstream_cache_entries", 0)
                    ),
                },
            }
            write_bo_importances_file(bo_importances_path, bo_payload)
            logger.info(
                "tune: wrote hyperparameter importances: %s", bo_importances_path
            )
            run_root = os.path.dirname(cfg.PLOT_DIR)
            bo_importances_csv_path = os.path.join(
                run_root,
                str(
                    getattr(
                        cfg,
                        "BO_IMPORTANCE_CSV_FILENAME",
                        "bayes_hyperparam_importances.csv",
                    )
                ),
            )
            write_optuna_importance_csv(bo_importances_csv_path, bo_payload)
            logger.info(
                "tune: wrote hyperparameter importances csv: %s",
                bo_importances_csv_path,
            )
            storage_path = str(getattr(cfg, "BO_STORAGE_PATH", "") or "").strip()
            if storage_path:
                study_tag = str(getattr(cfg, "BO_STUDY_TAG", "") or "").strip()
                base_study_name = str(getattr(cfg, "BO_STUDY_NAME", "segedge_tuning"))

                def _study_name(suffix: str) -> str:
                    stem = f"{base_study_name}_{suffix}"
                    return f"{stem}_{study_tag}" if study_tag else stem

                refinement = (
                    stage2_info.get("refinement", {})
                    if isinstance(stage2_info.get("refinement"), dict)
                    else {}
                )
                trial_specs = [
                    {
                        "study_name": _study_name("stage1_raw"),
                        "stage": "stage1_raw",
                        "max_recent_trials": int(
                            getattr(cfg, "BO_STAGE1_TRIALS", 50) or 50
                        ),
                    },
                    {
                        "study_name": _study_name("stage2_crf_shadow_broad"),
                        "stage": "stage2_crf_shadow_broad",
                        "max_recent_trials": int(
                            refinement.get("broad_trials", 0) or 0
                        ),
                    },
                    {
                        "study_name": _study_name("stage2_crf_shadow_refine"),
                        "stage": "stage2_crf_shadow_refine",
                        "max_recent_trials": int(
                            refinement.get("refine_trials", 0) or 0
                        ),
                    },
                    {
                        "study_name": _study_name("stage3_bridge"),
                        "stage": "stage3_bridge",
                        "max_recent_trials": int(
                            getattr(cfg, "BO_STAGE3_TRIALS", 30) or 30
                        ),
                    },
                ]
                bo_trials_csv_path = os.path.join(
                    run_root,
                    str(
                        getattr(
                            cfg,
                            "BO_TRIALS_CSV_FILENAME",
                            "bayes_trials_timeseries.csv",
                        )
                    ),
                )
                trial_rows = collect_optuna_trials_from_storage(
                    optuna_mod=optuna_mod,
                    storage_path=storage_path,
                    study_specs=trial_specs,
                )
                write_optuna_trials_csv(bo_trials_csv_path, trial_rows)
                logger.info(
                    "tune: wrote bayes trial time-series csv: %s", bo_trials_csv_path
                )
                bo_phase_timing_csv_path = os.path.join(
                    run_root,
                    str(
                        getattr(
                            cfg,
                            "BO_TRIAL_PHASE_TIMING_CSV_FILENAME",
                            "bayes_trial_phase_timing.csv",
                        )
                    ),
                )
                write_bayes_trial_phase_timing_csv(bo_phase_timing_csv_path, trial_rows)
                logger.info(
                    "tune: wrote bayes trial phase timing csv: %s",
                    bo_phase_timing_csv_path,
                )
            logger.info(
                "tune: bayes selected champion=%s roads_penalty=%.3f top-p=(%.3f, %.3f, %.3f, %.3f)",
                stage1_bundle["champion_source"],
                float(stage1_bundle["roads_penalty"]),
                float(stage1_bundle["top_p_a"]),
                float(stage1_bundle["top_p_b"]),
                float(stage1_bundle["top_p_min"]),
                float(stage1_bundle["top_p_max"]),
            )
            _render_tuning_plots(
                val_contexts,
                model,
                processor,
                device,
                pos_bank,
                neg_bank,
                stage1_bundle["best_raw_config"],
                stage1_bundle["best_bst"],
                best_crf_cfg,
                best_shadow_cfg,
                stage1_bundle["champion_source"],
                float(stage1_bundle["roads_penalty"]),
                float(stage1_bundle["top_p_a"]),
                float(stage1_bundle["top_p_b"]),
                float(stage1_bundle["top_p_min"]),
                float(stage1_bundle["top_p_max"]),
                ps,
                tile_size,
                stride,
                feature_dir,
                context_radius,
                best_bridge_cfg=best_bridge_cfg,
            )
            return {
                "bst": stage1_bundle["best_bst"],
                "best_raw_config": stage1_bundle["best_raw_config"],
                "best_xgb_config": stage1_bundle["best_xgb_config"],
                "champion_source": stage1_bundle["champion_source"],
                "neg_alpha": float(
                    stage1_bundle.get("neg_alpha", getattr(cfg, "NEG_ALPHA", 1.0))
                ),
                "best_crf_config": {
                    **best_crf_cfg,
                    "k": stage1_bundle["best_raw_config"]["k"],
                },
                "shadow_cfg": best_shadow_cfg,
                "roads_penalty": float(stage1_bundle["roads_penalty"]),
                "top_p_a": float(stage1_bundle["top_p_a"]),
                "top_p_b": float(stage1_bundle["top_p_b"]),
                "top_p_min": float(stage1_bundle["top_p_min"]),
                "top_p_max": float(stage1_bundle["top_p_max"]),
                "silver_core_iou": float(stage1_bundle["silver_core_iou"]),
                "silver_core_dilate_px": int(stage1_bundle["silver_core_dilate_px"]),
                "best_bridge_config": {
                    key: best_bridge_cfg[key]
                    for key in [
                        "bridge_max_gap_px",
                        "bridge_max_pairs",
                        "bridge_max_avg_cost",
                        "bridge_width_px",
                        "bridge_min_component_px",
                        "bridge_spur_prune_iters",
                        "iou",
                    ]
                    if key in best_bridge_cfg
                },
            }

    roads_penalties = [float(p) for p in getattr(cfg, "ROADS_PENALTY_VALUES", [1.0])]
    a_values = getattr(cfg, "TOP_P_A_VALUES", None) or [getattr(cfg, "TOP_P_A", 0.0)]
    b_values = getattr(cfg, "TOP_P_B_VALUES", None) or [getattr(cfg, "TOP_P_B", 0.05)]
    pmin_values = getattr(cfg, "TOP_P_MIN_VALUES", None) or [
        getattr(cfg, "TOP_P_MIN", 0.02)
    ]
    pmax_values = getattr(cfg, "TOP_P_MAX_VALUES", None) or [
        getattr(cfg, "TOP_P_MAX", 0.08)
    ]
    top_p_candidates = [
        (float(a), float(b), float(pmin), float(pmax))
        for a in a_values
        for b in b_values
        for pmin in pmin_values
        for pmax in pmax_values
        if pmin <= pmax
    ]
    if not top_p_candidates:
        raise ValueError("top-p candidate list is empty")

    best_bundle = None
    best_core_iou = None

    for penalty in roads_penalties:
        logger.info("tune: roads penalty=%s", penalty)
        for a, b, p_min, p_max in top_p_candidates:
            logger.info(
                "tune: top-p params a=%.3f b=%.3f p_min=%.3f p_max=%.3f",
                a,
                b,
                p_min,
                p_max,
            )

            # kNN tuning (weighted-mean IoU across val tiles)
            best_raw_config = None
            for k in cfg.K_VALUES:
                ious = []
                weights = []
                f1s = []
                precs = []
                recalls = []
                for ctx in val_contexts:
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
                        neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
                        prefetched_tiles=ctx["prefetched_b"],
                        use_fp16_matmul=USE_FP16_KNN,
                        context_radius=context_radius,
                    )
                    score_full = _apply_roads_penalty(
                        score_full, ctx["roads_mask"], penalty
                    )
                    p_val = _compute_top_p(ctx["buffer_density"], a, b, p_min, p_max)
                    _, mask = _top_p_threshold(score_full, ctx["sh_buffer_mask"], p_val)
                    mask = median_filter(mask.astype(np.uint8), size=3) > 0
                    metrics = compute_metrics(mask, ctx["gt_mask_eval"])
                    ious.append(metrics["iou"])
                    f1s.append(metrics["f1"])
                    precs.append(metrics["precision"])
                    recalls.append(metrics["recall"])
                    weights.append(float(ctx["gt_weight"]))

                weighted_iou = weighted_mean(ious, weights)
                weighted_f1 = weighted_mean(f1s, weights)
                weighted_precision = weighted_mean(precs, weights)
                weighted_recall = weighted_mean(recalls, weights)
                if best_raw_config is None or weighted_iou > best_raw_config["iou"]:
                    best_raw_config = {
                        "k": k,
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
            for candidate in xgb_candidates:
                bst = candidate["bst"]
                params_used = candidate["params"]
                ious = []
                weights = []
                f1s = []
                precs = []
                recalls = []
                for ctx in val_contexts:
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
                    )
                    score_full = _apply_roads_penalty(
                        score_full, ctx["roads_mask"], penalty
                    )
                    p_val = _compute_top_p(ctx["buffer_density"], a, b, p_min, p_max)
                    _, mask = _top_p_threshold(score_full, ctx["sh_buffer_mask"], p_val)
                    mask = median_filter(mask.astype(np.uint8), size=3) > 0
                    metrics = compute_metrics(mask, ctx["gt_mask_eval"])
                    ious.append(metrics["iou"])
                    f1s.append(metrics["f1"])
                    precs.append(metrics["precision"])
                    recalls.append(metrics["recall"])
                    weights.append(float(ctx["gt_weight"]))

                weighted_iou = weighted_mean(ious, weights)
                weighted_f1 = weighted_mean(f1s, weights)
                weighted_precision = weighted_mean(precs, weights)
                weighted_recall = weighted_mean(recalls, weights)
                cand = {
                    "k": -1,
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

            # Evaluate silver core (kNN ∩ XGB)
            core_ious = []
            core_weights = []
            dilate_px = int(getattr(cfg, "SILVER_CORE_DILATE_PX", 1))
            for ctx in val_contexts:
                p_val = _compute_top_p(ctx["buffer_density"], a, b, p_min, p_max)
                score_knn, _ = zero_shot_knn_single_scale_B_with_saliency(
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
                    neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
                    prefetched_tiles=ctx["prefetched_b"],
                    use_fp16_matmul=USE_FP16_KNN,
                    context_radius=context_radius,
                )
                score_knn = _apply_roads_penalty(score_knn, ctx["roads_mask"], penalty)
                _, mask_knn = _top_p_threshold(score_knn, ctx["sh_buffer_mask"], p_val)
                mask_knn = median_filter(mask_knn.astype(np.uint8), size=3) > 0

                score_xgb = xgb_score_image_b(
                    ctx["img_b"],
                    best_bst,
                    ps,
                    tile_size,
                    stride,
                    feature_dir,
                    ctx["image_id"],
                    prefetched_tiles=ctx["prefetched_b"],
                    context_radius=context_radius,
                )
                score_xgb = _apply_roads_penalty(score_xgb, ctx["roads_mask"], penalty)
                _, mask_xgb = _top_p_threshold(score_xgb, ctx["sh_buffer_mask"], p_val)
                mask_xgb = median_filter(mask_xgb.astype(np.uint8), size=3) > 0

                core_mask = np.logical_and(mask_knn, mask_xgb)
                if dilate_px > 0:
                    core_mask = binary_dilation(core_mask, disk(dilate_px))
                metrics_core = compute_metrics(core_mask, ctx["gt_mask_eval"])
                core_ious.append(metrics_core["iou"])
                core_weights.append(float(ctx["gt_weight"]))

            core_iou = weighted_mean(core_ious, core_weights)
            if best_core_iou is None or core_iou > best_core_iou:
                best_core_iou = core_iou
                champion_source = (
                    "raw" if best_raw_config["iou"] >= best_xgb_config["iou"] else "xgb"
                )
                best_bundle = {
                    "roads_penalty": penalty,
                    "best_raw_config": best_raw_config,
                    "best_xgb_config": best_xgb_config,
                    "best_bst": best_bst,
                    "champion_source": champion_source,
                    "top_p_a": a,
                    "top_p_b": b,
                    "top_p_min": p_min,
                    "top_p_max": p_max,
                    "silver_core_iou": core_iou,
                }

    if best_bundle is None:
        raise ValueError("top-p tuning returned no results")

    roads_penalty = best_bundle["roads_penalty"]
    best_raw_config = best_bundle["best_raw_config"]
    best_xgb_config = best_bundle["best_xgb_config"]
    best_bst = best_bundle["best_bst"]
    champion_source = best_bundle["champion_source"]
    top_p_a = best_bundle["top_p_a"]
    top_p_b = best_bundle["top_p_b"]
    top_p_min = best_bundle["top_p_min"]
    top_p_max = best_bundle["top_p_max"]
    for ctx in val_contexts:
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
                neg_alpha=getattr(cfg, "NEG_ALPHA", 1.0),
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
            )
        score_full = _apply_roads_penalty(
            score_full, ctx["roads_mask"], float(roads_penalty)
        )
        p_val = _compute_top_p(
            ctx["buffer_density"], top_p_a, top_p_b, top_p_min, top_p_max
        )
        thr_center, _ = _top_p_threshold(score_full, ctx["sh_buffer_mask"], p_val)
        ctx["score_full"] = score_full
        ctx["thr_center"] = thr_center
        ctx["top_p"] = p_val

    # CRF tuning across val tiles
    crf_candidates = [
        (psf, pw, pxy, bw, bxy, brgb)
        for psf in cfg.PROB_SOFTNESS_VALUES
        for pw in cfg.POS_W_VALUES
        for pxy in cfg.POS_XY_STD_VALUES
        for bw in cfg.BILATERAL_W_VALUES
        for bxy in cfg.BILATERAL_XY_STD_VALUES
        for brgb in cfg.BILATERAL_RGB_STD_VALUES
    ]
    best_crf_cfg = None
    best_crf_iou = None
    crf_candidates = crf_candidates[:CRF_MAX_CONFIGS]
    num_workers = int(getattr(cfg, "CRF_NUM_WORKERS", 1) or 1)
    logger.info(
        "tune: CRF grid search configs=%s, workers=%s",
        len(crf_candidates),
        num_workers,
    )
    _init_crf_parallel(val_contexts)
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for med_iou, cand in ex.map(_eval_crf_config, crf_candidates):
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
    protect_scores = getattr(cfg, "SHADOW_PROTECT_SCORES", [0.5])
    for weights in cfg.SHADOW_WEIGHT_SETS:
        iou_by_key = {
            (thr, protect_score): {"sum": 0.0, "w": 0.0}
            for thr in cfg.SHADOW_THRESHOLDS
            for protect_score in protect_scores
        }
        for ctx in val_contexts:
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
            thr_arr = np.array(cfg.SHADOW_THRESHOLDS, dtype=np.float32).reshape(-1, 1)
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
                for i, thr in enumerate(cfg.SHADOW_THRESHOLDS):
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

    best_bridge_cfg = {
        "bridge_max_gap_px": int(getattr(cfg, "BRIDGE_MAX_GAP_PX", 25)),
        "bridge_max_pairs": int(getattr(cfg, "BRIDGE_MAX_PAIRS", 3)),
        "bridge_max_avg_cost": float(getattr(cfg, "BRIDGE_MAX_AVG_COST", 1.0)),
        "bridge_width_px": int(getattr(cfg, "BRIDGE_WIDTH_PX", 2)),
        "bridge_min_component_px": int(getattr(cfg, "BRIDGE_MIN_COMPONENT_PX", 300)),
        "bridge_spur_prune_iters": int(getattr(cfg, "BRIDGE_SPUR_PRUNE_ITERS", 15)),
    }
    logger.info("tune: roads penalty selected=%s", roads_penalty)
    _render_tuning_plots(
        val_contexts,
        model,
        processor,
        device,
        pos_bank,
        neg_bank,
        best_raw_config,
        best_bst,
        best_crf_cfg,
        best_shadow_cfg,
        champion_source,
        float(roads_penalty),
        float(top_p_a),
        float(top_p_b),
        float(top_p_min),
        float(top_p_max),
        ps,
        tile_size,
        stride,
        feature_dir,
        context_radius,
        best_bridge_cfg=best_bridge_cfg,
    )
    return {
        "bst": best_bst,
        "best_raw_config": best_raw_config,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "best_crf_config": {**best_crf_cfg, "k": best_raw_config["k"]},
        "shadow_cfg": best_shadow_cfg,
        "roads_penalty": float(roads_penalty),
        "top_p_a": top_p_a,
        "top_p_b": top_p_b,
        "top_p_min": top_p_min,
        "top_p_max": top_p_max,
        "silver_core_iou": best_bundle["silver_core_iou"],
        "silver_core_dilate_px": int(getattr(cfg, "SILVER_CORE_DILATE_PX", 1)),
        "best_bridge_config": best_bridge_cfg,
    }
