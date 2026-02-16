"""Run-time training configuration logging helpers.

Examples:
    >>> _fmt_values([1, 2, 3], max_items=2)
    '[1, 2, ...] (n=3)'
"""

from __future__ import annotations

import logging

import config as cfg

logger = logging.getLogger(__name__)


def _fmt_values(values, max_items: int = 6) -> str:
    """Return a compact preview for a list-like config value.

    Examples:
        >>> _fmt_values([1, 2, 3], max_items=2)
        '[1, 2, ...] (n=3)'
    """
    vals = list(values)
    if len(vals) <= max_items:
        return str(vals)
    preview = ", ".join(str(v) for v in vals[:max_items])
    return f"[{preview}, ...] (n={len(vals)})"


def log_training_ablation_summary(
    source_count: int,
    val_count: int,
    holdout_count: int,
    feature_cache_mode: str,
    source_prefetch_gt_only: bool,
    auto_split_mode_legacy: str,
) -> None:
    """Log active training/tuning settings in a compact, stable format.

    Examples:
        >>> log_training_ablation_summary(1, 1, 1, "memory", True, "legacy") is None
        True
    """
    tuning_mode = str(getattr(cfg, "TUNING_MODE", "grid")).strip().lower()
    logger.info(
        "train settings: split source=%s val=%s holdout=%s auto_split=%s mode=%s",
        source_count,
        val_count,
        holdout_count,
        bool(getattr(cfg, "AUTO_SPLIT_TILES", False)),
        str(getattr(cfg, "AUTO_SPLIT_MODE", auto_split_mode_legacy)),
    )
    logger.info(
        "train settings: features cache_mode=%s source_prefetch_gt_only=%s "
        "batch=%s resample=%s patch=%s tile=%s stride=%s context=%s "
        "timing_tile_logs=%s feature_dir=%s bank_cache_dir=%s",
        feature_cache_mode,
        source_prefetch_gt_only,
        int(getattr(cfg, "FEATURE_BATCH_SIZE", 1) or 1),
        int(getattr(cfg, "RESAMPLE_FACTOR", 1) or 1),
        int(getattr(cfg, "PATCH_SIZE", 16) or 16),
        int(getattr(cfg, "TILE_SIZE", 1024) or 1024),
        int(getattr(cfg, "STRIDE", 1024) or 1024),
        int(getattr(cfg, "FEAT_CONTEXT_RADIUS", 0) or 0),
        bool(getattr(cfg, "TIMING_TILE_LOGS", False)),
        str(getattr(cfg, "FEATURE_DIR", "")),
        str(getattr(cfg, "BANK_CACHE_DIR", "")),
    )
    logger.info(
        "train settings: postprocess bridge=%s shadow_weights=%s shadow_thresholds=%s roads_penalties=%s",
        bool(getattr(cfg, "ENABLE_GAP_BRIDGING", False)),
        _fmt_values(getattr(cfg, "SHADOW_WEIGHT_SETS", []), max_items=3),
        _fmt_values(getattr(cfg, "SHADOW_THRESHOLDS", []), max_items=6),
        _fmt_values(getattr(cfg, "ROADS_PENALTY_VALUES", [1.0]), max_items=6),
    )
    logger.info(
        "train settings: xgb_labels pos_frac=%s neg_frac_max=%.4f "
        "scale_pos_weight=%s sample_weights=%s class_weight_max=%.2f "
        "max_neg_bank=%s max_neg_per_tile=%s kfold=%s/%s",
        (
            getattr(cfg, "XGB_POS_FRAC_THRESH", None)
            if getattr(cfg, "XGB_POS_FRAC_THRESH", None) is not None
            else getattr(cfg, "POS_FRAC_THRESH", 0.1)
        ),
        float(getattr(cfg, "XGB_NEG_FRAC_MAX", 0.0) or 0.0),
        bool(getattr(cfg, "XGB_USE_SCALE_POS_WEIGHT", True)),
        bool(getattr(cfg, "XGB_USE_SAMPLE_WEIGHTS", True)),
        float(getattr(cfg, "XGB_CLASS_WEIGHT_MAX", 25.0) or 25.0),
        int(getattr(cfg, "MAX_NEG_BANK", 8000) or 8000),
        int(getattr(cfg, "XGB_MAX_NEG_PER_TILE", 0) or 0),
        bool(getattr(cfg, "XGB_USE_KFOLD", False)),
        int(getattr(cfg, "XGB_KFOLD_SPLITS", 3) or 3),
    )
    if tuning_mode == "bayes":
        logger.info(
            "train ablation (bayes): stage_trials=%s/%s/%s "
            "objective_w=(gt=%.3f, sh=%.3f) fresh_study=%s trial_separators=%s",
            int(getattr(cfg, "BO_STAGE1_TRIALS", 50) or 50),
            int(getattr(cfg, "BO_STAGE2_TRIALS", 40) or 40),
            int(getattr(cfg, "BO_STAGE3_TRIALS", 30) or 30),
            float(getattr(cfg, "BO_OBJECTIVE_W_GT", 0.8)),
            float(getattr(cfg, "BO_OBJECTIVE_W_SH", 0.2)),
            bool(getattr(cfg, "BO_FORCE_NEW_STUDY", True)),
            bool(getattr(cfg, "BO_VERBOSE_TRIAL_SEPARATORS", True)),
        )
        logger.info(
            "train ablation (bayes ranges): k=%s neg_alpha=%s roads=%s top_p_a=%s top_p_b=%s",
            getattr(cfg, "BO_K_RANGE", None),
            getattr(cfg, "BO_NEG_ALPHA_RANGE", None),
            getattr(cfg, "BO_ROADS_PENALTY_RANGE", None),
            getattr(cfg, "BO_TOP_P_A_RANGE", None),
            getattr(cfg, "BO_TOP_P_B_RANGE", None),
        )
    else:
        logger.info(
            "train ablation (grid): k=%s top_p_a=%s top_p_b=%s top_p_min=%s top_p_max=%s",
            _fmt_values(getattr(cfg, "K_VALUES", []), max_items=8),
            _fmt_values(
                getattr(cfg, "TOP_P_A_VALUES", [getattr(cfg, "TOP_P_A", 0.0)]),
                max_items=8,
            ),
            _fmt_values(
                getattr(cfg, "TOP_P_B_VALUES", [getattr(cfg, "TOP_P_B", 0.05)]),
                max_items=8,
            ),
            _fmt_values(
                getattr(cfg, "TOP_P_MIN_VALUES", [getattr(cfg, "TOP_P_MIN", 0.02)]),
                max_items=8,
            ),
            _fmt_values(
                getattr(cfg, "TOP_P_MAX_VALUES", [getattr(cfg, "TOP_P_MAX", 0.08)]),
                max_items=8,
            ),
        )
