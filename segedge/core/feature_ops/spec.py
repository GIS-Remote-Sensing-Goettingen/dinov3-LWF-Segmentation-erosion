"""Hybrid feature specification helpers."""

from __future__ import annotations

import hashlib
import json

from ..config_loader import cfg


def hybrid_feature_spec_dict() -> dict:
    """Build a JSON-serializable hybrid feature specification.

    Returns:
        dict: Hybrid feature spec used for reproducibility/hash.

    Examples:
        >>> isinstance(hybrid_feature_spec_dict(), dict)
        True
    """
    hybrid = cfg.model.hybrid_features
    blocks = hybrid.blocks
    return {
        "enabled": bool(hybrid.enabled),
        "knn_l2_normalize": bool(hybrid.knn_l2_normalize),
        "xgb_zscore": bool(hybrid.xgb_zscore),
        "zscore_eps": float(hybrid.zscore_eps),
        "blocks": {
            "dino": {
                "enabled": bool(blocks.dino.enabled),
                "weight_knn": float(blocks.dino.weight_knn),
                "weight_xgb": float(blocks.dino.weight_xgb),
            },
            "rgb_stats": {
                "enabled": bool(blocks.rgb_stats.enabled),
                "weight_knn": float(blocks.rgb_stats.weight_knn),
                "weight_xgb": float(blocks.rgb_stats.weight_xgb),
            },
            "hsv_mean": {
                "enabled": bool(blocks.hsv_mean.enabled),
                "weight_knn": float(blocks.hsv_mean.weight_knn),
                "weight_xgb": float(blocks.hsv_mean.weight_xgb),
            },
            "grad_stats": {
                "enabled": bool(blocks.grad_stats.enabled),
                "weight_knn": float(blocks.grad_stats.weight_knn),
                "weight_xgb": float(blocks.grad_stats.weight_xgb),
            },
            "grad_orient_hist": {
                "enabled": bool(blocks.grad_orient_hist.enabled),
                "bins": int(blocks.grad_orient_hist.bins or 8),
                "weight_knn": float(blocks.grad_orient_hist.weight_knn),
                "weight_xgb": float(blocks.grad_orient_hist.weight_xgb),
            },
            "lbp_hist": {
                "enabled": bool(blocks.lbp_hist.enabled),
                "bins": int(blocks.lbp_hist.bins or 16),
                "weight_knn": float(blocks.lbp_hist.weight_knn),
                "weight_xgb": float(blocks.lbp_hist.weight_xgb),
            },
        },
    }


def hybrid_feature_spec_hash() -> str:
    """Return a stable short hash of the hybrid feature specification.

    Returns:
        str: Short SHA1 hash.

    Examples:
        >>> len(hybrid_feature_spec_hash()) >= 8
        True
    """
    payload = json.dumps(
        hybrid_feature_spec_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
