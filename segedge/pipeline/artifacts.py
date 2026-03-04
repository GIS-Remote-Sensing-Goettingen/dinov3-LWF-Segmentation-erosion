"""Model-bundle persistence helpers for train-once/infer-many workflows."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xgboost as xgb
import yaml

from ..core.features import hybrid_feature_spec_hash

_BUNDLE_VERSION = 1


def _normalize(value: Any):
    """Normalize nested values for safe YAML serialization.

    Examples:
        >>> _normalize({"x": (1, 2)}) == {"x": [1, 2]}
        True
    """
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_model_bundle(
    bundle_dir: str,
    tuned: dict,
    pos_bank: np.ndarray,
    neg_bank: np.ndarray | None,
    *,
    model_name: str,
    patch_size: int,
    resample_factor: int,
    tile_size: int,
    stride: int,
    context_radius: int,
) -> dict[str, Any]:
    """Persist trained artifacts into a single bundle directory.

    Args:
        bundle_dir (str): Output directory for the bundle.
        tuned (dict): Tuned configuration payload.
        pos_bank (np.ndarray): Positive kNN bank.
        neg_bank (np.ndarray | None): Negative kNN bank.
        model_name (str): Backbone name.
        patch_size (int): Patch size used by the model.
        resample_factor (int): Resample factor.
        tile_size (int): Runtime tile size.
        stride (int): Runtime stride.
        context_radius (int): Feature context radius.

    Returns:
        dict[str, Any]: Bundle metadata suitable for logging/checkpointing.

    Examples:
        >>> callable(save_model_bundle)
        True
    """
    os.makedirs(bundle_dir, exist_ok=True)

    pos_bank_file = "pos_bank.npy"
    neg_bank_file = "neg_bank.npy" if neg_bank is not None else None
    xgb_model_file = "xgb_model.json" if tuned.get("bst") is not None else None
    manifest_file = "manifest.yml"

    pos_bank_path = os.path.join(bundle_dir, pos_bank_file)
    np.save(pos_bank_path, pos_bank.astype(np.float32))
    if neg_bank is not None:
        np.save(os.path.join(bundle_dir, neg_bank_file), neg_bank.astype(np.float32))

    if tuned.get("bst") is not None:
        tuned["bst"].save_model(os.path.join(bundle_dir, xgb_model_file))

    created_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "bundle_version": _BUNDLE_VERSION,
        "created_utc": created_utc,
        "model": {
            "name": model_name,
            "patch_size": int(patch_size),
            "resample_factor": int(resample_factor),
        },
        "runtime": {
            "tile_size": int(tile_size),
            "stride": int(stride),
            "context_radius": int(context_radius),
        },
        "feature_spec_hash": hybrid_feature_spec_hash(),
        "model_toggles": {
            "knn_enabled": bool(tuned.get("knn_enabled", True)),
            "xgb_enabled": bool(tuned.get("xgb_enabled", True)),
            "crf_enabled": bool(tuned.get("crf_enabled", True)),
        },
        "tuned": {
            "best_raw_config": tuned.get("best_raw_config"),
            "best_xgb_config": tuned.get("best_xgb_config"),
            "best_crf_config": tuned.get("best_crf_config"),
            "shadow_cfg": tuned.get("shadow_cfg"),
            "champion_source": tuned.get("champion_source"),
            "roads_penalty": tuned.get("roads_penalty"),
            "xgb_feature_stats": tuned.get("xgb_feature_stats"),
            "feature_layout": tuned.get("feature_layout"),
        },
        "artifacts": {
            "manifest": manifest_file,
            "pos_bank": pos_bank_file,
            "neg_bank": neg_bank_file,
            "xgb_model": xgb_model_file,
        },
    }
    manifest_path = os.path.join(bundle_dir, manifest_file)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            _normalize(manifest),
            fh,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        )
    return {
        "path": bundle_dir,
        "version": _BUNDLE_VERSION,
        "manifest": manifest_path,
        "created_utc": created_utc,
    }


def load_model_bundle(bundle_dir: str) -> dict[str, Any]:
    """Load a persisted model bundle for inference-only execution.

    Args:
        bundle_dir (str): Bundle directory path.

    Returns:
        dict[str, Any]: Loaded artifacts and metadata.

    Examples:
        >>> callable(load_model_bundle)
        True
    """
    manifest_path = os.path.join(bundle_dir, "manifest.yml")
    if not os.path.exists(manifest_path):
        raise ValueError(f"model bundle manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh) or {}
    artifacts = manifest.get("artifacts", {})
    model_toggles = manifest.get("model_toggles", {})
    tuned_payload = manifest.get("tuned", {})

    pos_bank_file = artifacts.get("pos_bank", "pos_bank.npy")
    pos_bank_path = os.path.join(bundle_dir, pos_bank_file)
    if not os.path.exists(pos_bank_path):
        raise ValueError(f"model bundle missing pos_bank artifact: {pos_bank_path}")
    pos_bank = np.load(pos_bank_path).astype(np.float32, copy=False)

    neg_bank = None
    neg_bank_file = artifacts.get("neg_bank")
    if neg_bank_file:
        neg_bank_path = os.path.join(bundle_dir, neg_bank_file)
        if not os.path.exists(neg_bank_path):
            raise ValueError(f"model bundle missing neg_bank artifact: {neg_bank_path}")
        neg_bank = np.load(neg_bank_path).astype(np.float32, copy=False)

    bst = None
    xgb_enabled = bool(model_toggles.get("xgb_enabled", True))
    xgb_model_file = artifacts.get("xgb_model")
    if xgb_enabled:
        if not xgb_model_file:
            raise ValueError(
                "model bundle declares xgb_enabled=true but xgb_model artifact is missing"
            )
        xgb_model_path = os.path.join(bundle_dir, xgb_model_file)
        if not os.path.exists(xgb_model_path):
            raise ValueError(
                f"model bundle missing xgb_model artifact: {xgb_model_path}"
            )
        bst = xgb.Booster()
        bst.load_model(xgb_model_path)

    tuned = {
        "bst": bst,
        "best_raw_config": tuned_payload.get("best_raw_config"),
        "best_xgb_config": tuned_payload.get("best_xgb_config"),
        "best_crf_config": tuned_payload.get("best_crf_config"),
        "shadow_cfg": tuned_payload.get("shadow_cfg"),
        "champion_source": tuned_payload.get("champion_source"),
        "roads_penalty": tuned_payload.get("roads_penalty", 1.0),
        "xgb_feature_stats": tuned_payload.get("xgb_feature_stats"),
        "feature_layout": tuned_payload.get("feature_layout"),
        "knn_enabled": bool(model_toggles.get("knn_enabled", True)),
        "xgb_enabled": xgb_enabled,
        "crf_enabled": bool(model_toggles.get("crf_enabled", True)),
    }
    return {
        "manifest": manifest,
        "tuned": tuned,
        "pos_bank": pos_bank,
        "neg_bank": neg_bank,
    }


def validate_bundle_compatibility(
    manifest: dict[str, Any],
    *,
    model_name: str,
    patch_size: int,
    resample_factor: int,
    tile_size: int,
    stride: int,
    context_radius: int,
) -> None:
    """Validate that loaded bundle matches active runtime shape/config.

    Raises:
        ValueError: If any required runtime compatibility check fails.

    Examples:
        >>> callable(validate_bundle_compatibility)
        True
    """
    mismatches: list[str] = []
    model_payload = manifest.get("model", {})
    runtime_payload = manifest.get("runtime", {})

    checks = [
        ("model.name", model_payload.get("name"), model_name),
        ("model.patch_size", model_payload.get("patch_size"), int(patch_size)),
        (
            "model.resample_factor",
            model_payload.get("resample_factor"),
            int(resample_factor),
        ),
        ("runtime.tile_size", runtime_payload.get("tile_size"), int(tile_size)),
        ("runtime.stride", runtime_payload.get("stride"), int(stride)),
        (
            "runtime.context_radius",
            runtime_payload.get("context_radius"),
            int(context_radius),
        ),
        (
            "feature_spec_hash",
            manifest.get("feature_spec_hash"),
            hybrid_feature_spec_hash(),
        ),
    ]
    for field, actual, expected in checks:
        if actual != expected:
            mismatches.append(f"{field}: bundle={actual!r} runtime={expected!r}")

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"model bundle is incompatible with runtime config: {details}")
