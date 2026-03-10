"""Config-loader validation tests for inference-only mode."""

from __future__ import annotations

from pathlib import Path

import yaml

from segedge.core.config_loader import load_config


def _load_repo_config() -> dict:
    """Load repository config.yml as raw mapping.

    Examples:
        >>> isinstance(_load_repo_config(), dict)
        True
    """
    cfg_path = Path(__file__).resolve().parents[1] / "config.yml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def test_inference_only_allows_null_bundle_dir(tmp_path):
    """io.training=false may defer bundle resolution to runtime.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["io"]["training"] = False
    raw["io"].setdefault("inference", {})
    raw["io"]["inference"]["model_bundle_dir"] = None
    raw["io"]["inference"]["tiles"] = ["tile_a.tif"]
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.io.training is False
    assert loaded.io.inference.model_bundle_dir is None


def test_inference_group_defaults_are_applied(tmp_path):
    """io.inference section should parse with defaults when omitted.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["io"].pop("inference", None)
    raw["io"]["training"] = True
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.io.training is True
    assert loaded.io.inference.model_bundle_dir is None
    assert loaded.io.inference.tile_glob == "*.tif"
    assert loaded.io.inference.tiles == []
    assert loaded.io.inference.save_bundle is True
    assert loaded.io.inference.score_prior.enabled is False
    assert loaded.io.inference.score_prior.apply_to == "xgb"
    assert loaded.io.inference.score_prior.target == "source_labels"
    assert loaded.io.inference.score_prior.mode == "multiply"
    assert loaded.io.inference.score_prior.factor == 1.15
    assert loaded.io.inference.score_prior.clip_max == 1.0


def test_inference_score_prior_parses_from_config(tmp_path):
    """io.inference.score_prior should parse when explicitly configured.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["io"].setdefault("inference", {})
    raw["io"]["inference"]["score_prior"] = {
        "enabled": True,
        "apply_to": "xgb",
        "target": "source_labels",
        "mode": "multiply",
        "factor": 1.3,
        "clip_max": 0.95,
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.io.inference.score_prior.enabled is True
    assert loaded.io.inference.score_prior.factor == 1.3
    assert loaded.io.inference.score_prior.clip_max == 0.95
