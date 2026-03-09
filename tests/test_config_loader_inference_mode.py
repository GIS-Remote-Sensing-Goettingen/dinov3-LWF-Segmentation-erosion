"""Config-loader validation tests for inference-only mode."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_inference_only_requires_bundle_dir(tmp_path):
    """io.training=false must require io.inference.model_bundle_dir.

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
    with pytest.raises(ValueError, match="io.inference.model_bundle_dir"):
        load_config(cfg_path)


def test_inference_group_defaults_are_applied(tmp_path):
    """io.inference section should parse with defaults when omitted.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["io"].pop("inference", None)
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.io.training is True
    assert loaded.io.inference.model_bundle_dir is None
    assert loaded.io.inference.tile_glob == "*.tif"
    assert loaded.io.inference.tiles == []
    assert loaded.io.inference.save_bundle is True
