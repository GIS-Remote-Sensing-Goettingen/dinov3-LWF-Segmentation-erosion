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
    expected_trimap = list(raw["search"]["crf"]["trimap_band_pixels_values"])
    expected_fill_holes_xgb = bool(raw["postprocess"].get("fill_holes_xgb", False))
    raw["io"].pop("inference", None)
    raw["io"]["training"] = True
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.io.training is True
    assert loaded.io.inference.model_bundle_dir is None
    assert loaded.io.inference.tile_glob == "*.tif"
    assert loaded.io.inference.tiles == []
    assert loaded.io.inference.plot_every == 1
    assert loaded.io.inference.plots.unified is True
    assert loaded.io.inference.plots.qualitative_core is True
    assert loaded.io.inference.plots.score_threshold is True
    assert loaded.io.inference.plots.disagreement_entropy is True
    assert loaded.io.inference.plots.proposal_overlay is True
    assert loaded.io.inference.save_bundle is True
    assert loaded.io.inference.score_prior.enabled is False
    assert loaded.io.inference.score_prior.apply_to == "xgb"
    assert loaded.io.inference.score_prior.target == "source_labels"
    assert loaded.io.inference.score_prior.mode == "multiply"
    assert loaded.io.inference.score_prior.inside_factor == 1.15
    assert loaded.io.inference.score_prior.outside_factor == 1.0
    assert loaded.io.inference.score_prior.clip_max == 1.0
    assert loaded.postprocess.fill_holes_xgb is expected_fill_holes_xgb
    assert loaded.search.crf.trimap_band_pixels_values == expected_trimap


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
    assert loaded.io.inference.score_prior.inside_factor == 1.3
    assert loaded.io.inference.score_prior.outside_factor == 1.0
    assert loaded.io.inference.score_prior.clip_max == 0.95


def test_inference_score_prior_parses_inside_and_outside_factors(tmp_path):
    """io.inference.score_prior should support separate inside/outside factors.

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
        "inside_factor": 1.4,
        "outside_factor": 0.8,
        "clip_max": 0.9,
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.io.inference.score_prior.enabled is True
    assert loaded.io.inference.score_prior.inside_factor == 1.4
    assert loaded.io.inference.score_prior.outside_factor == 0.8
    assert loaded.io.inference.score_prior.clip_max == 0.9


def test_inference_plot_every_must_be_positive(tmp_path):
    """io.inference.plot_every should reject non-positive values.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["io"].setdefault("inference", {})
    raw["io"]["inference"]["plot_every"] = 0
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    try:
        load_config(cfg_path)
    except ValueError as exc:
        assert "io.inference.plot_every" in str(exc)
    else:
        raise AssertionError("expected plot_every validation error")


def test_inference_plot_toggles_parse_from_config(tmp_path):
    """io.inference.plots should parse individual plot toggles.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["io"].setdefault("inference", {})
    raw["io"]["inference"]["plots"] = {
        "unified": False,
        "qualitative_core": True,
        "score_threshold": False,
        "disagreement_entropy": False,
        "proposal_overlay": True,
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)
    assert loaded.io.inference.plots.unified is False
    assert loaded.io.inference.plots.qualitative_core is True
    assert loaded.io.inference.plots.score_threshold is False
    assert loaded.io.inference.plots.disagreement_entropy is False
    assert loaded.io.inference.plots.proposal_overlay is True


def test_crf_trimap_band_pixels_parse_from_config(tmp_path):
    """search.crf.trimap_band_pixels_values should parse when explicitly configured.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["search"]["crf"]["trimap_band_pixels_values"] = [8, 16, 24]
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)

    assert loaded.search.crf.trimap_band_pixels_values == [8, 16, 24]


def test_postprocess_fill_holes_xgb_parses_from_config(tmp_path):
    """postprocess.fill_holes_xgb should parse when explicitly configured.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["postprocess"]["fill_holes_xgb"] = True
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)

    assert loaded.postprocess.fill_holes_xgb is True


def test_novel_proposals_width_bonus_and_hard_cap_parse_from_config(tmp_path):
    """Novel proposal width compensation settings should parse explicitly.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["postprocess"]["novel_proposals"]["width_bonus_per_pca"] = 1.25
    raw["postprocess"]["novel_proposals"]["hard_width_cap_m"] = 18.0
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    loaded = load_config(cfg_path)

    assert loaded.postprocess.novel_proposals.width_bonus_per_pca == 1.25
    assert loaded.postprocess.novel_proposals.hard_width_cap_m == 18.0


def test_novel_proposals_hard_width_cap_must_cover_base_width(tmp_path):
    """Hard width cap must be at least the base width allowance.

    Examples:
        >>> True
        True
    """
    raw = _load_repo_config()
    raw["postprocess"]["novel_proposals"]["max_width_m"] = 12.0
    raw["postprocess"]["novel_proposals"]["hard_width_cap_m"] = 10.0
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    try:
        load_config(cfg_path)
    except ValueError as exc:
        assert "hard_width_cap_m" in str(exc)
    else:
        raise AssertionError("expected hard_width_cap_m validation error")
