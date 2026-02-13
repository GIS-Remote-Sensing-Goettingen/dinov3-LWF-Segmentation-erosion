"""Unit tests for auto-split mode and inference-tile capping."""

from __future__ import annotations

import pytest

from segedge.pipeline.common import (
    AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT,
    AUTO_SPLIT_MODE_LEGACY,
    _cap_inference_tiles,
    _split_tiles_from_gt_presence,
    tile_has_gt_overlap,
)


def test_cap_inference_tiles_disabled_returns_sorted() -> None:
    """Capping disabled should return all holdout tiles in sorted order.

    Examples:
        >>> True
        True
    """
    holdout = ["tile_c", "tile_a", "tile_b"]
    selected = _cap_inference_tiles(holdout, cap_enabled=False, cap=1, seed=42)
    assert selected == ["tile_a", "tile_b", "tile_c"]


def test_cap_inference_tiles_enabled_reproducible_subset() -> None:
    """Capped selection should be deterministic for a fixed seed.

    Examples:
        >>> True
        True
    """
    holdout = [f"tile_{i:03d}" for i in range(20)]
    selected_1 = _cap_inference_tiles(holdout, cap_enabled=True, cap=5, seed=7)
    selected_2 = _cap_inference_tiles(holdout, cap_enabled=True, cap=5, seed=7)
    assert selected_1 == selected_2
    assert len(selected_1) == 5
    assert selected_1 == sorted(selected_1)
    assert set(selected_1).issubset(set(holdout))


def test_cap_inference_tiles_enabled_requires_positive_cap() -> None:
    """Enabled capping rejects non-positive cap values.

    Examples:
        >>> True
        True
    """
    with pytest.raises(ValueError, match="INFERENCE_TILE_CAP must be > 0"):
        _cap_inference_tiles(["tile_a"], cap_enabled=True, cap=0, seed=42)


def test_split_tiles_mode_gt_to_val_cap_holdout() -> None:
    """GT->val mode puts all GT tiles into validation and caps holdout tiles.

    Examples:
        >>> True
        True
    """
    source, val, holdout = _split_tiles_from_gt_presence(
        gt_tiles=["g3", "g1", "g2"],
        holdout_tiles=["h4", "h2", "h1", "h3"],
        mode=AUTO_SPLIT_MODE_GT_TO_VAL_CAP_HOLDOUT,
        val_fraction=0.5,
        seed=42,
        cap_enabled=True,
        cap=2,
        cap_seed=42,
    )
    assert source == []
    assert val == ["g1", "g2", "g3"]
    assert len(holdout) == 2
    assert holdout == sorted(holdout)
    assert set(holdout).issubset({"h1", "h2", "h3", "h4"})


def test_split_tiles_mode_legacy_preserves_source_plus_val_count() -> None:
    """Legacy mode still splits GT tiles into source/validation sets.

    Examples:
        >>> True
        True
    """
    source, val, holdout = _split_tiles_from_gt_presence(
        gt_tiles=["g1", "g2", "g3", "g4"],
        holdout_tiles=["h2", "h1"],
        mode=AUTO_SPLIT_MODE_LEGACY,
        val_fraction=0.5,
        seed=42,
        cap_enabled=True,
        cap=1,
        cap_seed=42,
    )
    assert len(source) + len(val) == 4
    assert set(source).isdisjoint(set(val))
    assert holdout == ["h1", "h2"]


def test_tile_has_gt_overlap_returns_false_without_vectors() -> None:
    """Public GT-overlap helper should short-circuit when vectors are missing.

    Examples:
        >>> True
        True
    """
    assert tile_has_gt_overlap("missing_tile.tif", [], downsample_factor=1) is False
