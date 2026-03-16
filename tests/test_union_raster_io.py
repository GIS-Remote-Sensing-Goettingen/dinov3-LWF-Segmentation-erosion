"""Tests for rolling union GeoTIFF helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import segedge.core.io_utils as io_utils


def _write_tile(
    path: Path, *, left: float, top: float, width: int = 2, height: int = 2
):
    """Write a small single-band tile raster.

    Examples:
        >>> True
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=from_origin(left, top, 1.0, 1.0),
        nodata=0,
    ) as dst:
        dst.write(np.zeros((height, width), dtype=np.uint8), 1)


def test_union_raster_profile_and_append_merge_adjacent_tiles(tmp_path):
    """Adjacent tiles should build one mosaic grid and merge masks by OR.

    Examples:
        >>> True
        True
    """
    tile_a = tmp_path / "tile_a.tif"
    tile_b = tmp_path / "tile_b.tif"
    _write_tile(tile_a, left=0.0, top=2.0)
    _write_tile(tile_b, left=2.0, top=2.0)

    profile = io_utils.build_union_raster_profile([str(tile_a), str(tile_b)])
    union_path = tmp_path / "union.tif"
    io_utils.initialize_union_raster(str(union_path), profile)

    io_utils.append_mask_to_union_raster(
        np.array([[1, 0], [0, 0]], dtype=np.uint8),
        str(tile_a),
        str(union_path),
    )
    io_utils.append_mask_to_union_raster(
        np.array([[0, 1], [0, 0]], dtype=np.uint8),
        str(tile_b),
        str(union_path),
    )

    with rasterio.open(union_path) as src:
        merged = src.read(1)

    assert merged.shape == (2, 4)
    np.testing.assert_array_equal(
        merged,
        np.array([[1, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8),
    )


def test_validate_union_raster_compatibility_rejects_mismatch(tmp_path):
    """Resume validation should fail when the stored raster grid does not match.

    Examples:
        >>> True
        True
    """
    tile_path = tmp_path / "tile.tif"
    _write_tile(tile_path, left=0.0, top=2.0)
    profile = io_utils.build_union_raster_profile([str(tile_path)])
    union_path = tmp_path / "union.tif"
    io_utils.initialize_union_raster(str(union_path), profile)

    bad_profile = dict(profile)
    bad_profile["width"] = int(profile["width"]) + 1

    with pytest.raises(ValueError, match="incompatible"):
        io_utils.validate_union_raster_compatibility(str(union_path), bad_profile)


def test_backup_union_raster_copies_geotiff(tmp_path):
    """Raster backups should copy the GeoTIFF into the backup directory.

    Examples:
        >>> True
        True
    """
    tile_path = tmp_path / "tile.tif"
    _write_tile(tile_path, left=0.0, top=2.0)
    profile = io_utils.build_union_raster_profile([str(tile_path)])
    union_path = tmp_path / "union.tif"
    io_utils.initialize_union_raster(str(union_path), profile)

    io_utils.backup_union_raster(
        str(union_path),
        str(tmp_path / "backup"),
        step=2,
        backup_name="union_backup",
    )

    assert (tmp_path / "backup" / "union_backup.tif").exists()
