"""Tests for source-label reprojection optimizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

import segedge.core.io_utils as io_utils


def _write_raster(
    path: Path,
    *,
    data: np.ndarray,
    crs: str,
    left: float,
    top: float,
    pixel_size_x: float,
    pixel_size_y: float | None = None,
) -> None:
    """Write a small single-band GeoTIFF.

    Examples:
        >>> True
        True
    """
    pixel_size_y = pixel_size_x if pixel_size_y is None else pixel_size_y
    transform = from_origin(left, top, pixel_size_x, pixel_size_y)
    raster_data = np.asarray(data, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=raster_data.shape[1],
        height=raster_data.shape[0],
        count=1,
        dtype=raster_data.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(raster_data, 1)


def _legacy_reproject_labels_to_image(
    ref_img_path: str, labels_path: str, downsample_factor: int = 1
) -> np.ndarray:
    """Legacy MemoryFile-based reprojection path for regression comparison.

    Examples:
        >>> callable(_legacy_reproject_labels_to_image)
        True
    """
    with rasterio.open(ref_img_path) as ref, rasterio.open(labels_path) as src:
        if downsample_factor > 1:
            dst_width = ref.width // downsample_factor
            dst_height = ref.height // downsample_factor
            dst_transform = ref.transform * ref.transform.scale(
                ref.width / dst_width,
                ref.height / dst_height,
            )
        else:
            dst_width = ref.width
            dst_height = ref.height
            dst_transform = ref.transform
        dst_meta = ref.meta.copy()
        dst_meta.update(
            dtype=src.dtypes[0],
            count=1,
            width=dst_width,
            height=dst_height,
            transform=dst_transform,
        )
        memfile = MemoryFile()
        with memfile.open(**dst_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=ref.crs,
                dst_width=dst_width,
                dst_height=dst_height,
                src_nodata=src.nodata,
                dst_nodata=src.nodata if src.nodata is not None else 0,
                resampling=Resampling.nearest,
            )
            return dst.read(1)


def test_reproject_labels_to_image_matches_legacy_on_aligned_grid(
    tmp_path, monkeypatch
):
    """Aligned same-CRS tiles should use direct window reads and match legacy output.

    Examples:
        >>> True
        True
    """
    io_utils._clear_source_label_cache()
    labels_path = tmp_path / "labels.tif"
    ref_path = tmp_path / "ref.tif"
    labels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    ref = np.zeros((4, 4), dtype=np.uint8)
    _write_raster(
        labels_path,
        data=labels,
        crs="EPSG:3857",
        left=0,
        top=8,
        pixel_size_x=1,
    )
    _write_raster(
        ref_path,
        data=ref,
        crs="EPSG:3857",
        left=2,
        top=6,
        pixel_size_x=1,
    )

    def _unexpected_reproject(*args, **kwargs):
        raise AssertionError("aligned fast path should not call reproject()")

    monkeypatch.setattr(io_utils, "reproject", _unexpected_reproject)
    optimized = io_utils.reproject_labels_to_image(
        str(ref_path), str(labels_path), downsample_factor=2
    )
    legacy = _legacy_reproject_labels_to_image(
        str(ref_path), str(labels_path), downsample_factor=2
    )

    np.testing.assert_array_equal(optimized, legacy)
    assert optimized.shape == (2, 2)


def test_reproject_labels_to_image_matches_legacy_on_crs_mismatch(
    tmp_path, monkeypatch
):
    """CRS-mismatched tiles should fall back to rasterio.warp.reproject().

    Examples:
        >>> True
        True
    """
    io_utils._clear_source_label_cache()
    labels_path = tmp_path / "labels_wgs84.tif"
    ref_path = tmp_path / "ref_mercator.tif"
    labels = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    _write_raster(
        labels_path,
        data=labels,
        crs="EPSG:4326",
        left=0.0,
        top=1.0,
        pixel_size_x=0.1,
    )
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    left, bottom = transformer.transform(0.05, 0.55)
    right, top = transformer.transform(0.45, 0.95)
    pixel_size_x = (right - left) / 4
    pixel_size_y = (top - bottom) / 4
    _write_raster(
        ref_path,
        data=np.zeros((4, 4), dtype=np.uint8),
        crs="EPSG:3857",
        left=left,
        top=top,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
    )
    reproject_calls = {"count": 0}
    original_reproject = io_utils.reproject

    def _counting_reproject(*args, **kwargs):
        reproject_calls["count"] += 1
        return original_reproject(*args, **kwargs)

    monkeypatch.setattr(io_utils, "reproject", _counting_reproject)
    optimized = io_utils.reproject_labels_to_image(str(ref_path), str(labels_path))
    legacy = _legacy_reproject_labels_to_image(str(ref_path), str(labels_path))

    assert reproject_calls["count"] == 1
    np.testing.assert_array_equal(optimized, legacy)


def test_reproject_labels_to_image_reuses_cached_source_label_handle(
    tmp_path, monkeypatch
):
    """Repeated use of the same label raster should reuse the cached source handle.

    Examples:
        >>> True
        True
    """
    io_utils._clear_source_label_cache()
    labels_path = tmp_path / "labels.tif"
    ref_a_path = tmp_path / "ref_a.tif"
    ref_b_path = tmp_path / "ref_b.tif"
    _write_raster(
        labels_path,
        data=np.ones((6, 6), dtype=np.uint8),
        crs="EPSG:3857",
        left=0,
        top=6,
        pixel_size_x=1,
    )
    _write_raster(
        ref_a_path,
        data=np.zeros((3, 3), dtype=np.uint8),
        crs="EPSG:3857",
        left=0,
        top=6,
        pixel_size_x=1,
    )
    _write_raster(
        ref_b_path,
        data=np.zeros((3, 3), dtype=np.uint8),
        crs="EPSG:3857",
        left=3,
        top=6,
        pixel_size_x=1,
    )
    original_open = io_utils.rasterio.open
    open_counts = {"labels": 0}

    def _counting_open(path, *args, **kwargs):
        if str(path) == str(labels_path):
            open_counts["labels"] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(io_utils.rasterio, "open", _counting_open)
    io_utils.reproject_labels_to_image(str(ref_a_path), str(labels_path))
    io_utils.reproject_labels_to_image(str(ref_b_path), str(labels_path))

    assert open_counts["labels"] == 1
    io_utils._clear_source_label_cache()
