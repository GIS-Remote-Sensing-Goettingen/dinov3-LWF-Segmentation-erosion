"""Merge shard union rasters into one final union family.

Examples:
    >>> sorted(UNION_VARIANTS)
    ['crf', 'raw', 'shadow', 'shadow_with_proposals']
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

UNION_VARIANTS = ("raw", "crf", "shadow", "shadow_with_proposals")
MERGE_WINDOW_SIZE = 1024


def _transforms_match(left, right) -> bool:
    """Return whether two affine transforms are numerically equivalent.

    Examples:
        >>> _transforms_match((1, 0, 0, 0, -1, 0, 0, 0, 1), (1, 0, 0, 0, -1, 0, 0, 0, 1))
        True
    """
    return all(
        math.isclose(float(a), float(b), abs_tol=1e-9) for a, b in zip(left, right)
    )


def _is_close_to_int(value: float) -> bool:
    """Return whether a floating-point value is effectively integral.

    Examples:
        >>> _is_close_to_int(3.0000001)
        True
    """
    return math.isclose(value, round(value), abs_tol=1e-6)


def _validate_axis_aligned_transform(transform, source_path: Path) -> None:
    """Validate that a transform is north-up and non-rotated.

    Examples:
        >>> callable(_validate_axis_aligned_transform)
        True
    """
    if not math.isclose(float(transform.b), 0.0, abs_tol=1e-9) or not math.isclose(
        float(transform.d), 0.0, abs_tol=1e-9
    ):
        raise ValueError(f"union raster must be axis-aligned: {source_path}")
    if float(transform.a) <= 0 or float(transform.e) >= 0:
        raise ValueError(
            f"union raster must use positive x and negative y resolution: {source_path}"
        )


def _validate_aligned_union_transform(
    *,
    transform,
    reference_transform,
    source_path: Path,
) -> None:
    """Validate that a transform is aligned to the reference pixel grid.

    Examples:
        >>> callable(_validate_aligned_union_transform)
        True
    """
    _validate_axis_aligned_transform(transform, source_path)
    if not math.isclose(
        float(transform.a),
        float(reference_transform.a),
        abs_tol=1e-9,
    ) or not math.isclose(
        float(transform.e),
        float(reference_transform.e),
        abs_tol=1e-9,
    ):
        raise ValueError(f"incompatible union raster resolution: {source_path}")
    col_offset = (float(transform.c) - float(reference_transform.c)) / float(
        reference_transform.a
    )
    row_offset = (float(transform.f) - float(reference_transform.f)) / float(
        reference_transform.e
    )
    if not (_is_close_to_int(col_offset) and _is_close_to_int(row_offset)):
        raise ValueError(f"incompatible union raster grid alignment: {source_path}")


def _build_mosaic_profile(source_paths: list[Path]) -> dict[str, object]:
    """Return the output profile for a mosaic of aligned union rasters.

    Examples:
        >>> callable(_build_mosaic_profile)
        True
    """
    with rasterio.open(source_paths[0]) as src0:
        profile = src0.profile.copy()
        profile.update(compress="lzw")
        first_transform = src0.transform
        first_crs = src0.crs
        first_dtype = src0.dtypes[0]
        first_nodata = src0.nodata
        first_count = src0.count
        left = src0.bounds.left
        bottom = src0.bounds.bottom
        right = src0.bounds.right
        top = src0.bounds.top
        _validate_axis_aligned_transform(first_transform, source_paths[0])

    for source_path in source_paths[1:]:
        with rasterio.open(source_path) as src:
            if (
                src.crs != first_crs
                or src.count != first_count
                or src.dtypes[0] != first_dtype
                or src.nodata != first_nodata
            ):
                raise ValueError(f"incompatible union raster profile: {source_path}")
            _validate_aligned_union_transform(
                transform=src.transform,
                reference_transform=first_transform,
                source_path=source_path,
            )
            left = min(left, src.bounds.left)
            bottom = min(bottom, src.bounds.bottom)
            right = max(right, src.bounds.right)
            top = max(top, src.bounds.top)

    pixel_width = float(first_transform.a)
    pixel_height = abs(float(first_transform.e))
    width_float = (right - left) / pixel_width
    height_float = (top - bottom) / pixel_height
    if not (_is_close_to_int(width_float) and _is_close_to_int(height_float)):
        raise ValueError(
            "merged union raster bounds are not aligned to the shared grid"
        )
    profile.update(
        width=int(round(width_float)),
        height=int(round(height_float)),
        transform=from_origin(left, top, pixel_width, pixel_height),
    )
    return profile


def _window_in_destination(
    *,
    source_dataset: rasterio.io.DatasetReader,
    destination_dataset: rasterio.io.DatasetWriter,
    source_path: Path,
) -> Window:
    """Return the destination window for one aligned source raster.

    Examples:
        >>> callable(_window_in_destination)
        True
    """
    if source_dataset.crs != destination_dataset.crs:
        raise ValueError(f"incompatible union raster CRS: {source_path}")
    _validate_aligned_union_transform(
        transform=source_dataset.transform,
        reference_transform=destination_dataset.transform,
        source_path=source_path,
    )
    col_offset = (
        float(source_dataset.transform.c) - float(destination_dataset.transform.c)
    ) / float(destination_dataset.transform.a)
    row_offset = (
        float(source_dataset.transform.f) - float(destination_dataset.transform.f)
    ) / float(destination_dataset.transform.e)
    col_off = int(round(col_offset))
    row_off = int(round(row_offset))
    if (
        col_off < 0
        or row_off < 0
        or col_off + source_dataset.width > destination_dataset.width
        or row_off + source_dataset.height > destination_dataset.height
    ):
        raise ValueError(f"union raster falls outside merged bounds: {source_path}")
    return Window(
        col_off=col_off,
        row_off=row_off,
        width=source_dataset.width,
        height=source_dataset.height,
    )


def _iter_chunk_windows(*, width: int, height: int) -> list[Window]:
    """Return fixed-size windows that cover a raster without full-array reads.

    Examples:
        >>> len(_iter_chunk_windows(width=2, height=1))
        1
    """
    windows: list[Window] = []
    for row_off in range(0, height, MERGE_WINDOW_SIZE):
        window_height = min(MERGE_WINDOW_SIZE, height - row_off)
        for col_off in range(0, width, MERGE_WINDOW_SIZE):
            window_width = min(MERGE_WINDOW_SIZE, width - col_off)
            windows.append(
                Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=window_width,
                    height=window_height,
                )
            )
    return windows


def merge_union_variant(
    *,
    variant: str,
    shard_run_dirs: list[str],
    output_dir: str,
) -> Path | None:
    """Merge one union variant from multiple shard run directories.
    Examples:
        >>> merge_union_variant(variant="raw", shard_run_dirs=[], output_dir="/tmp") is None
        True
    """
    source_paths = [
        Path(run_dir) / "shapes" / "unions" / variant / "union.tif"
        for run_dir in shard_run_dirs
    ]
    source_paths = [path for path in source_paths if path.exists()]
    if not source_paths:
        return None

    profile = _build_mosaic_profile(source_paths)
    out_path = Path(output_dir) / variant / "union.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        for window in _iter_chunk_windows(
            width=int(profile["width"]),
            height=int(profile["height"]),
        ):
            dst.write(
                np.zeros((int(window.height), int(window.width)), dtype=np.uint8),
                1,
                window=window,
            )
    with rasterio.open(out_path, "r+") as dst:
        for source_path in source_paths:
            with rasterio.open(source_path) as src:
                dst_window = _window_in_destination(
                    source_dataset=src,
                    destination_dataset=dst,
                    source_path=source_path,
                )
                for src_window in _iter_chunk_windows(
                    width=src.width,
                    height=src.height,
                ):
                    target_window = Window(
                        col_off=float(dst_window.col_off) + float(src_window.col_off),
                        row_off=float(dst_window.row_off) + float(src_window.row_off),
                        width=src_window.width,
                        height=src_window.height,
                    )
                    source_pixels = src.read(1, window=src_window)
                    existing = dst.read(1, window=target_window)
                    dst.write(
                        np.maximum(existing, source_pixels),
                        1,
                        window=target_window,
                    )
    return out_path


def merge_shard_unions(
    *,
    shard_run_dirs: list[str],
    output_dir: str,
) -> list[Path]:
    """Merge all supported union variants from shard outputs.

    Examples:
        >>> merge_shard_unions(shard_run_dirs=[], output_dir="/tmp")
        []
    """
    merged_paths: list[Path] = []
    for variant in UNION_VARIANTS:
        merged = merge_union_variant(
            variant=variant,
            shard_run_dirs=shard_run_dirs,
            output_dir=output_dir,
        )
        if merged is not None:
            merged_paths.append(merged)
    return merged_paths


def main() -> None:
    """CLI entrypoint for merging shard union rasters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged union rasters are written.",
    )
    parser.add_argument(
        "shard_run_dirs",
        nargs="+",
        help="Completed shard run directories to merge.",
    )
    args = parser.parse_args()
    merged_paths = merge_shard_unions(
        shard_run_dirs=[os.path.abspath(path) for path in args.shard_run_dirs],
        output_dir=args.output_dir,
    )
    print(f"merged {len(merged_paths)} union rasters into {args.output_dir}")


if __name__ == "__main__":
    main()
