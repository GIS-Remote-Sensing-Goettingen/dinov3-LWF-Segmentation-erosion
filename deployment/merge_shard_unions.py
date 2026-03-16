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

UNION_VARIANTS = ("raw", "crf", "shadow", "shadow_with_proposals")


def _transforms_match(left, right) -> bool:
    """Return whether two affine transforms are numerically equivalent.

    Examples:
        >>> _transforms_match((1, 0, 0, 0, -1, 0, 0, 0, 1), (1, 0, 0, 0, -1, 0, 0, 0, 1))
        True
    """
    return all(
        math.isclose(float(a), float(b), abs_tol=1e-9) for a, b in zip(left, right)
    )


def _validate_compatible_union_rasters(
    source_paths: list[Path],
) -> dict[str, object]:
    """Return a shared profile after validating all union rasters match.

    Examples:
        >>> callable(_validate_compatible_union_rasters)
        True
    """
    with rasterio.open(source_paths[0]) as src0:
        profile = src0.profile.copy()
        profile.update(compress="lzw")
        first_transform = src0.transform
        first_crs = src0.crs
        first_width = src0.width
        first_height = src0.height
        first_dtype = src0.dtypes[0]
        first_nodata = src0.nodata
        first_count = src0.count

    for source_path in source_paths[1:]:
        with rasterio.open(source_path) as src:
            if (
                src.crs != first_crs
                or src.width != first_width
                or src.height != first_height
                or src.count != first_count
                or src.dtypes[0] != first_dtype
                or src.nodata != first_nodata
                or not _transforms_match(src.transform, first_transform)
            ):
                raise ValueError(f"incompatible union raster profile: {source_path}")
    return profile


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

    profile = _validate_compatible_union_rasters(source_paths)
    out_path = Path(output_dir) / variant / "union.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        with rasterio.open(source_paths[0]) as src0:
            for _, window in src0.block_windows(1):
                merged = np.zeros(
                    (int(window.height), int(window.width)),
                    dtype=np.uint8,
                )
                for source_path in source_paths:
                    with rasterio.open(source_path) as src:
                        merged = np.maximum(merged, src.read(1, window=window))
                dst.write(merged, 1, window=window)
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
