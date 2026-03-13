"""Merge shard union shapefiles into one final union family.

Examples:
    >>> sorted(UNION_VARIANTS)
    ['crf', 'raw', 'shadow', 'shadow_with_proposals']
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import fiona

UNION_VARIANTS = ("raw", "crf", "shadow", "shadow_with_proposals")
SHAPEFILE_EXTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def _remove_shapefile_set(path: Path) -> None:
    """Delete a shapefile and its sidecars if they exist.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     shp = Path(d) / "demo.shp"
        ...     _ = shp.write_text("", encoding="utf-8")
        ...     _remove_shapefile_set(shp)
        ...     shp.exists()
        False
    """
    base = path.with_suffix("")
    for ext in SHAPEFILE_EXTS:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            candidate.unlink()


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
        Path(run_dir) / "shapes" / "unions" / variant / "union.shp"
        for run_dir in shard_run_dirs
    ]
    source_paths = [path for path in source_paths if path.exists()]
    if not source_paths:
        return None

    out_path = Path(output_dir) / variant / "union.shp"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_shapefile_set(out_path)

    with fiona.open(source_paths[0], "r") as src0:
        schema = src0.schema.copy()
        crs = src0.crs

    with fiona.open(
        out_path,
        "w",
        driver="ESRI Shapefile",
        crs=crs,
        schema=schema,
    ) as dst:
        for source_path in source_paths:
            with fiona.open(source_path, "r") as src:
                for feature in src:
                    dst.write(feature)
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
    """CLI entrypoint for merging shard union shapefiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged union shapefiles are written.",
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
    print(f"merged {len(merged_paths)} union shapefiles into {args.output_dir}")


if __name__ == "__main__":
    main()
