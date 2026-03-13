"""Build deterministic inference shard files from the configured tile source.

Examples:
    >>> _round_robin_shards(["a", "b", "c", "d"], 2)
    [['a', 'c'], ['b', 'd']]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segedge.core.config_loader import cfg  # noqa: E402
from segedge.pipeline.common import filter_tiles_by_source_label_presence  # noqa: E402

logger = logging.getLogger(__name__)


def _round_robin_shards(tiles: list[str], shard_count: int) -> list[list[str]]:
    """Partition tiles across shards in deterministic round-robin order.

    Examples:
        >>> _round_robin_shards(["a", "b", "c"], 2)
        [['a', 'c'], ['b']]
    """
    shards = [[] for _ in range(shard_count)]
    for idx, tile in enumerate(tiles):
        shards[idx % shard_count].append(tile)
    return shards


def build_inference_shards(
    *,
    shard_count: int,
    output_dir: str,
    job_name: str,
) -> tuple[Path, list[Path]]:
    """Resolve inference tiles once and write shard files plus a manifest.

    Examples:
        >>> callable(build_inference_shards)
        True
    """
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")
    shard_root = Path(output_dir) / job_name
    shard_root.mkdir(parents=True, exist_ok=True)
    tiles, inference_dir, inference_glob = _resolve_inference_tiles_for_shards()
    shards = _round_robin_shards(tiles, shard_count)
    shard_files: list[Path] = []
    for idx, shard_tiles in enumerate(shards):
        shard_path = shard_root / f"tiles_shard_{idx:03d}.txt"
        shard_path.write_text(
            "\n".join(shard_tiles) + ("\n" if shard_tiles else ""),
            encoding="utf-8",
        )
        shard_files.append(shard_path)
    manifest = {
        "assignment_method": "round_robin",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inference_dir": inference_dir,
        "inference_glob": inference_glob,
        "job_name": job_name,
        "num_shards": shard_count,
        "source_label_raster": cfg.io.paths.source_label_raster,
        "total_tiles": len(tiles),
        "tiles_file_paths": [str(path) for path in shard_files],
        "tiles_per_shard": [len(shard_tiles) for shard_tiles in shards],
    }
    (shard_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return shard_root, shard_files


def _load_tiles_file(path: str) -> list[str]:
    """Load one tile path per line.

    Examples:
        >>> _load_tiles_file.__name__
        '_load_tiles_file'
    """
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _resolve_inference_tiles_for_shards() -> tuple[list[str], str | None, str]:
    """Resolve shard input tiles without importing the full inference runtime.

    Examples:
        >>> callable(_resolve_inference_tiles_for_shards)
        True
    """
    infer_tiles_file = str(cfg.io.inference.tiles_file or "").strip()
    infer_tiles_dir = str(cfg.io.inference.tiles_dir or "").strip()
    infer_tile_glob = str(
        cfg.io.inference.tile_glob or cfg.io.paths.inference_glob or "*.tif"
    )
    explicit_tiles = [str(tile) for tile in cfg.io.inference.tiles]
    legacy_inference_dir = str(cfg.io.paths.inference_dir or "").strip()
    legacy_holdout_tiles = [str(tile) for tile in cfg.io.paths.holdout_tiles]

    if infer_tiles_file:
        if not os.path.isfile(infer_tiles_file):
            raise ValueError(f"inference tiles_file not found: {infer_tiles_file}")
        tiles = _load_tiles_file(infer_tiles_file)
        logger.info(
            "inference tiles file: %s -> %s tiles",
            infer_tiles_file,
            len(tiles),
        )
        return tiles, None, infer_tile_glob

    tiles_dir = infer_tiles_dir or legacy_inference_dir
    if tiles_dir:
        if not os.path.isdir(tiles_dir):
            raise ValueError(f"inference tiles_dir not found: {tiles_dir}")
        tiles = sorted(glob.glob(os.path.join(tiles_dir, infer_tile_glob)))
        logger.info(
            "inference dir: %s (glob=%s) -> %s tiles",
            tiles_dir,
            infer_tile_glob,
            len(tiles),
        )
        filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(tiles)
        if excluded_count:
            logger.info(
                "inference: excluded %s tiles with no SOURCE_LABEL_RASTER labels inside tile",
                excluded_count,
            )
        return filtered_tiles, tiles_dir, infer_tile_glob

    if explicit_tiles:
        filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
            explicit_tiles
        )
        if excluded_count:
            logger.info(
                "inference: excluded %s explicit tiles with no SOURCE_LABEL_RASTER labels inside tile",
                excluded_count,
            )
        return filtered_tiles, None, infer_tile_glob

    filtered_tiles, excluded_count = filter_tiles_by_source_label_presence(
        legacy_holdout_tiles
    )
    if excluded_count:
        logger.info(
            "inference: excluded %s legacy holdout tiles with no SOURCE_LABEL_RASTER labels inside tile",
            excluded_count,
        )
    return filtered_tiles, None, infer_tile_glob


def main() -> None:
    """CLI entrypoint for building shard tile lists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, required=True, help="Number of shards.")
    parser.add_argument(
        "--output-dir",
        default="output/shards",
        help="Directory where shard files and manifest are written.",
    )
    parser.add_argument(
        "--job-name",
        required=True,
        help="Subdirectory name for this shard manifest.",
    )
    args = parser.parse_args()
    shard_root, shard_files = build_inference_shards(
        shard_count=args.shards,
        output_dir=args.output_dir,
        job_name=args.job_name,
    )
    print(f"wrote {len(shard_files)} shard files under {shard_root}")


if __name__ == "__main__":
    main()
