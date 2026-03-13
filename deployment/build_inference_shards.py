"""Build deterministic inference shard files from the configured tile source.

Examples:
    >>> _round_robin_shards(["a", "b", "c", "d"], 2)
    [['a', 'c'], ['b', 'd']]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from segedge.core.config_loader import cfg
from segedge.pipeline.inference_flow import resolve_inference_tiles

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
    tiles, inference_dir, inference_glob = resolve_inference_tiles(
        infer_tiles_file=None,
        infer_tiles_dir=cfg.io.inference.tiles_dir,
        infer_tile_glob=cfg.io.inference.tile_glob,
        infer_tiles=list(cfg.io.inference.tiles),
        legacy_inference_dir=cfg.io.paths.inference_dir,
        legacy_inference_glob=cfg.io.paths.inference_glob,
        legacy_holdout_tiles=list(cfg.io.paths.holdout_tiles),
        logger=logger,
    )
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
