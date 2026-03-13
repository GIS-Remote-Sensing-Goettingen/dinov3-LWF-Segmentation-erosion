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
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from rasterio import open as rio_open
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segedge.core.config_loader import cfg  # noqa: E402

logger = logging.getLogger(__name__)
_SOURCE_LABEL_FILTER_CACHE_VERSION = 1
_FILTER_PROGRESS_STRIDE = 250


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
    logger.info(
        "build shards: job=%s shards=%s output=%s",
        job_name,
        shard_count,
        shard_root,
    )
    tiles, inference_dir, inference_glob = _resolve_inference_tiles_for_shards(
        shard_root=shard_root
    )
    logger.info(
        "build shards: filtered tiles=%s assignment=round_robin",
        len(tiles),
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
        logger.info(
            "build shards: wrote shard %03d with %s tiles -> %s",
            idx,
            len(shard_tiles),
            shard_path,
        )
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
    logger.info("build shards: manifest written -> %s", shard_root / "manifest.json")
    return shard_root, shard_files


def _load_tiles_file(path: str) -> list[str]:
    """Load one tile path per line.

    Examples:
        >>> _load_tiles_file.__name__
        '_load_tiles_file'
    """
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _source_label_filter_cache_path(shard_root: Path) -> Path:
    """Return the deployment-local cache path for source-label tile filtering."""
    cache_dir = shard_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "source_label_tile_presence_cache.json"


def _stat_signature(path: str) -> dict[str, int | str]:
    """Return a compact metadata signature for cache invalidation."""
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
    }


def _load_source_label_filter_cache(
    *,
    cache_path: Path,
    raster_path: str,
) -> tuple[dict[str, dict], dict[str, int | str]]:
    """Load the persisted source-label cache if it matches the raster."""
    raster_signature = _stat_signature(raster_path)
    if not cache_path.exists():
        return {}, raster_signature
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}, raster_signature
    if payload.get("version") != _SOURCE_LABEL_FILTER_CACHE_VERSION:
        return {}, raster_signature
    if payload.get("raster_signature") != raster_signature:
        return {}, raster_signature
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}, raster_signature
    return entries, raster_signature


def _save_source_label_filter_cache(
    *,
    cache_path: Path,
    raster_signature: dict[str, int | str],
    entries: dict[str, dict],
) -> None:
    """Persist the source-label cache atomically."""
    payload = {
        "version": _SOURCE_LABEL_FILTER_CACHE_VERSION,
        "raster_signature": raster_signature,
        "entries": entries,
    }
    tmp_path = cache_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(cache_path)


def _filter_tiles_by_source_label_presence(
    *,
    tile_paths: list[str],
    raster_path: str | None,
    shard_root: Path,
) -> tuple[list[str], int]:
    """Filter tiles to those containing positive source-label pixels."""
    if not tile_paths:
        return [], 0
    if not raster_path:
        return list(tile_paths), 0

    cache_path = _source_label_filter_cache_path(shard_root)
    try:
        cache_entries, raster_signature = _load_source_label_filter_cache(
            cache_path=cache_path,
            raster_path=raster_path,
        )
    except OSError:
        cache_entries, raster_signature = {}, {}
    cache_hits = 0
    cache_dirty = False
    filtered_paths: list[str] = []
    excluded_count = 0
    start_ts = time.monotonic()

    logger.info(
        "source-label filter: start tiles=%s raster=%s",
        len(tile_paths),
        raster_path,
    )

    with rio_open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        nodata = src.nodata
        for idx, tile_path in enumerate(tile_paths, start=1):
            tile_signature = _stat_signature(tile_path)
            cache_entry = cache_entries.get(tile_path)
            if (
                cache_entry is not None
                and cache_entry.get("tile_signature") == tile_signature
            ):
                cache_hits += 1
                if cache_entry.get("has_labels", False):
                    filtered_paths.append(tile_path)
                else:
                    excluded_count += 1
                continue

            with rio_open(tile_path) as tile_src:
                tile_bounds = tile_src.bounds
                tile_crs = tile_src.crs

            if (
                raster_crs is not None
                and tile_crs is not None
                and tile_crs != raster_crs
            ):
                tile_bounds = transform_bounds(
                    tile_crs,
                    raster_crs,
                    *tile_bounds,
                    densify_pts=21,
                )
            tb_left, tb_bottom, tb_right, tb_top = tile_bounds
            rb_left, rb_bottom, rb_right, rb_top = raster_bounds
            overlap_left = max(tb_left, rb_left)
            overlap_bottom = max(tb_bottom, rb_bottom)
            overlap_right = min(tb_right, rb_right)
            overlap_top = min(tb_top, rb_top)
            if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
                has_labels = False
            else:
                window = from_bounds(
                    overlap_left,
                    overlap_bottom,
                    overlap_right,
                    overlap_top,
                    transform=src.transform,
                )
                label_data = src.read(1, window=window, masked=True)
                if label_data.size == 0:
                    has_labels = False
                else:
                    positive_mask = np.logical_and(
                        ~np.ma.getmaskarray(label_data),
                        np.asarray(label_data) > 0,
                    )
                    if nodata is not None:
                        positive_mask &= np.asarray(label_data) != nodata
                    has_labels = bool(np.any(positive_mask))

            cache_entries[tile_path] = {
                "tile_signature": tile_signature,
                "has_labels": has_labels,
            }
            cache_dirty = True
            if has_labels:
                filtered_paths.append(tile_path)
            else:
                excluded_count += 1

            if idx % _FILTER_PROGRESS_STRIDE == 0 or idx == len(tile_paths):
                elapsed_s = time.monotonic() - start_ts
                logger.info(
                    "source-label filter: processed %s/%s tiles kept=%s excluded=%s cache_hits=%s elapsed=%.1fs",
                    idx,
                    len(tile_paths),
                    len(filtered_paths),
                    excluded_count,
                    cache_hits,
                    elapsed_s,
                )

    if cache_dirty:
        try:
            _save_source_label_filter_cache(
                cache_path=cache_path,
                raster_signature=raster_signature,
                entries=cache_entries,
            )
        except OSError:
            logger.debug(
                "source-label filter cache save failed: %s",
                cache_path,
                exc_info=True,
            )
    if cache_hits:
        logger.info(
            "source-label filter cache: reused %s/%s tile decisions",
            cache_hits,
            len(tile_paths),
        )
    logger.info(
        "source-label filter: done kept=%s excluded=%s elapsed=%.1fs",
        len(filtered_paths),
        excluded_count,
        time.monotonic() - start_ts,
    )
    return filtered_paths, excluded_count


def _resolve_inference_tiles_for_shards(
    *,
    shard_root: Path,
) -> tuple[list[str], str | None, str]:
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
    source_label_raster = str(cfg.io.paths.source_label_raster or "").strip() or None
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
        filtered_tiles, excluded_count = _filter_tiles_by_source_label_presence(
            tile_paths=tiles,
            raster_path=source_label_raster,
            shard_root=shard_root,
        )
        if excluded_count:
            logger.info(
                "inference: excluded %s tiles with no SOURCE_LABEL_RASTER labels inside tile",
                excluded_count,
            )
        return filtered_tiles, tiles_dir, infer_tile_glob

    if explicit_tiles:
        filtered_tiles, excluded_count = _filter_tiles_by_source_label_presence(
            tile_paths=explicit_tiles,
            raster_path=source_label_raster,
            shard_root=shard_root,
        )
        if excluded_count:
            logger.info(
                "inference: excluded %s explicit tiles with no SOURCE_LABEL_RASTER labels inside tile",
                excluded_count,
            )
        return filtered_tiles, None, infer_tile_glob

    filtered_tiles, excluded_count = _filter_tiles_by_source_label_presence(
        tile_paths=legacy_holdout_tiles,
        raster_path=source_label_raster,
        shard_root=shard_root,
    )
    if excluded_count:
        logger.info(
            "inference: excluded %s legacy holdout tiles with no SOURCE_LABEL_RASTER labels inside tile",
            excluded_count,
        )
    return filtered_tiles, None, infer_tile_glob


def main() -> None:
    """CLI entrypoint for building shard tile lists."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
