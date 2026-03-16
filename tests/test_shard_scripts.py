"""Tests for shard-building and shard-union merge scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

_BUILD_PATH = (
    Path(__file__).resolve().parents[1] / "deployment" / "build_inference_shards.py"
)
_BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_inference_shards", _BUILD_PATH
)
assert _BUILD_SPEC is not None
assert _BUILD_SPEC.loader is not None
_BUILD_MODULE = importlib.util.module_from_spec(_BUILD_SPEC)
sys.modules[_BUILD_SPEC.name] = _BUILD_MODULE
_BUILD_SPEC.loader.exec_module(_BUILD_MODULE)

_MERGE_PATH = (
    Path(__file__).resolve().parents[1] / "deployment" / "merge_shard_unions.py"
)
_MERGE_SPEC = importlib.util.spec_from_file_location("merge_shard_unions", _MERGE_PATH)
assert _MERGE_SPEC is not None
assert _MERGE_SPEC.loader is not None
_MERGE_MODULE = importlib.util.module_from_spec(_MERGE_SPEC)
sys.modules[_MERGE_SPEC.name] = _MERGE_MODULE
_MERGE_SPEC.loader.exec_module(_MERGE_MODULE)

build_inference_shards = _BUILD_MODULE.build_inference_shards
merge_shard_unions = _MERGE_MODULE.merge_shard_unions


def _write_union_raster(path: Path, pixels: np.ndarray) -> None:
    """Write a tiny GeoTIFF union raster for one stage variant.

    Examples:
        >>> True
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=pixels.shape[1],
        height=pixels.shape[0],
        count=1,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=from_origin(0.0, float(pixels.shape[0]), 1.0, 1.0),
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(np.asarray(pixels, dtype=np.uint8), 1)


def test_build_inference_shards_writes_manifest_and_round_robin_files(
    tmp_path,
    monkeypatch,
):
    """Shard builder should write deterministic round-robin tile lists.

    Examples:
        >>> True
        True
    """
    monkeypatch.setattr(
        _BUILD_MODULE,
        "_resolve_inference_tiles_for_shards",
        lambda **_kwargs: (
            ["tile_00.tif", "tile_01.tif", "tile_02.tif", "tile_03.tif"],
            "/tmp/tiles",
            "*.tif",
        ),
    )
    monkeypatch.setattr(
        _BUILD_MODULE.cfg.io.paths,
        "source_label_raster",
        "/tmp/source_labels.tif",
    )

    shard_root, shard_files = build_inference_shards(
        shard_count=2,
        output_dir=str(tmp_path / "shards"),
        job_name="demo",
    )

    assert shard_root == tmp_path / "shards" / "demo"
    assert [path.name for path in shard_files] == [
        "tiles_shard_000.txt",
        "tiles_shard_001.txt",
    ]
    assert shard_files[0].read_text(encoding="utf-8").splitlines() == [
        "tile_00.tif",
        "tile_02.tif",
    ]
    assert shard_files[1].read_text(encoding="utf-8").splitlines() == [
        "tile_01.tif",
        "tile_03.tif",
    ]

    manifest = json.loads((shard_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["assignment_method"] == "round_robin"
    assert manifest["num_shards"] == 2
    assert manifest["total_tiles"] == 4
    assert manifest["tiles_file_paths"] == [str(path) for path in shard_files]


def test_merge_shard_unions_combines_stage_union_files(tmp_path):
    """Union merge should OR all shard pixels for each stage variant.

    Examples:
        >>> True
        True
    """
    shard_a = tmp_path / "run_a"
    shard_b = tmp_path / "run_b"
    for variant in ("raw", "shadow_with_proposals"):
        _write_union_raster(
            shard_a / "shapes" / "unions" / variant / "union.tif",
            np.array([[1, 0]], dtype=np.uint8),
        )
        _write_union_raster(
            shard_b / "shapes" / "unions" / variant / "union.tif",
            np.array([[0, 1]], dtype=np.uint8),
        )

    merged = merge_shard_unions(
        shard_run_dirs=[str(shard_a), str(shard_b)],
        output_dir=str(tmp_path / "merged"),
    )

    merged_names = sorted(path.parent.name for path in merged)
    assert merged_names == ["raw", "shadow_with_proposals"]

    raw_union = tmp_path / "merged" / "raw" / "union.tif"
    with rasterio.open(raw_union) as src:
        merged_pixels = src.read(1)
    np.testing.assert_array_equal(merged_pixels, np.array([[1, 1]], dtype=np.uint8))
