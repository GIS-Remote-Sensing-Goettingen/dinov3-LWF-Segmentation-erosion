"""I/O helpers for raster/vector data and artifacts."""

from __future__ import annotations

import logging
import math
import os
import shutil

import fiona
import numpy as np
import rasterio
import rasterio.features as rfeatures
import yaml
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from skimage.morphology import dilation, disk
from skimage.transform import resize

from .config_loader import cfg
from .timing_utils import time_end, time_start

logger = logging.getLogger(__name__)
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


def load_dop20_image(path: str, downsample_factor: int = 1) -> np.ndarray:
    """Load a GeoTIFF orthophoto into an RGB array.

    Args:
        path (str): Path to a GeoTIFF image.
        downsample_factor (int): Integer downsample factor.

    Returns:
        np.ndarray: HxWx3 RGB array.

    Examples:
        >>> callable(load_dop20_image)
        True
    """
    t0 = time_start()
    with rasterio.open(path) as src:
        if downsample_factor > 1:
            out_h = src.height // downsample_factor
            out_w = src.width // downsample_factor
            arr = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.bilinear,
            )
        else:
            arr = src.read()
    img = reshape_as_image(arr)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    time_end(f"load_dop20_image[{os.path.basename(path)}]", t0)
    return img


def reproject_labels_to_image(
    ref_img_path: str, labels_path: str, downsample_factor: int = 1
) -> np.ndarray:
    """Reproject a raster label map onto a reference image grid.

    Args:
        ref_img_path (str): Reference image path.
        labels_path (str): Label raster path.
        downsample_factor (int): Integer downsample factor.

    Returns:
        np.ndarray: Reprojected label array.

    Examples:
        >>> callable(reproject_labels_to_image)
        True
    """
    t0 = time_start()
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
            count=src.count,
            width=dst_width,
            height=dst_height,
            transform=dst_transform,
        )
        memfile = MemoryFile()
        with memfile.open(**dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=ref.crs,
                    dst_width=dst_width,
                    dst_height=dst_height,
                    resampling=Resampling.nearest,
                )
            labels_arr = dst.read()
    labels_2d = labels_arr[0]
    expected_shape = (dst_height, dst_width)
    if labels_2d.shape != expected_shape:
        logger.warning(
            "labels raster shape %s != expected %s for %s; resizing to match",
            labels_2d.shape,
            expected_shape,
            labels_path,
        )
        labels_2d = resize(
            labels_2d,
            expected_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(labels_2d.dtype)
    time_end(
        f"reproject_labels_to_image[{os.path.basename(labels_path)} -> {os.path.basename(ref_img_path)}]",
        t0,
    )
    return labels_2d


def rasterize_vector_labels(
    vector_path: str | list[str],
    ref_raster_path: str,
    burn_value: int = 1,
    downsample_factor: int = 1,
) -> np.ndarray:
    """Rasterize one or more vector layers onto the reference raster grid.

    If vector_path is a list, each file is rasterized and combined with logical OR.

    Args:
        vector_path (str | list[str]): Vector path or list of paths.
        ref_raster_path (str): Reference raster path.
        burn_value (int): Burn value for rasterization.
        downsample_factor (int): Integer downsample factor.

    Returns:
        np.ndarray: Rasterized mask array.

    Examples:
        >>> callable(rasterize_vector_labels)
        True
    """
    t0 = time_start()
    vector_paths = vector_path if isinstance(vector_path, list) else [vector_path]
    with rasterio.open(ref_raster_path) as src:
        if downsample_factor > 1:
            out_shape = (
                src.height // downsample_factor,
                src.width // downsample_factor,
            )
            transform = src.transform * src.transform.scale(
                src.width / out_shape[1],
                src.height / out_shape[0],
            )
        else:
            out_shape = (src.height, src.width)
            transform = src.transform
        raster_crs = src.crs
    gt_mask = np.zeros(out_shape, dtype="uint8")

    for vp in vector_paths:
        with fiona.open(vp, "r") as shp:
            vec_crs = shp.crs
            if not vec_crs:
                logger.warning(
                    "vector CRS missing for %s; assuming EPSG:4326 (WGS84)", vp
                )
                vec_crs = CRS.from_epsg(4326).to_dict()
            transformer = None
            if raster_crs and vec_crs and vec_crs != raster_crs.to_dict():
                logger.info(
                    "reprojecting vector geometries from %s -> %s for %s",
                    vec_crs,
                    raster_crs.to_dict(),
                    vp,
                )
                transformer = Transformer.from_crs(
                    vec_crs, raster_crs.to_dict(), always_xy=True
                )
            shapes = []
            for feat in shp:
                geom = feat["geometry"]
                if transformer is not None:
                    geom_obj = shape(geom)
                    geom_obj = shp_transform(transformer.transform, geom_obj)
                    geom = mapping(geom_obj)
                shapes.append((geom, burn_value))

        if not shapes:
            logger.warning("no geometries found in %s", vp)
            continue

        mask_i = rfeatures.rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        gt_mask = np.maximum(gt_mask, mask_i)
    time_end("rasterize_vector_labels", t0)
    return gt_mask


def build_roads_raster_from_vector(
    vector_path: str,
    out_raster_path: str,
    bounds: tuple[float, float, float, float],
    pixel_size: tuple[float, float],
    target_crs,
) -> str:
    """Rasterize road vectors into a single mask GeoTIFF.

    Args:
        vector_path (str): Roads vector path.
        out_raster_path (str): Output GeoTIFF path.
        bounds (tuple[float, float, float, float]): Target bounds (left, bottom, right, top).
        pixel_size (tuple[float, float]): Pixel size (x, y).
        target_crs: Target CRS for rasterization.

    Returns:
        str: Output raster path.

    Examples:
        >>> callable(build_roads_raster_from_vector)
        True
    """
    t0 = time_start()
    left, bottom, right, top = bounds
    res_x, res_y = pixel_size
    if res_x <= 0 or res_y <= 0:
        raise ValueError("pixel_size must be positive")
    width = int(math.ceil((right - left) / res_x))
    height = int(math.ceil((top - bottom) / res_y))
    transform = from_origin(left, top, res_x, res_y)
    crs_obj = CRS.from_user_input(target_crs) if target_crs is not None else None

    shapes = []
    with fiona.open(vector_path, "r") as shp:
        vec_crs = shp.crs
        transformer = None
        if vec_crs and crs_obj is not None:
            vec_crs_obj = CRS.from_user_input(vec_crs)
            if vec_crs_obj != crs_obj:
                logger.info(
                    "reprojecting road geometries from %s -> %s for %s",
                    vec_crs_obj.to_string(),
                    crs_obj.to_string(),
                    vector_path,
                )
                transformer = Transformer.from_crs(vec_crs_obj, crs_obj, always_xy=True)
        for feat in shp:
            geom = feat.get("geometry")
            if not geom:
                continue
            geom_obj = shape(geom)
            if geom_obj.is_empty:
                continue
            if transformer is not None:
                geom_obj = shp_transform(transformer.transform, geom_obj)
            shapes.append((mapping(geom_obj), 1))

    if not shapes:
        raise ValueError(f"no geometries found in {vector_path}")

    mask = rfeatures.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=False,
    )

    os.makedirs(os.path.dirname(out_raster_path), exist_ok=True)
    with rasterio.open(
        out_raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs_obj,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(mask, 1)

    time_end("build_roads_raster_from_vector", t0)
    logger.info(
        "roads raster written: %s (shape=%s, features=%s)",
        out_raster_path,
        mask.shape,
        len(shapes),
    )
    return out_raster_path


def build_sh_buffer_mask(labels_sh: np.ndarray, buffer_pixels: int) -> np.ndarray:
    """Dilate SH_2022 raster by buffer_pixels using cv2 or skimage.

    Args:
        labels_sh (np.ndarray): SH label raster.
        buffer_pixels (int): Buffer radius in pixels.

    Returns:
        np.ndarray: Buffered mask.

    Examples:
        >>> import numpy as np
        >>> labels = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        >>> mask = build_sh_buffer_mask(labels, buffer_pixels=0)
        >>> mask.astype(int).tolist()
        [[0, 1], [0, 0]]
    """
    t0 = time_start()
    base = labels_sh > 0
    if buffer_pixels <= 0:
        time_end("build_sh_buffer_mask", t0)
        return base
    if _HAS_CV2 and cv2 is not None:
        ksize = 2 * buffer_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        buf = cv2.dilate(base.astype(np.uint8), kernel).astype(bool)
    else:
        buf = dilation(base.astype(bool), disk(buffer_pixels))
    time_end("build_sh_buffer_mask", t0)
    return buf


def export_mask_to_shapefile(mask: np.ndarray, ref_raster_path: str, out_path: str):
    """Vectorize a binary mask and save polygons to a shapefile.

    Args:
        mask (np.ndarray): Binary mask.
        ref_raster_path (str): Reference raster path for CRS/transform.
        out_path (str): Output shapefile path.

    Examples:
        >>> callable(export_mask_to_shapefile)
        True
    """
    t0 = time_start()
    mask_uint8 = mask.astype("uint8")
    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        crs = src.crs
    shape_generator = rfeatures.shapes(
        mask_uint8, mask=mask_uint8 == 1, transform=transform
    )
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with fiona.open(
        out_path,
        mode="w",
        driver="ESRI Shapefile",
        crs=crs.to_dict() if crs is not None else None,
        schema=schema,
    ) as shp:
        idx = 0
        for geom, value in shape_generator:
            if value != 1:
                continue
            shp.write({"geometry": geom, "properties": {"id": int(idx)}})
            idx += 1
    time_end("export_mask_to_shapefile", t0)
    logger.info("shapefile written to: %s", out_path)


def export_masks_to_shapefile_union(
    masks_with_refs: list[tuple[np.ndarray, str]], out_path: str
):
    """Write multiple masks into a single shapefile without dissolving overlaps.

    Args:
        masks_with_refs (list[tuple[np.ndarray, str]]): Mask and reference raster pairs.
        out_path (str): Output shapefile path.

    Examples:
        >>> callable(export_masks_to_shapefile_union)
        True
    """
    t0 = time_start()
    if not masks_with_refs:
        logger.warning("export_masks_to_shapefile_union: no masks provided")
        return
    _, first_ref = masks_with_refs[0]
    with rasterio.open(first_ref) as src:
        crs = src.crs
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    idx = 0
    with fiona.open(
        out_path,
        mode="w",
        driver="ESRI Shapefile",
        crs=crs.to_dict() if crs is not None else None,
        schema=schema,
    ) as shp:
        for mask, ref_raster_path in masks_with_refs:
            mask_uint8 = mask.astype("uint8")
            with rasterio.open(ref_raster_path) as src:
                transform = src.transform
            shape_generator = rfeatures.shapes(
                mask_uint8, mask=mask_uint8 == 1, transform=transform
            )
            for geom, value in shape_generator:
                if value != 1:
                    continue
                shp.write({"geometry": geom, "properties": {"id": int(idx)}})
                idx += 1
    time_end("export_masks_to_shapefile_union", t0)
    logger.info("union shapefile written to: %s (features=%s)", out_path, idx)


def append_mask_to_union_shapefile(
    mask: np.ndarray,
    ref_raster_path: str,
    out_path: str,
    start_id: int = 0,
) -> int:
    """Append a mask's polygons into a union shapefile.

    Args:
        mask (np.ndarray): Binary mask.
        ref_raster_path (str): Reference raster path for CRS/transform.
        out_path (str): Union shapefile path (created if missing).
        start_id (int): Starting feature id.

    Returns:
        int: Next feature id after appending.

    Examples:
        >>> callable(append_mask_to_union_shapefile)
        True
    """
    mask_uint8 = mask.astype("uint8")
    if mask_uint8.max() == 0:
        return start_id
    t0 = time_start()
    with rasterio.open(ref_raster_path) as src:
        transform = src.transform
        crs = src.crs
    shape_generator = rfeatures.shapes(
        mask_uint8, mask=mask_uint8 == 1, transform=transform
    )
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if os.path.exists(out_path) else "w"
    idx = start_id
    with fiona.open(
        out_path,
        mode=mode,
        driver="ESRI Shapefile",
        crs=crs.to_dict() if crs is not None else None,
        schema=schema,
    ) as shp:
        for geom, value in shape_generator:
            if value != 1:
                continue
            shp.write({"geometry": geom, "properties": {"id": int(idx)}})
            idx += 1
    time_end("append_mask_to_union_shapefile", t0)
    logger.info("union shapefile appended: %s (+%s features)", out_path, idx - start_id)
    return idx


def backup_union_shapefile(out_path: str, backup_dir: str, step: int) -> None:
    """Copy a union shapefile set to a backup directory.

    Args:
        out_path (str): Union shapefile path.
        backup_dir (str): Directory for backups.
        step (int): Step index for naming.

    Examples:
        >>> callable(backup_union_shapefile)
        True
    """
    if not os.path.exists(out_path):
        logger.warning("union shapefile missing; skipping backup")
        return
    os.makedirs(backup_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(out_path))[0]
    backup_base = os.path.join(backup_dir, f"{base}_step_{step:05d}")
    exts = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
    for ext in exts:
        src_path = os.path.splitext(out_path)[0] + ext
        if not os.path.exists(src_path):
            continue
        dst_path = backup_base + ext
        shutil.copy2(src_path, dst_path)
    logger.info("union shapefile backup written: %s", backup_base + ".shp")


def count_shapefile_features(path: str) -> int:
    """Count features in a shapefile (0 if missing).

    Args:
        path (str): Shapefile path.

    Returns:
        int: Number of features.

    Examples:
        >>> callable(count_shapefile_features)
        True
    """
    if not os.path.exists(path):
        return 0
    try:
        with fiona.open(path, "r") as shp:
            return sum(1 for _ in shp)
    except Exception:
        return 0


def consolidate_features_for_image(
    feature_dir: str, image_id: str, output_suffix: str = "_features_full.npy"
):
    """Concatenate all tile feature .npy files for an image into a single array.

    Args:
        feature_dir (str): Directory containing feature tiles.
        image_id (str): Image identifier.
        output_suffix (str): Output filename suffix.

    Returns:
        str | None: Path to consolidated feature array, or None if missing.

    Examples:
        >>> import numpy as np
        >>> import os
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     np.save(os.path.join(d, "img_y0_x0_features.npy"), np.zeros((1, 1, 2), dtype=np.float32))
        ...     np.save(os.path.join(d, "img_y0_x1_features.npy"), np.ones((1, 1, 2), dtype=np.float32))
        ...     out_path = consolidate_features_for_image(d, "img")
        ...     np.load(out_path).shape
        (2, 2)
    """
    t0 = time_start()
    if not os.path.isdir(feature_dir):
        logger.warning("feature_dir does not exist: %s", feature_dir)
        return None
    prefix = f"{image_id}_y"
    suffix = "_features.npy"
    files = [
        f
        for f in os.listdir(feature_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    if not files:
        logger.warning(
            "no feature tiles found for image_id=%s in %s", image_id, feature_dir
        )
        return None
    files = sorted(files)
    feats_list = []
    for fname in files:
        fpath = os.path.join(feature_dir, fname)
        arr = np.load(fpath)
        feats_list.append(arr.reshape(-1, arr.shape[-1]))
    feats_full = np.concatenate(feats_list, axis=0).astype(np.float32)
    out_path = os.path.join(feature_dir, f"{image_id}{output_suffix}")
    np.save(out_path, feats_full)
    time_end(f"consolidate_features_for_image[{image_id}]", t0)
    logger.info(
        "consolidated %s tiles for %s -> %s, shape=%s",
        len(files),
        image_id,
        out_path,
        feats_full.shape,
    )
    return out_path


def export_best_settings(
    best_raw_config,
    best_crf_config,
    model_name,
    img_path,
    img2_path,
    buffer_m,
    pixel_size_m,
    shadow_cfg=None,
    best_xgb_config: dict | None = None,
    champion_source: str | None = None,
    xgb_model_info: dict | None = None,
    model_info: dict | None = None,
    extra_settings: dict | None = None,
    best_settings_path: str | None = None,
):
    """Write a minimal YAML with the champion configurations and context.

    Args:
        best_raw_config (dict): Best raw configuration.
        best_crf_config (dict): Best CRF configuration.
        model_name (str): Model name.
        img_path (str): Image A path.
        img2_path (str): Image B path.
        buffer_m (float): Buffer in meters.
        pixel_size_m (float): Pixel size in meters.
        shadow_cfg (dict | None): Shadow filter configuration.
        extra_settings (dict | None): Extra settings to write.
        best_settings_path (str | None): Output path override.

    Examples:
        >>> import os
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "best.yml")
        ...     export_best_settings(
        ...         {"k": 1},
        ...         {"prob_softness": 0.1},
        ...         "model",
        ...         "a.tif",
        ...         "b.tif",
        ...         8.0,
        ...         0.2,
        ...         best_settings_path=path,
        ...     )
        ...     os.path.exists(path)
        True
    """
    best_settings = {
        "best_raw_config": best_raw_config,
        "best_crf_config": best_crf_config,
        "best_shadow_config": shadow_cfg,
        "best_xgb_config": best_xgb_config,
        "champion_source": champion_source,
        "xgb_model_info": xgb_model_info,
        "model_info": model_info,
        "model_name": model_name,
        "img_a": img_path,
        "img_b": img2_path,
        "buffer_m": buffer_m,
        "pixel_size_m": pixel_size_m,
    }
    if extra_settings:
        best_settings["extra"] = extra_settings
    out_path = best_settings_path or cfg.io.paths.best_settings_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _normalize(value):
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [_normalize(v) for v in value]
        if isinstance(value, tuple):
            return [_normalize(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _normalize(best_settings),
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        )
    logger.info("best settings written to %s", out_path)
