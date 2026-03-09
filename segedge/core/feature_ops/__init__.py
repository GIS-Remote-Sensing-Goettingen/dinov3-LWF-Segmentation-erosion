"""Feature extraction building blocks split from the legacy features module."""

from .cache import (
    load_tile_features_if_valid,
    save_tile_features,
    tile_feature_meta_path,
    tile_feature_path,
)
from .extraction import (
    extract_patch_features_batch_single_scale,
    extract_patch_features_single_scale,
    prefetch_features_single_scale_image,
)
from .fusion import (
    _apply_xgb_stats_hwc,
    _compute_lbp_codes,
    _mode_weight,
    _rgb_to_hsv_image,
    add_local_context_mean,
    apply_xgb_feature_stats,
    deserialize_xgb_feature_stats,
    fit_xgb_feature_stats,
    fuse_patch_features,
    l2_normalize,
    serialize_xgb_feature_stats,
)
from .spec import hybrid_feature_spec_dict, hybrid_feature_spec_hash
from .tiling import crop_to_multiple_of_ps, labels_to_patch_masks, tile_iterator

__all__ = [
    "_apply_xgb_stats_hwc",
    "_compute_lbp_codes",
    "_mode_weight",
    "_rgb_to_hsv_image",
    "add_local_context_mean",
    "apply_xgb_feature_stats",
    "crop_to_multiple_of_ps",
    "deserialize_xgb_feature_stats",
    "extract_patch_features_batch_single_scale",
    "extract_patch_features_single_scale",
    "fit_xgb_feature_stats",
    "fuse_patch_features",
    "hybrid_feature_spec_dict",
    "hybrid_feature_spec_hash",
    "labels_to_patch_masks",
    "l2_normalize",
    "load_tile_features_if_valid",
    "prefetch_features_single_scale_image",
    "save_tile_features",
    "serialize_xgb_feature_stats",
    "tile_feature_meta_path",
    "tile_feature_path",
    "tile_iterator",
]
