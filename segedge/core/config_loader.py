"""Load and validate the runtime YAML configuration.

Examples:
    >>> isinstance(cfg.model.backbone.patch_size, int)
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class ThresholdRangeConfig:
    """Range specification for score thresholds."""

    start: float
    stop: float
    count: int

    @property
    def values(self) -> list[float]:
        """Expand to an explicit list of threshold values."""
        return [float(x) for x in np.linspace(self.start, self.stop, self.count)]


@dataclass
class IOPathsConfig:
    """Filesystem paths and tile lists."""

    source_tile: str
    target_tile: str
    source_label_raster: str
    eval_gt_vectors: list[str]
    source_tiles: list[str] | None
    val_tiles: list[str]
    holdout_tiles: list[str]
    feature_dir: str
    bank_cache_dir: str
    output_dir: str
    plot_dir: str
    best_settings_path: str
    log_path: str
    roads_mask_path: str | None = None


@dataclass
class IOAutoSplitConfig:
    """Auto split settings."""

    enabled: bool
    tiles_dir: str
    tile_glob: str
    val_split_fraction: float
    split_seed: int
    gt_presence_downsample: int | None
    gt_presence_workers: int | None


@dataclass
class IOConfig:
    """I/O config group."""

    paths: IOPathsConfig
    auto_split: IOAutoSplitConfig


@dataclass
class BackboneConfig:
    """Backbone model settings."""

    name: str
    patch_size: int
    resample_factor: int


@dataclass
class TilingConfig:
    """Tile traversal settings."""

    tile_size: int
    stride: int


@dataclass
class PriorsConfig:
    """Spatial priors and GT clipping settings."""

    buffer_m: float
    clip_gt_to_buffer: bool


@dataclass
class BanksConfig:
    """Feature bank settings."""

    pos_frac_thresh: float
    neg_alpha: float
    feat_context_radius: int
    bank_erosion_radius: int
    max_neg_bank: int


@dataclass
class AugmentationConfig:
    """Source-tile augmentation settings."""

    enabled: bool
    include_identity: bool
    horizontal_flip: bool
    vertical_flip: bool
    rotations_deg: list[int]


@dataclass
class ModelConfig:
    """Model and representation settings."""

    backbone: BackboneConfig
    tiling: TilingConfig
    priors: PriorsConfig
    banks: BanksConfig
    augmentation: AugmentationConfig


@dataclass
class LOOConfig:
    """Leave-one-out training settings."""

    enabled: bool
    min_train_tiles: int


@dataclass
class TrainingConfig:
    """Training strategy settings."""

    loo: LOOConfig


@dataclass
class KNNConfig:
    """kNN search settings."""

    k_values: list[int]
    thresholds: ThresholdRangeConfig
    use_fp16_knn: bool
    use_gpu_threshold_metrics: bool
    threshold_batch_size: int
    threshold_cpu_batch_size: int


@dataclass
class CRFConfig:
    """CRF search settings."""

    prob_softness_values: list[float]
    pos_w_values: list[float]
    pos_xy_std_values: list[float]
    bilateral_w_values: list[float]
    bilateral_xy_std_values: list[float]
    bilateral_rgb_std_values: list[float]
    num_workers: int
    max_configs: int


@dataclass
class XGBConfig:
    """XGBoost search settings."""

    use_gpu: bool
    val_fraction: float
    num_boost_round: int
    early_stop: int
    verbose_eval: int
    param_grid: list[dict[str, Any]]


@dataclass
class SearchConfig:
    """Search/tuning config group."""

    knn: KNNConfig
    crf: CRFConfig
    xgb: XGBConfig


@dataclass
class ShadowConfig:
    """Shadow filter settings."""

    weight_sets: list[tuple[float, float, float]]
    thresholds: list[float]
    protect_scores: list[float]


@dataclass
class RoadsConfig:
    """Roads penalty settings."""

    penalty_values: list[float]


@dataclass
class PostprocessConfig:
    """Postprocess config group."""

    shadow: ShadowConfig
    roads: RoadsConfig


@dataclass
class RuntimeConfig:
    """Runtime and diagnostics settings."""

    feature_cache_mode: str
    feature_batch_size: int
    resume_run: bool
    resume_run_dir: str | None
    union_backup_every: int
    union_backup_dir: str | None
    debug_timing: bool
    debug_timing_verbose: bool
    compact_timing_logs: bool


@dataclass
class Config:
    """Root runtime config."""

    io: IOConfig
    model: ModelConfig
    training: TrainingConfig
    search: SearchConfig
    postprocess: PostprocessConfig
    runtime: RuntimeConfig


def _require_mapping(value: Any, section: str) -> dict[str, Any]:
    """Return mapping value or raise a readable config error."""
    if not isinstance(value, dict):
        raise ValueError(f"config section '{section}' must be a mapping")
    return value


def _as_list_str(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list")
    return [str(v) for v in value]


def _as_list_float(value: Any, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list")
    return [float(v) for v in value]


def _as_list_int(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list")
    return [int(v) for v in value]


def _load_thresholds(data: dict[str, Any]) -> ThresholdRangeConfig:
    return ThresholdRangeConfig(
        start=float(data["start"]),
        stop=float(data["stop"]),
        count=int(data["count"]),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load the repository config YAML into a typed config object."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yml"
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    root = _require_mapping(raw, "root")
    io = _require_mapping(root["io"], "io")
    io_paths = _require_mapping(io["paths"], "io.paths")
    io_auto_split = _require_mapping(io["auto_split"], "io.auto_split")

    model = _require_mapping(root["model"], "model")
    model_backbone = _require_mapping(model["backbone"], "model.backbone")
    model_tiling = _require_mapping(model["tiling"], "model.tiling")
    model_priors = _require_mapping(model["priors"], "model.priors")
    model_banks = _require_mapping(model["banks"], "model.banks")
    model_augmentation = _require_mapping(
        model.get("augmentation", {}), "model.augmentation"
    )
    training = _require_mapping(root.get("training", {}), "training")
    training_loo = _require_mapping(training.get("loo", {}), "training.loo")

    search = _require_mapping(root["search"], "search")
    search_knn = _require_mapping(search["knn"], "search.knn")
    search_crf = _require_mapping(search["crf"], "search.crf")
    search_xgb = _require_mapping(search["xgb"], "search.xgb")

    postprocess = _require_mapping(root["postprocess"], "postprocess")
    post_shadow = _require_mapping(postprocess["shadow"], "postprocess.shadow")
    post_roads = _require_mapping(postprocess["roads"], "postprocess.roads")

    runtime = _require_mapping(root["runtime"], "runtime")

    io_paths_cfg = IOPathsConfig(
        source_tile=str(io_paths["source_tile"]),
        target_tile=str(io_paths["target_tile"]),
        source_label_raster=str(io_paths["source_label_raster"]),
        eval_gt_vectors=_as_list_str(
            io_paths["eval_gt_vectors"], "io.paths.eval_gt_vectors"
        ),
        source_tiles=(
            None
            if io_paths.get("source_tiles") is None
            else _as_list_str(io_paths["source_tiles"], "io.paths.source_tiles")
        ),
        val_tiles=_as_list_str(io_paths["val_tiles"], "io.paths.val_tiles"),
        holdout_tiles=_as_list_str(io_paths["holdout_tiles"], "io.paths.holdout_tiles"),
        feature_dir=str(io_paths["feature_dir"]),
        bank_cache_dir=str(io_paths["bank_cache_dir"]),
        output_dir=str(io_paths["output_dir"]),
        plot_dir=str(io_paths["plot_dir"]),
        best_settings_path=str(io_paths["best_settings_path"]),
        log_path=str(io_paths["log_path"]),
        roads_mask_path=(
            str(io_paths["roads_mask_path"])
            if io_paths.get("roads_mask_path") is not None
            else None
        ),
    )

    io_auto_split_cfg = IOAutoSplitConfig(
        enabled=bool(io_auto_split["enabled"]),
        tiles_dir=str(io_auto_split["tiles_dir"]),
        tile_glob=str(io_auto_split["tile_glob"]),
        val_split_fraction=float(io_auto_split["val_split_fraction"]),
        split_seed=int(io_auto_split["split_seed"]),
        gt_presence_downsample=(
            int(io_auto_split["gt_presence_downsample"])
            if io_auto_split.get("gt_presence_downsample") is not None
            else None
        ),
        gt_presence_workers=(
            int(io_auto_split["gt_presence_workers"])
            if io_auto_split.get("gt_presence_workers") is not None
            else None
        ),
    )

    model_cfg = ModelConfig(
        backbone=BackboneConfig(
            name=str(model_backbone["name"]),
            patch_size=int(model_backbone["patch_size"]),
            resample_factor=int(model_backbone["resample_factor"]),
        ),
        tiling=TilingConfig(
            tile_size=int(model_tiling["tile_size"]),
            stride=int(model_tiling["stride"]),
        ),
        priors=PriorsConfig(
            buffer_m=float(model_priors["buffer_m"]),
            clip_gt_to_buffer=bool(model_priors["clip_gt_to_buffer"]),
        ),
        banks=BanksConfig(
            pos_frac_thresh=float(model_banks["pos_frac_thresh"]),
            neg_alpha=float(model_banks["neg_alpha"]),
            feat_context_radius=int(model_banks["feat_context_radius"]),
            bank_erosion_radius=int(model_banks["bank_erosion_radius"]),
            max_neg_bank=int(model_banks["max_neg_bank"]),
        ),
        augmentation=AugmentationConfig(
            enabled=bool(model_augmentation.get("enabled", False)),
            include_identity=bool(model_augmentation.get("include_identity", True)),
            horizontal_flip=bool(model_augmentation.get("horizontal_flip", False)),
            vertical_flip=bool(model_augmentation.get("vertical_flip", False)),
            rotations_deg=_as_list_int(
                model_augmentation.get("rotations_deg", []),
                "model.augmentation.rotations_deg",
            ),
        ),
    )
    training_cfg = TrainingConfig(
        loo=LOOConfig(
            enabled=bool(training_loo.get("enabled", True)),
            min_train_tiles=int(training_loo.get("min_train_tiles", 1)),
        )
    )

    search_cfg = SearchConfig(
        knn=KNNConfig(
            k_values=[int(v) for v in search_knn["k_values"]],
            thresholds=_load_thresholds(
                _require_mapping(search_knn["thresholds"], "search.knn.thresholds")
            ),
            use_fp16_knn=bool(search_knn["use_fp16_knn"]),
            use_gpu_threshold_metrics=bool(search_knn["use_gpu_threshold_metrics"]),
            threshold_batch_size=int(search_knn["threshold_batch_size"]),
            threshold_cpu_batch_size=int(search_knn["threshold_cpu_batch_size"]),
        ),
        crf=CRFConfig(
            prob_softness_values=_as_list_float(
                search_crf["prob_softness_values"], "search.crf.prob_softness_values"
            ),
            pos_w_values=_as_list_float(
                search_crf["pos_w_values"], "search.crf.pos_w_values"
            ),
            pos_xy_std_values=_as_list_float(
                search_crf["pos_xy_std_values"], "search.crf.pos_xy_std_values"
            ),
            bilateral_w_values=_as_list_float(
                search_crf["bilateral_w_values"], "search.crf.bilateral_w_values"
            ),
            bilateral_xy_std_values=_as_list_float(
                search_crf["bilateral_xy_std_values"],
                "search.crf.bilateral_xy_std_values",
            ),
            bilateral_rgb_std_values=_as_list_float(
                search_crf["bilateral_rgb_std_values"],
                "search.crf.bilateral_rgb_std_values",
            ),
            num_workers=int(search_crf["num_workers"]),
            max_configs=int(search_crf["max_configs"]),
        ),
        xgb=XGBConfig(
            use_gpu=bool(search_xgb["use_gpu"]),
            val_fraction=float(search_xgb["val_fraction"]),
            num_boost_round=int(search_xgb["num_boost_round"]),
            early_stop=int(search_xgb["early_stop"]),
            verbose_eval=int(search_xgb["verbose_eval"]),
            param_grid=list(search_xgb["param_grid"]),
        ),
    )

    weight_sets_raw = post_shadow["weight_sets"]
    if not isinstance(weight_sets_raw, list):
        raise ValueError("'postprocess.shadow.weight_sets' must be a list")
    weight_sets = [tuple(float(x) for x in row) for row in weight_sets_raw]

    postprocess_cfg = PostprocessConfig(
        shadow=ShadowConfig(
            weight_sets=weight_sets,
            thresholds=_as_list_float(
                post_shadow["thresholds"], "postprocess.shadow.thresholds"
            ),
            protect_scores=_as_list_float(
                post_shadow["protect_scores"], "postprocess.shadow.protect_scores"
            ),
        ),
        roads=RoadsConfig(
            penalty_values=_as_list_float(
                post_roads["penalty_values"], "postprocess.roads.penalty_values"
            )
        ),
    )

    runtime_cfg = RuntimeConfig(
        feature_cache_mode=str(runtime["feature_cache_mode"]),
        feature_batch_size=int(runtime["feature_batch_size"]),
        resume_run=bool(runtime["resume_run"]),
        resume_run_dir=(
            str(runtime["resume_run_dir"])
            if runtime.get("resume_run_dir") is not None
            else None
        ),
        union_backup_every=int(runtime["union_backup_every"]),
        union_backup_dir=(
            str(runtime["union_backup_dir"])
            if runtime.get("union_backup_dir") is not None
            else None
        ),
        debug_timing=bool(runtime["debug_timing"]),
        debug_timing_verbose=bool(runtime["debug_timing_verbose"]),
        compact_timing_logs=bool(runtime.get("compact_timing_logs", True)),
    )

    return Config(
        io=IOConfig(paths=io_paths_cfg, auto_split=io_auto_split_cfg),
        model=model_cfg,
        training=training_cfg,
        search=search_cfg,
        postprocess=postprocess_cfg,
        runtime=runtime_cfg,
    )


cfg = load_config()
