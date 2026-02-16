# SegEdge zero-shot pipeline configuration.
import logging

# Organized by: IO paths, model/tiling, scoring/search, evaluation/runtime.

# -----------------------------------------------------------------------------
# 1) IO paths and datasets
# -----------------------------------------------------------------------------

# Single-tile fallbacks (used when *_TILES lists are None).
SOURCE_TILE = "data/tiles/dop20_596000_5974000_1km_20cm.tif"
TARGET_TILE = "data/dop20_592000_5982000_1km_20cm.tif"

# Tile discovery + auto split (optional)
TILES_DIR = "/mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt/folder_1"
TILE_GLOB = "*.tif"
AUTO_SPLIT_TILES = True
# Auto-split mode:
# - "gt_to_val_cap_holdout": all GT-overlap tiles are validation; source comes from
#   SOURCE_TILES; holdout (non-GT) can be capped.
# - "legacy_gt_source_val_holdout": split GT-overlap tiles into source/validation.
# For XGB generalization, prefer legacy mode so training uses most GT-overlap tiles.
AUTO_SPLIT_MODE = "legacy_gt_source_val_holdout"
VAL_SPLIT_FRACTION = 0.2
SPLIT_SEED = 42
# Holdout cap (applies in gt_to_val_cap_holdout mode).
INFERENCE_TILE_CAP_ENABLED = True
INFERENCE_TILE_CAP = 50
INFERENCE_TILE_CAP_SEED = 42
# Downsample factor for GT presence checks (None uses RESAMPLE_FACTOR).
GT_PRESENCE_DOWNSAMPLE = None
# Worker count for GT presence checks (None uses SLURM_CPUS_PER_TASK or os.cpu_count()).
GT_PRESENCE_WORKERS = None

# Label raster used to build banks on SOURCE_* tiles.
SOURCE_LABEL_RASTER = "data/lables/planet_labels_2022.tif"
# Source supervision mode for kNN-bank and XGB training labels:
# - "gt_if_available": use GT vectors when available, otherwise SOURCE_LABEL_RASTER
# - "gt_only": require GT vectors for source supervision
# - "source_raster": always use SOURCE_LABEL_RASTER
SOURCE_SUPERVISION_MODE = "gt_if_available"
# Optional override of GT vectors used for source supervision; None uses EVAL_GT_VECTORS.
SOURCE_TRAIN_GT_VECTORS = None
# Minimum GT-positive pixels required before accepting GT supervision in gt_if_available mode.
SOURCE_SUPERVISION_MIN_POS_PIXELS = 1
# Anti-leak checks between source training and validation.
ANTI_LEAK_CHECKS_ENABLED = True
# Raise instead of warn when anti-leak checks detect problems.
ANTI_LEAK_FAIL_FAST = True
# Spatial overlap threshold (relative to smaller tile area) to flag source/validation overlap.
ANTI_LEAK_TILE_OVERLAP_MIN_RATIO = 0.0
# Limit number of overlapping source/validation tile pairs scanned/logged.
ANTI_LEAK_MAX_LOGGED_PAIRS = 10
# Auto-derive source-train GT vectors from EVAL_GT_VECTORS when SOURCE_TRAIN_GT_VECTORS is unset.
AUTO_DERIVE_SOURCE_TRAIN_GT_VECTORS = True
# Exclusion buffer (meters) around validation footprint removed from source-train GT derivation.
AUTO_GT_DERIVE_EXCLUSION_BUFFER_M = 1.0
# Optional area filter after clipping derived GT geometries (0 disables).
AUTO_GT_DERIVE_MIN_GEOM_AREA_M2 = 0.0
# Optionally write a clipped eval GT copy into the run's derived_gt directory.
AUTO_GT_DERIVE_WRITE_EVAL_COPY = False

# Evaluation GT vectors (union-merged). Use EVAL_GT_VECTORS when available,
EVAL_GT_VECTORS = [
    "data/lables/lables_1.shp",
    "data/lables/lables_2.shp",
    "data/lables/lables_3.shp",
    "data/lables/labels_final.shp",
]

# Multi-tile inputs (set to None to use SOURCE_TILE).
# When AUTO_SPLIT_TILES=True, these are overridden by the auto split.
SOURCE_TILES = [
    "data/tiles/dop20_596000_5974000_1km_20cm.tif",
    "data/tiles/dop20_596000_5975000_1km_20cm.tif",
    "data/tiles/dop20_596000_5976000_1km_20cm.tif",
    "data/tiles/dop20_596000_5977000_1km_20cm.tif",
    "data/tiles/dop20_596000_5983000_1km_20cm.tif",
]

# Validation tiles (first is used for tuning; the rest are evaluated with fixed settings).
# When AUTO_SPLIT_TILES=True, these are overridden by the auto split.
VAL_TILES = [
    "data/tiles/dop20_596000_5974000_1km_20cm.tif",
    "data/tiles/dop20_596000_5977000_1km_20cm.tif",
    "data/tiles/dop20_596000_5983000_1km_20cm.tif",
]
# Holdout tiles (used for final inference). Overridden by auto split.
HOLDOUT_TILES = [
    "data/tiles/dop20_596000_5975000_1km_20cm.tif",
    "data/tiles/dop20_596000_5976000_1km_20cm.tif",
]


# Output locations (everything except DINO features).
# Persist DINO feature/tile cache under project storage for long runs.
FEATURE_DIR = "/mnt/ceph-hdd/projects/mthesis_davide_mattioli/dino_features"
BANK_CACHE_DIR = "/mnt/ceph-hdd/projects/mthesis_davide_mattioli/dino_features/banks"
OUTPUT_DIR = "output"
PLOT_DIR = "output/plots"
TUNING_PLOT_MAX_TILES = 10
BEST_SETTINGS_PATH = "output/best_settings.yml"
LOG_PATH = "output/run.log"
LOG_LEVEL = logging.INFO
DEBUG_REPROJECT = True
# Incremental timing telemetry CSV outputs.
TIMING_CSV_ENABLED = True
TIMING_CSV_FILENAME = "tile_phase_timing.csv"
TIMING_SUMMARY_CSV_FILENAME = "timing_opportunity_cost.csv"
# Flush cadence for detailed rows (1 updates files after every completed tile).
TIMING_CSV_FLUSH_EVERY = 1
# Runtime timing logs.
DEBUG_TIMING = True
DEBUG_TIMING_VERBOSE = False
# Prefer CUDA bilinear interpolation for patch->pixel resize hot paths (kNN/XGB).
USE_GPU_RESIZE = True
# If False, suppress per-tile timing lines and keep image-level timing summaries.
TIMING_TILE_LOGS = False
# Explainability outputs (Tier 1, no SHAP dependency).
XAI_ENABLED = True
XAI_SAVE_JSON = True
XAI_SAVE_PLOTS = True
XAI_DIRNAME = "xai"
XAI_SUMMARY_FILENAME = "xai_summary.csv"
XAI_TOP_FEATURES = 20
XAI_TOP_PATCHES = 50
XAI_INCLUDE_XGB = True
XAI_INCLUDE_KNN = True
XAI_HOLDOUT_CAP_ENABLED = True
XAI_HOLDOUT_CAP = 10
XAI_HOLDOUT_CAP_SEED = 42
# Post-training PCA overlays for XGB-relevant embedding components.
XAI_PCA_XGB_ENABLED = True
XAI_PCA_SOURCE_TILE_INDEX = 0
XAI_PCA_TOP_COMPONENTS = 5
XAI_PCA_MAX_COMPONENTS = 32
# Resume previous run (requires RESUME_RUN_DIR).
RESUME_RUN = False
RESUME_RUN_DIR = None
# Rolling union shapefile backups (0 disables backups).
UNION_BACKUP_EVERY = 10
UNION_BACKUP_DIR = None

# Feature cache mode: "disk" caches tiles to FEATURE_DIR, "memory" reuses per-image in RAM.
FEATURE_CACHE_MODE = "disk"  # "disk" | "memory"
# If True, prefetch/cache source features only for source tiles overlapping GT vectors.
SOURCE_PREFETCH_GT_ONLY = True
# Batch size for feature extraction (1 disables batching).
FEATURE_BATCH_SIZE = 4

# -----------------------------------------------------------------------------
# 2) Model + tiling
# -----------------------------------------------------------------------------
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
RESAMPLE_FACTOR = 1  # 3x downsample: 0.2 m/px -> 0.6 m/px
PATCH_SIZE = 16  # ViT patch size
TILE_SIZE = 2048
STRIDE = 512
BUFFER_M = 5.0  # spatial prior buffer (meters)

# Patch labeling for banks.
POS_FRAC_THRESH = 0.05  # patch is positive if FG fraction >= this
NEG_ALPHA = 1.0  # kNN negative bank weight
# Global cap for negative samples in banks/XGB dataset building.
MAX_NEG_BANK = 40000

# Optional patch-context smoothing (applies to banks, kNN, XGB).
FEAT_CONTEXT_RADIUS = 0  # 0 disables; try 1 or 2 for more context

# Bank label erosion (in pixels on the resampled grid). Set to 0 to keep thin positives.
BANK_EROSION_RADIUS = 0

# -----------------------------------------------------------------------------
# 3) Scoring/search
# -----------------------------------------------------------------------------

# kNN grid search
# K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 50, 75, 100, 150, 200, 300, 500]
K_VALUES = [175, 200, 250]
THRESHOLDS = [float(x) for x in __import__("numpy").linspace(0.01, 0.9, 100)]

# CRF grid search
PROB_SOFTNESS_VALUES = [0.08, 0.12, 0.2]
POS_W_VALUES = [1.0, 2.0, 3.0]
POS_XY_STD_VALUES = [3.0, 5.0]
BILATERAL_W_VALUES = [1.0, 3.0, 5.0]
BILATERAL_XY_STD_VALUES = [25.0]
BILATERAL_RGB_STD_VALUES = [3.0, 5.0]
CRF_NUM_WORKERS = 16

# Shadow filtering (RGB weighted sum) after CRF
SHADOW_WEIGHT_SETS = [
    (1.0, 1.0, 1.0),
    (0.7, 1.0, 1.0),
]
SHADOW_THRESHOLDS = [
    20,
    40,
    60,
    80,
    100,
    120,
]
SHADOW_PROTECT_SCORES = [0.3, 0.5]

# Roads mask penalty (multiplicative) for kNN/XGB scores
ROADS_MASK_PATH = "data/roads/roads_mask.shp"
ROADS_PENALTY_VALUES = [0.8, 0.7, 0.6]
ROADS_SIMPLIFY_TOLERANCE_M = 0.2

# Adaptive top-p selection inside buffer
TOP_P_A = 0.2
TOP_P_B = 0.04
TOP_P_MIN = 0.02
TOP_P_MAX = 0.12
TOP_P_A_VALUES = [0.0, 0.2, 0.4]
TOP_P_B_VALUES = [0.02, 0.04, 0.06]
TOP_P_MIN_VALUES = [0.02, 0.03]
TOP_P_MAX_VALUES = [0.06, 0.08, 0.1, 0.12]
SILVER_CORE_DILATE_PX = 1
SILVER_CORE_DILATE_PX_VALUES = [0, 1, 2]

# Tuning strategy
# - "grid": exhaustive Cartesian search (legacy behavior).
# - "bayes": staged Optuna TPE search with optional perturbation robustness.
TUNING_MODE = "bayes"

# Bayesian optimization controls
BO_SEED = 42
BO_STAGE1_TRIALS = 40
BO_STAGE2_TRIALS = 20
BO_STAGE3_TRIALS = 20
BO_TIMEOUT_S = None
BO_ENABLE_PRUNING = True
BO_EARLY_STOP_PATIENCE = 20
BO_EARLY_STOP_MIN_DELTA = 0.0
BO_N_STARTUP_TRIALS = 12
BO_PRUNER_MIN_RESOURCE = 1
BO_PRUNER_REDUCTION_FACTOR = 2
BO_PRUNER_WARMUP_STEPS = 1
BO_SAMPLER = "tpe"  # "tpe" | "cmaes"
BO_TPE_MULTIVARIATE = True
BO_TPE_GROUP = True
BO_STORAGE_PATH = "output/optuna_tuning.db"
BO_STUDY_NAME = "segedge_tuning"
BO_STUDY_TAG = "v2_ranges"
# If True, force a fresh study namespace for each run (prevents accidental resume).
BO_FORCE_NEW_STUDY = True
# Optional explicit run tag for fresh studies (None => auto timestamp token).
BO_STUDY_RUN_TAG = None
# If True, print clearer trial separators in Bayes tuning logs.
BO_VERBOSE_TRIAL_SEPARATORS = True
BO_IMPORTANCE_FILENAME = "bayes_hyperparam_importances.json"
BO_IMPORTANCE_CSV_FILENAME = "bayes_hyperparam_importances.csv"
BO_TRIALS_CSV_FILENAME = "bayes_trials_timeseries.csv"
BO_TRIAL_PHASE_TIMING_CSV_FILENAME = "bayes_trial_phase_timing.csv"
# Robust objective = w_gt * IoU_GT + w_sh * IoU_SH
BO_OBJECTIVE_W_GT = 0.8
BO_OBJECTIVE_W_SH = 0.2
BO_PERTURBATIONS_PER_TILE = 1
BO_PERTURB_SEED = 42
# Keep tuning logs compact and append a single timing suffix per trial line.
BO_TIMING_SUMMARY_LOG = True
BO_TIMING_LOG_TOP_PHASES = 1
BO_TIMING_LOG_MIN_PHASE_S = 0.5
# If False, suppress kNN/XGB image-level timing logs inside Bayes tuning loops.
BO_EMIT_COMPONENT_TIMING_LOGS = False
# Optional dynamic threshold calibration (validation objective only).
BO_USE_DYNAMIC_F1_THRESHOLD = False
BO_DYNAMIC_THRESHOLD_BINS = 64
# Stage-2 refinement controls.
BO_STAGE2_TOP_N = 10
BO_STAGE2_BROAD_FRACTION = 0.6
# Bridge optimization gate.
BO_TUNE_BRIDGE = False

# Range-first Optuna search space (if set, takes precedence over *_VALUES).
BO_K_RANGE = (10, 300)
BO_NEG_ALPHA_RANGE = (0.1, 5.0)
BO_ROADS_PENALTY_RANGE = (0.5, 1.0)
BO_TOP_P_A_RANGE = (0.0, 1.0)
BO_TOP_P_B_RANGE = (0.01, 0.15)
BO_TOP_P_MIN_RANGE = (0.01, 0.06)
BO_TOP_P_MAX_RANGE = (0.05, 0.2)
BO_SILVER_CORE_DILATE_PX_RANGE = (0, 3, 1)
BO_CRF_PROB_SOFTNESS_RANGE = (0.05, 0.5)
BO_CRF_POS_W_RANGE = (0.5, 8.0)
BO_CRF_POS_XY_STD_RANGE = (1.0, 10.0)
BO_CRF_BILATERAL_W_RANGE = (0.5, 10.0)
BO_CRF_BILATERAL_XY_STD_RANGE = (10.0, 100.0)
BO_CRF_BILATERAL_RGB_STD_RANGE = (1.0, 20.0)
BO_SHADOW_THRESHOLD_RANGE = (10, 140, 5)
BO_SHADOW_PROTECT_SCORE_RANGE = (0.1, 0.9)
BO_BRIDGE_MAX_GAP_PX_RANGE = (100, 600, 50)
BO_BRIDGE_MAX_PAIRS_RANGE = (1, 5, 1)
BO_BRIDGE_MAX_AVG_COST_RANGE = (0.2, 1.0)
BO_BRIDGE_WIDTH_PX_RANGE = (80, 240, 20)
BO_BRIDGE_MIN_COMPONENT_PX_RANGE = (50, 300, 10)
BO_BRIDGE_SPUR_PRUNE_ITERS_RANGE = (5, 25, 1)

# Continuity bridging (post-CRF, pre-shadow)
ENABLE_GAP_BRIDGING = True
BRIDGE_MAX_GAP_PX = 500
BRIDGE_MAX_PAIRS = 3
BRIDGE_MAX_AVG_COST = 0.5
BRIDGE_WIDTH_PX = 200
BRIDGE_MIN_COMPONENT_PX = 100
BRIDGE_SPUR_PRUNE_ITERS = 15
BRIDGE_MAX_GAP_PX_VALUES = [200, 350, 500]
BRIDGE_MAX_PAIRS_VALUES = [2, 3, 4]
BRIDGE_MAX_AVG_COST_VALUES = [0.4, 0.5, 0.7]
BRIDGE_WIDTH_PX_VALUES = [120, 160, 200]
BRIDGE_MIN_COMPONENT_PX_VALUES = [80, 100, 150]
BRIDGE_SPUR_PRUNE_ITERS_VALUES = [10, 15, 20]

# XGBoost (patch classifier)
XGB_USE_GPU = True
XGB_VAL_FRACTION = 0.2
XGB_NUM_BOOST_ROUND = 15
XGB_EARLY_STOP = 100
XGB_VERBOSE_EVAL = 20
# Optional k-fold selection for XGB candidate robustness checks.
XGB_USE_KFOLD = True
XGB_KFOLD_SPLITS = 3
XGB_KFOLD_GROUP_BY_TILE = True
XGB_KFOLD_SHUFFLE = True
# Candidate screening controls (pre-grid/Bayes downstream scoring).
XGB_CANDIDATE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
XGB_SELECTION_USE_ALL_VAL_TILES = True
# Final refit after candidate selection: best_round_mean | holdout_es | fixed_rounds.
XGB_REFIT_MODE = "best_round_mean"
XGB_REFIT_HOLDOUT_FRACTION = 0.1
# Override patch-purity threshold for XGB supervision (None => POS_FRAC_THRESH).
XGB_POS_FRAC_THRESH = 0.1
# Mark a patch as negative only if positive fraction is <= this threshold.
# Values > 0 add a strict "ignore band" between negatives and positives.
XGB_NEG_FRAC_MAX = 0.0
# Class-imbalance controls for XGB training.
XGB_USE_SCALE_POS_WEIGHT = True
XGB_USE_SAMPLE_WEIGHTS = True
# Clamp auto class weight ratio to avoid unstable overweighting on sparse GT tiles.
XGB_CLASS_WEIGHT_MAX = 25.0
# Warn when per-tile or merged XGB datasets are too sparse in positives.
XGB_MIN_POS_SAMPLES_WARN = 200
XGB_MIN_POS_RATIO_WARN = 0.02
# Optional per-tile cap before global MAX_NEG_BANK subsampling (0 disables).
XGB_MAX_NEG_PER_TILE = 5000
XGB_PARAM_GRID = [
    # 1) Baseline
    {
        "max_depth": 6,
        "eta": 0.05,
        "colsample_bytree": 0.3,
        "subsample": 0.9,
        "reg_alpha": 0.05,
        "min_child_weight": 1,
    },
    # 2) Deep & slow (needs more rounds)
    {
        "max_depth": 7,
        "eta": 0.03,
        "colsample_bytree": 0.3,
        "subsample": 0.9,
        "reg_alpha": 0.05,
        "min_child_weight": 1,
    },
    # 3) More regularized
    {
        "max_depth": 6,
        "eta": 0.05,
        "colsample_bytree": 0.3,
        "subsample": 0.9,
        "reg_alpha": 0.1,
        "min_child_weight": 1,
    },
]

# -----------------------------------------------------------------------------
# 4) Evaluation and runtime
# -----------------------------------------------------------------------------

# If True, evaluation ignores GT outside the SH buffer (upper bound can reach 1.0).
CLIP_GT_TO_BUFFER = True

# Split evaluation settings (legacy).
