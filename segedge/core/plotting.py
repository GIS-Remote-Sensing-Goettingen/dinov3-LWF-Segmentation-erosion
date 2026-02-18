"""Plotting helpers for SegEdge outputs."""

from __future__ import annotations

import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import binary_dilation, binary_erosion

logger = logging.getLogger(__name__)


def _normalize_heatmap(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a heatmap to [0, 1].

    Examples:
        >>> import numpy as np
        >>> _normalize_heatmap(np.array([[1.0, 3.0]])).shape
        (1, 2)
    """
    vals = values.astype(np.float32)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax - vmin <= 1e-8:
        return np.zeros_like(vals)
    return (vals - vmin) / (vmax - vmin)


def _overlay_mask(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a binary mask onto an RGB image.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((2, 2, 3), dtype=np.uint8)
        >>> out = _overlay_mask(img, np.array([[1, 0], [0, 0]]), (255, 0, 0))
        >>> out.shape
        (2, 2, 3)
    """
    out = img_rgb.copy()
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return out
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out_f = out.astype(np.float32)
    out_f[mask_bool] = (1.0 - alpha) * out_f[mask_bool] + alpha * c
    return np.clip(out_f, 0, 255).astype(img_rgb.dtype)


def save_core_qualitative_plot(
    img_b: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    plot_dir: str,
    image_id_b: str,
    gt_available: bool,
    boundary_band_px: int = 10,
) -> None:
    """Save core 4-panel qualitative diagnostics for one tile.

    Examples:
        >>> callable(save_core_qualitative_plot)
        True
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].imshow(img_b)
    axs[0, 0].set_title("RGB")
    axs[0, 0].axis("off")

    if gt_available:
        gt_overlay = _overlay_mask(img_b, gt_mask, (0, 0, 255), alpha=0.45)
        axs[0, 1].imshow(gt_overlay)
        axs[0, 1].set_title("GT overlay")
    else:
        axs[0, 1].imshow(np.zeros(img_b.shape[:2], dtype=np.uint8), cmap="gray")
        axs[0, 1].set_title("GT unavailable")
    axs[0, 1].axis("off")

    pred_overlay = _overlay_mask(img_b, pred_mask, (255, 0, 0), alpha=0.45)
    axs[1, 0].imshow(pred_overlay)
    axs[1, 0].set_title("Prediction overlay")
    axs[1, 0].axis("off")

    if gt_available:
        gt_bool = gt_mask.astype(bool)
        pred_bool = pred_mask.astype(bool)
        boundary = np.logical_xor(gt_bool, binary_erosion(gt_bool))
        if boundary_band_px > 0:
            band = binary_dilation(boundary, iterations=int(boundary_band_px))
        else:
            band = boundary
        fp = np.logical_and(np.logical_and(pred_bool, ~gt_bool), band)
        fn = np.logical_and(np.logical_and(~pred_bool, gt_bool), band)
        err_overlay = img_b.copy()
        err_overlay = _overlay_mask(err_overlay, fp, (255, 0, 0), alpha=0.65)
        err_overlay = _overlay_mask(err_overlay, fn, (0, 0, 255), alpha=0.65)
        axs[1, 1].imshow(err_overlay)
        axs[1, 1].set_title("Boundary-band errors (FP red / FN blue)")
    else:
        axs[1, 1].imshow(img_b)
        axs[1, 1].set_title("Boundary-band errors unavailable")
    axs[1, 1].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{image_id_b}_qualitative_core.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved core qualitative to %s", out_path)


def save_score_threshold_plot(
    score_map: np.ndarray,
    threshold: float,
    sh_buffer_mask: np.ndarray,
    plot_dir: str,
    image_id_b: str,
) -> None:
    """Save score heatmap with threshold and in/out buffer histogram inset.

    Examples:
        >>> callable(save_score_threshold_plot)
        True
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    norm = _normalize_heatmap(score_map)
    im = ax.imshow(norm, cmap="magma")
    ax.set_title(f"Champion score map (thr={threshold:.3f})")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("normalized score", rotation=270, labelpad=12)

    in_vals = score_map[sh_buffer_mask.astype(bool)].ravel()
    out_vals = score_map[~sh_buffer_mask.astype(bool)].ravel()
    ins = inset_axes(ax, width="38%", height="38%", loc="lower left", borderpad=1.2)
    if in_vals.size > 0:
        ins.hist(in_vals, bins=40, alpha=0.6, color="tab:blue", label="in-buffer")
    if out_vals.size > 0:
        ins.hist(out_vals, bins=40, alpha=0.6, color="tab:orange", label="out-buffer")
    ins.axvline(float(threshold), color="white", linestyle="--", linewidth=1.2)
    ins.set_facecolor((0.1, 0.1, 0.1, 0.8))
    ins.tick_params(axis="both", labelsize=7, colors="white")
    for spine in ins.spines.values():
        spine.set_color("white")
    if in_vals.size > 0 or out_vals.size > 0:
        ins.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{image_id_b}_score_threshold.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved score threshold to %s", out_path)


def save_disagreement_entropy_plot(
    disagreement_map: np.ndarray,
    entropy_map: np.ndarray,
    candidate_mask: np.ndarray,
    plot_dir: str,
    image_id_b: str,
) -> None:
    """Save disagreement and entropy diagnostics with proposal contour.

    Examples:
        >>> callable(save_disagreement_entropy_plot)
        True
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    panels = [
        ("|score_xgb - score_knn|", disagreement_map, "inferno"),
        ("Entropy uncertainty", entropy_map, "viridis"),
    ]
    for ax, (title, data, cmap) in zip(axs, panels, strict=True):
        norm = _normalize_heatmap(data)
        im = ax.imshow(norm, cmap=cmap)
        if candidate_mask is not None and candidate_mask.any():
            ax.contour(
                candidate_mask.astype(np.uint8),
                levels=[0.5],
                colors="yellow",
                linewidths=0.6,
            )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{image_id_b}_disagreement_entropy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved disagreement/entropy to %s", out_path)


def save_proposal_overlay_plot(
    img_b: np.ndarray,
    prediction_mask: np.ndarray,
    candidate_mask: np.ndarray,
    accepted_mask: np.ndarray,
    rejected_mask: np.ndarray,
    plot_dir: str,
    image_id_b: str,
    accept_rgb: tuple[int, int, int],
    reject_rgb: tuple[int, int, int],
    candidate_rgb: tuple[int, int, int],
) -> None:
    """Save proposal overlay plot with accepted/rejected colors.

    Examples:
        >>> callable(save_proposal_overlay_plot)
        True
    """
    overlay = _overlay_mask(img_b, prediction_mask, (255, 0, 0), alpha=0.3)
    overlay = _overlay_mask(overlay, candidate_mask, candidate_rgb, alpha=0.25)
    overlay = _overlay_mask(overlay, rejected_mask, reject_rgb, alpha=0.65)
    overlay = _overlay_mask(overlay, accepted_mask, accept_rgb, alpha=0.65)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title("Proposals: accepted vs rejected")
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{image_id_b}_proposal_overlay.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved proposal overlay to %s", out_path)


def save_dino_channel_importance_plot(
    bst,
    feature_layout: dict | None,
    plot_dir: str,
    top_k: int = 20,
) -> str | None:
    """Save top DINO channel importances from trained XGB model.

    Examples:
        >>> callable(save_dino_channel_importance_plot)
        True
    """
    if bst is None:
        return None
    feature_names = (
        list(feature_layout.get("feature_names", []))
        if feature_layout is not None
        else []
    )
    if not feature_names:
        logger.warning("skip dino importance plot: missing feature names")
        return None

    gain = bst.get_score(importance_type="gain")
    if not gain:
        logger.warning("skip dino importance plot: empty gain map")
        return None

    dino_rows: list[tuple[str, float]] = []
    for name, val in gain.items():
        if not name.startswith("dino_"):
            continue
        try:
            idx = int(name.split("_", 1)[1])
        except (ValueError, IndexError):
            idx = -1
        dino_rows.append((f"dino_{idx}", float(val)))
    if not dino_rows:
        logger.warning("skip dino importance plot: no dino_* features in gain map")
        return None

    dino_rows.sort(key=lambda t: t[1], reverse=True)
    top = dino_rows[: max(1, int(top_k))]
    labels = [r[0] for r in top][::-1]
    values = np.array([r[1] for r in top][::-1], dtype=np.float32)
    cum = np.cumsum(values) / max(float(values.sum()), 1e-8)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    y = np.arange(len(labels))
    ax1.barh(y, values, color="#2f7fb8", alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("XGB gain")
    ax1.set_title("Top DINO channels by XGB gain")

    ax2 = ax1.twiny()
    ax2.plot(cum, y, color="#f06a3a", linewidth=1.8)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("cumulative gain share")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, "xgb_dino_channel_importance.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved dino channel importance to %s", out_path)
    return out_path


def save_plot(
    img_b,
    gt_mask_B,
    mask_raw_best,
    best_raw_config,
    best_crf_mask,
    best_crf_config,
    thr_center_for_crf,
    plot_dir,
    image_id_b,
    best_shadow=None,
    labels_sh=None,
):
    """Save comparison figure (RGB, GT, raw, CRF, optional shadow/labels).

    Args:
        img_b (np.ndarray): Image B RGB array.
        gt_mask_B (np.ndarray): Ground-truth mask for B.
        mask_raw_best (np.ndarray): Best raw mask.
        best_raw_config (dict): Best raw configuration metrics.
        best_crf_mask (np.ndarray): Best CRF mask.
        best_crf_config (dict): Best CRF configuration metrics.
        thr_center_for_crf (float): Threshold center for CRF.
        plot_dir (str): Output directory for plots.
        image_id_b (str): Image identifier.
        best_shadow (dict | None): Optional shadow filter result.
        labels_sh (np.ndarray | None): Optional SH labels.

    Examples:
        >>> callable(save_plot)
        True
    """
    show_labels = labels_sh is not None
    show_shadow = best_shadow is not None
    cols = 3 if (show_labels or show_shadow) else 2
    fig, axs = plt.subplots(2, cols, figsize=(8 * cols, 12))
    if cols == 2 and axs.ndim == 1:
        axs = np.array([[axs[0], axs[1]], [axs[2], axs[3]]])
    axs[0, 0].imshow(img_b)
    axs[0, 0].set_title("Image B (RGB)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gt_mask_B > 0, cmap="gray")
    axs[0, 1].set_title("Ground truth (union)")
    axs[0, 1].axis("off")

    if show_labels:
        axs[0, 2].imshow(labels_sh > 0, cmap="gray")
        axs[0, 2].set_title("SOURCE_LABEL_RASTER (reprojected)")
        axs[0, 2].axis("off")

    overlay_raw = img_b.copy()
    overlay_raw[mask_raw_best] = (
        0.5 * overlay_raw[mask_raw_best] + 0.5 * np.array([0, 255, 0])
    ).astype(overlay_raw.dtype)
    axs[1, 0].imshow(overlay_raw)
    axs[1, 0].set_title(
        f"Raw kNN (k={best_raw_config['k']}, thr={best_raw_config['threshold']:.3f})\n"
        f"IoU={best_raw_config['iou']:.3f}, F1={best_raw_config['f1']:.3f}"
    )
    axs[1, 0].axis("off")

    overlay_crf = img_b.copy()
    overlay_crf[best_crf_mask] = (
        0.5 * overlay_crf[best_crf_mask] + 0.5 * np.array([255, 0, 0])
    ).astype(overlay_crf.dtype)
    axs[1, 1].imshow(overlay_crf)
    axs[1, 1].set_title(
        f"CRF (k={best_crf_config['k']}, center_thr={thr_center_for_crf:.3f})\n"
        f"IoU={best_crf_config['iou']:.3f}, F1={best_crf_config['f1']:.3f}"
    )
    axs[1, 1].axis("off")

    if show_shadow:
        shadow_mask = best_shadow["mask"]
        shadow_cfg = best_shadow["cfg"]
        overlay_shadow = img_b.copy()
        overlay_shadow[shadow_mask] = (
            0.5 * overlay_shadow[shadow_mask] + 0.5 * np.array([255, 255, 0])
        ).astype(overlay_shadow.dtype)
        axs[1, 2].imshow(overlay_shadow)
        axs[1, 2].set_title(
            f"Shadow filter w={shadow_cfg['weights']}, thr={shadow_cfg['threshold']}\n"
            f"IoU={shadow_cfg['iou']:.3f}, F1={shadow_cfg['f1']:.3f}"
        )
        axs[1, 2].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_raw_crf.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved to %s", plot_path)


def save_best_model_plot(
    img_b,
    gt_mask,
    pred_mask,
    title,
    plot_dir,
    image_id_b,
    filename_suffix="champion.png",
):
    """Save a simple overlay plot of the champion mask vs GT.

    Args:
        img_b (np.ndarray): Image B RGB array.
        gt_mask (np.ndarray): Ground-truth mask.
        pred_mask (np.ndarray): Predicted mask.
        title (str): Plot title.
        plot_dir (str): Output directory.
        image_id_b (str): Image identifier.
        filename_suffix (str): Filename suffix.

    Examples:
        >>> callable(save_best_model_plot)
        True
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img_b)
    axs[0].set_title("Image B (RGB)")
    axs[0].axis("off")

    axs[1].imshow(gt_mask.astype(bool), cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    overlay = img_b.copy()
    overlay[pred_mask] = (
        0.5 * overlay[pred_mask] + 0.5 * np.array([255, 0, 0])
    ).astype(overlay.dtype)
    axs[2].imshow(overlay)
    axs[2].set_title(title)
    axs[2].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_{filename_suffix}")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved champion overlay to %s", plot_path)


def save_knn_xgb_gt_plot(
    img_b,
    gt_mask,
    mask_knn,
    mask_xgb,
    plot_dir,
    image_id_b,
    title_knn="kNN",
    title_xgb="XGBoost",
    filename_suffix="knn_xgb_gt.png",
):
    """Save three panels: RGB+GT overlay, RGB+kNN overlay, RGB+XGB overlay.

    Args:
        img_b (np.ndarray): Image B RGB array.
        gt_mask (np.ndarray): Ground-truth mask.
        mask_knn (np.ndarray): kNN mask.
        mask_xgb (np.ndarray): XGB mask.
        plot_dir (str): Output directory.
        image_id_b (str): Image identifier.
        title_knn (str): kNN panel title.
        title_xgb (str): XGB panel title.
        filename_suffix (str): Filename suffix.

    Examples:
        >>> callable(save_knn_xgb_gt_plot)
        True
    """
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    # GT overlay
    overlay_gt = img_b.copy()
    overlay_gt[gt_mask.astype(bool)] = (
        0.5 * overlay_gt[gt_mask.astype(bool)] + 0.5 * np.array([0, 0, 255])
    ).astype(overlay_gt.dtype)
    axs[0].imshow(overlay_gt)
    axs[0].set_title("GT overlay")
    axs[0].axis("off")

    # kNN overlay
    overlay_knn = img_b.copy()
    overlay_knn[mask_knn.astype(bool)] = (
        0.5 * overlay_knn[mask_knn.astype(bool)] + 0.5 * np.array([0, 255, 0])
    ).astype(overlay_knn.dtype)
    axs[1].imshow(overlay_knn)
    axs[1].set_title(title_knn)
    axs[1].axis("off")

    # XGB overlay
    overlay_xgb = img_b.copy()
    overlay_xgb[mask_xgb.astype(bool)] = (
        0.5 * overlay_xgb[mask_xgb.astype(bool)] + 0.5 * np.array([255, 0, 0])
    ).astype(overlay_xgb.dtype)
    axs[2].imshow(overlay_xgb)
    axs[2].set_title(title_xgb)
    axs[2].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_{filename_suffix}")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved kNN/XGB/GT overlay to %s", plot_path)


def save_unified_plot(
    img_b,
    gt_mask,
    labels_sh,
    masks: dict,
    metrics: dict,
    plot_dir,
    image_id_b,
    show_metrics: bool,
    gt_available: bool,
    similarity_map=None,
    score_maps: dict | None = None,
    skeleton=None,
    endpoints=None,
    proposal_masks: dict | None = None,
):
    """Save a unified multi-panel plot for all phases.

    Args:
        img_b (np.ndarray): Image B RGB array.
        gt_mask (np.ndarray): Ground-truth mask.
        labels_sh (np.ndarray): Source label raster mask.
        masks (dict): Mask dict keyed by phase.
        metrics (dict): Metrics dict keyed by phase.
        plot_dir (str): Output directory.
        image_id_b (str): Image identifier.
        show_metrics (bool): Whether to include metrics in titles.
        gt_available (bool): Whether GT labels are available.
        similarity_map (np.ndarray | None): Optional DINO similarity map.
        score_maps (dict | None): Optional score heatmaps (kNN/XGB).
        skeleton (np.ndarray | None): Optional skeleton mask.
        endpoints (np.ndarray | None): Optional endpoint coordinates.

    Examples:
        >>> callable(save_unified_plot)
        True
    """

    def _title(base: str, key: str | None = None) -> str:
        if not show_metrics or key is None or key not in metrics:
            return base
        m = metrics[key]
        return f"{base} IoU={m['iou']:.3f}, F1={m['f1']:.3f}"

    panels = []
    panels.append(("RGB", img_b, None))
    if labels_sh is not None:
        panels.append(("Source labels", labels_sh > 0, "gray"))
    if gt_available:
        panels.append(("GT" if show_metrics else "GT", gt_mask > 0, "gray"))
    if similarity_map is not None:
        panels.append(
            ("DINO similarity", _normalize_heatmap(similarity_map), "coolwarm")
        )
    if score_maps:
        if "knn" in score_maps:
            panels.append(
                ("kNN score", _normalize_heatmap(score_maps["knn"]), "coolwarm")
            )
        if "xgb" in score_maps:
            panels.append(
                ("XGB score", _normalize_heatmap(score_maps["xgb"]), "coolwarm")
            )

    phase_order = [
        ("knn_raw", "kNN raw"),
        ("knn_crf", "kNN CRF"),
        ("knn_shadow", "kNN shadow"),
        ("xgb_raw", "XGB raw"),
        ("xgb_crf", "XGB CRF"),
        ("xgb_shadow", "XGB shadow"),
        ("champion_raw", "Champion raw"),
        ("champion_crf", "Champion CRF"),
    ]
    phase_order.append(("champion_shadow", "Champion shadow"))

    for key, label in phase_order:
        if key not in masks:
            continue
        overlay = img_b.copy()
        overlay[masks[key].astype(bool)] = (
            0.5 * overlay[masks[key].astype(bool)] + 0.5 * np.array([255, 0, 0])
        ).astype(overlay.dtype)
        panels.append((_title(label, key), overlay, None))

    if skeleton is not None:
        overlay_skel = img_b.copy()
        overlay_skel[skeleton.astype(bool)] = (
            0.5 * overlay_skel[skeleton.astype(bool)] + 0.5 * np.array([0, 255, 255])
        ).astype(overlay_skel.dtype)
        panels.append(("Skeleton + endpoints", overlay_skel, None))

    if proposal_masks:
        if "candidate" in proposal_masks:
            panels.append(
                (
                    "Proposal candidates",
                    _overlay_mask(
                        img_b, proposal_masks["candidate"], (255, 255, 0), 0.5
                    ),
                    None,
                )
            )
        if "accepted" in proposal_masks:
            panels.append(
                (
                    "Proposals accepted",
                    _overlay_mask(
                        img_b, proposal_masks["accepted"], (0, 255, 255), 0.6
                    ),
                    None,
                )
            )
        if "rejected" in proposal_masks:
            panels.append(
                (
                    "Proposals rejected",
                    _overlay_mask(
                        img_b, proposal_masks["rejected"], (255, 165, 0), 0.6
                    ),
                    None,
                )
            )

    cols = 3
    rows = int(math.ceil(len(panels) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axs = np.array(axs).reshape(rows, cols)

    for idx, (title, img, cmap) in enumerate(panels):
        r = idx // cols
        c = idx % cols
        if cmap is None:
            axs[r, c].imshow(img)
        else:
            axs[r, c].imshow(img, cmap=cmap)
        axs[r, c].set_title(title)
        axs[r, c].axis("off")
        if (
            skeleton is not None
            and title == "Skeleton + endpoints"
            and endpoints is not None
        ):
            if len(endpoints) > 0:
                axs[r, c].scatter(
                    endpoints[:, 1],
                    endpoints[:, 0],
                    s=3,
                    c="red",
                    marker="o",
                )

    for idx in range(len(panels), rows * cols):
        r = idx // cols
        c = idx % cols
        axs[r, c].axis("off")

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{image_id_b}_unified.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot saved unified to %s", plot_path)
