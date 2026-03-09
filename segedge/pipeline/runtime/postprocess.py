"""Shadow filtering and proposal heuristics."""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import label as ndi_label
from skimage.morphology import skeletonize

from ...core.config_loader import cfg

logger = logging.getLogger(__name__)


def _apply_shadow_filter(
    img_b: np.ndarray,
    base_mask: np.ndarray,
    weights,
    threshold: float,
    score_full: np.ndarray,
    protect_score: float | None,
) -> np.ndarray:
    """Apply a shadow filter with optional score protection.

    Examples:
        >>> mask = _apply_shadow_filter(
        ...     np.ones((1, 1, 3), dtype=np.uint8),
        ...     np.array([[True]]),
        ...     [1, 1, 1],
        ...     0.0,
        ...     np.array([[0.0]], dtype=np.float32),
        ...     None,
        ... )
        >>> mask.shape
        (1, 1)
    """
    img_float = img_b.astype(np.float32)
    w = np.array(weights, dtype=np.float32).reshape(1, 1, 3)
    wsum = (img_float * w).sum(axis=2)
    shadow_pass = wsum >= threshold
    if protect_score is None:
        return np.logical_and(base_mask, shadow_pass)
    return np.logical_and(base_mask, shadow_pass | (score_full >= protect_score))


def _compute_component_shape_metrics(
    component_mask: np.ndarray,
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    pixel_size_m: float,
) -> dict[str, float]:
    """Compute shape/context metrics for a connected component.

    Examples:
        >>> metrics = _compute_component_shape_metrics(np.array([[True]]), np.array([[1.0]]), None, 1.0)
        >>> metrics["area_px"]
        1.0
    """
    area_px = int(component_mask.sum())
    if area_px <= 0:
        return {
            "area_px": 0.0,
            "length_px": 0.0,
            "length_m": 0.0,
            "mean_width_px": 0.0,
            "mean_width_m": 0.0,
            "skeleton_ratio": 0.0,
            "pca_ratio": 0.0,
            "circularity": 1.0,
            "mean_score": 0.0,
            "road_overlap": 0.0,
            "centroid_row": 0.0,
            "centroid_col": 0.0,
        }

    coords = np.argwhere(component_mask)
    centroid_row = float(coords[:, 0].mean())
    centroid_col = float(coords[:, 1].mean())
    pca_ratio = 1.0
    if coords.shape[0] >= 2:
        centered = coords.astype(np.float32) - coords.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        if cov.ndim == 0:
            eigvals = np.array([float(cov), 0.0], dtype=np.float32)
        else:
            eigvals = np.linalg.eigvalsh(cov).astype(np.float32)
            if eigvals.shape[0] == 1:
                eigvals = np.array([eigvals[0], 0.0], dtype=np.float32)
        eigvals = np.sort(eigvals)[::-1]
        lam1 = max(float(eigvals[0]), 1e-8)
        lam2 = max(float(eigvals[1]), 1e-8)
        pca_ratio = lam1 / lam2

    boundary = np.logical_and(component_mask, ~binary_erosion(component_mask))
    perimeter_px = float(boundary.sum())
    circularity = float(4.0 * np.pi * area_px / (perimeter_px**2 + 1e-8))

    skel = skeletonize(component_mask)
    length_px = float(skel.sum())
    if length_px > 0:
        dt = distance_transform_edt(component_mask)
        widths = 2.0 * dt[skel]
        mean_width_px = float(widths.mean()) if widths.size > 0 else 0.0
    else:
        mean_width_px = 0.0
    skeleton_ratio = float(length_px / max(mean_width_px, 1e-6))

    mean_score = (
        float(score_map[component_mask].mean()) if score_map is not None else 0.0
    )
    road_overlap = (
        float(roads_mask[component_mask].mean()) if roads_mask is not None else 0.0
    )
    return {
        "area_px": float(area_px),
        "length_px": float(length_px),
        "length_m": float(length_px * pixel_size_m),
        "mean_width_px": float(mean_width_px),
        "mean_width_m": float(mean_width_px * pixel_size_m),
        "skeleton_ratio": float(skeleton_ratio),
        "pca_ratio": float(pca_ratio),
        "circularity": float(circularity),
        "mean_score": float(mean_score),
        "road_overlap": float(road_overlap),
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
    }


def filter_novel_proposals(
    champion_mask: np.ndarray,
    labels_sh: np.ndarray,
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    pixel_size_m: float,
    sh_buffer_mask: np.ndarray | None = None,
    proposal_source_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Filter candidate components using elongated-object heuristics.

    Examples:
        >>> callable(filter_novel_proposals)
        True
    """
    p = cfg.postprocess.novel_proposals
    if proposal_source_mask is None:
        source_mask = champion_mask.astype(bool)
    else:
        source_mask = proposal_source_mask.astype(bool)
    if p.search_scope == "sh_buffer":
        if sh_buffer_mask is None:
            scope_mask = np.ones_like(source_mask, dtype=bool)
        else:
            scope_mask = sh_buffer_mask.astype(bool)
    else:
        scope_mask = np.ones_like(source_mask, dtype=bool)
    candidate_mask = np.logical_and(source_mask, scope_mask)
    candidate_mask = np.logical_and(candidate_mask, ~(labels_sh > 0))

    if sh_buffer_mask is None:
        buffer_mask = np.zeros_like(candidate_mask, dtype=bool)
    else:
        buffer_mask = sh_buffer_mask.astype(bool)
    candidate_inside = np.logical_and(candidate_mask, buffer_mask)
    candidate_outside = np.logical_and(candidate_mask, ~buffer_mask)
    if not bool(p.enabled):
        return {
            "candidate_mask": candidate_mask,
            "candidate_inside_mask": candidate_inside,
            "candidate_outside_mask": candidate_outside,
            "accepted_inside_mask": np.zeros_like(candidate_mask, dtype=bool),
            "accepted_outside_mask": np.zeros_like(candidate_mask, dtype=bool),
            "evaluated_outside_mask": np.zeros_like(candidate_mask, dtype=bool),
            "accepted_mask": np.zeros_like(candidate_mask, dtype=bool),
            "rejected_mask": np.zeros_like(candidate_mask, dtype=bool),
            "records": [],
        }

    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        if int(p.connectivity) <= 1
        else np.ones((3, 3), dtype=np.uint8)
    )

    accepted_inside_mask = np.zeros_like(candidate_mask, dtype=bool)
    accepted_outside_mask = np.zeros_like(candidate_mask, dtype=bool)
    rejected_mask = np.zeros_like(candidate_mask, dtype=bool)
    records = []
    component_id = 0

    inside_labels, inside_count = ndi_label(
        candidate_inside.astype(np.uint8), structure=structure
    )
    for local_id in range(1, int(inside_count) + 1):
        component_id += 1
        comp = inside_labels == local_id
        metrics = _compute_component_shape_metrics(
            comp, score_map, roads_mask, pixel_size_m
        )
        accepted_inside_mask |= comp
        records.append(
            {
                "component_id": int(component_id),
                "accepted": True,
                "acceptance_score": 1.0,
                "reject_reasons": [],
                "zone": "inside_buffer_auto",
                **metrics,
            }
        )

    outside_labels, outside_count = ndi_label(
        candidate_outside.astype(np.uint8), structure=structure
    )
    total_checks = 8.0
    for local_id in range(1, int(outside_count) + 1):
        component_id += 1
        comp = outside_labels == local_id
        metrics = _compute_component_shape_metrics(
            comp, score_map, roads_mask, pixel_size_m
        )
        checks = {
            "min_area_px": metrics["area_px"] >= float(p.min_area_px),
            "min_length_m": metrics["length_m"] >= float(p.min_length_m),
            "max_width_m": metrics["mean_width_m"] <= float(p.max_width_m),
            "min_skeleton_ratio": metrics["skeleton_ratio"]
            >= float(p.min_skeleton_ratio),
            "min_pca_ratio": metrics["pca_ratio"] >= float(p.min_pca_ratio),
            "max_circularity": metrics["circularity"] <= float(p.max_circularity),
            "min_mean_score": metrics["mean_score"] >= float(p.min_mean_score),
            "max_road_overlap": metrics["road_overlap"] <= float(p.max_road_overlap),
        }
        passed_checks = float(sum(1 for ok in checks.values() if ok))
        acceptance_score = passed_checks / total_checks
        accepted = all(checks.values())
        if accepted:
            accepted_outside_mask |= comp
        else:
            rejected_mask |= comp
        records.append(
            {
                "component_id": int(component_id),
                "accepted": bool(accepted),
                "acceptance_score": float(acceptance_score),
                "reject_reasons": [k for k, ok in checks.items() if not ok],
                "zone": "outside_evaluated",
                **metrics,
            }
        )

    accepted_mask = np.logical_or(accepted_inside_mask, accepted_outside_mask)
    evaluated_outside_mask = candidate_outside
    logger.info(
        "novel proposals: candidates=%s inside_auto=%s outside_eval=%s accepted=%s rejected=%s preset=%s",
        int(component_id),
        int(inside_count),
        int(outside_count),
        int(sum(1 for r in records if r["accepted"])),
        int(sum(1 for r in records if not r["accepted"])),
        str(getattr(p, "heuristic_preset", "unknown")),
    )
    return {
        "candidate_mask": candidate_mask,
        "candidate_inside_mask": candidate_inside,
        "candidate_outside_mask": candidate_outside,
        "accepted_inside_mask": accepted_inside_mask,
        "accepted_outside_mask": accepted_outside_mask,
        "evaluated_outside_mask": evaluated_outside_mask,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
        "records": records,
    }
