"""Shadow filtering and proposal heuristics."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects
from scipy.ndimage import label as ndi_label
from skimage.morphology import skeletonize

from ...core.config_loader import cfg
from ...core.timing_utils import perf_span

logger = logging.getLogger(__name__)
_PROPOSAL_MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))
_PROPOSAL_PARALLEL_MIN_COMPONENTS = 8


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
    *,
    row_offset: int = 0,
    col_offset: int = 0,
    basic_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute shape/context metrics for a connected component.

    Examples:
        >>> metrics = _compute_component_shape_metrics(np.array([[True]]), np.array([[1.0]]), None, 1.0)
        >>> metrics["area_px"]
        1.0
    """
    metrics = _compute_component_basic_metrics(
        component_mask,
        score_map,
        roads_mask,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    area_px = int(metrics["area_px"])
    if area_px <= 0:
        return metrics
    if basic_metrics is not None:
        metrics.update(basic_metrics)

    with perf_span("compute_component_shape_metrics", substage="boundary_metrics"):
        boundary = np.logical_and(component_mask, ~binary_erosion(component_mask))
        perimeter_px = float(boundary.sum())
        circularity = float(4.0 * np.pi * area_px / (perimeter_px**2 + 1e-8))

    with perf_span("compute_component_shape_metrics", substage="skeletonize"):
        skel = skeletonize(component_mask)
        length_px = float(skel.sum())
    if length_px > 0:
        with perf_span(
            "compute_component_shape_metrics", substage="distance_transform"
        ):
            dt = distance_transform_edt(component_mask)
            widths = 2.0 * dt[skel]
            mean_width_px = float(widths.mean()) if widths.size > 0 else 0.0
    else:
        mean_width_px = 0.0
    skeleton_ratio = float(length_px / max(mean_width_px, 1e-6))

    metrics.update(
        {
            "length_px": float(length_px),
            "length_m": float(length_px * pixel_size_m),
            "mean_width_px": float(mean_width_px),
            "mean_width_m": float(mean_width_px * pixel_size_m),
            "skeleton_ratio": float(skeleton_ratio),
            "circularity": float(circularity),
        }
    )
    return metrics


def _empty_component_metrics(
    *,
    row_offset: int = 0,
    col_offset: int = 0,
) -> dict[str, float]:
    """Return default metrics for an empty component."""
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
        "centroid_row": float(row_offset),
        "centroid_col": float(col_offset),
    }


def _compute_pca_ratio(coords: np.ndarray) -> float:
    """Return the PCA elongation ratio for component coordinates."""
    if coords.shape[0] < 2:
        return 1.0
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
    return lam1 / lam2


def _compute_component_basic_metrics(
    component_mask: np.ndarray,
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    *,
    row_offset: int = 0,
    col_offset: int = 0,
) -> dict[str, float]:
    """Compute the cheap component metrics on a local component crop."""
    area_px = int(component_mask.sum())
    if area_px <= 0:
        return _empty_component_metrics(
            row_offset=row_offset,
            col_offset=col_offset,
        )
    coords = np.argwhere(component_mask)
    centroid_row = float(coords[:, 0].mean() + row_offset)
    centroid_col = float(coords[:, 1].mean() + col_offset)
    mean_score = (
        float(score_map[component_mask].mean()) if score_map is not None else 0.0
    )
    road_overlap = (
        float(roads_mask[component_mask].mean()) if roads_mask is not None else 0.0
    )
    with perf_span("compute_component_basic_metrics", substage="pca_ratio"):
        pca_ratio = float(_compute_pca_ratio(coords))
    return {
        "area_px": float(area_px),
        "length_px": 0.0,
        "length_m": 0.0,
        "mean_width_px": 0.0,
        "mean_width_m": 0.0,
        "skeleton_ratio": 0.0,
        "pca_ratio": pca_ratio,
        "circularity": 1.0,
        "mean_score": float(mean_score),
        "road_overlap": float(road_overlap),
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
    }


def _component_local_mask(
    labels: np.ndarray,
    local_id: int,
    bbox_slice: tuple[slice, slice],
) -> tuple[np.ndarray, int, int]:
    """Return a local boolean mask and offsets for one connected component."""
    row_slice, col_slice = bbox_slice
    return (
        labels[bbox_slice] == local_id,
        int(row_slice.start or 0),
        int(col_slice.start or 0),
    )


def _evaluate_inside_component(
    local_id: int,
    bbox_slice: tuple[slice, slice],
    labels: np.ndarray,
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    pixel_size_m: float,
) -> tuple[int, tuple[slice, slice], np.ndarray, dict[str, float]]:
    """Compute metrics for an auto-accepted inside-buffer component."""
    comp_local, row_offset, col_offset = _component_local_mask(
        labels,
        local_id,
        bbox_slice,
    )
    score_local = score_map[bbox_slice]
    roads_local = roads_mask[bbox_slice] if roads_mask is not None else None
    metrics = _compute_component_shape_metrics(
        comp_local,
        score_local,
        roads_local,
        pixel_size_m,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    return local_id, bbox_slice, comp_local, metrics


def _evaluate_outside_component(
    local_id: int,
    bbox_slice: tuple[slice, slice],
    labels: np.ndarray,
    score_map: np.ndarray,
    roads_mask: np.ndarray | None,
    pixel_size_m: float,
    proposal_cfg,
) -> tuple[int, tuple[slice, slice], np.ndarray, dict[str, float], dict[str, bool]]:
    """Evaluate one outside-buffer proposal component with staged metrics."""
    comp_local, row_offset, col_offset = _component_local_mask(
        labels,
        local_id,
        bbox_slice,
    )
    score_local = score_map[bbox_slice]
    roads_local = roads_mask[bbox_slice] if roads_mask is not None else None
    basic_metrics = _compute_component_basic_metrics(
        comp_local,
        score_local,
        roads_local,
        row_offset=row_offset,
        col_offset=col_offset,
    )
    checks = {
        "min_area_px": basic_metrics["area_px"] >= float(proposal_cfg.min_area_px),
        "min_length_m": False,
        "max_width_m": False,
        "min_skeleton_ratio": False,
        "min_pca_ratio": basic_metrics["pca_ratio"]
        >= float(proposal_cfg.min_pca_ratio),
        "max_circularity": False,
        "min_mean_score": basic_metrics["mean_score"]
        >= float(proposal_cfg.min_mean_score),
        "max_road_overlap": basic_metrics["road_overlap"]
        <= float(proposal_cfg.max_road_overlap),
    }
    cheap_fail = not all(
        checks[key]
        for key in (
            "min_area_px",
            "min_pca_ratio",
            "min_mean_score",
            "max_road_overlap",
        )
    )
    if cheap_fail:
        return local_id, bbox_slice, comp_local, basic_metrics, checks
    full_metrics = _compute_component_shape_metrics(
        comp_local,
        score_local,
        roads_local,
        pixel_size_m,
        row_offset=row_offset,
        col_offset=col_offset,
        basic_metrics=basic_metrics,
    )
    allowed_width_m = _allowed_component_width_m(
        pca_ratio=full_metrics["pca_ratio"],
        proposal_cfg=proposal_cfg,
    )
    checks.update(
        {
            "min_length_m": full_metrics["length_m"]
            >= float(proposal_cfg.min_length_m),
            "max_width_m": full_metrics["mean_width_m"] <= allowed_width_m,
            "min_skeleton_ratio": full_metrics["skeleton_ratio"]
            >= float(proposal_cfg.min_skeleton_ratio),
            "max_circularity": full_metrics["circularity"]
            <= float(proposal_cfg.max_circularity),
        }
    )
    full_metrics["allowed_width_m"] = float(allowed_width_m)
    return local_id, bbox_slice, comp_local, full_metrics, checks


def _allowed_component_width_m(*, pca_ratio: float, proposal_cfg) -> float:
    """Return the elongation-adjusted width allowance for one proposal.

    Examples:
        >>> class P:
        ...     max_width_m = 10.0
        ...     min_pca_ratio = 3.0
        ...     width_bonus_per_pca = 1.0
        ...     hard_width_cap_m = 20.0
        >>> _allowed_component_width_m(pca_ratio=3.0, proposal_cfg=P())
        10.0
        >>> _allowed_component_width_m(pca_ratio=8.0, proposal_cfg=P())
        15.0
    """
    base_width_m = float(proposal_cfg.max_width_m)
    hard_cap_m = float(proposal_cfg.hard_width_cap_m)
    bonus_per_pca = float(proposal_cfg.width_bonus_per_pca)
    pca_bonus = max(float(pca_ratio) - float(proposal_cfg.min_pca_ratio), 0.0)
    return min(hard_cap_m, base_width_m + bonus_per_pca * pca_bonus)


def _component_map(
    items: list[tuple[int, tuple[slice, slice]]],
    evaluator,
):
    """Map component evaluators with bounded thread parallelism."""
    if len(items) < _PROPOSAL_PARALLEL_MIN_COMPONENTS or _PROPOSAL_MAX_WORKERS <= 1:
        return [evaluator(*item) for item in items]
    with ThreadPoolExecutor(
        max_workers=min(_PROPOSAL_MAX_WORKERS, len(items))
    ) as executor:
        return list(executor.map(lambda item: evaluator(*item), items))


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

    with perf_span("filter_novel_proposals", substage="connected_components"):
        inside_labels, inside_count = ndi_label(
            candidate_inside.astype(np.uint8), structure=structure
        )
        outside_labels, outside_count = ndi_label(
            candidate_outside.astype(np.uint8), structure=structure
        )
        inside_items = [
            (local_id, bbox_slice)
            for local_id, bbox_slice in enumerate(find_objects(inside_labels), start=1)
            if bbox_slice is not None
        ]
        outside_items = [
            (local_id, bbox_slice)
            for local_id, bbox_slice in enumerate(find_objects(outside_labels), start=1)
            if bbox_slice is not None
        ]

    with perf_span(
        "filter_novel_proposals",
        substage="inside_buffer_components",
        extra={"component_count": len(inside_items)},
    ):
        inside_results = _component_map(
            inside_items,
            lambda local_id, bbox_slice: _evaluate_inside_component(
                local_id,
                bbox_slice,
                inside_labels,
                score_map,
                roads_mask,
                pixel_size_m,
            ),
        )
    for _local_id, bbox_slice, comp_local, metrics in inside_results:
        component_id += 1
        accepted_inside_mask[bbox_slice] |= comp_local
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

    total_checks = 8.0
    with perf_span(
        "filter_novel_proposals",
        substage="outside_buffer_components",
        extra={"component_count": len(outside_items)},
    ):
        outside_results = _component_map(
            outside_items,
            lambda local_id, bbox_slice: _evaluate_outside_component(
                local_id,
                bbox_slice,
                outside_labels,
                score_map,
                roads_mask,
                pixel_size_m,
                p,
            ),
        )
    for _local_id, bbox_slice, comp_local, metrics, checks in outside_results:
        component_id += 1
        passed_checks = float(sum(1 for ok in checks.values() if ok))
        acceptance_score = passed_checks / total_checks
        accepted = all(checks.values())
        if accepted:
            accepted_outside_mask[bbox_slice] |= comp_local
        else:
            rejected_mask[bbox_slice] |= comp_local
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
