from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import convolve
from skimage.graph import route_through_array
from skimage.morphology import binary_dilation, disk, remove_small_objects, skeletonize

logger = logging.getLogger(__name__)


def _skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return skeleton endpoints as (row, col) array.

    Args:
        skel (np.ndarray): Boolean skeleton mask.

    Returns:
        np.ndarray: Endpoint coordinates (N, 2).

    Examples:
        >>> import numpy as np
        >>> sk = np.zeros((3, 3), dtype=bool)
        >>> sk[1, 1] = True
        >>> _skeleton_endpoints(sk).shape[1]
        2
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbors = neighbors - skel.astype(np.uint8)
    return np.argwhere(skel & (neighbors == 1))


def skeletonize_with_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute skeleton and endpoint coordinates from a binary mask.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        tuple[np.ndarray, np.ndarray]: Skeleton mask and endpoints (N, 2).

    Examples:
        >>> import numpy as np
        >>> m = np.zeros((3, 3), dtype=bool)
        >>> m[1, 1] = True
        >>> sk, pts = skeletonize_with_endpoints(m)
        >>> sk.shape == m.shape and pts.shape[1] == 2
        True
    """
    skel = skeletonize(mask.astype(bool))
    endpoints = _skeleton_endpoints(skel)
    return skel, endpoints


def _prune_skeleton_spurs(skel: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return skel
    pruned = skel.copy()
    for _ in range(iters):
        endpoints = _skeleton_endpoints(pruned)
        if endpoints.size == 0:
            break
        pruned[endpoints[:, 0], endpoints[:, 1]] = False
    return pruned


def bridge_skeleton_gaps(
    mask: np.ndarray,
    prob: np.ndarray,
    max_gap_px: int,
    max_pairs_per_endpoint: int,
    max_avg_cost: float,
    bridge_width_px: int,
    min_component_area_px: int,
    spur_prune_iters: int,
) -> np.ndarray:
    """Bridge skeleton gaps using shortest paths on a cost map.

    Args:
        mask (np.ndarray): Input binary mask.
        prob (np.ndarray): Probability map in [0, 1].
        max_gap_px (int): Max endpoint distance to consider.
        max_pairs_per_endpoint (int): Candidate pairs per endpoint.
        max_avg_cost (float): Max average cost per path pixel.
        bridge_width_px (int): Dilation radius for bridged skeleton.
        min_component_area_px (int): Remove components below this size.
        spur_prune_iters (int): Iterative pruning of skeleton endpoints.

    Returns:
        np.ndarray: Bridged mask.

    Examples:
        >>> isinstance(bridge_skeleton_gaps.__name__, str)
        True
    """
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return mask_bool
    if prob.shape != mask_bool.shape:
        raise ValueError("prob must match mask shape")

    eps = 1e-6
    prob_clipped = np.clip(prob.astype(np.float32), eps, 1.0)
    cost = -np.log(prob_clipped)

    skel = skeletonize(mask_bool)
    endpoints = _skeleton_endpoints(skel)
    if endpoints.size == 0:
        return mask_bool

    pairs: set[tuple[int, int]] = set()
    for i, (y, x) in enumerate(endpoints):
        dy = endpoints[:, 0] - y
        dx = endpoints[:, 1] - x
        dist = np.sqrt(dy * dy + dx * dx)
        candidates = np.where((dist > 0) & (dist <= max_gap_px))[0]
        if candidates.size == 0:
            continue
        order = candidates[np.argsort(dist[candidates])]
        for j in order[:max_pairs_per_endpoint]:
            a, b = sorted((i, int(j)))
            pairs.add((a, b))

    bridged = skel.copy()
    h, w = mask_bool.shape
    for i, j in pairs:
        y1, x1 = endpoints[i]
        y2, x2 = endpoints[j]
        y_min = max(0, min(y1, y2) - max_gap_px)
        y_max = min(h - 1, max(y1, y2) + max_gap_px)
        x_min = max(0, min(x1, x2) - max_gap_px)
        x_max = min(w - 1, max(x1, x2) + max_gap_px)
        window = cost[y_min : y_max + 1, x_min : x_max + 1]
        start = (int(y1 - y_min), int(x1 - x_min))
        end = (int(y2 - y_min), int(x2 - x_min))
        try:
            path, path_cost = route_through_array(
                window,
                start,
                end,
                fully_connected=True,
                geometric=True,
            )
        except Exception:
            continue
        if not path:
            continue
        avg_cost = float(path_cost) / max(1, len(path))
        if avg_cost > max_avg_cost:
            continue
        for py, px in path:
            bridged[y_min + py, x_min + px] = True

    bridged = _prune_skeleton_spurs(bridged, spur_prune_iters)
    bridged_ribbon = binary_dilation(bridged, disk(max(1, int(bridge_width_px))))
    out = np.logical_or(mask_bool, bridged_ribbon)
    if min_component_area_px > 0:
        out = remove_small_objects(out, min_size=int(min_component_area_px))
    return out
