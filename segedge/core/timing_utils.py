"""Timing helpers for optional performance logging."""

from __future__ import annotations

import logging
import time

from .config_loader import cfg

# Defaults with config overrides
DEBUG_TIMING = cfg.runtime.debug_timing
DEBUG_TIMING_VERBOSE = cfg.runtime.debug_timing_verbose
COMPACT_TIMING_LOGS = cfg.runtime.compact_timing_logs

logger = logging.getLogger(__name__)

_AGG_STATS: dict[str, dict[str, float]] = {}


def _update_agg(label: str, dt: float) -> None:
    """Accumulate compact timing stats for repetitive labels.

    Examples:
        >>> _AGG_STATS.clear()
        >>> _update_agg("demo", 1.0)
        >>> int(_AGG_STATS["demo"]["count"])
        1
    """
    stats = _AGG_STATS.get(label)
    if stats is None:
        _AGG_STATS[label] = {
            "count": 1.0,
            "total": dt,
            "min": dt,
            "max": dt,
        }
        return
    stats["count"] += 1.0
    stats["total"] += dt
    stats["min"] = min(stats["min"], dt)
    stats["max"] = max(stats["max"], dt)


def _pop_agg(label: str) -> str | None:
    """Return a compact summary string and clear the aggregated label.

    Examples:
        >>> _AGG_STATS.clear()
        >>> _update_agg("demo", 1.0)
        >>> "demo n=1" in (_pop_agg("demo") or "")
        True
    """
    stats = _AGG_STATS.pop(label, None)
    if not stats:
        return None
    count = int(stats["count"])
    total = stats["total"]
    mean = total / max(1, count)
    return (
        f"{label} n={count} total={total:.3f}s "
        f"mean={mean:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s"
    )


def time_start():
    """Return a perf counter start if timing is enabled.

    Returns:
        float | None: Start time if timing enabled, else None.

    Examples:
        >>> isinstance(time_start(), (float, type(None)))
        True
    """
    if not DEBUG_TIMING:
        return None
    return time.perf_counter()


def time_end(label: str, t0):
    """Log elapsed time for a block if timing is enabled.

    Args:
        label (str): Label for the timing block.
        t0 (float | None): Start time.

    Examples:
        >>> time_end("noop", None) is None
        True
    """
    if not DEBUG_TIMING or t0 is None:
        return
    dt = time.perf_counter() - t0

    if COMPACT_TIMING_LOGS:
        if label.startswith("extract_patch_features_batch_single_scale"):
            _update_agg("extract_patch_features_batch_single_scale", dt)
            return
        if label.startswith("extract_patch_features_single_scale"):
            _update_agg("extract_patch_features_single_scale", dt)
            return
        if label.startswith("zero_shot_tile("):
            _update_agg("zero_shot_tile", dt)
            return

        suffix_parts = []
        if label.startswith("prefetch_features_single_scale_image"):
            for key in (
                "extract_patch_features_batch_single_scale",
                "extract_patch_features_single_scale",
            ):
                summary = _pop_agg(key)
                if summary:
                    suffix_parts.append(summary)
        elif label.startswith("zero_shot_knn_single_scale_B_with_saliency"):
            summary = _pop_agg("zero_shot_tile")
            if summary:
                suffix_parts.append(summary)

        suffix = f" | {'; '.join(suffix_parts)}" if suffix_parts else ""
        logger.info("time %s: %.3f s%s", label, dt, suffix)
        return

    logger.info("time %s: %.3f s", label, dt)
