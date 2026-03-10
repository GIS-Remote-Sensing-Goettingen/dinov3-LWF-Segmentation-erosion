"""Timing helpers for optional performance logging."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from .config_loader import cfg

# Defaults with config overrides
DEBUG_TIMING = cfg.runtime.debug_timing
DEBUG_TIMING_VERBOSE = cfg.runtime.debug_timing_verbose
COMPACT_TIMING_LOGS = cfg.runtime.compact_timing_logs

logger = logging.getLogger(__name__)

_AGG_STATS: dict[str, dict[str, float]] = {}
_PERF_AGG_STATS: dict[tuple[str | None, str, str | None], dict[str, float]] = {}
_PERF_LOG_PATH: str | None = None
_PERF_RUN_ID: str | None = None
_PERF_PHASE: ContextVar[str | None] = ContextVar("segedge_perf_phase", default=None)
_PERF_TILE: ContextVar[str | None] = ContextVar("segedge_perf_tile", default=None)
_PERF_IMAGE_ID: ContextVar[str | None] = ContextVar(
    "segedge_perf_image_id", default=None
)


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


def configure_performance_logging(
    log_path: str | None,
    *,
    run_id: str | None = None,
) -> None:
    """Configure the structured performance log destination.

    Examples:
        >>> configure_performance_logging(None)
    """
    global _PERF_LOG_PATH, _PERF_RUN_ID
    _PERF_LOG_PATH = log_path
    _PERF_RUN_ID = run_id
    _PERF_AGG_STATS.clear()
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)


@contextmanager
def performance_context(
    *,
    phase: str | None = None,
    tile: str | None = None,
    image_id: str | None = None,
):
    """Temporarily attach phase/tile context to performance records.

    Examples:
        >>> with performance_context(phase="demo", tile="tile_a"):
        ...     _PERF_PHASE.get(), _PERF_TILE.get()
        ('demo', 'tile_a')
    """
    phase_token = tile_token = image_token = None
    if phase is not None:
        phase_token = _PERF_PHASE.set(phase)
    if tile is not None:
        tile_token = _PERF_TILE.set(tile)
    if image_id is not None:
        image_token = _PERF_IMAGE_ID.set(image_id)
    try:
        yield
    finally:
        if image_token is not None:
            _PERF_IMAGE_ID.reset(image_token)
        if tile_token is not None:
            _PERF_TILE.reset(tile_token)
        if phase_token is not None:
            _PERF_PHASE.reset(phase_token)


def _update_perf_agg(
    phase: str | None,
    stage: str,
    substage: str | None,
    dt: float,
) -> None:
    """Accumulate structured performance stats."""
    key = (phase, stage, substage)
    stats = _PERF_AGG_STATS.get(key)
    if stats is None:
        _PERF_AGG_STATS[key] = {
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


def _write_perf_record(record: dict[str, object]) -> None:
    """Append one structured performance record when enabled."""
    if not _PERF_LOG_PATH:
        return
    with open(_PERF_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def record_performance(
    stage: str,
    duration_s: float,
    *,
    substage: str | None = None,
    phase: str | None = None,
    tile: str | None = None,
    image_id: str | None = None,
    extra: dict[str, object] | None = None,
    kind: str = "span",
) -> None:
    """Write one structured performance span.

    Examples:
        >>> configure_performance_logging(None)
        >>> record_performance("demo", 0.1)
    """
    active_phase = phase if phase is not None else _PERF_PHASE.get()
    active_tile = tile if tile is not None else _PERF_TILE.get()
    active_image_id = image_id if image_id is not None else _PERF_IMAGE_ID.get()
    _update_perf_agg(active_phase, stage, substage, duration_s)
    _write_perf_record(
        {
            "duration_s": float(duration_s),
            "extra": extra or {},
            "image_id": active_image_id,
            "kind": kind,
            "phase": active_phase,
            "run_id": _PERF_RUN_ID,
            "stage": stage,
            "substage": substage,
            "tile": active_tile,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )


@contextmanager
def perf_span(
    stage: str,
    *,
    substage: str | None = None,
    phase: str | None = None,
    tile: str | None = None,
    image_id: str | None = None,
    extra: dict[str, object] | None = None,
):
    """Measure and record a structured performance span.

    Examples:
        >>> configure_performance_logging(None)
        >>> with perf_span("demo", substage="unit"):
        ...     _ = 1 + 1
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_performance(
            stage,
            time.perf_counter() - t0,
            substage=substage,
            phase=phase,
            tile=tile,
            image_id=image_id,
            extra=extra,
        )


def emit_performance_summary(
    reason: str,
    *,
    tile_index: int | None = None,
) -> None:
    """Write an aggregate summary snapshot to the performance log.

    Examples:
        >>> configure_performance_logging(None)
        >>> emit_performance_summary("demo")
    """
    summary: dict[str, dict[str, float | int | None]] = {}
    sorted_items = sorted(
        _PERF_AGG_STATS.items(),
        key=lambda item: (
            item[0][0] or "",
            item[0][1],
            item[0][2] or "",
        ),
    )
    for (phase, stage, substage), stats in sorted_items:
        key = "|".join(
            part for part in (phase or "", stage, substage or "") if part != ""
        )
        count = int(stats["count"])
        summary[key] = {
            "count": count,
            "max_s": float(stats["max"]),
            "mean_s": float(stats["total"] / max(1, count)),
            "min_s": float(stats["min"]),
            "phase": phase,
            "stage": stage,
            "substage": substage,
            "total_s": float(stats["total"]),
        }
    top_by_total = [
        {
            "phase": phase,
            "stage": stage,
            "substage": substage,
            "total_s": float(stats["total"]),
            "mean_s": float(stats["total"] / max(1, int(stats["count"]))),
        }
        for (phase, stage, substage), stats in sorted(
            _PERF_AGG_STATS.items(),
            key=lambda item: item[1]["total"],
            reverse=True,
        )[:5]
    ]
    _write_perf_record(
        {
            "kind": "summary",
            "reason": reason,
            "run_id": _PERF_RUN_ID,
            "stats": summary,
            "top_by_total_s": top_by_total,
            "tile_index": tile_index,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
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
    record_performance(label, dt, kind="timer")

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
