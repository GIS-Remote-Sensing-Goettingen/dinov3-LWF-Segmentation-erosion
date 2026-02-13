"""Timing helpers for optional performance logging."""

from __future__ import annotations

import logging
import time

import config as cfg

# Defaults with config overrides
DEBUG_TIMING = getattr(cfg, "DEBUG_TIMING", True)
DEBUG_TIMING_VERBOSE = getattr(cfg, "DEBUG_TIMING_VERBOSE", False)

logger = logging.getLogger(__name__)


def tile_timing_enabled() -> bool:
    """Return whether verbose tile-level timing lines are enabled.

    Examples:
        >>> isinstance(tile_timing_enabled(), bool)
        True
    """
    return bool(getattr(cfg, "TIMING_TILE_LOGS", False))


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
    logger.info("time %s: %.3f s", label, dt)


def time_end_tile(label: str, t0):
    """Log elapsed time only when tile-level timing is enabled.

    Examples:
        >>> time_end_tile("noop", None) is None
        True
    """
    if not tile_timing_enabled():
        return
    time_end(label, t0)
