"""Timing helpers for optional performance logging."""

from __future__ import annotations

import logging
import time

import config as cfg

# Defaults with config overrides
DEBUG_TIMING = getattr(cfg, "DEBUG_TIMING", True)
DEBUG_TIMING_VERBOSE = getattr(cfg, "DEBUG_TIMING_VERBOSE", False)

logger = logging.getLogger(__name__)


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
