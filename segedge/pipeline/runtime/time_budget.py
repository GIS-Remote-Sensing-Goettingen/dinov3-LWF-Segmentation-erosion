"""Time-budget utilities for training and inference cutover."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def compute_budget_deadline(start_ts: float, hours: float) -> float:
    """Return the deadline timestamp for a wall-clock budget.

    Examples:
        >>> compute_budget_deadline(10.0, 1.0)
        3610.0
    """
    return float(start_ts) + float(hours) * 3600.0


def is_budget_exceeded(deadline_ts: float | None, now_ts: float | None = None) -> bool:
    """Return True when the current time has reached/exceeded the deadline.

    Examples:
        >>> is_budget_exceeded(5.0, now_ts=6.0)
        True
    """
    if deadline_ts is None:
        return False
    now = time.time() if now_ts is None else float(now_ts)
    return now >= float(deadline_ts)


def remaining_budget_s(
    deadline_ts: float | None,
    now_ts: float | None = None,
) -> float | None:
    """Return remaining seconds before deadline (clamped at >=0).

    Examples:
        >>> remaining_budget_s(10.0, now_ts=7.5)
        2.5
    """
    if deadline_ts is None:
        return None
    now = time.time() if now_ts is None else float(now_ts)
    return max(0.0, float(deadline_ts) - now)


def deadline_ts_to_utc_iso(deadline_ts: float | None) -> str | None:
    """Convert epoch seconds to UTC ISO timestamp string.

    Examples:
        >>> deadline_ts_to_utc_iso(None) is None
        True
    """
    if deadline_ts is None:
        return None
    return datetime.fromtimestamp(float(deadline_ts), tz=timezone.utc).isoformat()


def parse_utc_iso_to_epoch(value: str | None) -> float | None:
    """Parse ISO UTC timestamp to epoch seconds.

    Examples:
        >>> parse_utc_iso_to_epoch("") is None
        True
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def build_time_budget_status(
    *,
    enabled: bool,
    hours: float,
    scope: str,
    cutover_mode: str,
    deadline_ts: float | None,
    clock_start_ts: float | None,
    cutover_triggered: bool,
    cutover_stage: str,
) -> dict | None:
    """Build a serializable snapshot of time-budget status.

    Examples:
        >>> build_time_budget_status(
        ...     enabled=False,
        ...     hours=1.0,
        ...     scope="x",
        ...     cutover_mode="y",
        ...     deadline_ts=None,
        ...     clock_start_ts=None,
        ...     cutover_triggered=False,
        ...     cutover_stage="none",
        ... ) is None
        True
    """
    if not enabled:
        return None
    now = time.time()
    rem = remaining_budget_s(deadline_ts, now_ts=now)
    elapsed = None
    if clock_start_ts is not None:
        elapsed = max(0.0, now - float(clock_start_ts))
    return {
        "enabled": bool(enabled),
        "hours": float(hours),
        "scope": str(scope),
        "cutover_mode": str(cutover_mode),
        "deadline_utc": deadline_ts_to_utc_iso(deadline_ts),
        "remaining_s": float(rem) if rem is not None else None,
        "elapsed_s": float(elapsed) if elapsed is not None else None,
        "cutover_triggered": bool(cutover_triggered),
        "cutover_stage": str(cutover_stage),
    }
