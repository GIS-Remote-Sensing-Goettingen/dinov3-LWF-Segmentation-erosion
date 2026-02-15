"""Compact per-trial timing helpers for Bayesian tuning.

Examples:
    >>> timer = TrialPhaseTimer()
    >>> timer.add_duration("score_base_knn_xgb_s", 1.0)
    >>> snap = timer.finalize(total_s=2.0)
    >>> round(float(snap["scoring_share_pct"]), 1)
    50.0
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

PHASE_COLUMNS = (
    "score_base_knn_xgb_s",
    "score_perturb_knn_xgb_s",
    "threshold_base_s",
    "threshold_perturb_s",
    "metrics_base_s",
    "metrics_perturb_s",
    "crf_s",
    "bridge_s",
    "shadow_s",
    "trial_total_s",
    "perturb_share_pct",
    "scoring_share_pct",
)


class TrialPhaseTimer:
    """Accumulate coarse trial timings without per-item log spam.

    Examples:
        >>> timer = TrialPhaseTimer()
        >>> isinstance(timer, TrialPhaseTimer)
        True
    """

    def __init__(self) -> None:
        """Initialize empty phase accumulators.

        Examples:
            >>> timer = TrialPhaseTimer()
            >>> timer.finalize(total_s=0.0)["trial_total_s"]
            0.0
        """
        self._durations: dict[str, float] = {}

    @contextmanager
    def phase(self, phase_name: str) -> Iterator[None]:
        """Time a block and add elapsed time under `phase_name`.

        Examples:
            >>> timer = TrialPhaseTimer()
            >>> with timer.phase("score_base_knn_xgb_s"):
            ...     _ = 1 + 1
            >>> timer.finalize(total_s=1.0)["score_base_knn_xgb_s"] > 0.0
            True
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_duration(phase_name, time.perf_counter() - t0)

    def add_duration(self, phase_name: str, seconds: float) -> None:
        """Add elapsed seconds to a phase bucket.

        Examples:
            >>> timer = TrialPhaseTimer()
            >>> timer.add_duration("score_base_knn_xgb_s", 1.5)
            >>> timer.add_duration("score_base_knn_xgb_s", 0.5)
            >>> timer.finalize(total_s=3.0)["score_base_knn_xgb_s"]
            2.0
        """
        self._durations[phase_name] = self._durations.get(phase_name, 0.0) + float(
            max(0.0, seconds)
        )

    def finalize(self, *, total_s: float) -> dict[str, float]:
        """Return normalized trial timing snapshot with derived percentages.

        Examples:
            >>> timer = TrialPhaseTimer()
            >>> timer.add_duration("score_base_knn_xgb_s", 2.0)
            >>> snap = timer.finalize(total_s=4.0)
            >>> round(snap["scoring_share_pct"], 1)
            50.0
        """
        out = {key: float(self._durations.get(key, 0.0)) for key in PHASE_COLUMNS}
        out["trial_total_s"] = float(max(0.0, total_s))
        perturb_s = (
            out["score_perturb_knn_xgb_s"]
            + out["threshold_perturb_s"]
            + out["metrics_perturb_s"]
        )
        scoring_s = out["score_base_knn_xgb_s"] + out["score_perturb_knn_xgb_s"]
        total = out["trial_total_s"]
        out["perturb_share_pct"] = (
            float((perturb_s / total) * 100.0) if total > 0 else 0.0
        )
        out["scoring_share_pct"] = (
            float((scoring_s / total) * 100.0) if total > 0 else 0.0
        )
        return out

    def summary(self, *, top_n: int = 1, min_phase_s: float = 0.5) -> str:
        """Return compact timing summary for a single trial log line.

        Examples:
            >>> timer = TrialPhaseTimer()
            >>> timer.add_duration("score_base_knn_xgb_s", 1.2)
            >>> "score_base_knn_xgb_s" in timer.summary()
            True
        """
        pairs = [
            (name, dur)
            for name, dur in self._durations.items()
            if dur >= float(min_phase_s)
        ]
        pairs.sort(key=lambda item: item[1], reverse=True)
        top = pairs[: max(0, int(top_n))]
        if not top:
            return ""
        summary_parts = [f"{name}:{dur:.2f}s" for name, dur in top]
        return ", ".join(summary_parts)
