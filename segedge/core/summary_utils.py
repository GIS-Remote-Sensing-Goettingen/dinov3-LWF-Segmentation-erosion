from __future__ import annotations

import numpy as np


def weighted_mean(values: list[float], weights: list[float]) -> float:
    """Compute weighted mean with safe fallback.

    Examples:
        >>> weighted_mean([1.0, 3.0], [1.0, 1.0])
        2.0
    """
    total_weight = float(np.sum(weights)) if weights else 0.0
    if total_weight <= 0:
        return float(np.mean(values)) if values else 0.0
    return float(np.sum(np.array(values) * np.array(weights)) / total_weight)


def phase_metrics_summary(
    acc: dict[str, list[dict]], bridge_enabled: bool
) -> dict[str, dict]:
    """Return weighted and median metrics per phase.

    Examples:
        >>> acc = {
        ...     "knn_raw": [
        ...         {"iou": 0.5, "f1": 0.6, "precision": 0.7, "recall": 0.8, "_weight": 2.0}
        ...     ]
        ... }
        >>> summary = phase_metrics_summary(acc, bridge_enabled=False)
        >>> round(summary["knn_raw"]["weighted"]["iou"], 3)
        0.5
    """
    metric_keys = ["iou", "f1", "precision", "recall"]
    phase_order = [
        "knn_raw",
        "knn_crf",
        "knn_shadow",
        "xgb_raw",
        "xgb_crf",
        "xgb_shadow",
        "silver_core",
        "champion_raw",
        "champion_crf",
    ]
    if bridge_enabled:
        phase_order.append("champion_bridge")
    phase_order.append("champion_shadow")
    summary: dict[str, dict] = {}
    for phase in phase_order:
        if phase not in acc or not acc[phase]:
            continue
        weights = [float(m.get("_weight", 0.0)) for m in acc[phase]]
        vals = {k: [m.get(k, 0.0) for m in acc[phase]] for k in metric_keys}
        weighted = {k: weighted_mean(v, weights) for k, v in vals.items()}
        median = {k: float(np.median(v)) for k, v in vals.items()}
        summary[phase] = {
            "count": int(len(acc[phase])),
            "weight_sum": float(sum(weights)),
            "weighted": weighted,
            "median": median,
        }
    return summary


def phase_delta_summary(
    phase_summary: dict[str, dict], bridge_enabled: bool
) -> dict[str, dict[str, dict[str, float]]]:
    """Return weighted-metric deltas between consecutive phases.

    Examples:
        >>> phase_summary = {
        ...     "knn_raw": {"weighted": {"iou": 0.1, "f1": 0.2, "precision": 0.3, "recall": 0.4}},
        ...     "knn_crf": {"weighted": {"iou": 0.2, "f1": 0.3, "precision": 0.4, "recall": 0.5}},
        ... }
        >>> deltas = phase_delta_summary(phase_summary, bridge_enabled=False)
        >>> round(deltas["knn"]["knn_raw_to_knn_crf"]["delta_iou"], 3)
        0.1
    """
    metric_keys = ["iou", "f1", "precision", "recall"]
    chains = {
        "knn": ["knn_raw", "knn_crf", "knn_shadow"],
        "xgb": ["xgb_raw", "xgb_crf", "xgb_shadow"],
        "champion": ["champion_raw", "champion_crf"],
    }
    if bridge_enabled:
        chains["champion"].append("champion_bridge")
    chains["champion"].append("champion_shadow")
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for chain_name, chain in chains.items():
        chain_deltas: dict[str, dict[str, float]] = {}
        for prev, curr in zip(chain, chain[1:]):
            if prev not in phase_summary or curr not in phase_summary:
                continue
            prev_metrics = phase_summary[prev]["weighted"]
            curr_metrics = phase_summary[curr]["weighted"]
            key = f"{prev}_to_{curr}"
            chain_deltas[key] = {
                f"delta_{metric}": float(curr_metrics[metric] - prev_metrics[metric])
                for metric in metric_keys
            }
        if chain_deltas:
            deltas[chain_name] = chain_deltas
    return deltas


def aggregate_timings(timings: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate timing dictionaries into summary stats.

    Examples:
        >>> summary = aggregate_timings([{"step": 1.0}, {"step": 3.0}])
        >>> round(summary["step"]["mean_s"], 3)
        2.0
    """
    keys = sorted({key for timing in timings for key in timing.keys()})
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(timing[key])
            for timing in timings
            if key in timing and timing[key] is not None
        ]
        if not values:
            continue
        stats[key] = {
            "count": int(len(values)),
            "total_s": float(np.sum(values)),
            "mean_s": float(np.mean(values)),
            "median_s": float(np.median(values)),
            "min_s": float(np.min(values)),
            "max_s": float(np.max(values)),
        }
    return stats


def timing_summary(timings: list[dict]) -> dict[str, object]:
    """Return a timing summary with totals and step stats.

    Examples:
        >>> summary = timing_summary([{"total_s": 2.0}, {"total_s": 4.0}])
        >>> round(summary["total"]["mean_s"], 3)
        3.0
    """
    summary: dict[str, object] = {"tiles": int(len(timings)), "steps": {}}
    if not timings:
        return summary
    summary["steps"] = aggregate_timings(timings)
    total_vals = [
        float(timing["total_s"])
        for timing in timings
        if "total_s" in timing and timing["total_s"] is not None
    ]
    if total_vals:
        summary["total"] = {
            "total_s": float(np.sum(total_vals)),
            "mean_s": float(np.mean(total_vals)),
            "median_s": float(np.median(total_vals)),
            "min_s": float(np.min(total_vals)),
            "max_s": float(np.max(total_vals)),
        }
    return summary
