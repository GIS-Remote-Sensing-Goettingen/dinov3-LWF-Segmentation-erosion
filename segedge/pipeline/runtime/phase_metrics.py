"""Phase-level metric accumulation and summaries."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Compute weighted mean with safe fallback.

    Examples:
        >>> _weighted_mean([1.0, 3.0], [1.0, 1.0])
        2.0
    """
    total_w = float(np.sum(weights))
    if total_w <= 0:
        return float(np.mean(values)) if values else 0.0
    return float(np.sum(np.array(values) * np.array(weights)) / total_w)


def _log_phase(kind: str, name: str) -> None:
    """Log a phase marker with ANSI color.

    Examples:
        >>> _log_phase("START", "demo")
    """
    msg = f"PHASE {kind}: {name}".upper()
    logger.info("\033[31m%s\033[0m", msg)


def _update_phase_metrics(acc: dict[str, list[dict]], metrics_map: dict) -> None:
    """Append metrics into a phase accumulator.

    Examples:
        >>> acc = {}
        >>> _update_phase_metrics(acc, {"demo": {"iou": 1.0}})
        >>> list(acc)
        ['demo']
    """
    for key, metrics in metrics_map.items():
        acc.setdefault(key, []).append(metrics)


def _summarize_phase_metrics(acc: dict[str, list[dict]], label: str) -> None:
    """Write weighted and median summaries for each phase.

    Examples:
        >>> _summarize_phase_metrics({}, "demo")
    """
    if not acc:
        logger.info("summary %s: no metrics", label)
        return
    metric_keys = ["iou", "f1", "precision", "recall"]
    phase_order = [
        "knn_raw",
        "knn_crf",
        "knn_shadow",
        "xgb_raw",
        "xgb_crf",
        "xgb_shadow",
        "champion_raw",
        "champion_crf",
        "champion_shadow",
    ]

    logger.info("summary %s: phase metrics", label)
    for phase in phase_order:
        if phase not in acc or not acc[phase]:
            continue
        weights = [float(m.get("_weight", 0.0)) for m in acc[phase]]
        vals = {k: [m.get(k, 0.0) for m in acc[phase]] for k in metric_keys}
        mean_vals = {k: _weighted_mean(v, weights) for k, v in vals.items()}
        med_vals = {k: float(np.median(v)) for k, v in vals.items()}
        logger.info(
            "summary %s %s wmean IoU=%.3f F1=%.3f P=%.3f R=%.3f | median IoU=%.3f F1=%.3f",
            label,
            phase,
            mean_vals["iou"],
            mean_vals["f1"],
            mean_vals["precision"],
            mean_vals["recall"],
            med_vals["iou"],
            med_vals["f1"],
        )

    champ_chain = ["champion_raw", "champion_crf", "champion_shadow"]
    for prev, curr in zip(champ_chain, champ_chain[1:], strict=True):
        if prev not in acc or curr not in acc:
            continue
        prev_weights = [float(m.get("_weight", 0.0)) for m in acc[prev]]
        curr_weights = [float(m.get("_weight", 0.0)) for m in acc[curr]]
        prev_iou = _weighted_mean([m.get("iou", 0.0) for m in acc[prev]], prev_weights)
        curr_iou = _weighted_mean([m.get("iou", 0.0) for m in acc[curr]], curr_weights)
        prev_f1 = _weighted_mean([m.get("f1", 0.0) for m in acc[prev]], prev_weights)
        curr_f1 = _weighted_mean([m.get("f1", 0.0) for m in acc[curr]], curr_weights)
        logger.info(
            "summary %s delta %s→%s IoU=%.3f F1=%.3f",
            label,
            prev,
            curr,
            float(curr_iou - prev_iou),
            float(curr_f1 - prev_f1),
        )


def summarize_phase_metrics_mean_std(
    phase_metrics: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Summarize phase metrics as mean/std pairs across runs.

    Examples:
        >>> summarize_phase_metrics_mean_std({})
        {}
    """
    out: dict[str, dict[str, float]] = {}
    metric_keys = ["iou", "f1", "precision", "recall"]
    for phase, rows in phase_metrics.items():
        if not rows:
            continue
        phase_summary: dict[str, float] = {}
        for key in metric_keys:
            vals = [float(r.get(key, 0.0)) for r in rows]
            phase_summary[f"{key}_mean"] = float(np.mean(vals))
            phase_summary[f"{key}_std"] = float(np.std(vals))
        out[phase] = phase_summary
    return out
