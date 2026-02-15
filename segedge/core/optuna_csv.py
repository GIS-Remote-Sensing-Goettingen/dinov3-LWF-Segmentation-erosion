"""Helpers for exporting Optuna study telemetry to CSV artifacts.

Examples:
    >>> isinstance(collect_optuna_trials_from_storage.__name__, str)
    True
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable
from datetime import datetime


def _normalize_storage_url(storage_path: str) -> str:
    """Normalize storage path/URL for optuna APIs.

    Examples:
        >>> _normalize_storage_url("output/x.db").startswith("sqlite:///")
        True
    """
    value = str(storage_path or "").strip()
    if "://" in value:
        return value
    return f"sqlite:///{value}"


def _trial_duration_s(trial) -> float | None:
    """Compute trial duration in seconds when timestamps are present.

    Examples:
        >>> isinstance(_trial_duration_s.__name__, str)
        True
    """
    dt_start = getattr(trial, "datetime_start", None)
    dt_end = getattr(trial, "datetime_complete", None)
    if isinstance(dt_start, datetime) and isinstance(dt_end, datetime):
        return float((dt_end - dt_start).total_seconds())
    return None


def _scalarize(value) -> str | int | float | bool:
    """Convert non-scalar objects into stable JSON strings.

    Examples:
        >>> _scalarize((1, 2))
        '[1, 2]'
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _study_trials_to_rows(
    *,
    study,
    stage: str,
    max_recent_trials: int | None = None,
) -> list[dict[str, object]]:
    """Convert study trials into wide CSV-ready rows.

    Examples:
        >>> isinstance(_study_trials_to_rows.__name__, str)
        True
    """
    trials = sorted(
        list(getattr(study, "trials", []) or []), key=lambda t: int(t.number)
    )
    if max_recent_trials is not None and max_recent_trials > 0:
        trials = trials[-int(max_recent_trials) :]

    rows: list[dict[str, object]] = []
    best_so_far = float("-inf")
    for idx, trial in enumerate(trials):
        value = None if trial.value is None else float(trial.value)
        state = str(trial.state).split(".")[-1]
        if value is not None and value > best_so_far:
            best_so_far = value
            is_best_so_far = 1
        else:
            is_best_so_far = 0
        dt_end = getattr(trial, "datetime_complete", None)
        dt_start = getattr(trial, "datetime_start", None)
        ts = dt_end if isinstance(dt_end, datetime) else dt_start
        row: dict[str, object] = {
            "stage": str(stage),
            "study_name": str(getattr(study, "study_name", "")),
            "trial_index_stage": int(idx),
            "trial_number_global": int(trial.number),
            "state": state,
            "objective": value if value is not None else "",
            "duration_s": _trial_duration_s(trial) or "",
            "is_best_so_far": int(is_best_so_far),
            "timestamp_utc": ts.isoformat() if isinstance(ts, datetime) else "",
        }
        for key, val in sorted((trial.params or {}).items(), key=lambda kv: kv[0]):
            row[f"param__{key}"] = _scalarize(val)
        for key, val in sorted((trial.user_attrs or {}).items(), key=lambda kv: kv[0]):
            row[f"attr__{key}"] = _scalarize(val)
        rows.append(row)
    return rows


def collect_optuna_trials_from_storage(
    *,
    optuna_mod,
    storage_path: str | None,
    study_specs: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Load studies and return merged per-trial rows.

    Study spec keys:
    - `study_name` (required)
    - `stage` (optional)
    - `max_recent_trials` (optional)

    Examples:
        >>> isinstance(collect_optuna_trials_from_storage.__name__, str)
        True
    """
    if optuna_mod is None or not storage_path:
        return []
    storage_url = _normalize_storage_url(str(storage_path))
    out_rows: list[dict[str, object]] = []
    for spec in study_specs:
        study_name = str(spec.get("study_name", "")).strip()
        if not study_name:
            continue
        stage = str(spec.get("stage", study_name))
        max_recent = spec.get("max_recent_trials")
        try:
            study = optuna_mod.load_study(study_name=study_name, storage=storage_url)
        except Exception:
            continue
        out_rows.extend(
            _study_trials_to_rows(
                study=study,
                stage=stage,
                max_recent_trials=(
                    None if max_recent in (None, "", 0) else int(max_recent)
                ),
            )
        )
    out_rows.sort(
        key=lambda r: (
            str(r.get("stage", "")),
            int(r.get("trial_index_stage", 0)),
        )
    )
    return out_rows


def write_optuna_trials_csv(output_path: str, rows: list[dict[str, object]]) -> None:
    """Write Optuna trial telemetry CSV.

    Examples:
        >>> isinstance(write_optuna_trials_csv.__name__, str)
        True
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_cols = [
        "timestamp_utc",
        "stage",
        "study_name",
        "trial_index_stage",
        "trial_number_global",
        "state",
        "objective",
        "duration_s",
        "is_best_so_far",
    ]
    dynamic_cols = sorted(
        {str(k) for row in rows for k in row.keys() if str(k) not in set(base_cols)}
    )
    fieldnames = base_cols + dynamic_cols
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_bayes_trial_phase_timing_csv(
    output_path: str,
    rows: list[dict[str, object]],
) -> None:
    """Write compact one-row-per-trial Bayesian phase timing CSV.

    Examples:
        >>> isinstance(write_bayes_trial_phase_timing_csv.__name__, str)
        True
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    phase_cols = [
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
    ]
    fieldnames = [
        "timestamp_utc",
        "stage",
        "study_name",
        "trial_index_stage",
        "trial_number_global",
        "state",
        "objective",
        "duration_s",
        "is_best_so_far",
    ] + phase_cols

    def _as_float(value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "timestamp_utc": row.get("timestamp_utc", ""),
                "stage": row.get("stage", ""),
                "study_name": row.get("study_name", ""),
                "trial_index_stage": row.get("trial_index_stage", ""),
                "trial_number_global": row.get("trial_number_global", ""),
                "state": row.get("state", ""),
                "objective": row.get("objective", ""),
                "duration_s": row.get("duration_s", ""),
                "is_best_so_far": row.get("is_best_so_far", ""),
            }
            for phase in phase_cols:
                attr_key = f"attr__{phase}"
                out[phase] = row.get(attr_key, "")
            if out.get("perturb_share_pct", "") in ("", None):
                total = _as_float(out.get("trial_total_s"))
                perturb = (
                    _as_float(out.get("score_perturb_knn_xgb_s"))
                    + _as_float(out.get("threshold_perturb_s"))
                    + _as_float(out.get("metrics_perturb_s"))
                )
                out["perturb_share_pct"] = (
                    (perturb / total * 100.0) if total > 0 else 0.0
                )
            if out.get("scoring_share_pct", "") in ("", None):
                total = _as_float(out.get("trial_total_s"))
                scoring = _as_float(out.get("score_base_knn_xgb_s")) + _as_float(
                    out.get("score_perturb_knn_xgb_s")
                )
                out["scoring_share_pct"] = (
                    (scoring / total * 100.0) if total > 0 else 0.0
                )
            writer.writerow(out)


def write_optuna_importance_csv(output_path: str, payload: dict) -> None:
    """Write stage-wise parameter importances to CSV.

    Examples:
        >>> isinstance(write_optuna_importance_csv.__name__, str)
        True
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows: list[dict[str, object]] = []
    for stage_key in ("stage1", "stage2", "stage3"):
        stage_obj = payload.get(stage_key, {})
        importances = (
            stage_obj.get("importances", {}) if isinstance(stage_obj, dict) else {}
        )
        rank = 1
        for param_name, score in sorted(
            ((str(k), float(v)) for k, v in importances.items()),
            key=lambda kv: kv[1],
            reverse=True,
        ):
            rows.append(
                {
                    "stage": stage_key,
                    "rank": rank,
                    "param_name": param_name,
                    "importance": score,
                }
            )
            rank += 1
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["stage", "rank", "param_name", "importance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
