"""Helpers for incremental timing CSV telemetry.

Examples:
    >>> rows = build_tile_timing_rows(
    ...     run_dir="output/run_001",
    ...     stage="holdout_inference",
    ...     tile_role="holdout",
    ...     tile_path="tile.tif",
    ...     image_id="tile",
    ...     timings={"crf_s": 1.2, "total_s": 2.5},
    ...     gt_available=False,
    ...     source_mode="manual",
    ...     auto_split_mode="gt_to_val_cap_holdout",
    ...     resample_factor=1,
    ...     tile_size=2048,
    ...     stride=512,
    ...     status="done",
    ...     timestamp_utc="2026-02-10T00:00:00",
    ... )
    >>> rows[0]["phase_name"], rows[0]["duration_s"]
    ('crf_s', 1.2)
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from statistics import mean, median

DETAIL_COLUMNS = [
    "run_dir",
    "timestamp_utc",
    "stage",
    "tile_role",
    "tile_path",
    "image_id",
    "phase_name",
    "duration_s",
    "gt_available",
    "source_mode",
    "auto_split_mode",
    "resample_factor",
    "tile_size",
    "stride",
    "status",
]

SUMMARY_COLUMNS = [
    "scope",
    "phase_name",
    "count",
    "total_s",
    "mean_s",
    "median_s",
    "min_s",
    "max_s",
    "runtime_share_pct",
    "optional_phase",
    "phase_group",
    "opportunity_rank",
]


def _phase_group(phase_name: str) -> str:
    """Return a stable phase group label for a timing key.

    Examples:
        >>> _phase_group("crf_s")
        'crf'
        >>> _phase_group("prefetch_features_s")
        'features'
    """
    if phase_name.startswith("load_") or "context" in phase_name:
        return "data_loading"
    if "prefetch" in phase_name or "feature" in phase_name:
        return "features"
    if phase_name.startswith("knn_") or phase_name == "knn_s":
        return "knn"
    if phase_name.startswith("xgb_") or phase_name == "xgb_s":
        return "xgb"
    if "threshold" in phase_name:
        return "thresholding"
    if "crf" in phase_name:
        return "crf"
    if "bridge" in phase_name or "skeleton" in phase_name:
        return "bridge"
    if "shadow" in phase_name:
        return "shadow"
    if "plot" in phase_name:
        return "plotting"
    if phase_name.startswith("export") or "append_" in phase_name:
        return "io_export"
    return "other"


def _is_optional_phase(phase_name: str) -> bool:
    """Return whether a phase is optional for runtime opportunity analysis.

    Examples:
        >>> _is_optional_phase("bridge_s")
        True
        >>> _is_optional_phase("prefetch_features_s")
        False
    """
    return (
        "bridge" in phase_name
        or "shadow" in phase_name
        or "plot" in phase_name
        or phase_name.startswith("roads_")
    )


def build_tile_timing_rows(
    run_dir: str,
    stage: str,
    tile_role: str,
    tile_path: str,
    image_id: str,
    timings: dict[str, float],
    gt_available: bool,
    source_mode: str,
    auto_split_mode: str,
    resample_factor: int,
    tile_size: int,
    stride: int,
    status: str,
    timestamp_utc: str,
) -> list[dict[str, object]]:
    """Expand a timing dictionary into normalized CSV rows.

    Args:
        run_dir (str): Run directory.
        stage (str): Pipeline stage label.
        tile_role (str): Tile role label.
        tile_path (str): Tile path.
        image_id (str): Tile image id.
        timings (dict[str, float]): Timing map.
        gt_available (bool): Whether GT is available for the tile.
        source_mode (str): Source tile mode.
        auto_split_mode (str): Auto split mode.
        resample_factor (int): Resample factor.
        tile_size (int): Tile size.
        stride (int): Tile stride.
        status (str): Tile status.
        timestamp_utc (str): UTC timestamp.

    Returns:
        list[dict[str, object]]: One row per timing key.

    Examples:
        >>> rows = build_tile_timing_rows(
        ...     run_dir="output/run_001",
        ...     stage="source_training",
        ...     tile_role="source",
        ...     tile_path="tile.tif",
        ...     image_id="tile",
        ...     timings={"a_s": 1.0, "b_s": 2.0},
        ...     gt_available=True,
        ...     source_mode="manual",
        ...     auto_split_mode="disabled",
        ...     resample_factor=1,
        ...     tile_size=2048,
        ...     stride=512,
        ...     status="done",
        ...     timestamp_utc="2026-02-10T00:00:00",
        ... )
        >>> len(rows)
        2
    """
    rows: list[dict[str, object]] = []
    for phase_name in sorted(timings):
        duration = float(timings[phase_name])
        rows.append(
            {
                "run_dir": run_dir,
                "timestamp_utc": timestamp_utc,
                "stage": stage,
                "tile_role": tile_role,
                "tile_path": tile_path,
                "image_id": image_id,
                "phase_name": phase_name,
                "duration_s": duration,
                "gt_available": bool(gt_available),
                "source_mode": source_mode,
                "auto_split_mode": auto_split_mode,
                "resample_factor": int(resample_factor),
                "tile_size": int(tile_size),
                "stride": int(stride),
                "status": status,
            }
        )
    return rows


def append_csv_rows(
    out_path: str, columns: list[str], rows: list[dict[str, object]]
) -> None:
    """Append CSV rows and write header once.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "x.csv")
        ...     append_csv_rows(path, ["a", "b"], [{"a": 1, "b": 2}])
        ...     append_csv_rows(path, ["a", "b"], [{"a": 3, "b": 4}])
        ...     with open(path, "r", encoding="utf-8") as f:
        ...         len([ln for ln in f.read().splitlines() if ln.strip()])
        3
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()


def read_timing_detail_csv(out_path: str) -> list[dict[str, object]]:
    """Read detailed timing rows from CSV when present.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "detail.csv")
        ...     append_tile_timing_csv_rows(path, [{"phase_name": "x", "duration_s": 1.0}])
        ...     rows = read_timing_detail_csv(path)
        ...     len(rows)
        1
    """
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return []
    rows: list[dict[str, object]] = []
    with open(out_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = dict(row)
            if "duration_s" in parsed:
                try:
                    parsed["duration_s"] = float(parsed["duration_s"])
                except (TypeError, ValueError):
                    parsed["duration_s"] = 0.0
            rows.append(parsed)
    return rows


def summarize_timing_rows(
    detail_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate detailed timing rows into opportunity-cost summary rows.

    Examples:
        >>> summary = summarize_timing_rows(
        ...     [
        ...         {
        ...             "stage": "holdout_inference",
        ...             "phase_name": "crf_s",
        ...             "duration_s": 2.0,
        ...             "status": "done",
        ...         },
        ...         {
        ...             "stage": "holdout_inference",
        ...             "phase_name": "knn_s",
        ...             "duration_s": 1.0,
        ...             "status": "done",
        ...         },
        ...     ]
        ... )
        >>> any(r["scope"] == "holdout_inference" and r["phase_name"] == "crf_s" for r in summary)
        True
    """
    rows_done = [r for r in detail_rows if str(r.get("status", "done")) == "done"]
    if not rows_done:
        return []

    scope_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows_done:
        stage = str(row.get("stage", "unknown"))
        scope_rows[stage].append(row)
        scope_rows["all"].append(row)
        if stage in {"validation_inference", "holdout_inference"}:
            scope_rows["all_inference"].append(row)

    out_rows: list[dict[str, object]] = []
    for scope in sorted(scope_rows):
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in scope_rows[scope]:
            grouped[str(row["phase_name"])].append(float(row["duration_s"]))

        totals_by_phase = {k: float(sum(v)) for k, v in grouped.items()}
        total_scope_s = float(sum(totals_by_phase.values()))
        ranked = sorted(totals_by_phase.items(), key=lambda kv: kv[1], reverse=True)
        rank_map = {phase: idx + 1 for idx, (phase, _) in enumerate(ranked)}

        for phase_name in sorted(grouped):
            vals = grouped[phase_name]
            total_s = float(sum(vals))
            runtime_share_pct = (
                (100.0 * total_s / total_scope_s) if total_scope_s > 0 else 0.0
            )
            out_rows.append(
                {
                    "scope": scope,
                    "phase_name": phase_name,
                    "count": int(len(vals)),
                    "total_s": total_s,
                    "mean_s": float(mean(vals)),
                    "median_s": float(median(vals)),
                    "min_s": float(min(vals)),
                    "max_s": float(max(vals)),
                    "runtime_share_pct": runtime_share_pct,
                    "optional_phase": (
                        "yes" if _is_optional_phase(phase_name) else "no"
                    ),
                    "phase_group": _phase_group(phase_name),
                    "opportunity_rank": int(rank_map[phase_name]),
                }
            )
    return out_rows


def write_timing_summary_csv(
    out_path: str, detail_rows: list[dict[str, object]]
) -> None:
    """Write (overwrite) timing summary CSV from detailed rows.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "summary.csv")
        ...     write_timing_summary_csv(
        ...         path,
        ...         [
        ...             {
        ...                 "stage": "holdout_inference",
        ...                 "phase_name": "total_s",
        ...                 "duration_s": 1.0,
        ...                 "status": "done",
        ...             }
        ...         ],
        ...     )
        ...     os.path.exists(path)
        True
    """
    summary_rows = summarize_timing_rows(detail_rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)


def append_tile_timing_csv_rows(
    out_path: str,
    rows: list[dict[str, object]],
) -> None:
    """Append tile timing rows into the canonical detailed CSV schema.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "detail.csv")
        ...     append_tile_timing_csv_rows(path, [{"phase_name": "x", "duration_s": 1.0}])
        ...     os.path.exists(path)
        True
    """
    append_csv_rows(out_path, DETAIL_COLUMNS, rows)
