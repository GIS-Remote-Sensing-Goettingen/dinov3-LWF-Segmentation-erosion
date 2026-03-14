"""Analyze structured SegEdge performance logs.

Examples:
    >>> parse_kind_filter("span")
    {'span'}
    >>> parse_phase_filter(None)
    'holdout_inference'
    >>> summarize_durations([1.0, 2.0, 3.0])["median_s"]
    2.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PerformanceRecord:
    """One parsed performance-log row."""

    duration_s: float
    extra: dict[str, object]
    image_id: str | None
    kind: str
    phase: str | None
    run_id: str | None
    stage: str | None
    substage: str | None
    tile: str | None
    ts: str | None


def parse_kind_filter(kind: str) -> set[str]:
    """Return the record kinds selected by the CLI flag.

    Examples:
        >>> sorted(parse_kind_filter("all"))
        ['span', 'timer']
    """
    normalized = kind.strip().lower()
    if normalized == "all":
        return {"span", "timer"}
    if normalized not in {"span", "timer"}:
        raise ValueError(f"unsupported kind filter: {kind}")
    return {normalized}


def parse_phase_filter(phase: str | None) -> str | None:
    """Normalize the phase filter used by the CLI.

    Examples:
        >>> parse_phase_filter("all") is None
        True
    """
    if phase is None:
        return "holdout_inference"
    normalized = phase.strip()
    if normalized.lower() == "all":
        return None
    return normalized


def load_performance_records(path: Path) -> list[PerformanceRecord]:
    """Load structured performance records from a JSONL file.

    Examples:
        >>> isinstance(load_performance_records(Path("scripts/analyze_performance_log.py")), list)
        True
    """
    records: list[PerformanceRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        records.append(
            PerformanceRecord(
                duration_s=float(payload.get("duration_s", 0.0)),
                extra=dict(payload.get("extra", {})),
                image_id=payload.get("image_id"),
                kind=str(payload.get("kind", "")),
                phase=payload.get("phase"),
                run_id=payload.get("run_id"),
                stage=payload.get("stage"),
                substage=payload.get("substage"),
                tile=payload.get("tile"),
                ts=payload.get("ts"),
            )
        )
    return records


def filter_records(
    records: Iterable[PerformanceRecord],
    *,
    kinds: set[str],
    phase: str | None,
    include_tile_null: bool = True,
    focus: str | None = None,
) -> list[PerformanceRecord]:
    """Filter out summary rows and keep only the requested kinds.

    Examples:
        >>> rows = [PerformanceRecord(1.0, {}, None, "summary", None, None, "a", None, None, None)]
        >>> filter_records(rows, kinds={"span"}, phase=None)
        []
    """
    out: list[PerformanceRecord] = []
    for record in records:
        if record.kind not in kinds:
            continue
        if phase is not None and record.phase != phase:
            continue
        if not include_tile_null and not record.tile:
            continue
        if focus:
            token = f"{record.stage or ''}::{record.substage or ''}".lower()
            focus_lower = focus.lower()
            if (
                focus_lower not in token
                and focus_lower not in (record.stage or "").lower()
            ):
                continue
        out.append(record)
    return out


def _nearest_rank(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile from a list of numeric values.

    Examples:
        >>> _nearest_rank([1.0, 2.0, 3.0], 0.9)
        3.0
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize_durations(values: list[float]) -> dict[str, float | int]:
    """Compute summary statistics for a list of durations.

    Examples:
        >>> summarize_durations([2.0])["count"]
        1
    """
    if not values:
        return {
            "count": 0,
            "max_s": 0.0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "min_s": 0.0,
            "p90_s": 0.0,
            "total_s": 0.0,
        }
    return {
        "count": len(values),
        "max_s": max(values),
        "mean_s": sum(values) / len(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "p90_s": _nearest_rank(values, 0.9),
        "total_s": sum(values),
    }


def build_stage_summary(
    records: Iterable[PerformanceRecord],
) -> list[dict[str, float | int | str]]:
    """Group records by stage.

    Examples:
        >>> rows = [PerformanceRecord(1.0, {}, None, "span", None, None, "x", None, None, None)]
        >>> build_stage_summary(rows)[0]["stage"]
        'x'
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.stage is None:
            continue
        grouped[record.stage].append(record.duration_s)
    rows: list[dict[str, float | int | str]] = []
    for stage, durations in grouped.items():
        rows.append({"stage": stage, **summarize_durations(durations)})
    return sorted(rows, key=lambda row: float(row["total_s"]), reverse=True)


def build_substage_summary(
    records: Iterable[PerformanceRecord],
) -> list[dict[str, float | int | str]]:
    """Group records by stage and substage.

    Examples:
        >>> rows = [PerformanceRecord(1.0, {}, None, "span", None, None, "x", None, None, None)]
        >>> build_substage_summary(rows)[0]["substage"]
        '<none>'
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.stage is None:
            continue
        substage = record.substage or "<none>"
        grouped[(record.stage, substage)].append(record.duration_s)
    rows: list[dict[str, float | int | str]] = []
    for (stage, substage), durations in grouped.items():
        rows.append(
            {
                "stage": stage,
                "substage": substage,
                **summarize_durations(durations),
            }
        )
    return sorted(rows, key=lambda row: float(row["total_s"]), reverse=True)


def build_tile_summary(
    records: Iterable[PerformanceRecord],
) -> tuple[list[dict[str, float | int | str]], dict[str, dict[str, float]]]:
    """Build traced-total summaries per tile.

    Examples:
        >>> rows = [PerformanceRecord(1.0, {}, None, "span", None, None, "x", "a", "tile", None)]
        >>> build_tile_summary(rows)[0][0]["tile"]
        'tile'
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    contributors: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in records:
        if not record.tile:
            continue
        grouped[record.tile].append(record.duration_s)
        key = f"{record.stage or '<none>'}::{record.substage or '<none>'}"
        contributors[record.tile][key] += record.duration_s
    rows: list[dict[str, float | int | str]] = []
    for tile, durations in grouped.items():
        rows.append({"tile": tile, **summarize_durations(durations)})
    rows.sort(key=lambda row: float(row["total_s"]), reverse=True)
    return rows, contributors


def overview(records: list[PerformanceRecord]) -> dict[str, object]:
    """Return high-level dataset counts for reporting.

    Examples:
        >>> rows = [
        ...     PerformanceRecord(
        ...         1.0,
        ...         {},
        ...         "run_1",
        ...         "span",
        ...         "holdout_inference",
        ...         "img",
        ...         "x",
        ...         "sub",
        ...         "tile",
        ...         None,
        ...     )
        ... ]
        >>> overview(rows)["tile_count"]
        1
    """
    kinds = Counter(record.kind for record in records)
    phases = Counter(record.phase or "<none>" for record in records)
    tiles = {record.tile for record in records if record.tile}
    run_ids = sorted({record.run_id for record in records if record.run_id})
    return {
        "kind_counts": dict(kinds),
        "phase_counts": dict(phases),
        "null_phase_count": sum(1 for record in records if record.phase is None),
        "null_tile_count": sum(1 for record in records if not record.tile),
        "row_count": len(records),
        "run_ids": run_ids,
        "tile_count": len(tiles),
    }


def build_exclusion_summary(
    records: list[PerformanceRecord],
    selected: list[PerformanceRecord],
    *,
    selected_phase: str | None,
    include_tile_null: bool,
) -> dict[str, int]:
    """Summarize how many rows were excluded by the active filters.

    Examples:
        >>> rec = PerformanceRecord(
        ...     1.0, {}, None, "span", "holdout_inference", None, "x", None, "tile", None
        ... )
        >>> build_exclusion_summary(
        ...     [rec],
        ...     [rec],
        ...     selected_phase="holdout_inference",
        ...     include_tile_null=False,
        ... )["selected_rows"]
        1
    """
    selected_ids = {id(record) for record in selected}
    excluded = [record for record in records if id(record) not in selected_ids]
    selected_kinds = Counter(record.kind for record in selected)
    excluded_phase_null = sum(1 for record in excluded if record.phase is None)
    excluded_other_phase = 0
    if selected_phase is not None:
        excluded_other_phase = sum(
            1
            for record in excluded
            if record.phase is not None and record.phase != selected_phase
        )
    excluded_tile_null = 0
    if not include_tile_null:
        excluded_tile_null = sum(
            1
            for record in excluded
            if not record.tile
            and (selected_phase is None or record.phase == selected_phase)
        )
    return {
        "excluded_rows": len(excluded),
        "excluded_phase_null": excluded_phase_null,
        "excluded_other_phase": excluded_other_phase,
        "excluded_tile_null": excluded_tile_null,
        "selected_rows": len(selected),
        "selected_timer_rows": selected_kinds.get("timer", 0),
        "selected_span_rows": selected_kinds.get("span", 0),
    }


def build_outlier_rows(
    records: Iterable[PerformanceRecord],
    *,
    stage: str,
    substage: str,
) -> list[dict[str, object]]:
    """Return rows for tiles with the largest cumulative duration for a stage/substage.

    Examples:
        >>> rec = PerformanceRecord(
        ...     2.0,
        ...     {},
        ...     None,
        ...     "span",
        ...     "holdout_inference",
        ...     None,
        ...     "roads_mask",
        ...     "tree_query",
        ...     "tile_a",
        ...     None,
        ... )
        >>> build_outlier_rows([rec], stage="roads_mask", substage="tree_query")[0]["tile"]
        'tile_a'
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not record.tile:
            continue
        if record.stage != stage or (record.substage or "<none>") != substage:
            continue
        grouped[record.tile].append(record.duration_s)
    rows: list[dict[str, object]] = []
    for tile, durations in grouped.items():
        rows.append({"tile": tile, **summarize_durations(durations)})
    return sorted(rows, key=lambda row: float(row["total_s"]), reverse=True)


def build_non_inference_summary(
    records: Iterable[PerformanceRecord],
) -> list[dict[str, object]]:
    """Summarize excluded non-inference rows by stage."""
    return build_stage_summary(list(records))


def format_table(
    rows: list[dict[str, object]],
    columns: list[tuple[str, str]],
    *,
    limit: int,
) -> str:
    """Format a simple fixed-width table for console output."""
    display_rows = rows[:limit]
    widths = []
    for key, title in columns:
        width = len(title)
        for row in display_rows:
            width = max(width, len(_format_cell(row.get(key))))
        widths.append(width)
    header = "  ".join(
        title.ljust(width) for (_, title), width in zip(columns, widths, strict=True)
    )
    sep = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(
            _format_cell(row.get(key)).ljust(width)
            for (key, _), width in zip(columns, widths, strict=True)
        )
        for row in display_rows
    ]
    return "\n".join([header, sep, *body]) if body else "\n".join([header, sep])


def _format_cell(value: object) -> str:
    """Format one table cell."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write summary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_text(
    stage_rows: list[dict[str, object]],
    substage_rows: list[dict[str, object]],
) -> str:
    """Build a short EDA interpretation block.

    Examples:
        >>> build_summary_text(
        ...     [{"stage": "x", "mean_s": 1.0, "total_s": 1.0}],
        ...     [{
        ...         "stage": "x",
        ...         "substage": "y",
        ...         "mean_s": 0.5,
        ...         "median_s": 0.5,
        ...         "max_s": 0.5,
        ...         "total_s": 0.5,
        ...     }],
        ... ).startswith("Interpretation")
        True
    """
    top_stages = ", ".join(
        f"{row['stage']} ({float(row['mean_s']):.2f}s avg)" for row in stage_rows[:3]
    )
    top_substages = ", ".join(
        (f"{row['stage']}::{row['substage']}" f" ({float(row['mean_s']):.2f}s avg)")
        for row in substage_rows[:3]
    )
    top_targets = ", ".join(
        f"{row['stage']}::{row['substage']} ({float(row['total_s']):.1f}s total)"
        for row in substage_rows[:5]
    )
    outlier_warning = "No major stage-outlier spread detected."
    for row in substage_rows:
        median = float(row["median_s"])
        max_s = float(row["max_s"])
        if max_s >= max(60.0, median * 5.0):
            outlier_warning = (
                "Outlier warning: "
                f"{row['stage']}::{row['substage']} peaks at {max_s:.2f}s "
                f"vs median {median:.2f}s."
            )
            break
    return "\n".join(
        [
            "Interpretation",
            "--------------",
            f"Top stage bottlenecks: {top_stages or 'n/a'}",
            f"Top substage bottlenecks: {top_substages or 'n/a'}",
            f"Primary optimization targets: {top_targets or 'n/a'}",
            outlier_warning,
            (
                "Nested spans are reported separately; do not add parent and child "
                "totals directly when comparing stages."
            ),
        ]
    )


def main() -> int:
    """Run CLI analysis for a performance log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to performance.jsonl")
    parser.add_argument(
        "--kind",
        default="span",
        choices=["span", "timer", "all"],
        help="Record kind to analyze (default: span).",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help=(
            "Phase filter. Defaults to holdout_inference for mixed train+infer logs. "
            "Use 'all' to include every phase."
        ),
    )
    parser.add_argument(
        "--include-tile-null",
        action="store_true",
        help="Include records with tile=null in the stage summaries.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of rows to print for stage/substage tables.",
    )
    parser.add_argument(
        "--tile-limit",
        type=int,
        default=5,
        help="Number of hottest tiles to print.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV outputs.",
    )
    parser.add_argument(
        "--focus",
        default=None,
        help=(
            "Optional substring filter over stage/substage names, e.g. "
            "'load_context' or 'roads_mask'."
        ),
    )
    args = parser.parse_args()

    records = load_performance_records(args.log_path)
    kinds = parse_kind_filter(args.kind)
    phase_filter = parse_phase_filter(args.phase)
    filtered = filter_records(
        records,
        kinds=kinds,
        phase=phase_filter,
        include_tile_null=args.include_tile_null,
        focus=args.focus,
    )
    stage_rows = build_stage_summary(filtered)
    substage_rows = build_substage_summary(filtered)
    tile_rows, contributors = build_tile_summary(filtered)
    exclusions = build_exclusion_summary(
        records,
        filtered,
        selected_phase=phase_filter,
        include_tile_null=args.include_tile_null,
    )
    excluded_rows = [
        record
        for record in records
        if record.kind in kinds
        and (
            (phase_filter is not None and record.phase != phase_filter)
            or (not args.include_tile_null and not record.tile)
        )
    ]
    excluded_stage_rows = build_non_inference_summary(excluded_rows)
    outlier_rows = build_outlier_rows(
        filtered,
        stage="infer_on_holdout",
        substage="load_context",
    )

    meta = overview(records)
    filtered_meta = overview(filtered)
    print("Performance Log Overview")
    print("------------------------")
    print(f"log_path: {args.log_path}")
    print(f"rows: {meta['row_count']}")
    print(f"selected_rows: {filtered_meta['row_count']}")
    print(f"run_ids: {', '.join(meta['run_ids']) or '<none>'}")
    print(f"kind_counts: {meta['kind_counts']}")
    print(f"selected_kind_counts: {filtered_meta['kind_counts']}")
    print(f"phase_counts: {meta['phase_counts']}")
    print(f"tile_count: {filtered_meta['tile_count']}")
    if args.focus:
        print(f"focus: {args.focus}")
    if phase_filter == "holdout_inference" and not args.include_tile_null:
        print("selection: inference-only spans (phase=holdout_inference, tile!=null)")
    print()

    print("Excluded Records")
    print("----------------")
    print(f"excluded_rows: {exclusions['excluded_rows']}")
    print(f"excluded_phase_null: {exclusions['excluded_phase_null']}")
    print(f"excluded_other_phase: {exclusions['excluded_other_phase']}")
    print(f"excluded_tile_null: {exclusions['excluded_tile_null']}")
    print()

    print("Average Time By Stage")
    print("---------------------")
    print(
        format_table(
            stage_rows,
            [
                ("stage", "stage"),
                ("count", "count"),
                ("mean_s", "mean_s"),
                ("median_s", "median_s"),
                ("p90_s", "p90_s"),
                ("total_s", "total_s"),
                ("max_s", "max_s"),
            ],
            limit=args.top,
        )
    )
    print()
    print("Average Time By Stage/Substage")
    print("------------------------------")
    print(
        format_table(
            substage_rows,
            [
                ("stage", "stage"),
                ("substage", "substage"),
                ("count", "count"),
                ("mean_s", "mean_s"),
                ("median_s", "median_s"),
                ("p90_s", "p90_s"),
                ("total_s", "total_s"),
                ("max_s", "max_s"),
            ],
            limit=args.top,
        )
    )
    print()

    if excluded_stage_rows:
        print("Non-Selected Stage Overview")
        print("---------------------------")
        print(
            format_table(
                excluded_stage_rows,
                [
                    ("stage", "stage"),
                    ("count", "count"),
                    ("mean_s", "mean_s"),
                    ("median_s", "median_s"),
                    ("p90_s", "p90_s"),
                    ("total_s", "total_s"),
                    ("max_s", "max_s"),
                ],
                limit=min(5, args.top),
            )
        )
        print()

    if tile_rows:
        tile_totals = [float(row["total_s"]) for row in tile_rows]
        print("Per-Tile Traced Totals")
        print("----------------------")
        print(f"mean_total_s: {sum(tile_totals) / len(tile_totals):.3f}")
        print(f"median_total_s: {statistics.median(tile_totals):.3f}")
        print(f"min_total_s: {min(tile_totals):.3f}")
        print(f"max_total_s: {max(tile_totals):.3f}")
        print()
        print("Hottest Tiles")
        print("-------------")
        for row in tile_rows[: args.tile_limit]:
            tile = str(row["tile"])
            print(f"{tile}")
            print(
                "  traced_total_s="
                f"{float(row['total_s']):.3f} across {int(row['count'])} records"
            )
            top_parts = sorted(
                contributors[tile].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            for key, duration in top_parts:
                print(f"  {key}: {duration:.3f}s")
            print()

    if outlier_rows:
        print("Outlier Tiles By load_context")
        print("-----------------------------")
        print(
            format_table(
                outlier_rows,
                [
                    ("tile", "tile"),
                    ("count", "count"),
                    ("mean_s", "mean_s"),
                    ("median_s", "median_s"),
                    ("p90_s", "p90_s"),
                    ("total_s", "total_s"),
                    ("max_s", "max_s"),
                ],
                limit=min(5, args.tile_limit),
            )
        )
        print()

    print(build_summary_text(stage_rows, substage_rows))

    if args.out_dir is not None:
        write_csv(
            args.out_dir / "stage_summary.csv",
            stage_rows,
            [
                "stage",
                "count",
                "mean_s",
                "median_s",
                "p90_s",
                "min_s",
                "max_s",
                "total_s",
            ],
        )
        write_csv(
            args.out_dir / "substage_summary.csv",
            substage_rows,
            [
                "stage",
                "substage",
                "count",
                "mean_s",
                "median_s",
                "p90_s",
                "min_s",
                "max_s",
                "total_s",
            ],
        )
        write_csv(
            args.out_dir / "tile_summary.csv",
            tile_rows,
            [
                "tile",
                "count",
                "mean_s",
                "median_s",
                "p90_s",
                "min_s",
                "max_s",
                "total_s",
            ],
        )
        hottest_rows: list[dict[str, object]] = []
        for row in tile_rows[: args.tile_limit]:
            tile = str(row["tile"])
            top_parts = sorted(
                contributors[tile].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            for rank, (key, duration) in enumerate(top_parts, start=1):
                hottest_rows.append(
                    {
                        "tile": tile,
                        "rank": rank,
                        "contributor": key,
                        "duration_s": duration,
                    }
                )
        write_csv(
            args.out_dir / "hottest_tiles.csv",
            hottest_rows,
            ["tile", "rank", "contributor", "duration_s"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
