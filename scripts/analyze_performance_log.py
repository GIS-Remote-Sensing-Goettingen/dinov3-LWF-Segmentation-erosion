"""Analyze structured SegEdge performance logs.

Examples:
    >>> parse_kind_filter("span")
    {'span'}
    >>> summarize_durations([1.0, 2.0, 3.0])["median_s"]
    2.0
"""

from __future__ import annotations

import argparse
import csv
import json
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
        out.append(record)
    return out


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
            "total_s": 0.0,
        }
    return {
        "count": len(values),
        "max_s": max(values),
        "mean_s": sum(values) / len(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
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
    """Return high-level dataset counts for reporting."""
    kinds = Counter(record.kind for record in records)
    phases = Counter(record.phase or "<none>" for record in records)
    tiles = {record.tile for record in records if record.tile}
    run_ids = sorted({record.run_id for record in records if record.run_id})
    return {
        "kind_counts": dict(kinds),
        "phase_counts": dict(phases),
        "row_count": len(records),
        "run_ids": run_ids,
        "tile_count": len(tiles),
    }


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
        ...     [{"stage": "x", "mean_s": 1.0}],
        ...     [{"stage": "x", "substage": "y", "mean_s": 0.5}],
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
    return "\n".join(
        [
            "Interpretation",
            "--------------",
            f"Top stage bottlenecks: {top_stages or 'n/a'}",
            f"Top substage bottlenecks: {top_substages or 'n/a'}",
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
        help="Optional phase filter, e.g. holdout_inference.",
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
    args = parser.parse_args()

    records = load_performance_records(args.log_path)
    kinds = parse_kind_filter(args.kind)
    filtered = filter_records(records, kinds=kinds, phase=args.phase)
    stage_rows = build_stage_summary(filtered)
    substage_rows = build_substage_summary(filtered)
    tile_rows, contributors = build_tile_summary(filtered)

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
                ("total_s", "total_s"),
                ("max_s", "max_s"),
            ],
            limit=args.top,
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

    print(build_summary_text(stage_rows, substage_rows))

    if args.out_dir is not None:
        write_csv(
            args.out_dir / "stage_summary.csv",
            stage_rows,
            ["stage", "count", "mean_s", "median_s", "min_s", "max_s", "total_s"],
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
                "min_s",
                "max_s",
                "total_s",
            ],
        )
        write_csv(
            args.out_dir / "tile_summary.csv",
            tile_rows,
            ["tile", "count", "mean_s", "median_s", "min_s", "max_s", "total_s"],
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
