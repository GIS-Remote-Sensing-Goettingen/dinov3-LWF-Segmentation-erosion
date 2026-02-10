"""Tests for timing CSV helper utilities."""

from __future__ import annotations

import csv
from pathlib import Path

from segedge.core.timing_csv import (
    DETAIL_COLUMNS,
    SUMMARY_COLUMNS,
    append_tile_timing_csv_rows,
    build_tile_timing_rows,
    read_timing_detail_csv,
    summarize_timing_rows,
    write_timing_summary_csv,
)


def test_build_tile_timing_rows_expands_per_phase() -> None:
    """Build rows should expand one row per phase key.

    Examples:
        >>> True
        True
    """
    rows = build_tile_timing_rows(
        run_dir="output/run_001",
        stage="validation_inference",
        tile_role="validation",
        tile_path="/tmp/tile_a.tif",
        image_id="tile_a",
        timings={"knn_s": 1.0, "crf_s": 2.0},
        gt_available=True,
        source_mode="manual",
        auto_split_mode="disabled",
        resample_factor=1,
        tile_size=2048,
        stride=512,
        status="done",
        timestamp_utc="2026-02-10T00:00:00",
    )
    assert len(rows) == 2
    assert all(set(DETAIL_COLUMNS).issubset(set(r.keys())) for r in rows)
    assert sorted(r["phase_name"] for r in rows) == ["crf_s", "knn_s"]


def test_append_tile_timing_csv_rows_writes_header_once(tmp_path: Path) -> None:
    """Appending rows twice should keep a single CSV header.

    Examples:
        >>> True
        True
    """
    out_path = tmp_path / "tile_phase_timing.csv"
    row = {
        "run_dir": "output/run_001",
        "timestamp_utc": "2026-02-10T00:00:00",
        "stage": "holdout_inference",
        "tile_role": "holdout",
        "tile_path": "/tmp/tile_b.tif",
        "image_id": "tile_b",
        "phase_name": "total_s",
        "duration_s": 3.5,
        "gt_available": False,
        "source_mode": "manual",
        "auto_split_mode": "gt_to_val_cap_holdout",
        "resample_factor": 1,
        "tile_size": 2048,
        "stride": 512,
        "status": "done",
    }
    append_tile_timing_csv_rows(str(out_path), [row])
    append_tile_timing_csv_rows(str(out_path), [row])
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].split(",") == DETAIL_COLUMNS


def test_summarize_timing_rows_scopes_and_runtime_share() -> None:
    """Summary rows should include stage scope and all-inference scope.

    Examples:
        >>> True
        True
    """
    detail_rows = [
        {
            "stage": "validation_inference",
            "phase_name": "knn_s",
            "duration_s": 2.0,
            "status": "done",
        },
        {
            "stage": "validation_inference",
            "phase_name": "crf_s",
            "duration_s": 3.0,
            "status": "done",
        },
        {
            "stage": "holdout_inference",
            "phase_name": "knn_s",
            "duration_s": 1.0,
            "status": "done",
        },
    ]
    summary = summarize_timing_rows(detail_rows)
    scopes = {str(r["scope"]) for r in summary}
    assert "validation_inference" in scopes
    assert "holdout_inference" in scopes
    assert "all_inference" in scopes
    val_knn = [
        r
        for r in summary
        if r["scope"] == "validation_inference" and r["phase_name"] == "knn_s"
    ][0]
    assert float(val_knn["runtime_share_pct"]) > 0.0


def test_read_timing_detail_csv_parses_duration(tmp_path: Path) -> None:
    """Reader should parse numeric duration values from detail CSV.

    Examples:
        >>> True
        True
    """
    out_path = tmp_path / "tile_phase_timing.csv"
    append_tile_timing_csv_rows(
        str(out_path),
        [
            {
                "run_dir": "output/run_001",
                "timestamp_utc": "2026-02-10T00:00:00",
                "stage": "source_training",
                "tile_role": "source",
                "tile_path": "/tmp/tile_x.tif",
                "image_id": "tile_x",
                "phase_name": "source_tile_total_s",
                "duration_s": 9.5,
                "gt_available": True,
                "source_mode": "manual",
                "auto_split_mode": "disabled",
                "resample_factor": 1,
                "tile_size": 2048,
                "stride": 512,
                "status": "done",
            }
        ],
    )
    rows = read_timing_detail_csv(str(out_path))
    assert len(rows) == 1
    assert isinstance(rows[0]["duration_s"], float)


def test_write_timing_summary_csv_writes_expected_columns(tmp_path: Path) -> None:
    """Summary CSV writer should persist the canonical summary schema.

    Examples:
        >>> True
        True
    """
    out_path = tmp_path / "timing_opportunity_cost.csv"
    write_timing_summary_csv(
        str(out_path),
        [
            {
                "stage": "source_training",
                "phase_name": "build_banks_s",
                "duration_s": 4.2,
                "status": "done",
            }
        ],
    )
    with out_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == SUMMARY_COLUMNS
        rows = list(reader)
    assert len(rows) >= 1
