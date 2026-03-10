"""Tests for the performance-log EDA script."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_performance_log.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "analyze_performance_log",
    _MODULE_PATH,
)
assert _MODULE_SPEC is not None
assert _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

build_stage_summary = _MODULE.build_stage_summary
build_substage_summary = _MODULE.build_substage_summary
build_summary_text = _MODULE.build_summary_text
build_tile_summary = _MODULE.build_tile_summary
filter_records = _MODULE.filter_records
load_performance_records = _MODULE.load_performance_records
parse_kind_filter = _MODULE.parse_kind_filter
write_csv = _MODULE.write_csv


def _write_records(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_load_and_filter_records_ignore_summary_by_default(tmp_path):
    log_path = tmp_path / "performance.jsonl"
    _write_records(
        log_path,
        [
            {
                "duration_s": 4.0,
                "kind": "span",
                "phase": "holdout_inference",
                "stage": "infer_on_holdout",
                "substage": "xgb_stream",
                "tile": "tile_a.tif",
            },
            {
                "duration_s": 9.0,
                "kind": "summary",
                "phase": "holdout_inference",
                "stage": "rolling_summary",
                "tile": "tile_a.tif",
            },
        ],
    )

    records = load_performance_records(log_path)
    filtered = filter_records(records, kinds=parse_kind_filter("span"), phase=None)

    assert len(records) == 2
    assert len(filtered) == 1
    assert filtered[0].stage == "infer_on_holdout"


def test_stage_and_substage_summaries_group_expected_rows(tmp_path):
    log_path = tmp_path / "performance.jsonl"
    _write_records(
        log_path,
        [
            {
                "duration_s": 10.0,
                "kind": "span",
                "phase": "holdout_inference",
                "stage": "infer_on_holdout",
                "substage": "xgb_stream",
                "tile": "tile_a.tif",
            },
            {
                "duration_s": 20.0,
                "kind": "span",
                "phase": "holdout_inference",
                "stage": "infer_on_holdout",
                "tile": "tile_a.tif",
            },
            {
                "duration_s": 3.0,
                "kind": "span",
                "phase": "holdout_inference",
                "stage": "load_b_tile_context",
                "substage": "rasterize_gt",
                "tile": "tile_b.tif",
            },
        ],
    )

    records = filter_records(
        load_performance_records(log_path),
        kinds=parse_kind_filter("span"),
        phase="holdout_inference",
    )

    stage_rows = build_stage_summary(records)
    substage_rows = build_substage_summary(records)

    assert stage_rows[0]["stage"] == "infer_on_holdout"
    assert stage_rows[0]["total_s"] == 30.0
    assert stage_rows[0]["mean_s"] == 15.0

    substage_map = {
        (str(row["stage"]), str(row["substage"])): row for row in substage_rows
    }
    assert substage_map[("infer_on_holdout", "<none>")]["total_s"] == 20.0
    assert substage_map[("infer_on_holdout", "xgb_stream")]["total_s"] == 10.0
    assert substage_map[("load_b_tile_context", "rasterize_gt")]["total_s"] == 3.0


def test_tile_summary_and_csv_output(tmp_path):
    log_path = tmp_path / "performance.jsonl"
    _write_records(
        log_path,
        [
            {
                "duration_s": 8.0,
                "kind": "span",
                "phase": "holdout_inference",
                "stage": "infer_on_holdout",
                "substage": "xgb_stream",
                "tile": "tile_a.tif",
            },
            {
                "duration_s": 5.0,
                "kind": "span",
                "phase": "holdout_inference",
                "stage": "infer_on_holdout",
                "substage": "crf_stage",
                "tile": "tile_a.tif",
            },
            {
                "duration_s": 6.0,
                "kind": "span",
                "phase": "holdout_inference",
                "stage": "infer_on_holdout",
                "substage": "plot_exports",
                "tile": "tile_b.tif",
            },
        ],
    )

    records = filter_records(
        load_performance_records(log_path),
        kinds=parse_kind_filter("span"),
        phase="holdout_inference",
    )
    tile_rows, contributors = build_tile_summary(records)

    assert tile_rows[0]["tile"] == "tile_a.tif"
    assert tile_rows[0]["total_s"] == 13.0
    assert contributors["tile_a.tif"]["infer_on_holdout::xgb_stream"] == 8.0

    csv_path = tmp_path / "stage_summary.csv"
    stage_rows = build_stage_summary(records)
    write_csv(
        csv_path,
        stage_rows,
        ["stage", "count", "mean_s", "median_s", "min_s", "max_s", "total_s"],
    )

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["stage"] == "infer_on_holdout"
    assert rows[0]["total_s"] == "19.0"


def test_summary_text_mentions_nested_span_warning():
    summary = build_summary_text(
        [
            {"stage": "infer_on_holdout", "mean_s": 80.0},
            {"stage": "refine_with_densecrf", "mean_s": 45.0},
        ],
        [
            {
                "stage": "infer_on_holdout",
                "substage": "xgb_stream",
                "mean_s": 60.0,
            },
            {
                "stage": "xgb_score_image_b",
                "substage": "predict_inplace",
                "mean_s": 0.3,
            },
        ],
    )

    assert "Top stage bottlenecks: infer_on_holdout (80.00s avg)" in summary
    assert "infer_on_holdout::xgb_stream (60.00s avg)" in summary
    assert "Nested spans are reported separately" in summary
