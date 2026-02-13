"""Tests for Optuna CSV telemetry helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from segedge.core.optuna_csv import (
    collect_optuna_trials_from_storage,
    write_optuna_importance_csv,
    write_optuna_trials_csv,
)


@dataclass
class _FakeTrial:
    """Tiny fake trial object for helper tests."""

    number: int
    value: float | None
    state: str
    params: dict
    user_attrs: dict
    datetime_start: datetime | None
    datetime_complete: datetime | None


@dataclass
class _FakeStudy:
    """Tiny fake study object for helper tests."""

    study_name: str
    trials: list[_FakeTrial]


class _FakeOptuna:
    """Fake Optuna facade with load_study API."""

    def __init__(self, studies: dict[str, _FakeStudy]) -> None:
        self._studies = studies

    def load_study(self, *, study_name: str, storage: str):
        del storage
        if study_name not in self._studies:
            raise KeyError(study_name)
        return self._studies[study_name]


def test_collect_optuna_trials_from_storage_returns_wide_rows() -> None:
    """Collected rows should include base + param/attr columns.

    Examples:
        >>> True
        True
    """
    t0 = datetime(2026, 2, 11, 12, 0, 0)
    study = _FakeStudy(
        study_name="demo_stage1",
        trials=[
            _FakeTrial(
                number=0,
                value=0.2,
                state="TrialState.COMPLETE",
                params={"k": 120},
                user_attrs={"weighted_iou_gt": 0.1},
                datetime_start=t0,
                datetime_complete=t0 + timedelta(seconds=2),
            ),
            _FakeTrial(
                number=1,
                value=0.3,
                state="TrialState.COMPLETE",
                params={"k": 140},
                user_attrs={"weighted_iou_gt": 0.15},
                datetime_start=t0 + timedelta(seconds=3),
                datetime_complete=t0 + timedelta(seconds=5),
            ),
        ],
    )
    rows = collect_optuna_trials_from_storage(
        optuna_mod=_FakeOptuna({"demo_stage1": study}),
        storage_path="output/optuna_tuning.db",
        study_specs=[
            {
                "study_name": "demo_stage1",
                "stage": "stage1_raw",
                "max_recent_trials": 2,
            }
        ],
    )
    assert len(rows) == 2
    assert rows[0]["stage"] == "stage1_raw"
    assert "param__k" in rows[0]
    assert "attr__weighted_iou_gt" in rows[0]
    assert rows[1]["is_best_so_far"] == 1


def test_write_optuna_trials_csv_persists_header_and_rows(tmp_path: Path) -> None:
    """Trials CSV writer should persist rows with a stable base schema.

    Examples:
        >>> True
        True
    """
    out_path = tmp_path / "bayes_trials_timeseries.csv"
    rows = [
        {
            "timestamp_utc": "2026-02-11T12:00:00",
            "stage": "stage1_raw",
            "study_name": "demo_stage1",
            "trial_index_stage": 0,
            "trial_number_global": 1,
            "state": "COMPLETE",
            "objective": 0.42,
            "duration_s": 1.23,
            "is_best_so_far": 1,
            "param__k": 175,
        }
    ]
    write_optuna_trials_csv(str(out_path), rows)
    with out_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        saved = list(reader)
    assert reader.fieldnames is not None
    assert "stage" in reader.fieldnames
    assert "objective" in reader.fieldnames
    assert len(saved) == 1
    assert saved[0]["stage"] == "stage1_raw"


def test_write_optuna_importance_csv_writes_stage_rows(tmp_path: Path) -> None:
    """Importance CSV writer should flatten stage payloads.

    Examples:
        >>> True
        True
    """
    out_path = tmp_path / "bayes_hyperparam_importances.csv"
    payload = {
        "stage1": {"importances": {"k": 0.7, "roads_penalty": 0.3}},
        "stage2": {"importances": {"prob_softness": 0.4}},
        "stage3": {"importances": {}},
    }
    write_optuna_importance_csv(str(out_path), payload)
    with out_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 3
    assert rows[0]["stage"] == "stage1"
    assert rows[-1]["stage"] == "stage2"
