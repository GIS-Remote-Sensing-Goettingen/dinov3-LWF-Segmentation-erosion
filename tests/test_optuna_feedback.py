"""Tests for Optuna trial feedback callback formatting."""

from __future__ import annotations

from dataclasses import dataclass, field

from segedge.core.optuna_feedback import (
    build_optuna_callbacks_with_feedback,
    objective_weights,
)


@dataclass
class _State:
    name: str


@dataclass
class _Trial:
    number: int
    value: float | None
    state: _State
    user_attrs: dict = field(default_factory=dict)


@dataclass
class _Study:
    best_trial: _Trial
    trials: list[_Trial] = field(default_factory=list)

    def stop(self) -> None:
        """Match Optuna Study API shape for callback tests.

        Examples:
            >>> _Study(_Trial(0, 0.1, _State("COMPLETE"))).stop() is None
            True
        """
        return


def test_feedback_logs_separators_and_progress(caplog) -> None:
    """Feedback callback should emit readable separators and progress counters.

    Examples:
        >>> isinstance(test_feedback_logs_separators_and_progress.__name__, str)
        True
    """

    class _Cfg:
        BO_VERBOSE_TRIAL_SEPARATORS = True
        BO_EARLY_STOP_PATIENCE = 0
        BO_EARLY_STOP_MIN_DELTA = 0.0

    callbacks = build_optuna_callbacks_with_feedback(
        _Cfg,
        stage_name="Stage2 Broad",
        trial_total=10,
        trial_offset=2,
        default_patience=0,
    )
    assert len(callbacks) == 1
    cb = callbacks[0]
    trial = _Trial(
        number=44,
        value=0.26,
        state=_State("COMPLETE"),
        user_attrs={
            "weighted_iou_gt": 0.27,
            "weighted_iou_sh": 0.21,
        },
    )
    study = _Study(best_trial=trial, trials=[trial])
    with caplog.at_level("INFO", logger="segedge.core.optuna_feedback"):
        cb(study, trial)
    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "==== Stage2 Broad Trial 3/10 ====" in joined
    assert "progress=3/10" in joined
    assert "value=0.2600" in joined
    assert "loss=0.7400" in joined
    assert "==== End Stage2 Broad Trial 3/10 ====" in joined


def test_feedback_logs_pruned_without_separators_when_disabled(caplog) -> None:
    """Disabled separators should keep PRUNED feedback compact.

    Examples:
        >>> isinstance(test_feedback_logs_pruned_without_separators_when_disabled.__name__, str)
        True
    """

    class _Cfg:
        BO_VERBOSE_TRIAL_SEPARATORS = False
        BO_EARLY_STOP_PATIENCE = 0
        BO_EARLY_STOP_MIN_DELTA = 0.0

    callbacks = build_optuna_callbacks_with_feedback(
        _Cfg,
        stage_name="Stage1 Raw",
        trial_total=5,
        default_patience=0,
    )
    cb = callbacks[0]
    trial = _Trial(number=7, value=None, state=_State("PRUNED"))
    study = _Study(best_trial=_Trial(0, 0.1, _State("COMPLETE")))
    with caplog.at_level("INFO", logger="segedge.core.optuna_feedback"):
        cb(study, trial)
    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "state=PRUNED" in joined
    assert "progress=1/5" in joined
    assert "====" not in joined


def test_objective_weights_normalizes_or_falls_back() -> None:
    """Objective weight helper should normalize positive sums and guard zero-sum.

    Examples:
        >>> isinstance(test_objective_weights_normalizes_or_falls_back.__name__, str)
        True
    """

    class _Cfg:
        BO_OBJECTIVE_W_GT = 0.8
        BO_OBJECTIVE_W_SH = 0.2

    gt, sh = objective_weights(_Cfg)
    assert gt == 0.8
    assert sh == 0.2

    class _CfgZero:
        BO_OBJECTIVE_W_GT = 0.0
        BO_OBJECTIVE_W_SH = 0.0

    gt_zero, sh_zero = objective_weights(_CfgZero)
    assert gt_zero == 1.0
    assert sh_zero == 0.0


def test_feedback_appends_timing_summary_suffix(caplog) -> None:
    """Callback should append compact timing suffix when enabled by config.

    Examples:
        >>> isinstance(test_feedback_appends_timing_summary_suffix.__name__, str)
        True
    """

    class _Cfg:
        BO_VERBOSE_TRIAL_SEPARATORS = False
        BO_EARLY_STOP_PATIENCE = 0
        BO_EARLY_STOP_MIN_DELTA = 0.0
        BO_TIMING_SUMMARY_LOG = True

    callbacks = build_optuna_callbacks_with_feedback(
        _Cfg,
        stage_name="Stage1 Raw",
        trial_total=3,
        default_patience=0,
    )
    cb = callbacks[0]
    trial = _Trial(
        number=1,
        value=0.3,
        state=_State("COMPLETE"),
        user_attrs={"timing_summary": "score_base_knn_xgb_s:4.20s"},
    )
    study = _Study(best_trial=trial)
    with caplog.at_level("INFO", logger="segedge.core.optuna_feedback"):
        cb(study, trial)
    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "timing=score_base_knn_xgb_s:4.20s" in joined
