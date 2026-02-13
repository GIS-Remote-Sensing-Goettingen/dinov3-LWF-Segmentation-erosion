"""Tests for Optuna stagnation stop callbacks."""

from __future__ import annotations

from dataclasses import dataclass

from segedge.core.optuna_stop import (
    build_no_improvement_callbacks,
    build_no_improvement_callbacks_from_config,
)


@dataclass
class _Trial:
    """Minimal fake trial payload."""

    number: int
    value: float | None


class _Study:
    """Minimal fake study object implementing stop()."""

    def __init__(self, trials=None) -> None:
        self.trials = list(trials or [])
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_build_no_improvement_callbacks_none_when_disabled() -> None:
    """Zero patience should disable stagnation callbacks.

    Examples:
        >>> True
        True
    """
    assert build_no_improvement_callbacks(patience_steps=0) is None


def test_stagnation_callback_stops_after_patience() -> None:
    """Callback should stop when no trial improves within patience.

    Examples:
        >>> True
        True
    """
    callbacks = build_no_improvement_callbacks(patience_steps=2, min_delta=0.0)
    assert callbacks is not None
    cb = callbacks[0]
    study = _Study()
    cb(study, _Trial(number=0, value=0.5))
    assert study.stopped is False
    cb(study, _Trial(number=1, value=0.5))
    assert study.stopped is False
    cb(study, _Trial(number=2, value=0.5))
    assert study.stopped is True


def test_from_config_reads_patience_and_min_delta() -> None:
    """Config adapter should produce callbacks when patience is positive.

    Examples:
        >>> True
        True
    """

    class _Cfg:
        BO_EARLY_STOP_PATIENCE = 3
        BO_EARLY_STOP_MIN_DELTA = 0.01

    callbacks = build_no_improvement_callbacks_from_config(_Cfg)
    assert callbacks is not None
    study = _Study()
    cb = callbacks[0]
    cb(study, _Trial(number=0, value=1.0))
    cb(study, _Trial(number=1, value=1.005))
    cb(study, _Trial(number=2, value=1.007))
    cb(study, _Trial(number=3, value=1.008))
    assert study.stopped is True
