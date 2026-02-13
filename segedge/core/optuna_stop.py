"""Optuna stop callbacks for stagnation-based early termination.

Examples:
    >>> isinstance(build_no_improvement_callbacks.__name__, str)
    True
"""

from __future__ import annotations


def build_no_improvement_callbacks(
    *,
    patience_steps: int,
    min_delta: float = 0.0,
):
    """Return Optuna callbacks that stop a study on stagnation.

    The callback stops optimization when no improvement greater than `min_delta`
    has been observed for `patience_steps` trial numbers.

    Examples:
        >>> build_no_improvement_callbacks(patience_steps=0) is None
        True
    """
    patience = int(patience_steps)
    if patience <= 0:
        return None
    delta = float(min_delta)
    state = {
        "initialized": False,
        "best_value": None,
        "last_improvement_trial_number": -1,
    }

    def _initialize(study) -> None:
        if state["initialized"]:
            return
        state["initialized"] = True
        best_value = None
        last_idx = -1
        for t in getattr(study, "trials", []) or []:
            val = getattr(t, "value", None)
            if val is None:
                continue
            v = float(val)
            idx = int(getattr(t, "number", -1))
            if best_value is None or v > best_value + delta:
                best_value = v
                last_idx = idx
        state["best_value"] = best_value
        state["last_improvement_trial_number"] = int(last_idx)

    def _stop_on_stagnation(study, trial) -> None:
        _initialize(study)
        idx = int(getattr(trial, "number", -1))
        val = getattr(trial, "value", None)
        if val is not None:
            v = float(val)
            best = state["best_value"]
            if best is None or v > float(best) + delta:
                state["best_value"] = v
                state["last_improvement_trial_number"] = idx
        if (
            state["last_improvement_trial_number"] >= 0
            and idx - int(state["last_improvement_trial_number"]) >= patience
        ):
            study.stop()

    return [_stop_on_stagnation]


def build_no_improvement_callbacks_from_config(
    cfg_module,
    *,
    default_patience: int = 0,
):
    """Build stagnation callbacks from config-style attributes.

    Examples:
        >>> class _Cfg:
        ...     BO_EARLY_STOP_PATIENCE = 0
        ...     BO_EARLY_STOP_MIN_DELTA = 0.0
        >>> build_no_improvement_callbacks_from_config(_Cfg) is None
        True
    """
    patience = int(getattr(cfg_module, "BO_EARLY_STOP_PATIENCE", default_patience) or 0)
    min_delta = float(getattr(cfg_module, "BO_EARLY_STOP_MIN_DELTA", 0.0) or 0.0)
    return build_no_improvement_callbacks(
        patience_steps=patience,
        min_delta=min_delta,
    )
