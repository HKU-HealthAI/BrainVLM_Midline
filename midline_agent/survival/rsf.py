#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random survival forest for patient-level feature matrices."""
from __future__ import annotations

import numpy as np
from sksurv.ensemble import RandomSurvivalForest

RANDOM_STATE = 42
RSF_PARAMS = {
    "n_estimators": 500,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_features": 0.5,
    "n_jobs": -1,
}


def encode_outcome(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return ``(time, event)``. ``event=1`` is an observed event and ``event=0`` is censoring."""
    time_arr = np.asarray(time)
    event_arr = np.asarray(event)
    if time_arr.ndim != 1 or event_arr.ndim != 1 or time_arr.shape != event_arr.shape:
        raise ValueError(
            "time and event must be one-dimensional and the same length, "
            f"got {time_arr.shape} and {event_arr.shape}"
        )
    if time_arr.size == 0:
        raise ValueError("time and event must be non-empty")
    time_values = np.asarray(time_arr, dtype=float)
    if not bool(np.isfinite(time_values).all()) or bool(np.any(time_values <= 0)):
        raise ValueError("follow-up time must be finite and positive")
    if event_arr.dtype.kind in {"U", "S", "O"}:
        tokens = [str(value).strip() for value in event_arr.tolist()]
        if any(token not in {"0", "1"} for token in tokens):
            raise ValueError("event must be 0 for censoring or 1 for an observed event")
        event_values = np.asarray([int(token) for token in tokens], dtype=int)
    else:
        event_float = np.asarray(event_arr, dtype=float)
        if not bool(np.isfinite(event_float).all()) or not bool(np.isin(event_float, (0.0, 1.0)).all()):
            raise ValueError("event must be 0 for censoring or 1 for an observed event")
        event_values = event_float.astype(int)
    return time_values, event_values


def make_y(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Structured ``(event, time)`` array required by sksurv."""
    time_values, event_values = encode_outcome(time, event)
    observed = np.empty(len(time_values), dtype=[("event", bool), ("time", float)])
    observed["time"] = time_values
    observed["event"] = event_values.astype(bool)
    return observed


def fit_rsf(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    random_state: int = RANDOM_STATE,
    **overrides,
) -> RandomSurvivalForest:
    params = dict(RSF_PARAMS)
    params.update(overrides)
    model = RandomSurvivalForest(random_state=random_state, **params)
    model.fit(np.asarray(x, dtype=np.float64), make_y(time, event))
    return model


def predict_risk(model: RandomSurvivalForest, x: np.ndarray) -> np.ndarray:
    """Cumulative hazard. Larger values mean worse prognosis."""
    return np.asarray(model.predict(np.asarray(x, dtype=np.float64)), dtype=float)


def predict_survival(
    model: RandomSurvivalForest,
    x: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Return survival probabilities of shape ``(n, len(times))``."""
    times = np.asarray(times, dtype=float)
    curves = model.predict_survival_function(np.asarray(x, dtype=np.float64))
    return np.asarray([[fn(t) for t in times] for fn in curves], dtype=float)
