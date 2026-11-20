#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit a log-logistic AFT model to rank-normalized RSF risk scores."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

from .rsf import encode_outcome

AFT_MODELS = {
    "LogNormal": LogNormalAFTFitter,
    "Weibull": WeibullAFTFitter,
    "LogLogistic": LogLogisticAFTFitter,
}


def ensure_monotone(s: np.ndarray) -> np.ndarray:
    out = np.asarray(s, dtype=float).copy()
    for i in range(1, len(out)):
        if out[i] > out[i - 1]:
            out[i] = out[i - 1]
    return np.clip(out, 0.0, 1.0)


def rank_normalize_risk(risk_train: np.ndarray, risk_new: np.ndarray) -> np.ndarray:
    """Map new risk scores onto [0, 1] using the empirical distribution of the training scores."""
    sorted_train = np.sort(np.asarray(risk_train, dtype=float).reshape(-1))
    return np.interp(
        np.asarray(risk_new, dtype=float).reshape(-1),
        sorted_train,
        np.linspace(0.0, 1.0, len(sorted_train)),
    )


def _align_risk(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_values, event_values = encode_outcome(time, event)
    risk_values = np.asarray(risk, dtype=float).reshape(-1)
    if risk_values.shape != time_values.shape:
        raise ValueError(
            "risk, time, and event must be the same length, "
            f"got {risk_values.shape} and {time_values.shape}"
        )
    if not bool(np.isfinite(risk_values).all()):
        raise ValueError("risk scores must be finite")
    return risk_values, time_values, event_values


def fit_aft(
    risk_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    aft_cls=LogLogisticAFTFitter,
    penalizer: float = 0.05,
) -> Tuple[object, np.ndarray]:
    """Fit once. Return ``(model, sorted training risk scores)`` for later use of the same rank transform.

    Pass the penalizer used in the original fit to reproduce that fit.
    """
    risk, time_values, event_values = _align_risk(risk_train, time_train, event_train)
    frame = pd.DataFrame({
        "risk_score": rank_normalize_risk(risk, risk),
        "OS": time_values,
        "State": event_values,
    })
    model = aft_cls(penalizer=penalizer)
    model.fit(frame, duration_col="OS", event_col="State", ancillary=False)
    return model, np.sort(risk)


def _require_time_grid(times: np.ndarray, min_points: int = 1) -> np.ndarray:
    grid = np.asarray(times, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < min_points
        or not bool(np.isfinite(grid).all())
        or bool(np.any(grid < 0))
        or (grid.size > 1 and bool(np.any(np.diff(grid) <= 0)))
    ):
        raise ValueError("times must be a finite, non-negative, strictly increasing one-dimensional array")
    return grid


def survival_curves(
    model,
    risk_raw: np.ndarray,
    times: np.ndarray,
    sorted_train_risk: np.ndarray,
) -> np.ndarray:
    """Return survival probabilities of shape ``(n, len(times))`` and force each curve to be non-increasing."""
    risk_norm = rank_normalize_risk(sorted_train_risk, risk_raw)
    frame = pd.DataFrame({"risk_score": risk_norm})
    times = _require_time_grid(times)
    out = np.zeros((len(risk_norm), len(times)))
    for i, t in enumerate(times):
        values = np.asarray(
            model.predict_survival_function(frame, times=[float(t)]).values,
            dtype=float,
        ).reshape(-1)
        if values.shape != (len(risk_norm),) or not bool(np.isfinite(values).all()):
            raise RuntimeError(f"AFT survival prediction failed at t={float(t)}")
        out[:, i] = values
    for j in range(len(risk_norm)):
        out[j] = ensure_monotone(out[j])
    return out


def predict_median_os(
    model,
    risk_raw: np.ndarray,
    sorted_train_risk: np.ndarray,
) -> np.ndarray:
    """Predict median OS in the same units as the fitted follow-up times."""
    risk_norm = rank_normalize_risk(sorted_train_risk, risk_raw)
    median = model.predict_median(pd.DataFrame({"risk_score": np.asarray(risk_norm, dtype=float)}))
    values = np.asarray(median, dtype=float).reshape(-1)
    if values.shape != (len(risk_norm),) or not bool(np.isfinite(values).all()):
        raise RuntimeError("AFT median OS prediction failed")
    return values


def density_curves(survival: np.ndarray, times: np.ndarray) -> np.ndarray:
    grid = _require_time_grid(times, min_points=2)
    curves = np.asarray(survival, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != grid.size:
        raise ValueError("survival curve columns must match the number of time points")
    dt = np.diff(grid)
    t_mid = 0.5 * (grid[:-1] + grid[1:])
    out = np.zeros_like(curves, dtype=float)
    for j in range(curves.shape[0]):
        out[j] = np.interp(grid, t_mid, -np.diff(curves[j]) / dt)
    return np.clip(out, 0.0, None)
