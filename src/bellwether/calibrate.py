"""Calibration: nudging raw probabilities to be better-calibrated.

LLMs (especially RLHF'd chat models) tend to hedge toward 0.5, and averaging
several of them makes the result even less decisive. "Extremizing" counteracts this
by stretching probabilities away from 0.5 in log-odds space:

    logit(p) = ln(p / (1 - p));   p' = sigmoid(coef * logit(p))

``coef = 1`` is the identity. ``coef > 1`` makes forecasts more confident. The
forecasting literature finds a fixed ``coef`` around sqrt(3) ~= 1.73 is a good
default; once we have enough resolved questions we fit ``coef`` to our own history
to minimise Brier (``fit_extremize_coef``).
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_COEF = math.sqrt(3.0)
_EPS = 1e-6


def extremize(probs, coef: float = DEFAULT_COEF):
    """Push probabilities away from 0.5 (coef > 1) in log-odds space."""
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-coef * logit))


def fit_extremize_coef(probs, outcomes, bounds=(0.25, 4.0)) -> float:
    """Fit the extremizing coefficient that minimises Brier on resolved data."""
    from scipy.optimize import minimize_scalar  # lazy import; scipy ships with sklearn

    from .scoring import brier_score

    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    def objective(coef: float) -> float:
        return brier_score(extremize(p, coef), y)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(res.x)
