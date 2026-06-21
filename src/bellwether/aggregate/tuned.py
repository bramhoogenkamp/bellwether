"""Condition C: a tuned aggregator — the honest, hard baseline.

A weighted average of the agents (by stated confidence) that is then extremized to
counter the herd-toward-0.5 tendency. This is the baseline that matters most: the
forecasting literature shows a *tuned* aggregator can beat even a real prediction
market, so if Bellwether's market can't beat this, the right product is the
market+aggregator ensemble (condition E), not the market alone.

Note: a fuller "tuned" aggregator would also weight agents by their *track record*
and recalibrate against history. That needs cross-question history, so it's a
later addition; per-question we use confidence-weighting + extremizing.
"""

from __future__ import annotations

import numpy as np

from ..calibrate import DEFAULT_COEF, extremize


def confidence_weighted_mean(forecasts) -> float:
    if not forecasts:
        return 0.5
    probs = np.array([f.probability for f in forecasts], dtype=float)
    weights = np.array([max(f.confidence, 1e-6) for f in forecasts], dtype=float)
    return float(np.sum(weights * probs) / np.sum(weights))


def tuned_aggregate(forecasts, extremize_coef: float = DEFAULT_COEF) -> float:
    p = confidence_weighted_mean(forecasts)
    return float(extremize(p, extremize_coef))
