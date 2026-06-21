"""Condition B: the naive equal-weight mean of the agents' probabilities.

This is the baseline the market has to beat. It is deceptively strong — a simple
average of LLM forecasts already rivals a human crowd — so beating it is the whole
point of the exercise, not a gimme.
"""

from __future__ import annotations

import numpy as np


def naive_mean(forecasts) -> float:
    if not forecasts:
        return 0.5
    return float(np.mean([f.probability for f in forecasts]))
