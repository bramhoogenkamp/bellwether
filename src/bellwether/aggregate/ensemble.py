"""Condition E: ensemble of the market and the tuned aggregator.

The AIA Forecaster result is that a market + an AI forecaster beats either one
alone. This is our fallback-and-often-best product: if the market (D) doesn't beat
the tuned aggregator (C) on its own, their blend usually beats both.
"""

from __future__ import annotations


def ensemble(market_p: float, tuned_p: float, w_market: float = 0.5) -> float:
    return float(w_market * market_p + (1.0 - w_market) * tuned_p)
