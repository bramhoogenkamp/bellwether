"""Aggregators: turn a swarm's individual forecasts into one probability.

Each corresponds to a benchmark condition:
  B = naive_mean            (the wisdom-of-crowds baseline the market must beat)
  C = tuned_aggregate       (confidence-weighted + extremized — the honest hard baseline)
  D = market_price          (agents trade an LMSR market; the price is the answer)
  E = ensemble              (market + tuned — best-of-both, per the AIA finding)
"""

from .ensemble import ensemble
from .market import market_price
from .naive import naive_mean
from .tuned import confidence_weighted_mean, tuned_aggregate

__all__ = [
    "naive_mean",
    "confidence_weighted_mean",
    "tuned_aggregate",
    "market_price",
    "ensemble",
]
