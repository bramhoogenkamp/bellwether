"""The benchmark conditions A-G: each turns a swarm's forecasts into one probability.

  A  single LLM (one agent, no aggregation)        -- the floor
  B  naive mean of the agents                       -- wisdom-of-crowds baseline
  C  tuned aggregator (confidence-weighted + extremized) -- the honest hard baseline
  D  market price (agents trade an LMSR market)     -- the mechanism under test
  E  ensemble of market + tuned                     -- best-of-both (AIA finding)
  F  superforecaster baseline (from the question)   -- human gold standard (if present)
  G  company status-quo forecast (from the question) -- the incumbent to beat (if present)

``compute_conditions`` returns only the enabled conditions for which data exists
(F/G are skipped when the question carries no such baseline, e.g. mock questions
have G but not F).
"""

from __future__ import annotations

from .aggregate.ensemble import ensemble
from .aggregate.market import market_price
from .aggregate.naive import naive_mean
from .aggregate.tuned import tuned_aggregate
from .config import BenchmarkConfig
from .questions.base import Question


def single_llm(forecasts) -> float:
    """Condition A: just take the first agent — no aggregation at all."""
    return forecasts[0].probability if forecasts else 0.5


def compute_conditions(
    forecasts, question: Question, config: BenchmarkConfig, seed: int = 0
) -> dict[str, float]:
    enabled = set(config.conditions)
    coef = config.calibration.extremize_coef
    out: dict[str, float] = {}

    if "A" in enabled:
        out["A"] = single_llm(forecasts)
    if "B" in enabled:
        out["B"] = naive_mean(forecasts)

    # Build C and D only if needed (also reused by the ensemble E).
    tuned = tuned_aggregate(forecasts, coef) if {"C", "E"} & enabled else None
    market = market_price(forecasts, config.market, seed=seed) if {"D", "E"} & enabled else None

    if "C" in enabled:
        out["C"] = tuned
    if "D" in enabled:
        out["D"] = market
    if "E" in enabled and tuned is not None and market is not None:
        out["E"] = ensemble(market, tuned)

    if "F" in enabled and question.superforecaster_prob is not None:
        out["F"] = float(question.superforecaster_prob)
    if "G" in enabled and question.status_quo_prob is not None:
        out["G"] = float(question.status_quo_prob)

    return out
