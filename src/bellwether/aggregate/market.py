"""Condition D: the market price.

The agents trade an LMSR market. Each agent compares its belief to the current
price and, if it sees an edge, bets a fractional-Kelly slice of its bankroll —
scaled by its confidence, so more confident agents move the price more. We run a
few rounds so the price can settle. The final price is the swarm's market-aggregated
probability.

This is where a market can, in principle, beat plain averaging: confident/informed
agents get more weight (via stake), and the price reflects who is willing to back
their view — not just an unweighted vote.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import MarketConfig
from ..market.lmsr import LMSRMarket, liquidity_from_budget
from ..market.trading import kelly_intent, shares_for_stake


def market_price(forecasts, cfg: MarketConfig, seed: int = 0) -> float:
    if not forecasts:
        return 0.5

    b = liquidity_from_budget(cfg.max_loss_budget, n_outcomes=2)

    # Seed the market at the agents' consensus (their mean belief) rather than 0.5.
    # Starting at 0.5 biases the result toward the middle: with capped, finite-bankroll
    # trading the price can't always travel far enough to reach a strong consensus
    # (badly so on extreme-probability questions). Seeding at the prior lets trading
    # *refine* the consensus via confidence/stake-weighting instead of crawling to it.
    prior = float(np.clip(np.mean([f.probability for f in forecasts]), 1e-4, 1 - 1e-4))
    q0 = np.array([b * math.log(prior), b * math.log(1 - prior)])
    market = LMSRMarket(n_outcomes=2, b=b, q=q0)

    bankrolls = [float(cfg.starting_bankroll)] * len(forecasts)
    rng = np.random.default_rng(seed)

    for _ in range(cfg.rounds):
        for i in rng.permutation(len(forecasts)):
            if bankrolls[i] <= 1e-9:
                continue
            f = forecasts[i]
            price = market.prob_yes()
            intent = kelly_intent(
                f.probability, price, cfg.kelly_fraction, cfg.max_bet_fraction
            )
            if intent is None:
                continue
            # Confidence scales how much of the Kelly stake the agent actually deploys.
            stake = intent.fraction * f.confidence * bankrolls[i]
            if stake <= 1e-9:
                continue
            shares = shares_for_stake(market, intent.outcome, stake)
            cost = market.buy(intent.outcome, shares)
            bankrolls[i] -= cost

    return market.prob_yes()
