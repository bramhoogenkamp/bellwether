"""Position sizing for agent traders: fractional Kelly over an LMSR market.

An agent has a *belief* (its probability for YES) and faces the market's current
price. If they disagree there is an edge, and the agent should bet — but how much?
The Kelly criterion gives the bet fraction that maximises long-run growth of the
agent's bankroll. Full Kelly is famously aggressive (brutal drawdowns, and it
assumes your edge is perfectly calibrated), so in practice everyone uses
*fractional* Kelly (e.g. half). We also cap any single bet as a hard safety rail
so one over-confident agent cannot steamroll the price.

For a binary contract that pays 1 if the outcome happens and costs ``c`` (the
current price), an agent who believes the true probability is ``p`` has Kelly
fraction:

    buy YES (when p > c):  f = (p - c) / (1 - c)
    buy NO  (when p < c):  f = (c - p) / c

Both fall out of the standard Kelly formula ``f = (p*odds - (1-p)) / odds`` with
the contract's net odds. We scale by ``kelly_fraction`` and clip to ``max_fraction``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lmsr import LMSRMarket


@dataclass
class TradeIntent:
    """An agent's decision: stake ``fraction`` of its bankroll on ``outcome``."""

    outcome: int      # 0 = YES, 1 = NO
    fraction: float   # fraction of bankroll to stake (>= 0)


def kelly_intent(
    belief_yes: float,
    price_yes: float,
    kelly_fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> TradeIntent | None:
    """How much of bankroll to stake and on which side, or None if there's no edge."""
    c = float(price_yes)
    if not (0.0 < c < 1.0):
        raise ValueError("price_yes must be in (0, 1)")
    p = min(max(float(belief_yes), 1e-6), 1 - 1e-6)

    if abs(p - c) < 1e-9:
        return None  # agent agrees with the market -> no trade

    if p > c:
        f = (p - c) / (1.0 - c)
        outcome = 0  # YES looks underpriced
    else:
        f = (c - p) / c
        outcome = 1  # NO looks underpriced

    f *= kelly_fraction
    f = min(f, max_fraction)
    if f <= 0:
        return None
    return TradeIntent(outcome=outcome, fraction=f)


def shares_for_stake(
    market: LMSRMarket, outcome: int, stake: float, tol: float = 1e-9
) -> float:
    """How many shares of ``outcome`` cost approximately ``stake`` right now.

    LMSR cost rises monotonically (and convexly) with shares bought, so we bisect
    on the share count. We solve for a *stake* (money) rather than buying
    ``stake / price`` shares, because the price moves as you buy and the naive
    division overspends.
    """
    if stake <= 0:
        return 0.0

    def cost(shares: float) -> float:
        delta = np.zeros(market.n_outcomes)
        delta[outcome] = shares
        return market.cost_to_trade(delta)

    lo, hi = 0.0, 1.0
    while cost(hi) < stake:
        hi *= 2.0
        if hi > 1e12:  # absurd; bail out rather than loop forever
            break

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cost(mid) < stake:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
