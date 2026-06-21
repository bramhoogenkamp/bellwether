"""Logarithmic Market Scoring Rule (LMSR) automated market maker.

Why LMSR? In a thin market (few traders, low volume) an order book fails: there is
rarely a counterparty, spreads blow out, and a single order swings the price. LMSR
replaces the counterparty with a *market maker* that will always quote a price and
always take the other side of a trade. The price it quotes is, by construction, a
valid probability. That is exactly what we need for an agent-only market where our
handful of agents *are* the entire market.

"Liquidity" here is a single parameter ``b`` that we set ourselves, rather than
hoping enough volume shows up. Larger ``b`` -> deeper market, prices move less per
trade; smaller ``b`` -> jumpy prices. This is the "simulated liquidity" idea:
liquidity is a knob, not an emergent property of volume.

Core formulas (q = vector of net shares outstanding per outcome):

    cost   C(q) = b * ln( sum_i exp(q_i / b) )       # total money paid into the maker
    price  p_i  = exp(q_i / b) / sum_j exp(q_j / b)   # = softmax(q / b)
    trade  cost of moving q -> q' is C(q') - C(q)

Facts we rely on (and test in tests/test_lmsr.py):
  * prices are always in (0, 1) and sum to 1            -> they ARE probabilities
  * the log scoring rule is *strictly proper*           -> a risk-neutral trader
    maximises expected profit by moving the price to its true belief
  * the maker's worst-case loss (the subsidy we fund) is b * ln(N), i.e. b * ln(2)
    for a binary market. So to cap loss at budget L: b = L / ln(N).

Reference: Hanson (2007), "Logarithmic Market Scoring Rules for Modular
Combinatorial Information Aggregation".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable ln(sum(exp(x))).

    exp() overflows for moderately large inputs, and ``q_i / b`` grows large as a
    price approaches 0 or 1. Subtracting the max before exponentiating keeps every
    term in range without changing the result. Forgetting this is the single most
    common LMSR bug, so it lives in one place.
    """
    x = np.asarray(x, dtype=float)
    m = float(np.max(x))
    return m + math.log(float(np.sum(np.exp(x - m))))


def liquidity_from_budget(max_loss_budget: float, n_outcomes: int = 2) -> float:
    """The liquidity parameter ``b`` that caps market-maker loss at the budget.

    Worst-case subsidy = b * ln(N), so b = budget / ln(N).
    """
    if max_loss_budget <= 0:
        raise ValueError("max_loss_budget must be positive")
    if n_outcomes < 2:
        raise ValueError("need at least 2 outcomes")
    return max_loss_budget / math.log(n_outcomes)


@dataclass
class LMSRMarket:
    """A single LMSR market over ``n_outcomes`` mutually exclusive outcomes.

    For a binary question use the default 2 outcomes: index 0 = YES, 1 = NO.
    ``q`` holds net shares outstanding per outcome and starts at all-zeros, which
    corresponds to a uniform price (0.5 / 0.5 for binary).
    """

    n_outcomes: int = 2
    b: float = 100.0
    q: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError("b (liquidity) must be positive")
        if self.q is None:
            self.q = np.zeros(self.n_outcomes, dtype=float)
        else:
            self.q = np.asarray(self.q, dtype=float)
            if self.q.shape != (self.n_outcomes,):
                raise ValueError("q must have length n_outcomes")

    # --- pricing -----------------------------------------------------------
    def prices(self) -> np.ndarray:
        """Current prices = softmax(q / b). Always in (0, 1), sums to 1."""
        z = self.q / self.b
        z = z - np.max(z)  # stability; softmax is shift-invariant
        e = np.exp(z)
        return e / e.sum()

    def prob_yes(self) -> float:
        """Convenience for binary markets: the price of outcome 0 (YES)."""
        return float(self.prices()[0])

    # --- cost --------------------------------------------------------------
    def cost(self, q: np.ndarray | None = None) -> float:
        """LMSR cost function C(q). Defaults to the current state."""
        q = self.q if q is None else np.asarray(q, dtype=float)
        return self.b * _logsumexp(q / self.b)

    def cost_to_trade(self, delta: np.ndarray) -> float:
        """Cost to buy ``delta`` shares (vector, per outcome) from the current state.

        Positive entries buy shares of that outcome; the maker quotes the fair
        price C(q + delta) - C(q). Selling is just a negative delta.
        """
        delta = np.asarray(delta, dtype=float)
        return self.cost(self.q + delta) - self.cost(self.q)

    def buy(self, outcome: int, shares: float) -> float:
        """Buy ``shares`` of a single ``outcome``; mutate state; return the cost."""
        delta = np.zeros(self.n_outcomes)
        delta[outcome] = shares
        cost = self.cost_to_trade(delta)
        self.q = self.q + delta
        return cost

    def max_loss(self) -> float:
        """Worst-case subsidy the maker can ever pay out = b * ln(N)."""
        return self.b * math.log(self.n_outcomes)
