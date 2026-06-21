"""Tests for the LMSR market maker — the novel core, so we pin its identities."""

import math

import numpy as np
import pytest

from bellwether.market.lmsr import LMSRMarket, _logsumexp, liquidity_from_budget


def test_prices_sum_to_one_and_uniform_at_start():
    m = LMSRMarket(n_outcomes=2, b=100.0)
    p = m.prices()
    assert pytest.approx(p.sum(), abs=1e-12) == 1.0
    assert pytest.approx(p[0], abs=1e-12) == 0.5  # symmetric at q = 0
    assert pytest.approx(m.prob_yes(), abs=1e-12) == 0.5


def test_buying_yes_raises_yes_price():
    m = LMSRMarket(b=100.0)
    before = m.prob_yes()
    m.buy(outcome=0, shares=50.0)
    assert m.prob_yes() > before


def test_cost_to_trade_matches_cost_difference():
    m = LMSRMarket(b=100.0)
    delta = np.array([30.0, 0.0])
    expected = m.cost(m.q + delta) - m.cost(m.q)
    assert pytest.approx(m.cost_to_trade(delta), rel=1e-12) == expected


def test_buy_returns_cost_and_updates_state():
    m = LMSRMarket(b=100.0)
    q0 = m.q.copy()
    cost = m.buy(0, 40.0)
    assert cost > 0
    assert m.q[0] == q0[0] + 40.0


def test_logsumexp_matches_naive_on_safe_inputs():
    x = np.array([0.1, 0.2, -0.3, 0.05])
    naive = math.log(float(np.sum(np.exp(x))))
    assert pytest.approx(_logsumexp(x), rel=1e-12) == naive


def test_logsumexp_is_stable_on_large_inputs():
    # Naive exp() overflows here; the stable version must not.
    x = np.array([1000.0, 1001.0])
    val = _logsumexp(x)
    assert math.isfinite(val)
    assert pytest.approx(val, abs=1e-9) == 1001.0 + math.log(1 + math.e ** -1)


def test_liquidity_from_budget_binary():
    assert pytest.approx(liquidity_from_budget(100.0, 2), rel=1e-12) == 100.0 / math.log(2)


@pytest.mark.parametrize("shares", [1.0, 50.0, 500.0, 5000.0])
def test_market_maker_loss_never_exceeds_bound(shares):
    """If we push YES up by buying ``shares`` and YES then resolves true, the
    maker's loss = payout(shares) - revenue, and must stay below b*ln(2)."""
    b = 100.0
    m = LMSRMarket(b=b)
    revenue = m.buy(0, shares)        # money traders paid in
    payout = shares                   # each YES share pays 1 if YES resolves
    loss = payout - revenue
    assert 0.0 < loss < m.max_loss() + 1e-9
    assert pytest.approx(m.max_loss(), rel=1e-12) == b * math.log(2)


def test_loss_approaches_bound_for_large_trades():
    b = 100.0
    m = LMSRMarket(b=b)
    revenue = m.buy(0, 100_000.0)     # drive price ~ 1
    loss = 100_000.0 - revenue
    assert loss < m.max_loss()
    assert loss > 0.99 * m.max_loss()  # nearly saturates the bound
