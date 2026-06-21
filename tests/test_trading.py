"""Tests for fractional-Kelly position sizing and stake->shares solving."""

import pytest

from bellwether.market.lmsr import LMSRMarket
from bellwether.market.trading import kelly_intent, shares_for_stake


def test_no_edge_returns_none():
    assert kelly_intent(0.5, 0.5) is None


def test_buy_yes_when_belief_above_price():
    intent = kelly_intent(0.7, 0.5, kelly_fraction=1.0, max_fraction=1.0)
    assert intent is not None
    assert intent.outcome == 0
    assert pytest.approx(intent.fraction, rel=1e-9) == (0.7 - 0.5) / (1 - 0.5)


def test_buy_no_when_belief_below_price():
    intent = kelly_intent(0.3, 0.5, kelly_fraction=1.0, max_fraction=1.0)
    assert intent is not None
    assert intent.outcome == 1
    assert pytest.approx(intent.fraction, rel=1e-9) == (0.5 - 0.3) / 0.5


def test_fractional_kelly_scales_down():
    full = kelly_intent(0.8, 0.5, kelly_fraction=1.0, max_fraction=1.0).fraction
    half = kelly_intent(0.8, 0.5, kelly_fraction=0.5, max_fraction=1.0).fraction
    assert pytest.approx(half, rel=1e-9) == 0.5 * full


def test_max_fraction_caps_bet():
    intent = kelly_intent(0.99, 0.5, kelly_fraction=1.0, max_fraction=0.25)
    assert intent.fraction == 0.25


def test_shares_for_stake_costs_about_the_stake():
    m = LMSRMarket(b=100.0)
    stake = 10.0
    shares = shares_for_stake(m, outcome=0, stake=stake)
    import numpy as np

    delta = np.array([shares, 0.0])
    assert pytest.approx(m.cost_to_trade(delta), abs=1e-3) == stake


def test_zero_stake_buys_nothing():
    m = LMSRMarket(b=100.0)
    assert shares_for_stake(m, 0, 0.0) == 0.0
