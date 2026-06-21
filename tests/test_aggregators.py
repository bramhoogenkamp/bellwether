"""Tests for the aggregators and the A-G condition registry."""

import pytest

from bellwether.aggregate.market import market_price
from bellwether.aggregate.naive import naive_mean
from bellwether.aggregate.tuned import confidence_weighted_mean, tuned_aggregate
from bellwether.agents.agent import Forecast
from bellwether.conditions import compute_conditions, single_llm
from bellwether.config import BenchmarkConfig, MarketConfig
from bellwether.questions.mock_internal import MockInternalQuestionSource


def _fc(p, c=0.7):
    return Forecast(probability=p, confidence=c)


def test_naive_mean():
    assert naive_mean([_fc(0.4), _fc(0.6)]) == pytest.approx(0.5)


def test_confidence_weighting_pulls_toward_confident_agent():
    fs = [_fc(0.2, c=0.1), _fc(0.8, c=0.9)]
    assert confidence_weighted_mean(fs) > 0.5  # the confident 0.8 dominates


def test_tuned_extremizes_away_from_half():
    fs = [_fc(0.65), _fc(0.7), _fc(0.6)]
    naive = naive_mean(fs)
    tuned = tuned_aggregate(fs, extremize_coef=1.73)
    assert tuned > naive  # all above 0.5 -> extremizing pushes higher


def test_market_moves_toward_consensus_yes_and_no():
    cfg = MarketConfig()
    up = market_price([_fc(0.8)] * 6, cfg, seed=0)
    down = market_price([_fc(0.2)] * 6, cfg, seed=0)
    assert up > 0.55 and down < 0.45
    assert up > down


def test_market_is_deterministic_given_seed():
    fs = [_fc(0.7), _fc(0.3), _fc(0.6), _fc(0.55)]
    assert market_price(fs, MarketConfig(), seed=1) == market_price(fs, MarketConfig(), seed=1)


def test_market_stays_neutral_at_consensus_half():
    price = market_price([_fc(0.5)] * 5, MarketConfig(), seed=0)
    assert price == pytest.approx(0.5, abs=1e-6)  # no edge -> no trades


def test_single_llm_is_first_forecast():
    assert single_llm([_fc(0.42), _fc(0.9)]) == 0.42


def test_compute_conditions_keys_and_G_present_F_absent():
    q = MockInternalQuestionSource(seed=0).fetch(limit=1)[0]  # has status_quo, no superforecaster
    fs = [_fc(0.6), _fc(0.7), _fc(0.55), _fc(0.65)]
    cfg = BenchmarkConfig(conditions=["A", "B", "C", "D", "E", "F", "G"])
    out = compute_conditions(fs, q, cfg, seed=0)
    assert set(out) == {"A", "B", "C", "D", "E", "G"}  # F skipped (no baseline)
    for v in out.values():
        assert 0.0 <= v <= 1.0
