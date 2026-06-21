"""Tests for the scoring rules on hand-computed cases."""

import math

import pytest

from bellwether.scoring import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    log_loss,
    paired_bootstrap_brier_delta,
    score_all,
)


def test_brier_perfect_and_worst():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert pytest.approx(brier_score([0.5, 0.5], [1, 0])) == 0.25
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_bss_zero_for_base_rate_forecast():
    outcomes = [1, 0, 1, 1]
    base = sum(outcomes) / len(outcomes)
    probs = [base] * len(outcomes)
    assert pytest.approx(brier_skill_score(probs, outcomes), abs=1e-12) == 0.0


def test_bss_one_for_perfect_forecast():
    outcomes = [1, 0, 1, 0]
    assert pytest.approx(brier_skill_score([1, 0, 1, 0], outcomes)) == 1.0


def test_log_loss_known_value():
    # all 0.5 -> -ln(0.5) per item
    assert pytest.approx(log_loss([0.5, 0.5], [1, 0])) == -math.log(0.5)


def test_ece_zero_when_perfectly_calibrated():
    # 0% and 100% predictions that always match -> no calibration gap
    probs = [0.0, 0.0, 1.0, 1.0]
    outcomes = [0, 0, 1, 1]
    assert expected_calibration_error(probs, outcomes) == 0.0


def test_score_all_keys():
    s = score_all([0.6, 0.4, 0.7], [1, 0, 1])
    assert set(s) == {"brier", "bss", "log_loss", "ece", "n", "base_rate"}
    assert s["n"] == 3


def test_paired_bootstrap_detects_better_forecaster():
    outcomes = [1, 0, 1, 0, 1, 0, 1, 0]
    good = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
    bad = [0.5] * 8
    res = paired_bootstrap_brier_delta(good, bad, outcomes, n_boot=500, seed=1)
    assert res["mean_delta"] < 0          # good has lower Brier
    assert res["ci_high"] < 0             # significantly so
