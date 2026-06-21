"""Tests for the extremizing calibration transform."""

import numpy as np
import pytest

from bellwether.calibrate import extremize, fit_extremize_coef


def test_half_is_fixed_point():
    assert pytest.approx(extremize(0.5, coef=2.0)) == 0.5


def test_coef_one_is_identity():
    for p in (0.2, 0.5, 0.8):
        assert pytest.approx(extremize(p, coef=1.0), rel=1e-9) == p


def test_extremizing_pushes_away_from_half():
    assert extremize(0.6, coef=2.0) > 0.6   # above 0.5 -> higher
    assert extremize(0.4, coef=2.0) < 0.4   # below 0.5 -> lower


def test_outputs_stay_in_unit_interval():
    out = extremize([0.001, 0.5, 0.999], coef=3.0)
    assert np.all(out > 0) and np.all(out < 1)


def test_fit_recovers_extremizing_for_underconfident_data():
    # Build underconfident forecasts: outcomes are decisive but probs hug 0.5.
    rng = np.random.default_rng(0)
    outcomes = rng.integers(0, 2, 400)
    probs = np.where(outcomes == 1, 0.6, 0.4)  # right direction, too timid
    coef = fit_extremize_coef(probs, outcomes)
    assert coef > 1.0  # fitting should want to sharpen
