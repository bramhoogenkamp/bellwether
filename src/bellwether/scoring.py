"""Scoring rules for probabilistic forecasts.

Every forecaster (single LLM, naive mean, tuned aggregator, market, ...) is scored
the same way so configs are directly comparable in the MLflow runs-table. Headline
metrics:

  * Brier score - mean squared error between the probability and the 0/1 outcome.
    Lower is better; 0 is perfect, 0.25 is "always say 50%". Strictly proper, so
    honesty is optimal. The standard for binary forecasts.
  * Brier Skill Score (BSS) - "how much better than a baseline", 1 - BS/BS_ref.
    Reads as a percentage: 0.30 = 30% lower error than the baseline. We lead with
    this for non-technical audiences. Default baseline = climatology (the base
    rate): the honest "know nothing but the average" forecast.
  * log loss - like Brier but punishes confident-and-wrong far harder (unbounded).
  * calibration / ECE - "when we say 70%, does it happen ~70% of the time?".
    A low Brier can still hide miscalibration, so we always report this too.

We keep these as small, dependency-light functions (numpy only) so they are easy
to read and test. scikit-learn / netcal give equivalent results if preferred.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-15


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def brier_score(probs, outcomes) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs, outcomes) -> float:
    p = _clip(probs)
    y = np.asarray(outcomes, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_skill_score(probs, outcomes, reference=None) -> float:
    """1 - BS_forecast / BS_reference. Reference defaults to the base rate."""
    y = np.asarray(outcomes, dtype=float)
    bs = brier_score(probs, y)
    if reference is None:
        reference = np.full_like(y, float(np.mean(y)))  # climatology
    else:
        reference = np.asarray(reference, dtype=float)
    bs_ref = brier_score(reference, y)
    if bs_ref == 0:
        return 0.0
    return float(1 - bs / bs_ref)


def calibration_curve(probs, outcomes, n_bins: int = 10):
    """Return (mean_predicted, observed_freq, count) for each non-empty bin."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_pred.append(float(p[mask].mean()))
            obs_freq.append(float(y[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed frequency."""
    mean_pred, obs_freq, counts = calibration_curve(probs, outcomes, n_bins)
    if counts.sum() == 0:
        return 0.0
    return float(np.sum(counts * np.abs(mean_pred - obs_freq)) / counts.sum())


def score_all(probs, outcomes, reference=None) -> dict:
    """Bundle the standard metrics into one dict (logged to MLflow as metrics)."""
    m = murphy_decomposition(probs, outcomes)
    return {
        "brier": brier_score(probs, outcomes),
        "bss": brier_skill_score(probs, outcomes, reference),
        "log_loss": log_loss(probs, outcomes),
        "ece": expected_calibration_error(probs, outcomes),
        "reliability": m["reliability"],
        "resolution": m["resolution"],
        "n": int(len(outcomes)),
        "base_rate": float(np.mean(outcomes)),
    }


def murphy_decomposition(probs, outcomes, n_bins: int = 10) -> dict:
    """Decompose Brier = Reliability - Resolution + Uncertainty (Murphy 1973).

    reliability  (lower better; 0 = perfectly calibrated): calibration error.
    resolution   (higher better): how much outcome frequency varies across forecast
                 bins — the ability to discriminate yes-cases from no-cases.
    uncertainty  (fixed by the data): base_rate * (1 - base_rate).
    A forecast only adds value when resolution > reliability.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n = len(y)
    base = float(y.mean())
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    reliability = resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        p_bar = float(p[mask].mean())
        y_bar = float(y[mask].mean())
        reliability += nb * (p_bar - y_bar) ** 2
        resolution += nb * (y_bar - base) ** 2
    return {
        "reliability": reliability / n,
        "resolution": resolution / n,
        "uncertainty": base * (1 - base),
    }


def paired_bootstrap_brier_delta(
    probs_a, probs_b, outcomes, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Bootstrap 95% CI for (Brier_a - Brier_b), paired by question.

    Negative mean => A has the lower Brier (A is better). If the CI excludes 0 the
    difference is significant at ~5%. This is how we judge the headline D-vs-B and
    D-vs-C comparisons.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(outcomes, dtype=float)
    se_a = (np.asarray(probs_a, dtype=float) - y) ** 2
    se_b = (np.asarray(probs_b, dtype=float) - y) ** 2
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = se_a[idx].mean() - se_b[idx].mean()
    return {
        "mean_delta": float(se_a.mean() - se_b.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }
