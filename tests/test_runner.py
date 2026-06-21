"""Offline end-to-end smoke test of the benchmark runner (FakeLLM + mock data).

These assert pipeline *invariants* and robust sanity checks — not fragile claims
about which aggregator wins (that's an empirical question for the real benchmark).
We use enough questions that Brier scores are stable.
"""

from bellwether.config import BenchmarkConfig
from bellwether.runner import run_benchmark

_N = 120


def _config():
    return BenchmarkConfig(
        conditions=["A", "B", "C", "D", "E", "G"],
        questions={"source": "mock_internal", "limit": _N},
        swarm={
            "models": ["m1", "m2", "m3"],
            "n_agents": 6,
            "lenses": ["base_rate", "inside_view"],
        },
    )


def test_run_benchmark_scores_all_conditions():
    result = run_benchmark(_config(), log=lambda *_: None)
    assert result.n_scored == _N
    for cond in ["A", "B", "C", "D", "E", "G"]:
        assert cond in result.scores
        s = result.scores[cond]
        assert 0.0 <= s["brier"] <= 1.0
        assert s["n"] == _N


def test_run_benchmark_has_consistent_headline_deltas():
    result = run_benchmark(_config(), log=lambda *_: None)
    assert "D_vs_B" in result.deltas and "D_vs_C" in result.deltas
    for d in result.deltas.values():
        assert d["ci_low"] <= d["mean_delta"] <= d["ci_high"]


def test_run_benchmark_is_deterministic():
    a = run_benchmark(_config(), log=lambda *_: None)
    b = run_benchmark(_config(), log=lambda *_: None)
    assert a.scores["D"]["brier"] == b.scores["D"]["brier"]


def test_swarm_extracts_real_signal():
    # The swarm's basic aggregation should have positive skill (beat the base rate)
    # and beat the deliberately-biased status-quo baseline.
    result = run_benchmark(_config(), log=lambda *_: None)
    assert result.scores["B"]["bss"] > 0.0
    assert result.scores["B"]["brier"] < result.scores["G"]["brier"]
