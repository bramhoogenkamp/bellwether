"""Tests for the self-improving loop (offline, deterministic)."""

import numpy as np

from bellwether.loop import (
    Record,
    average,
    evaluate,
    format_ledger,
    log_pool,
    run_loop,
    split,
)


def test_log_pool_basics():
    assert abs(log_pool([0.5, 0.5]) - 0.5) < 1e-9
    # a near-certain NO drags the pool below the plain mean (this is the point)
    probs = [0.02, 0.5, 0.5, 0.5]
    assert log_pool(probs) < average(probs)
    # a sharpening coefficient pushes a >0.5 pool higher
    assert log_pool([0.6, 0.65, 0.7], coef=2.0) > log_pool([0.6, 0.65, 0.7], coef=1.0)
    out = log_pool([0.001, 0.999])
    assert 0.0 < out < 1.0


def test_evaluate_average_is_zero_against_itself():
    recs = [Record("comp-AND", 0.0, [0.02, 0.5, 0.5], 0.0),
            Record("comp-AND", 1.0, [0.7, 0.7, 0.7], 1.0)]
    e = evaluate(recs, average)
    assert abs(e["vs_average"]) < 1e-12


def test_split_partitions_all_records():
    recs = [Record("c", 0.0, [0.3, 0.4], 0.0) for _ in range(10)]
    tr, va = split(recs, train_frac=0.6, seed=0)
    assert len(tr) == 6 and len(tr) + len(va) == 10


def test_run_loop_history_is_reviewable_and_finds_the_pool():
    # Construct instances where a decisive NO agent should let the log pool beat the mean.
    rng = np.random.default_rng(0)
    recs = []
    for _ in range(60):
        y = float(rng.random() < 0.4)
        if y == 0.0:
            beliefs = [0.03] + [float(rng.uniform(0.3, 0.6)) for _ in range(2)]
            recs.append(Record("comp-AND", 0.0, beliefs, 0.0))
        else:
            recs.append(Record("comp-AND", 1.0, [float(rng.uniform(0.45, 0.7)) for _ in range(3)], 1.0))
    res = run_loop(recs)
    assert res["history"] and all("decision" in h for h in res["history"])
    assert res["n_train"] + res["n_val"] == 60
    # the log-opinion pool should beat the average on this data and be accepted somewhere
    assert any("log" in h["design"] and "accept" in h["decision"] for h in res["history"])
    assert "decision" in format_ledger(res).lower()
