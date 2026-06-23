#!/usr/bin/env python3
"""D1: the confidence-accuracy decoupling (the study's headline).

Across rounds of deliberation, track three things per round: accuracy (Brier of the
swarm mean toward the outcome, and the gap to the oracle), agreement (how close the
agents' beliefs are to each other), and confidence (how far their beliefs sit from
0.5). The claim is that agreement and confidence rise monotonically with rounds in
every condition, while accuracy improves only when a decisive dispersed signal is
present. So we run a redundant condition (substitutable, nothing to pool) alongside the
complementary ones (a decisive piece exists).

Per-instance belief vectors are logged every round so the predictor experiment (P1) can
reuse them.

    python scripts/run_decoupling.py --n 24 --rounds 3 --live --mlflow
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("LITELLM_LOG", "ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from bellwether import scoring  # noqa: E402
from bellwether.aggregate.naive import naive_mean  # noqa: E402
from bellwether.agents.llm import get_client  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.questions.synthetic import generate_info_instances  # noqa: E402

# condition -> (structure, n_agents, rule). Substitutable = redundant (nothing to pool);
# complementary = a decisive dispersed piece exists.
GRID = {
    "substitutable": ("substitutable", 4, None),
    "comp-AND": ("complementary", 4, "and"),
    "comp-OR": ("complementary", 4, "or"),
}


def _beliefs(forecasts):
    return [f.probability for f in forecasts]


def instance_rounds(swarm, inst, rounds):
    """Return (outcome, oracle, per_round_beliefs) where per_round_beliefs[0] is pre-deliberation."""
    private = swarm.forecast_each_private(inst.question, inst.slices)
    pooled = swarm.run(inst.question, inst.pooled)
    per_round = [_beliefs(private)]
    cur = private
    for _ in range(rounds):
        cur = swarm.run_debate_round(inst.question, inst.slices, cur)
        per_round.append(_beliefs(cur))
    return inst.question.outcome, (naive_mean(pooled) if pooled else 0.5), per_round


def run_cell(label, cfg, client, n, rounds, concurrency, log_path):
    structure, n_agents, rule = GRID[label]
    swarm = Swarm(cfg.swarm.model_copy(update={"n_agents": n_agents}), client)
    kw = {"n": n, "n_agents": n_agents, "structure": structure, "seed": cfg.seed}
    if rule:
        kw["rule"] = rule
    insts = generate_info_instances(**kw)
    rows, lock, total = [], threading.Lock(), len(insts)

    def work(inst):
        try:
            return instance_rounds(swarm, inst, rounds)
        except Exception as exc:
            print(f"[decoupling] {label} instance failed: {exc}", file=sys.stderr)
            return None

    def handle(r):
        with lock:
            rows.append(r)
            with log_path.open("a") as fh:
                fh.write(json.dumps({"cell": label, "outcome": r[0], "oracle": r[1], "rounds": r[2]}) + "\n")
            print(f"  [{label}] {len(rows)}/{total}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(work, inst) for inst in insts]
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                handle(r)
    return rows


def _agreement(belief_vectors):
    # 1 - mean pairwise absolute distance, averaged over instances (higher = more agreement)
    vals = []
    for b in belief_vectors:
        if len(b) >= 2:
            vals.append(1.0 - float(np.mean([abs(x - y) for x, y in itertools.combinations(b, 2)])))
    return float(np.mean(vals)) if vals else float("nan")


def _confidence(belief_vectors):
    # mean over agents and instances of distance from 0.5, scaled to [0,1]
    flat = [2 * abs(p - 0.5) for b in belief_vectors for p in b]
    return float(np.mean(flat)) if flat else float("nan")


def report(label, rows, rounds):
    ys = [r[0] for r in rows]
    oracle_b = scoring.brier_score([r[1] for r in rows], ys)
    print(f"\n=== {label} (n={len(rows)}, base {np.mean(ys):.2f}, oracle Brier {oracle_b:.3f}) ===")
    print(f"  {'round':<7}{'brier':>8}{'gap_oracle':>12}{'agreement':>11}{'confidence':>12}{'ece':>8}")
    for r in range(rounds + 1):
        means = [naive_mean_from(row[2][r]) for row in rows]
        bv = [row[2][r] for row in rows]
        brier = scoring.brier_score(means, ys)
        ece = scoring.expected_calibration_error(means, ys)
        print(f"  {('priv' if r==0 else 'r'+str(r)):<7}{brier:>8.3f}{brier-oracle_b:>12.3f}"
              f"{_agreement(bv):>11.3f}{_confidence(bv):>12.3f}{ece:>8.3f}")


def naive_mean_from(probs):
    return float(np.mean(probs)) if probs else 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infoagg.yaml")
    ap.add_argument("--cells", default="substitutable,comp-AND,comp-OR")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--out", default="data/decoupling.jsonl")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = get_client("litellm") if args.live else get_client("fake")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    log_path = ROOT / args.out
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    print(f"mode={'LIVE' if args.live else 'offline'} | cells={cells} | rounds={args.rounds} "
          f"| models={cfg.swarm.models}")

    results = {}
    for label in cells:
        results[label] = run_cell(label, cfg, client, args.n, args.rounds, args.concurrency, log_path)
    for label in cells:
        report(label, results[label], args.rounds)
    print(f"\nlog: {log_path}")


if __name__ == "__main__":
    main()
