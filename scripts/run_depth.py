#!/usr/bin/env python3
"""Iteration 3: deliberation depth.

Track how the deliberated average and a market over the deliberated beliefs move
toward the oracle as the number of deliberation rounds increases, on the complementary
cells. Each round, agents revise after seeing peers' rationales; we record the
aggregate after rounds 1, 2, and 3 in a single nested pass.

    python scripts/run_depth.py --cells comp-AND,comp-OR --n 24 --rounds 3 --live --mlflow

Offline (FakeLLM) is plumbing only, since FakeLLM cannot reason about the conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("LITELLM_LOG", "ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bellwether import scoring  # noqa: E402
from bellwether.aggregate.market import market_price  # noqa: E402
from bellwether.aggregate.naive import naive_mean  # noqa: E402
from bellwether.agents.llm import get_client  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.questions.synthetic import generate_info_instances  # noqa: E402

GRID = {  # label -> (structure, n_agents, rule)
    "comp-AND": ("complementary", 4, "and"),
    "comp-OR": ("complementary", 4, "or"),
    "comp-THRESH": ("complementary", 5, "threshold"),
    "unstr-AND": ("unstructured", 4, "and"),
    "unstr-OR": ("unstructured", 4, "or"),
    "unstr-THRESH": ("unstructured", 5, "threshold"),
}


def instance_conditions(swarm, cfg, inst, seed, rounds):
    private = swarm.forecast_each_private(inst.question, inst.slices)
    pooled = swarm.run(inst.question, inst.pooled)
    probs = {
        "average": naive_mean(private),
        "market": market_price(private, cfg.market, seed=seed),
        "oracle": naive_mean(pooled) if pooled else 0.5,
    }
    cur = private
    for r in range(1, rounds + 1):
        cur = swarm.run_debate_round(inst.question, inst.slices, cur)
        probs[f"debate_r{r}"] = naive_mean(cur)
        probs[f"market_r{r}"] = market_price(cur, cfg.market, seed=seed)
    return probs


def run_cell(label, cfg, client, n, rounds, concurrency, log_path):
    structure, n_agents, rule = GRID[label]
    swarm = Swarm(cfg.swarm.model_copy(update={"n_agents": n_agents}), client)
    insts = generate_info_instances(n=n, n_agents=n_agents, structure=structure, seed=cfg.seed, rule=rule)
    rows, lock, total = [], threading.Lock(), len(insts)

    def work(i, inst):
        try:
            return inst, instance_conditions(swarm, cfg, inst, cfg.seed + i, rounds)
        except Exception as exc:
            print(f"[depth] instance failed: {exc}", file=sys.stderr)
            return inst, None

    def handle(i, inst, probs):
        with lock:
            rows.append({"y": inst.question.outcome, **(probs or {})})
            with log_path.open("a") as fh:
                fh.write(json.dumps({"cell": label, "instance": i,
                                     "outcome": inst.question.outcome, "conditions": probs or {}}) + "\n")
            print(f"  [{label}] {len(rows)}/{total}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(work, i, inst): i for i, inst in enumerate(insts)}
        for fut in as_completed(futs):
            inst, probs = fut.result()
            handle(futs[fut], inst, probs)
    return rows


def _brier(rows, cond):
    pairs = [(r[cond], r["y"]) for r in rows if cond in r]
    if not pairs:
        return None
    return scoring.brier_score([p for p, _ in pairs], [y for _, y in pairs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infoagg.yaml")
    ap.add_argument("--cells", default="comp-AND,comp-OR")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = get_client("litellm") if args.live else get_client("fake")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    log_path = ROOT / "data" / "infoagg_depth.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    print(f"mode={'LIVE' if args.live else 'offline'} | cells={cells} | rounds={args.rounds} "
          f"| models={cfg.swarm.models}")
    for label in cells:
        rows = run_cell(label, cfg, client, args.n, args.rounds, args.concurrency, log_path)
        oracle = _brier(rows, "oracle")
        print(f"\n=== {label} (n={len(rows)}) ===")
        print(f"  average {_brier(rows,'average'):.3f}   market(private) {_brier(rows,'market'):.3f}   oracle {oracle:.3f}")
        print("  deliberation depth (does it close the gap to the oracle?):")
        for r in range(1, args.rounds + 1):
            print(f"    round {r}: debate {_brier(rows, f'debate_r{r}'):.3f}   market_over_debated {_brier(rows, f'market_r{r}'):.3f}")
    print(f"\nlog: {log_path}")


if __name__ == "__main__":
    main()
