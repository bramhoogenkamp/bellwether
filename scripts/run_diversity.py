#!/usr/bin/env python3
"""Iteration 6: model-mix / diversity sweep.

Does swarm composition change the deliberation result? We run the same
deliberation-depth comparison under a diverse mix of model families and under
homogeneous swarms of a single model. LLMs tend to be near-clones, so this checks
whether effective diversity is a hidden ceiling on how well deliberation pools
dispersed information. Run on the unstructured cell, where models must interpret messy
prose and so are most likely to differ.

    python scripts/run_diversity.py --cell unstr-AND --n 24 --rounds 3 --live --mlflow
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

GRID = {
    "comp-AND": ("complementary", 4, "and"),
    "comp-OR": ("complementary", 4, "or"),
    "unstr-AND": ("unstructured", 4, "and"),
    "unstr-OR": ("unstructured", 4, "or"),
}

COMPOSITIONS = {
    "diverse": ["openai/gpt-5-mini", "anthropic/claude-sonnet-4.5",
                "google/gemini-2.5-pro", "deepseek/deepseek-r1"],
    "homo-gpt5mini": ["openai/gpt-5-mini"],
    "homo-sonnet": ["anthropic/claude-sonnet-4.5"],
}


def instance_conditions(swarm, inst, rounds):
    private = swarm.forecast_each_private(inst.question, inst.slices)
    pooled = swarm.run(inst.question, inst.pooled)
    out = {
        "average": naive_mean(private),
        "oracle": naive_mean(pooled) if pooled else 0.5,
        "private_probs": [f.probability for f in private],
    }
    cur = private
    for r in range(1, rounds + 1):
        cur = swarm.run_debate_round(inst.question, inst.slices, cur)
        out[f"debate_r{r}"] = naive_mean(cur)
    return out


def run_composition(comp_name, models, label, cfg, client, n, rounds, concurrency, log_path):
    structure, n_agents, rule = GRID[label]
    swarm = Swarm(cfg.swarm.model_copy(update={"models": models, "n_agents": n_agents}), client)
    insts = generate_info_instances(n=n, n_agents=n_agents, structure=structure, seed=cfg.seed, rule=rule)
    rows, lock, total = [], threading.Lock(), len(insts)

    def work(inst):
        try:
            return inst, instance_conditions(swarm, inst, rounds)
        except Exception as exc:
            print(f"[diversity] {comp_name} instance failed: {exc}", file=sys.stderr)
            return inst, None

    def handle(inst, probs):
        with lock:
            rows.append({"y": inst.question.outcome, **(probs or {})})
            with log_path.open("a") as fh:
                fh.write(json.dumps({"composition": comp_name, "cell": label,
                                     "outcome": inst.question.outcome, "conditions": probs or {}}) + "\n")
            print(f"  [{comp_name}] {len(rows)}/{total}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(work, inst) for inst in insts]
        for fut in as_completed(futs):
            inst, probs = fut.result()
            handle(inst, probs)
    return rows


def _brier(rows, cond):
    pr = [(r[cond], r["y"]) for r in rows if cond in r]
    return scoring.brier_score([p for p, _ in pr], [y for _, y in pr]) if pr else None


def _disagreement(rows):
    # mean pairwise absolute difference of the agents' private forecasts (a diversity proxy)
    ds = []
    for r in rows:
        p = r.get("private_probs", [])
        if len(p) >= 2:
            ds.append(float(np.mean([abs(a - b) for a, b in itertools.combinations(p, 2)])))
    return float(np.mean(ds)) if ds else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infoagg.yaml")
    ap.add_argument("--cell", default="unstr-AND")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = get_client("litellm") if args.live else get_client("fake")
    log_path = ROOT / "data" / "infoagg_diversity.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    print(f"mode={'LIVE' if args.live else 'offline'} | cell={args.cell} | compositions={list(COMPOSITIONS)}")

    summary = {}
    for comp_name, models in COMPOSITIONS.items():
        rows = run_composition(comp_name, models, args.cell, cfg, client, args.n, args.rounds,
                               args.concurrency, log_path)
        summary[comp_name] = (rows, _disagreement(rows))

    print(f"\n=== {args.cell}: does swarm diversity change the deliberation result? ===")
    print(f"{'composition':<16}{'disagree':>9}{'average':>9}" + "".join(f"{'round'+str(r):>9}" for r in range(1, args.rounds + 1)) + f"{'oracle':>9}")
    for comp_name, (rows, dis) in summary.items():
        line = f"{comp_name:<16}{dis:>9.3f}{_brier(rows,'average'):>9.3f}"
        line += "".join(f"{_brier(rows, f'debate_r{r}'):>9.3f}" for r in range(1, args.rounds + 1))
        line += f"{_brier(rows,'oracle'):>9.3f}"
        print(line)
    print(f"\nlog: {log_path}")


if __name__ == "__main__":
    main()
