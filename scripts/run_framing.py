#!/usr/bin/env python3
"""Iteration 5: agent-framing A/B.

Neutral honest forecaster versus profit-maximizing trader, paired on the same
instances. Does framing the agents as gain-maximizers change their forecasts, the
market price, or their calibration? Each instance is forecast under both framings, and
we compare the average, the market, and calibration (ECE) between them.

    python scripts/run_framing.py --cells comp-AND,comp-OR --n 24 --live --mlflow
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

GRID = {
    "comp-AND": ("complementary", 4, "and"),
    "comp-OR": ("complementary", 4, "or"),
    "comp-THRESH": ("complementary", 5, "threshold"),
}


def run_cell(label, cfg, client, n, concurrency, log_path):
    structure, n_agents, rule = GRID[label]
    neutral = Swarm(cfg.swarm.model_copy(update={"n_agents": n_agents, "framing": "neutral"}), client)
    trader = Swarm(cfg.swarm.model_copy(update={"n_agents": n_agents, "framing": "trader"}), client)
    insts = generate_info_instances(n=n, n_agents=n_agents, structure=structure, seed=cfg.seed, rule=rule)
    rows, lock, total = [], threading.Lock(), len(insts)

    def work(i, inst):
        try:
            pn = neutral.forecast_each_private(inst.question, inst.slices)
            pt = trader.forecast_each_private(inst.question, inst.slices)
            pooled = neutral.run(inst.question, inst.pooled)
            return inst, {
                "avg_neutral": naive_mean(pn), "mkt_neutral": market_price(pn, cfg.market, seed=cfg.seed + i),
                "avg_trader": naive_mean(pt), "mkt_trader": market_price(pt, cfg.market, seed=cfg.seed + i),
                "oracle": naive_mean(pooled) if pooled else 0.5,
            }
        except Exception as exc:
            print(f"[framing] instance failed: {exc}", file=sys.stderr)
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


def _stat(rows, cond, fn):
    pr = [(r[cond], r["y"]) for r in rows if cond in r]
    return fn([p for p, _ in pr], [y for _, y in pr]) if pr else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infoagg.yaml")
    ap.add_argument("--cells", default="comp-AND,comp-OR")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = get_client("litellm") if args.live else get_client("fake")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    log_path = ROOT / "data" / "infoagg_framing.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    print(f"mode={'LIVE' if args.live else 'offline'} | cells={cells} | models={cfg.swarm.models}")

    for label in cells:
        rows = run_cell(label, cfg, client, args.n, args.concurrency, log_path)
        print(f"\n=== {label} (n={len(rows)}) ===")
        print(f"  oracle {_stat(rows,'oracle',scoring.brier_score):.3f}")
        print(f"  {'framing':<8}{'avg brier':>10}{'mkt brier':>10}{'avg ece':>9}{'mkt ece':>9}")
        for fr in ("neutral", "trader"):
            print(f"  {fr:<8}{_stat(rows,f'avg_{fr}',scoring.brier_score):>10.3f}"
                  f"{_stat(rows,f'mkt_{fr}',scoring.brier_score):>10.3f}"
                  f"{_stat(rows,f'avg_{fr}',scoring.expected_calibration_error):>9.3f}"
                  f"{_stat(rows,f'mkt_{fr}',scoring.expected_calibration_error):>9.3f}")
        for cond in ("avg", "mkt"):
            d = scoring.paired_bootstrap_brier_delta(
                [r[f"{cond}_trader"] for r in rows], [r[f"{cond}_neutral"] for r in rows],
                [r["y"] for r in rows], seed=cfg.seed)
            print(f"  trader - neutral ({cond}): {d['mean_delta']:+.4f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]")
    print(f"\nlog: {log_path}")


if __name__ == "__main__":
    main()
