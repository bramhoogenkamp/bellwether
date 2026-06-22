#!/usr/bin/env python3
"""Information-aggregation experiment: market vs. average vs. pooled-oracle.

For synthetic dispersed-private-information instances (known ground truth), each agent
sees only its private slice. We compare:
  * AVERAGE  (B): naive mean of the agents' private forecasts        [the baseline]
  * MARKET   (D): the agents trade an LMSR market on their slices     [under test]
  * ORACLE   (O): every agent sees the pooled evidence, then averaged  [upper bound]

Hypotheses (research/intro.md): for SUBSTITUTABLE signals, market ≈ average; for
COMPLEMENTARY signals, market >> average, toward the oracle.

    python scripts/run_infoagg.py --structure complementary --n 40        # offline (FakeLLM)
    python scripts/run_infoagg.py --structure both --n 40 --live          # real models

Offline (FakeLLM) validates plumbing; the scientific result needs --live (real reasoning).
"""

from __future__ import annotations

import argparse
import os
import sys
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


def run_structure(structure: str, swarm: Swarm, cfg: BenchmarkConfig, n: int) -> None:
    instances = generate_info_instances(
        n=n, n_agents=cfg.swarm.n_agents, structure=structure, seed=cfg.seed
    )
    rows = []
    for i, inst in enumerate(instances):
        private = swarm.run_private(inst.question, inst.slices)
        if not private:
            continue
        pooled = swarm.run(inst.question, inst.pooled)  # everyone informed = oracle
        rows.append({
            "y": inst.question.outcome,
            "average": naive_mean(private),
            "market": market_price(private, cfg.market, seed=cfg.seed + i),
            "oracle": naive_mean(pooled) if pooled else 0.5,
        })

    ys = [r["y"] for r in rows]
    scored = {k: scoring.score_all([r[k] for r in rows], ys) for k in ("average", "market", "oracle")}

    print(f"\n=== {structure.upper()} signals — {len(rows)} instances (base rate {scored['average']['base_rate']:.2f}) ===")
    print(f"{'method':<10}{'brier':>9}{'bss':>9}{'resolution':>12}{'ece':>8}")
    print("-" * 48)
    for k in ("average", "market", "oracle"):
        s = scored[k]
        print(f"{k:<10}{s['brier']:>9.4f}{s['bss']:>9.3f}{s['resolution']:>12.4f}{s['ece']:>8.3f}")

    d = scoring.paired_bootstrap_brier_delta(
        [r["market"] for r in rows], [r["average"] for r in rows], ys, seed=cfg.seed
    )
    sig = "significant" if (d["ci_high"] < 0 or d["ci_low"] > 0) else "n.s."
    print(f"\nMarket vs Average: ΔBrier={d['mean_delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] ({sig})")
    print("  (negative => market beats the average)")
    gap = scored["market"]["brier"] - scored["oracle"]["brier"]
    print(f"Market's gap to the oracle (upper bound): {gap:+.4f} Brier")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default="both", choices=["substitutable", "complementary", "both"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--config", default="configs/live_good.yaml")
    ap.add_argument("--live", action="store_true", help="use real models (default: FakeLLM offline)")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = get_client("litellm") if args.live else get_client("fake")
    swarm = Swarm(cfg.swarm, client)
    print(f"mode={'LIVE' if args.live else 'offline (FakeLLM)'} | agents={cfg.swarm.n_agents} | models={cfg.swarm.models}")

    structures = ["substitutable", "complementary"] if args.structure == "both" else [args.structure]
    for s in structures:
        run_structure(s, swarm, cfg, args.n)


if __name__ == "__main__":
    main()
