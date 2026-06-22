#!/usr/bin/env python3
"""Honest forward-test: forecast currently-OPEN markets with web-researched agents.

Because the markets are open, the outcome doesn't exist yet — so web retrieval cannot
leak the answer. We research each question, run the swarm, and compare our forecast to
the live market price (the crowd). We log every prediction so it can be Brier-scored
later, when the markets resolve.

    python scripts/forecast_open.py --config configs/live_good.yaml --limit 5

Needs OPENROUTER_API_KEY in .env. Spends real money (strong/reasoning models).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LITELLM_LOG", "ERROR")  # quiet litellm's per-call chatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bellwether.aggregate.market import market_price  # noqa: E402
from bellwether.aggregate.naive import naive_mean  # noqa: E402
from bellwether.aggregate.tuned import tuned_aggregate  # noqa: E402
from bellwether.agents.llm import LiteLLMClient  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.evidence.web import WebEvidenceSource  # noqa: E402
from bellwether.questions.polymarket import PolymarketQuestionSource  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live_good.yaml")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--research-model", default="openai/gpt-4o-mini:online")
    ap.add_argument(
        "--allow-odds",
        action="store_true",
        help="allow betting odds / market prices in research (default: stripped, to avoid circularity)",
    )
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = LiteLLMClient()
    swarm = Swarm(cfg.swarm, client)
    web = WebEvidenceSource(
        model=args.research_model,
        client=client,
        max_items=cfg.evidence.max_items,
        exclude_markets=not args.allow_odds,
    )
    print(f"Research mode: {'odds ALLOWED' if args.allow_odds else 'odds/markets STRIPPED'}")

    print(f"Fetching {args.limit} open Polymarket markets...")
    questions = PolymarketQuestionSource().fetch_open(limit=args.limit)
    print(f"Got {len(questions)} open markets. Models: {cfg.swarm.models}\n")

    log_path = ROOT / "data" / "forecasts_open.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    gaps = []
    with log_path.open("a") as logf:
        for i, q in enumerate(questions, 1):
            evidence = web.gather(q, cfg.evidence.max_items)
            forecasts = swarm.run(q, evidence)
            if not forecasts:
                print(f"[{i}] {q.text[:70]} — no forecasts (all agents failed); skipping\n")
                continue

            ours_market = market_price(forecasts, cfg.market, seed=cfg.seed + i)
            ours_naive = naive_mean(forecasts)
            ours_tuned = tuned_aggregate(forecasts, cfg.calibration.extremize_coef)
            gap = abs(ours_market - q.market_prob)
            gaps.append(gap)

            print(f"[{i}] {q.text}")
            print(f"    live market (crowd): {q.market_prob:.2f}   |   resolves: {q.resolution_date}")
            print(f"    Bellwether  market={ours_market:.2f}  naive={ours_naive:.2f}  tuned={ours_tuned:.2f}   (|gap to crowd|={gap:.2f})")
            print(f"    evidence ({len(evidence)} findings, odds-stripped={not args.allow_odds}):")
            for e in evidence[:4]:
                print(f"        · {e.text[:110]}")
            for f in forecasts[:3]:
                print(f"      - {f.model.split('/')[-1]}/{f.lens} p={f.probability:.2f}: {f.thesis[:90]}")
            print()

            logf.write(json.dumps({
                "ts": now,
                "id": q.id,
                "question": q.text,
                "resolution_date": str(q.resolution_date),
                "market_prob": q.market_prob,
                "bellwether_market": ours_market,
                "bellwether_naive": ours_naive,
                "bellwether_tuned": ours_tuned,
                "agents": [
                    {"model": f.model, "lens": f.lens, "p": f.probability, "thesis": f.thesis}
                    for f in forecasts
                ],
            }) + "\n")

    if gaps:
        print(f"Mean |Bellwether market − crowd| over {len(gaps)} questions: {sum(gaps)/len(gaps):.3f}")
    print(f"Predictions logged to {log_path} (score them once the markets resolve).")


if __name__ == "__main__":
    main()
