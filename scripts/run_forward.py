#!/usr/bin/env python3
"""Real-data forward test (leaves the synthetic lab).

Forecast currently-OPEN Polymarket questions that resolve within a short horizon, so we
get real Brier scores soon. The web-researched evidence (odds stripped) is partitioned
across a diverse swarm so each agent holds a different slice (dispersed private
information), the agents deliberate, and we log single / average / deliberated / market
forecasts alongside the crowd price. Outcomes do not exist yet, so this logs predictions
now and is scored later, when the markets resolve.

    python scripts/run_forward.py --config configs/forward.yaml --limit 40 --max-days 7 --rounds 2 --live

Needs OPENROUTER_API_KEY in .env. Spends real money.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("LITELLM_LOG", "ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bellwether.aggregate.market import market_price  # noqa: E402
from bellwether.aggregate.naive import naive_mean  # noqa: E402
from bellwether.agents.llm import LiteLLMClient, get_client  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.evidence.web import WebEvidenceSource  # noqa: E402
from bellwether.questions.polymarket import PolymarketQuestionSource  # noqa: E402


def partition(evidence, n):
    """Round-robin the evidence into n slices, one per agent (dispersed information)."""
    return [evidence[i::n] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/forward.yaml")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-days", type=int, default=7, help="only questions resolving within this many days")
    ap.add_argument("--rounds", type=int, default=2, help="deliberation rounds")
    ap.add_argument("--research-model", default="openai/gpt-4o-mini:online")
    ap.add_argument("--allow-odds", action="store_true")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = LiteLLMClient() if args.live else get_client("fake")
    swarm = Swarm(cfg.swarm, client)
    web = WebEvidenceSource(model=args.research_model, client=client,
                            max_items=cfg.evidence.max_items, exclude_markets=not args.allow_odds)

    cutoff = (datetime.now(timezone.utc).date() + timedelta(days=args.max_days))
    print(f"Fetching open Polymarket markets resolving on/before {cutoff} "
          f"(odds {'ALLOWED' if args.allow_odds else 'STRIPPED'})...")
    questions = PolymarketQuestionSource(scan_limit=500).fetch_open_within(days=args.max_days, limit=args.limit)
    print(f"{len(questions)} open markets resolve within {args.max_days} days. "
          f"Models: {cfg.swarm.models}\n")
    if not questions:
        print("No open markets resolve within the horizon; try a larger --max-days.")
        return

    log_path = ROOT / "data" / "forecasts_forward.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    now = datetime.now(timezone.utc).isoformat()
    rows, lock = [], threading.Lock()
    n_agents = cfg.swarm.n_agents

    def work(i, q):
        try:
            evidence = web.gather(q, cfg.evidence.max_items)
            slices = partition(evidence, n_agents)
            private = swarm.forecast_each_private(q, slices)
            cur = private
            for _ in range(args.rounds):
                cur = swarm.run_debate_round(q, slices, cur)
            cond = {
                "single": private[0].probability,
                "average": naive_mean(private),
                "deliberated": naive_mean(cur),
                "market": market_price(private, cfg.market, seed=cfg.seed + i),
                "market_deliberated": market_price(cur, cfg.market, seed=cfg.seed + i),
                "crowd": q.market_prob,
            }
            return i, q, cond, evidence, private
        except Exception as exc:
            print(f"[forward] {q.text[:60]} failed: {exc}", file=sys.stderr)
            return i, q, None, [], []

    def handle(i, q, cond, evidence, private):
        with lock:
            rows.append((q, cond))
            with log_path.open("a") as fh:
                fh.write(json.dumps({
                    "ts": now, "id": q.id, "question": q.text,
                    "resolution_date": str(q.resolution_date), "outcome": None,
                    "conditions": cond or {},
                    "n_evidence": len(evidence),
                    "evidence": [e.text[:200] for e in evidence],
                    "agents": [{"model": f.model, "p": f.probability, "thesis": f.thesis[:160]} for f in private],
                }) + "\n")
            if cond:
                print(f"[{len(rows)}/{len(questions)}] {q.text[:70]}  (resolves {q.resolution_date})")
                print(f"    crowd {cond['crowd']:.2f} | single {cond['single']:.2f}  avg {cond['average']:.2f}  "
                      f"deliberated {cond['deliberated']:.2f}  market {cond['market']:.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, i, q): i for i, q in enumerate(questions)}
        for fut in as_completed(futs):
            handle(*fut.result())

    done = [(q, c) for q, c in rows if c]
    if done:
        import numpy as np
        for cond in ("single", "average", "deliberated", "market"):
            gap = float(np.mean([abs(c[cond] - c["crowd"]) for _, c in done]))
            print(f"  mean |{cond} - crowd| = {gap:.3f}")
    print(f"\nLogged {len(done)} forecasts to {log_path}. Score with scripts/score_forward.py once they resolve.")


if __name__ == "__main__":
    main()
