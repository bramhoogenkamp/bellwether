#!/usr/bin/env python3
"""Score forecasts on ForecastBench resolved questions.

Two modes:

  # FREE, instant: score ForecastBench's own crowd probabilities (the bar to beat)
  python scripts/score_forecastbench.py --market-baseline-only

  # Run our swarm and score it (default: dataset questions, NO retrieval = honest).
  # Live retrieval on resolved questions LEAKS, so it's off by default here.
  python scripts/score_forecastbench.py --limit 20 --sources yfinance,fred,dbnomics,wikipedia

For an *informed* honest number, forecast an OPEN question set and score it on a
future resolution set instead (true forward-test).
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
from bellwether.agents.llm import get_client  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.conditions import compute_conditions  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.evidence.base import apply_leakage_guard, get_evidence_source  # noqa: E402
from bellwether.questions.forecastbench import ForecastBenchQuestionSource  # noqa: E402

_LABEL = {"A": "single LLM", "B": "naive mean", "C": "tuned", "D": "market (LMSR)",
          "E": "market+tuned", "M": "ForecastBench crowd"}


def _print_scores(title, scored: dict):
    print(f"\n{title}")
    print(f"{'cond':<5}{'name':<22}{'brier':>9}{'bss':>9}{'ece':>8}{'n':>6}")
    print("-" * 59)
    for c in sorted(scored, key=lambda k: scored[k]["brier"]):
        s = scored[c]
        print(f"{c:<5}{_LABEL.get(c, c):<22}{s['brier']:>9.4f}{s['bss']:>9.3f}{s['ece']:>8.3f}{s['n']:>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution-set", default=None, help="filename; default = latest")
    ap.add_argument("--question-set", default=None)
    ap.add_argument("--data-dir", default=str(ROOT / "data/forecastbench/datasets"))
    ap.add_argument("--sources", default="yfinance,fred,dbnomics,wikipedia")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--market-baseline-only", action="store_true")
    ap.add_argument("--config", default="configs/live_good.yaml")
    ap.add_argument("--retrieval", default="none", choices=["none", "web"])
    args = ap.parse_args()

    # --- market baseline: score ForecastBench's own crowd probabilities (free) ---
    if args.market_baseline_only:
        src = ForecastBenchQuestionSource(
            resolution_set=args.resolution_set, question_set=args.question_set,
            data_dir=args.data_dir, sources=["manifold", "metaculus", "polymarket", "infer"],
        )
        qs = [q for q in src.fetch(limit=10_000) if q.market_prob is not None]
        probs = [q.market_prob for q in qs]
        ys = [q.outcome for q in qs]
        _print_scores(
            f"ForecastBench CROWD baseline ({len(qs)} resolved market questions)",
            {"M": scoring.score_all(probs, ys)},
        )
        print("\nThis is the bar to beat. (Brier 0.25 = coin flip; lower is better.)")
        return

    # --- run our swarm and score it ---
    cfg = BenchmarkConfig.from_yaml(args.config)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    src = ForecastBenchQuestionSource(
        resolution_set=args.resolution_set, question_set=args.question_set,
        data_dir=args.data_dir, sources=sources,
    )
    questions = src.fetch(limit=args.limit)
    print(f"Loaded {len(questions)} resolved ForecastBench questions (sources={sources}).")
    print(f"Retrieval: {args.retrieval}  |  models: {cfg.swarm.models}")

    client = get_client("litellm")
    swarm = Swarm(cfg.swarm, client)
    evidence_src = None
    if args.retrieval == "web":
        from bellwether.evidence.web import WebEvidenceSource
        evidence_src = WebEvidenceSource(client=client, max_items=cfg.evidence.max_items,
                                         exclude_markets=True)

    rows = []
    for i, q in enumerate(questions, 1):
        evidence = []
        if evidence_src is not None:
            evidence = apply_leakage_guard(evidence_src.gather(q, cfg.evidence.max_items), q.issue_date)
        forecasts = swarm.run(q, evidence)
        if not forecasts:
            continue
        conds = compute_conditions(forecasts, q, cfg, seed=cfg.seed + i)
        rows.append({"outcome": q.outcome, **conds})
        print(f"  [{i}/{len(questions)}] y={q.outcome:.0f} " + " ".join(f"{k}={v:.2f}" for k, v in conds.items()))

    if not rows:
        print("No rows scored.")
        return
    scored = {}
    for c in ["A", "B", "C", "D", "E", "M"]:
        ps = [r[c] for r in rows if c in r]
        ys = [r["outcome"] for r in rows if c in r]
        if ps:
            scored[c] = scoring.score_all(ps, ys)
    _print_scores(f"Bellwether on ForecastBench ({len(rows)} questions)", scored)


if __name__ == "__main__":
    main()
