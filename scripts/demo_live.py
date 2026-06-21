#!/usr/bin/env python3
"""Watch a Bellwether market form: agents trade one question, price updates live.

This is the visual demo. By default it runs fully offline on a mock question with
the FakeLLM, printing each agent's thesis and how the LMSR price moves as they
trade. With --live it pulls one open Manifold question and uses real models.

    python scripts/demo_live.py                       # offline mock demo
    python scripts/demo_live.py --live --url <manifold-market-url>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bellwether.agents.llm import get_client  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.evidence.base import apply_leakage_guard, get_evidence_source  # noqa: E402
from bellwether.market.lmsr import LMSRMarket, liquidity_from_budget  # noqa: E402
from bellwether.market.trading import kelly_intent, shares_for_stake  # noqa: E402
from bellwether.questions.mock_internal import MockInternalQuestionSource  # noqa: E402


def run_market_verbose(question, forecasts, market_cfg) -> float:
    b = liquidity_from_budget(market_cfg.max_loss_budget, 2)
    market = LMSRMarket(b=b)
    bankrolls = [float(market_cfg.starting_bankroll)] * len(forecasts)

    print(f"\nQuestion: {question.text}")
    if question.is_resolved:
        print(f"(actual outcome: {'YES' if question.outcome else 'NO'})")
    print(f"\nStarting price: YES = {market.prob_yes():.3f}\n")
    print(f"{'agent':<28}{'belief':>8}{'side':>6}{'price->':>10}")
    print("-" * 52)

    for round_i in range(market_cfg.rounds):
        for i, f in enumerate(forecasts):
            price = market.prob_yes()
            intent = kelly_intent(
                f.probability, price, market_cfg.kelly_fraction, market_cfg.max_bet_fraction
            )
            if intent is None or bankrolls[i] <= 1e-9:
                continue
            stake = intent.fraction * f.confidence * bankrolls[i]
            shares = shares_for_stake(market, intent.outcome, stake)
            bankrolls[i] -= market.buy(intent.outcome, shares)
            side = "YES" if intent.outcome == 0 else "NO"
            label = f"{f.model.split('/')[-1]}/{f.lens}"
            print(f"{label:<28}{f.probability:>8.2f}{side:>6}{market.prob_yes():>10.3f}")

    print(f"\nFinal market probability: YES = {market.prob_yes():.3f}")
    return market.prob_yes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--url", default=None, help="Manifold market URL (with --live)")
    args = ap.parse_args()

    config = BenchmarkConfig.from_yaml(args.config)

    if args.live:
        client = get_client("litellm")
        # Minimal live path: fetch one open Manifold market by URL would go here.
        raise SystemExit("Live demo: wire a Manifold open-question fetch by --url, then reuse the loop below.")
    else:
        client = get_client("fake")
        question = MockInternalQuestionSource(seed=7).fetch(limit=1)[0]
        evidence = apply_leakage_guard(
            get_evidence_source("mock_internal", seed=config.seed).gather(
                question, config.evidence.max_items, as_of=question.issue_date
            ),
            question.issue_date,
        )

    swarm = Swarm(config.swarm, client)
    forecasts = swarm.run(question, evidence)

    print("\nAgent theses:")
    for f in forecasts:
        print(f"  - {f.model.split('/')[-1]}/{f.lens}: {f.probability:.2f}  {f.thesis}")

    run_market_verbose(question, forecasts, config.market)


if __name__ == "__main__":
    main()
