#!/usr/bin/env python3
"""Run the Bellwether benchmark and print a comparison of conditions A-G.

Examples:
    # offline, free (FakeLLM + mock questions):
    python scripts/run_benchmark.py --limit 20

    # log the run to MLflow (then: mlflow ui):
    python scripts/run_benchmark.py --mlflow

    # real models via OpenRouter (needs OPENROUTER_API_KEY in .env):
    python scripts/run_benchmark.py --live --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bellwether.agents.llm import get_client  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.runner import BenchmarkResult, run_benchmark  # noqa: E402

_COND_LABEL = {
    "A": "single LLM",
    "B": "naive mean",
    "C": "tuned aggregator",
    "D": "market (LMSR)",
    "E": "market+tuned",
    "F": "superforecaster",
    "G": "status quo",
}


def print_table(result: BenchmarkResult) -> None:
    print(f"\nScored {result.n_scored} questions.\n")
    print(f"{'cond':<5}{'name':<18}{'brier':>9}{'bss':>9}{'log_loss':>10}{'ece':>8}")
    print("-" * 59)
    for cond in sorted(result.scores, key=lambda c: result.scores[c]["brier"]):
        s = result.scores[cond]
        print(
            f"{cond:<5}{_COND_LABEL.get(cond, cond):<18}"
            f"{s['brier']:>9.4f}{s['bss']:>9.3f}{s['log_loss']:>10.4f}{s['ece']:>8.3f}"
        )

    if result.deltas:
        print("\nHeadline (paired bootstrap, negative => market is better):")
        for name, d in result.deltas.items():
            sig = "significant" if d["ci_high"] < 0 or d["ci_low"] > 0 else "not significant"
            print(
                f"  {name}: ΔBrier={d['mean_delta']:+.4f} "
                f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  ({sig})"
            )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Bellwether benchmark.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--live", action="store_true", help="use real LLMs via OpenRouter")
    ap.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    ap.add_argument("--quiet", action="store_true", help="suppress per-question lines")
    args = ap.parse_args()

    config = BenchmarkConfig.from_yaml(args.config)
    client = get_client("litellm") if args.live else get_client("fake")
    result = run_benchmark(
        config,
        client=client,
        limit=args.limit,
        mlflow_enabled=args.mlflow,
        log=(lambda *_: None) if args.quiet else print,
    )
    print_table(result)


if __name__ == "__main__":
    main()
