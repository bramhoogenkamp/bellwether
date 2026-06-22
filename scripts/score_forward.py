#!/usr/bin/env python3
"""Score the forward test, excluding questions whose evidence leaked a forecast.

Two guards are applied at analysis time:
  - probability leakage: drop any question whose retrieved evidence contained an
    external forecast probability (Opta-style "X% chance", "supercomputer",
    "simulations predict", "implied probability"). The odds filter at retrieval time
    catches bookmaker/market terms; this catches third-party model forecasts.
  - outcome leakage: report the future-resolving subset separately, since a same-day
    match may already have been played when the live web search ran.

Distance-to-crowd is shown now; Brier is computed for any market that has resolved
(re-queried from Polymarket Gamma).

    python scripts/score_forward.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Genuine external-forecast leakage (kept tight to avoid flagging "4.62 goals" or
# "their chances of advancing").
LEAK = re.compile(
    r"supercomputer|\d{1,3}(?:\.\d+)?\s*%\s*chance|simulations?\s+predict|"
    r"implied\s+probab|\bodds\s+of\b|\bimplied\s+odds\b",
    re.I,
)
CONDS = ["crowd", "single", "average", "deliberated", "market"]


def _resolved_outcome(gid):
    import httpx
    try:
        m = httpx.get(f"https://gamma-api.polymarket.com/markets/{gid}", timeout=15).json()
        if not m.get("closed"):
            return None
        return float(json.loads(m["outcomePrices"])[0])  # 1.0 if YES settled
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/forecasts_forward.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / args.log).read_text().splitlines() if l.strip()]
    done = [r for r in rows if r.get("conditions")]
    for r in done:
        r["_leaked"] = any(LEAK.search(e) for e in r.get("evidence", []))
    clean = [r for r in done if not r["_leaked"]]
    excluded = [r for r in done if r["_leaked"]]

    print(f"answered: {len(done)} | clean: {len(clean)} | excluded for forecast leakage: {len(excluded)}")
    for r in excluded:
        print(f"   excluded: {r['question'][:70]}")

    print("\n=== distance to crowd on the clean subset, mean |forecast - crowd| ===")
    for c in CONDS[1:]:
        g = np.mean([abs(r["conditions"][c] - r["conditions"]["crowd"]) for r in clean])
        print(f"  {c:<12} {g:.3f}")

    # Brier on resolved clean questions, with the leak-safe future split called out.
    today = datetime.now(timezone.utc).date().isoformat()
    scored = []
    for r in clean:
        o = _resolved_outcome(r["id"].replace("poly-open-", ""))
        if o in (0.0, 1.0):
            scored.append((r, o))
    print(f"\n=== resolved & scored (clean subset): {len(scored)} ===")
    if scored:
        future = [(r, o) for r, o in scored if str(r["resolution_date"]) > today]
        for label, subset in (("all clean resolved", scored), ("future-resolving only (leak-safe)", future)):
            if not subset:
                continue
            print(f"  {label} (n={len(subset)}):")
            for c in CONDS:
                b = np.mean([(r["conditions"][c] - o) ** 2 for r, o in subset])
                print(f"      {c:<12} Brier {b:.3f}")
    else:
        print("  none resolved yet; rerun after the markets settle.")


if __name__ == "__main__":
    main()
