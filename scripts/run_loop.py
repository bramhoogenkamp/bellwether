#!/usr/bin/env python3
"""Run the self-improving loop over cached agent beliefs (no new model calls).

Replays many market and aggregation designs over beliefs already logged by
run_infoagg, scores each against the known outcomes on a train and a held-out
validation split, and writes a reviewable ledger of designs, scores, and decisions.

    python scripts/run_loop.py --log data/infoagg_infoagg.jsonl --cells comp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bellwether.loop import format_ledger, load_beliefs, run_loop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "data/infoagg_infoagg.jsonl"))
    ap.add_argument("--cells", default="comp", help="comma-separated cell label prefixes")
    ap.add_argument("--out", default=str(ROOT / "experiments/loop/ledger.md"))
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    records = load_beliefs(args.log, cells)
    if not records:
        print(f"No cached beliefs in {args.log} for cells {cells}. Run scripts/run_infoagg.py first.")
        return
    print(f"Loaded {len(records)} cached instances from {args.log} (cells: {cells}).")

    result = run_loop(records)
    md = format_ledger(result)
    print("\n" + md)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n")
    (out.parent / "ledger.jsonl").write_text(
        "\n".join(json.dumps(h) for h in result["history"]) + "\n"
    )
    print(f"\nLedger written to {out} (machine-readable: {out.parent / 'ledger.jsonl'}).")


if __name__ == "__main__":
    main()
