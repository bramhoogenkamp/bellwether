#!/usr/bin/env python3
"""Cache public benchmark data locally for offline scoring (network required).

    python scripts/pull_questions.py manifold --limit 200
    python scripts/pull_questions.py forecastbench   # clones the datasets repo

Manifold resolved markets are saved to data/manifold_resolved.json. ForecastBench
is cloned to data/forecastbench (CC BY-SA); pick a question_set + matching
resolution_set from there for the runner.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).resolve().parents[1] / "data"
_FB_REPO = "https://github.com/forecastingresearch/forecastbench-datasets.git"


def pull_manifold(limit: int) -> None:
    from bellwether.questions.manifold import ManifoldQuestionSource

    DATA.mkdir(exist_ok=True)
    questions = ManifoldQuestionSource().fetch(limit=limit)
    out = DATA / "manifold_resolved.json"
    out.write_text(json.dumps([q.model_dump(mode="json") for q in questions], indent=2))
    print(f"saved {len(questions)} resolved Manifold markets -> {out}")


def pull_forecastbench() -> None:
    DATA.mkdir(exist_ok=True)
    dest = DATA / "forecastbench"
    if dest.exists():
        print(f"{dest} already exists; pulling latest")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "1", _FB_REPO, str(dest)], check=True)
    print(f"forecastbench datasets at {dest} (see datasets/question_sets + resolution_sets)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["manifold", "forecastbench"])
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    if args.source == "manifold":
        pull_manifold(args.limit)
    else:
        pull_forecastbench()


if __name__ == "__main__":
    main()
