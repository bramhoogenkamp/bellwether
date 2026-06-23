#!/usr/bin/env python3
"""P1: predict, before deliberating, whether deliberation will help or hurt.

Features come only from the pre-deliberation (round 0) beliefs: how dispersed the agents
are, how far they sit from 0.5, and whether a minority dissents. The label is whether
deliberation reduced the squared error of the swarm mean. We validate with
leave-one-condition-out cross-validation, report AUC, and evaluate a gating policy that
deliberates only when help is predicted, against always-deliberate and never-deliberate.

The strong test is transfer: train on one log (synthetic) and test on another (the
forward test, or HiddenBench). A rule learned on toy tasks that calls the sign of
deliberation on a dataset we did not build is the result that sells the predictor.

    python scripts/predict_deliberation.py --train data/decoupling.jsonl
    python scripts/predict_deliberation.py --train data/decoupling.jsonl --test data/other.jsonl
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]


def _features(round0):
    b = np.asarray(round0, float)
    mean = float(b.mean())
    return [
        float(b.std()),                       # dispersion
        float(b.max() - b.min()),             # spread
        float(np.mean(2 * np.abs(b - 0.5))),  # mean confidence (extremity)
        float(max(abs(x - mean) for x in b)), # largest dissent from the mean
        mean,                                 # mean belief
    ]


def _load(path):
    X, y_help, groups, before, after, outcomes = [], [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rounds = r.get("rounds")
        if not rounds or len(rounds) < 2:
            continue
        out = float(r["outcome"])
        m0, mK = float(np.mean(rounds[0])), float(np.mean(rounds[-1]))
        helped = 1 if (mK - out) ** 2 < (m0 - out) ** 2 else 0
        X.append(_features(rounds[0]))
        y_help.append(helped)
        groups.append(r.get("cell", "all"))
        before.append(m0)
        after.append(mK)
        outcomes.append(out)
    return (np.array(X), np.array(y_help), np.array(groups),
            np.array(before), np.array(after), np.array(outcomes))


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _policy_briers(pred_help, before, after, outcomes):
    gated = np.where(pred_help == 1, after, before)
    return {
        "never_deliberate": _brier(before, outcomes),
        "always_deliberate": _brier(after, outcomes),
        "gated (predicted)": _brier(gated, outcomes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/decoupling.jsonl")
    ap.add_argument("--test", default=None, help="optional held-out log for transfer (OOD) eval")
    args = ap.parse_args()

    X, y, g, before, after, out = _load(args.train)
    print(f"train: {len(X)} instances, {len(set(g))} conditions, deliberation helped {y.mean():.0%}")
    if len(set(y)) < 2:
        print("only one class for 'helped'; need both to train a predictor.")
        return

    # Leave-one-condition-out cross-validation -> out-of-fold predictions
    logo = LeaveOneGroupOut()
    oof = np.full(len(X), np.nan)
    if len(set(g)) >= 2:
        for tr, te in logo.split(X, y, g):
            if len(set(y[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    if mask.sum() and len(set(y[mask])) == 2:
        print(f"leave-one-condition-out AUC: {roc_auc_score(y[mask], oof[mask]):.3f}")
        pol = _policy_briers((oof[mask] >= 0.5).astype(int), before[mask], after[mask], out[mask])
        print("  policy Brier (lower better):")
        for k, v in pol.items():
            print(f"    {k:<22}{v:.3f}")

    if args.test:
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        Xt, yt, gt, bt, at, ot = _load(args.test)
        if len(Xt) == 0:
            print(f"\ntest log {args.test} has no usable rows.")
            return
        pt = clf.predict_proba(Xt)[:, 1]
        print(f"\nTRANSFER to {args.test}: {len(Xt)} instances, helped {yt.mean():.0%}")
        if len(set(yt)) == 2:
            print(f"  transfer AUC: {roc_auc_score(yt, pt):.3f}")
        pol = _policy_briers((pt >= 0.5).astype(int), bt, at, ot)
        print("  policy Brier (lower better):")
        for k, v in pol.items():
            print(f"    {k:<22}{v:.3f}")


if __name__ == "__main__":
    main()
