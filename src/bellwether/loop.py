"""Self-improving loop over market and aggregation designs.

The expensive step, eliciting agent beliefs from the LLMs, is done once and cached
(the per-instance agent forecasts that scripts/run_infoagg.py already logs). The loop
then replays many designs over those cached beliefs for free, scores each against the
known outcome, and records which beat the plain average and approach the oracle. This
is how we search for a market design that actually aggregates dispersed information,
without re-spending on the models each iteration.

Each design maps a list of per-agent probabilities to one aggregate probability:
- average: the plain mean (the baseline to beat).
- log-opinion pool: the equilibrium an LMSR with Kelly traders settles to (a weighted
  average in log-odds). A single near-certain agent pulls this pool hard toward its
  belief, which a plain mean cannot do.
- extremized log pool: the pool with a sharpening coefficient.
- simulated market: the actual LMSR trading loop from aggregate.market, capped or
  uncapped (uses a confidence proxy, since the cache stores only probabilities).

Guardrail against overfitting: designs are scored on a train split, and a win is only
accepted if it also holds on a held-out validation split.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aggregate.market import market_price
from .agents.agent import Forecast
from .config import MarketConfig
from .scoring import brier_score

_EPS = 1e-6


@dataclass
class Record:
    cell: str
    outcome: float
    beliefs: list[float]  # one private probability per agent
    oracle: float


def load_beliefs(log_path, cells_prefix: list[str] | None = None) -> list[Record]:
    """Load cached per-instance agent beliefs from a run_infoagg JSONL log."""
    out: list[Record] = []
    for line in Path(log_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if cells_prefix and not any(r["cell"].startswith(c) for c in cells_prefix):
            continue
        beliefs = [a["p"] for a in r.get("agents", [])]
        if not beliefs:
            continue
        out.append(Record(r["cell"], float(r["outcome"]), beliefs,
                          float(r.get("conditions", {}).get("oracle", 0.5))))
    return out


# --- designs (probabilities -> one aggregate probability) ---------------------

def average(probs) -> float:
    return float(np.mean(probs))


def log_pool(probs, coef: float = 1.0) -> float:
    """Equilibrium (logarithmic opinion) pool: average in log-odds, optionally sharpened."""
    p = np.clip(np.asarray(probs, float), _EPS, 1 - _EPS)
    z = coef * float(np.mean(np.log(p / (1 - p))))
    return 1.0 / (1.0 + math.exp(-z))


def _forecasts(probs):
    # confidence proxy: more decisive beliefs carry more confidence
    return [Forecast(probability=float(p), confidence=min(max(0.5 + abs(p - 0.5), 0.3), 0.99))
            for p in probs]


def sim_market(probs, kelly: float, cap: float, rounds: int, seed: int = 0) -> float:
    cfg = MarketConfig(kelly_fraction=kelly, max_bet_fraction=cap, rounds=rounds)
    return market_price(_forecasts(probs), cfg, seed=seed)


def base_designs() -> dict:
    return {
        "log-opinion pool (equilibrium)": lambda p: log_pool(p),
        "log pool extremized x1.73": lambda p: log_pool(p, math.sqrt(3)),
        "sim market capped (exp1 settings)": lambda p: sim_market(p, 0.5, 0.25, 3),
        "sim market uncapped (exp2 settings)": lambda p: sim_market(p, 1.0, 1.0, 5),
    }


# --- evaluation + split -------------------------------------------------------

def evaluate(records: list[Record], fn) -> dict:
    ys = [r.outcome for r in records]
    preds = [fn(r.beliefs) for r in records]
    avg = [average(r.beliefs) for r in records]
    orc = [r.oracle for r in records]
    b = brier_score(preds, ys)
    return {
        "brier": b,
        "vs_average": b - brier_score(avg, ys),     # negative => beats the average
        "gap_to_oracle": b - brier_score(orc, ys),   # smaller => closer to the oracle
        "n": len(records),
    }


def split(records, train_frac: float = 0.6, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    k = int(train_frac * len(records))
    train = {int(i) for i in idx[:k]}
    return ([r for i, r in enumerate(records) if i in train],
            [r for i, r in enumerate(records) if i not in train])


# --- the loop -----------------------------------------------------------------

def run_loop(records: list[Record], train_frac: float = 0.6, seed: int = 0) -> dict:
    """Try designs, accept only those that beat the average on train and validation,
    then adaptively tune the winning family. Returns a reviewable history."""
    train, val = split(records, train_frac, seed)
    history = []

    def step(name, fn, note):
        tr, va = evaluate(train, fn), evaluate(val, fn)
        if tr["vs_average"] < 0 and va["vs_average"] < 0:
            decision = "accept: beats average on train and validation"
        elif tr["vs_average"] < 0:
            decision = "reject: won on train but not validation (likely overfit)"
        else:
            decision = "reject: did not beat the average on train"
        history.append({"design": name, "note": note, "train": tr, "val": va, "decision": decision})
        return va["vs_average"] < 0 and tr["vs_average"] < 0

    accepted = []
    for name, fn in base_designs().items():
        if step(name, fn, "base design"):
            accepted.append(name)

    # Adaptive step: if the equilibrium pool family won, tune its sharpening coefficient.
    if any("log" in a for a in accepted):
        best_coef, best_val = 1.0, evaluate(val, log_pool)["vs_average"]
        for coef in (1.3, 1.7, 2.0, 2.5):
            if step(f"log pool coef={coef}", lambda p, c=coef: log_pool(p, c),
                    "proposed: the pool family won, so tune its extremizing coefficient"):
                v = evaluate(val, lambda p, c=coef: log_pool(p, c))["vs_average"]
                if v < best_val:
                    best_coef, best_val = coef, v
        history.append({"design": "SELECTED", "note": f"best validated pool coef={best_coef}",
                        "train": {}, "val": {"vs_average": best_val}, "decision": "final choice"})

    return {"history": history, "n_train": len(train), "n_val": len(val)}


def format_ledger(result: dict) -> str:
    lines = [
        "# Self-improving loop ledger",
        "",
        f"Train instances: {result['n_train']}, validation instances: {result['n_val']}.",
        "Negative vs_average means the design beats the plain average. gap_to_oracle "
        "closer to zero means closer to the fully informed upper bound.",
        "",
        f"| design | train vs_avg | val vs_avg | val gap_to_oracle | decision |",
        f"|---|---|---|---|---|",
    ]
    for h in result["history"]:
        tr = h["train"].get("vs_average")
        va = h["val"].get("vs_average")
        gap = h["val"].get("gap_to_oracle")
        lines.append(
            f"| {h['design']} | {tr:+.4f} | {va:+.4f} | {gap if gap is None else f'{gap:+.4f}'} | {h['decision']} |"
            if tr is not None else
            f"| {h['design']} |  |  {va:+.4f} |  | {h['decision']} |"
        )
    return "\n".join(lines)
