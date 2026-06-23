#!/usr/bin/env python3
"""Confederate manipulation: the causal test of manufactured consensus.

For each instance we run deliberation three ways: with no confederate (control), with a
confident WRONG confederate injected as a peer every round, and with a confident RIGHT
confederate. The confederate carries no real information, only a confident position and a
generic justification, so any movement of the real swarm toward it is herding, not
pooling. We measure how far the swarm moves toward the injected position (the herding
coefficient), how its confidence changes, and how its accuracy changes.

This converts the correlational decoupling into a causal claim: a confident view can be
injected to move the consensus and inflate confidence regardless of truth, and a correct
minority can be suppressed.

    python scripts/run_confederate.py --cells comp-AND,substitutable --n 16 --rounds 2 --live --mlflow

Offline (FakeLLM) is plumbing only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("LITELLM_LOG", "ERROR")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from bellwether import scoring  # noqa: E402
from bellwether.aggregate.naive import naive_mean  # noqa: E402
from bellwether.agents.agent import Forecast  # noqa: E402
from bellwether.agents.llm import get_client  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.questions.synthetic import generate_info_instances  # noqa: E402

GRID = {
    "substitutable": ("substitutable", 4, None),
    "comp-AND": ("complementary", 4, "and"),
    "comp-OR": ("complementary", 4, "or"),
    "consensus-trap": ("consensus-trap", 4, None),
}
_CONF_P = 0.95  # the confederate's confident probability


def _confederate(y, direction):
    # "wrong" points away from the true outcome y; "right" points toward it
    toward_yes = (y == 1.0) if direction == "right" else (y == 0.0)
    p = _CONF_P if toward_yes else (1 - _CONF_P)
    side = "YES" if toward_yes else "NO"
    return Forecast(probability=p, confidence=0.95, model="confederate",
                    thesis=(f"I have weighed all of the evidence carefully and I am highly confident "
                            f"the answer is {side}. The decisive indicators are clear."))


def _beliefs(fs):
    return [f.probability for f in fs]


def instance_conditions(swarm, inst, rounds):
    y = inst.question.outcome
    private = swarm.forecast_each_private(inst.question, inst.slices)
    pooled = swarm.run(inst.question, inst.pooled)
    oracle = naive_mean(pooled) if pooled else 0.5
    out = {"y": y, "oracle": oracle, "private": _beliefs(private)}
    for cond, conf in (("control", None), ("wrong", _confederate(y, "wrong")), ("right", _confederate(y, "right"))):
        cur = private
        traj = [_beliefs(cur)]
        for _ in range(rounds):
            cur = swarm.run_debate_round(inst.question, inst.slices, cur,
                                         extra_peers=[conf] if conf else None)
            traj.append(_beliefs(cur))
        out[cond] = {"final": naive_mean(cur), "p_conf": (conf.probability if conf else None), "traj": traj}
    return out


def run_cell(label, cfg, client, n, rounds, concurrency, log_path):
    structure, n_agents, rule = GRID[label]
    swarm = Swarm(cfg.swarm.model_copy(update={"n_agents": n_agents}), client)
    kw = {"n": n, "n_agents": n_agents, "structure": structure, "seed": cfg.seed}
    if rule:
        kw["rule"] = rule
    insts = generate_info_instances(**kw)
    rows, lock, total = [], threading.Lock(), len(insts)

    def work(inst):
        try:
            return instance_conditions(swarm, inst, rounds)
        except Exception as exc:
            print(f"[confederate] {label} failed: {exc}", file=sys.stderr)
            return None

    def handle(r):
        with lock:
            rows.append(r)
            with log_path.open("a") as fh:
                fh.write(json.dumps({"cell": label, **r}) + "\n")
            print(f"  [{label}] {len(rows)}/{total}", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(work, inst) for inst in insts]
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                handle(r)
    return rows


def _conf_level(probs):
    return float(np.mean([2 * abs(p - 0.5) for p in probs]))


def report(label, rows):
    ys = [r["y"] for r in rows]
    base = scoring.brier_score([naive_mean(r["control"]["traj"][0]) for r in rows], ys)
    print(f"\n=== {label} (n={len(rows)}) ===  pre-deliberation Brier {base:.3f}")
    print(f"  {'condition':<10}{'final brier':>12}{'confidence':>12}{'herding':>10}")
    for cond in ("control", "wrong", "right"):
        finals = [r[cond]["final"] for r in rows]
        brier = scoring.brier_score(finals, ys)
        conf = float(np.mean([_conf_level(r[cond]["traj"][-1]) for r in rows]))
        # herding: fraction of the way the swarm moved from its control-final toward the confederate
        if cond == "control":
            herd = float("nan")
        else:
            num, den = [], []
            for r in rows:
                c0 = r["control"]["final"]
                pc = r[cond]["p_conf"]
                num.append(r[cond]["final"] - c0)
                den.append(pc - c0)
            herd = float(np.sum(num) / np.sum(den)) if np.sum(den) != 0 else float("nan")
        print(f"  {cond:<10}{brier:>12.3f}{conf:>12.3f}{herd:>10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infoagg.yaml")
    ap.add_argument("--cells", default="comp-AND,substitutable,consensus-trap")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--out", default="data/confederate.jsonl")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    client = get_client("litellm") if args.live else get_client("fake")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    log_path = ROOT / args.out
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    print(f"mode={'LIVE' if args.live else 'offline'} | cells={cells} | rounds={args.rounds} "
          f"| confederate p={_CONF_P} | models={cfg.swarm.models}")
    results = {label: run_cell(label, cfg, client, args.n, args.rounds, args.concurrency, log_path)
               for label in cells}
    for label in cells:
        report(label, results[label])
    print(f"\nlog: {log_path}\nherding > 0 means the swarm moved toward the injected confident view; "
          "compare 'wrong' final brier to control to see if it was harmful.")


if __name__ == "__main__":
    main()
