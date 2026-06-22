#!/usr/bin/env python3
"""Information-aggregation experiment — the novel study (see research/intro.md).

For synthetic dispersed-private-information instances (KNOWN ground truth, so we score
immediately with no leakage), each agent sees only its private slice. We compare six
forecasters and ask: does the MARKET aggregate the private information better than
averaging — and when?

Conditions per instance:
  single   one agent's private forecast            (floor)
  average  naive mean of private forecasts          (the baseline — Schoenegger)
  tuned    confidence-weighted + extremized mean     (the hard baseline — Atanasov)
  market   agents trade an LMSR market               (UNDER TEST — Galanis-adjacent)
  debate   one round of deliberation, then mean      (does a market beat talk? — HiddenBench)
  oracle   every agent sees ALL slices, then mean    (upper bound)

Grid (the phase diagram): complementary signals at increasing conjunction depth k, and
substitutable signals at two noise levels. Hypotheses: substitutable -> market ≈ average;
complementary -> market >> average, toward the oracle, and market beats debate.

    python scripts/run_infoagg.py --preflight              # 2 live instances, prints reasoning
    python scripts/run_infoagg.py --live --n 24 --mlflow   # the full run
    python scripts/run_infoagg.py --n 30                    # offline plumbing (FakeLLM)
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
from bellwether.aggregate.market import market_price  # noqa: E402
from bellwether.aggregate.naive import naive_mean  # noqa: E402
from bellwether.aggregate.tuned import tuned_aggregate  # noqa: E402
from bellwether.agents.llm import get_client  # noqa: E402
from bellwether.agents.swarm import Swarm  # noqa: E402
from bellwether.config import BenchmarkConfig  # noqa: E402
from bellwether.questions.synthetic import generate_info_instances  # noqa: E402

CONDITIONS = ["single", "average", "tuned", "market", "debate", "oracle"]

DEFAULT_GRID = [
    # Control: redundant signals — expect market ≈ average.
    {"structure": "substitutable", "n_agents": 4, "noise": 0.10, "label": "sub-lo"},
    {"structure": "substitutable", "n_agents": 4, "noise": 0.25, "label": "sub-hi"},
    # The categorization that can change the result — how the information is structured:
    {"structure": "complementary", "rule": "and", "n_agents": 4, "label": "comp-AND"},
    {"structure": "complementary", "rule": "or", "n_agents": 4, "label": "comp-OR"},
    {"structure": "complementary", "rule": "threshold", "n_agents": 5, "label": "comp-THRESH"},
]


def _swarm_for(cfg, client, n_agents):
    return Swarm(cfg.swarm.model_copy(update={"n_agents": n_agents}), client)


def _conditions_for_instance(swarm, cfg, inst, seed):
    private = swarm.forecast_each_private(inst.question, inst.slices)
    pooled = swarm.run(inst.question, inst.pooled)            # everyone informed = oracle
    debate = swarm.run_debate_round(inst.question, inst.slices, private)
    probs = {
        "single": private[0].probability,
        "average": naive_mean(private),
        "tuned": tuned_aggregate(private, cfg.calibration.extremize_coef),
        "market": market_price(private, cfg.market, seed=seed),
        "debate": naive_mean(debate),
        "oracle": naive_mean(pooled) if pooled else 0.5,
    }
    return private, probs


def _mean_pairwise_error_corr(priv_matrix, ys):
    """Effective-diversity diagnostic: mean pairwise correlation of agents' errors.
    Lower = more decorrelated (genuinely diverse). High = clones (market is starved)."""
    err = np.asarray(priv_matrix) - np.asarray(ys)[:, None]
    n_agents = err.shape[1]
    if err.shape[0] < 3 or n_agents < 2:
        return float("nan")
    corrs = []
    for a in range(n_agents):
        for b in range(a + 1, n_agents):
            if err[:, a].std() > 1e-9 and err[:, b].std() > 1e-9:
                corrs.append(np.corrcoef(err[:, a], err[:, b])[0, 1])
    return float(np.mean(corrs)) if corrs else float("nan")


def _instance_result(swarm, cfg, inst, seed):
    """Compute all conditions for one instance. Self-contained + thread-safe (the
    swarm/agents/client are stateless per call), so instances run concurrently."""
    try:
        private, probs = _conditions_for_instance(swarm, cfg, inst, seed)
    except Exception as exc:  # one bad instance shouldn't kill the cell
        print(f"[infoagg] instance failed: {exc}", file=sys.stderr)
        private, probs = [], {c: 0.5 for c in CONDITIONS}
    return inst, private, probs


def run_cell(cell, cfg, client, n, log_path, concurrency=8):
    swarm = _swarm_for(cfg, client, cell["n_agents"])
    instances = generate_info_instances(
        n=n, n_agents=cell["n_agents"], structure=cell["structure"],
        seed=cfg.seed, noise=cell.get("noise", 0.08), rule=cell.get("rule", "and"),
    )

    # Run instances concurrently — each is ~3*n_agents sequential model calls, so this
    # is where the wall-clock savings live. We log + print each instance as it finishes
    # (under a lock) so progress is visible live.
    rows, priv_matrix, disagreements = [], [], []
    lock = threading.Lock()
    total = len(instances)

    def handle(i, inst, private, probs):
        ps = [f.probability for f in private]
        with lock:
            rows.append({"y": inst.question.outcome, **probs})
            priv_matrix.append(ps if ps else [0.5] * cell["n_agents"])
            disagreements.append(float(np.std(ps)) if ps else 0.0)
            with log_path.open("a") as fh:
                fh.write(json.dumps({
                    "cell": cell["label"], "structure": cell["structure"],
                    "n_agents": cell["n_agents"], "instance": i,
                    "outcome": inst.question.outcome, "conditions": probs,
                    "disagreement": disagreements[-1],
                    "agents": [{"model": f.model, "p": f.probability, "thesis": f.thesis[:160]}
                               for f in private],
                }) + "\n")
            print(f"  [{cell['label']}] {len(rows)}/{total} done", flush=True)

    if concurrency > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_instance_result, swarm, cfg, inst, cfg.seed + i): i
                    for i, inst in enumerate(instances)}
            for fut in as_completed(futs):
                inst, private, probs = fut.result()
                handle(futs[fut], inst, private, probs)
    else:
        for i, inst in enumerate(instances):
            inst, private, probs = _instance_result(swarm, cfg, inst, cfg.seed + i)
            handle(i, inst, private, probs)

    ys = [r["y"] for r in rows]
    scored = {c: scoring.score_all([r[c] for r in rows], ys) for c in CONDITIONS}
    deltas = {
        other: scoring.paired_bootstrap_brier_delta(
            [r["market"] for r in rows], [r[other] for r in rows], ys, seed=cfg.seed)
        for other in ("average", "tuned", "debate")
    }
    return {
        "cell": cell, "scored": scored, "deltas": deltas, "n": len(rows),
        "base_rate": scored["average"]["base_rate"],
        "mean_disagreement": float(np.mean(disagreements)),
        "error_corr": _mean_pairwise_error_corr(priv_matrix, ys),
    }


def print_cell(res):
    c = res["cell"]
    print(f"\n=== {c['label']}  ({c['structure']}, n_agents={c['n_agents']}, "
          f"n={res['n']}, base rate {res['base_rate']:.2f}) ===")
    print(f"   diversity: mean disagreement {res['mean_disagreement']:.3f}, "
          f"agent-error corr {res['error_corr']:.2f} (lower = more effective diversity)")
    print(f"   {'cond':<9}{'brier':>9}{'bss':>8}{'resol':>9}{'ece':>7}")
    for cond in sorted(CONDITIONS, key=lambda k: res["scored"][k]["brier"]):
        s = res["scored"][cond]
        print(f"   {cond:<9}{s['brier']:>9.4f}{s['bss']:>8.3f}{s['resolution']:>9.4f}{s['ece']:>7.3f}")
    for other, d in res["deltas"].items():
        sig = "*" if (d["ci_high"] < 0 or d["ci_low"] > 0) else " "
        print(f"   market vs {other:<8} ΔBrier={d['mean_delta']:+.4f} "
              f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]{sig}")


def print_summary(results):
    print("\n" + "=" * 78)
    print("SUMMARY (negative ΔBrier => market better; gap-to-oracle: how far above the upper bound)")
    print(f"{'cell':<10}{'mkt':>8}{'avg':>8}{'tuned':>8}{'oracle':>8}{'mkt-avg':>10}{'mkt-tuned':>11}{'gap2oracle':>12}")
    for r in results:
        s = r["scored"]
        print(f"{r['cell']['label']:<10}"
              f"{s['market']['brier']:>8.3f}{s['average']['brier']:>8.3f}"
              f"{s['tuned']['brier']:>8.3f}{s['oracle']['brier']:>8.3f}"
              f"{r['deltas']['average']['mean_delta']:>10.4f}"
              f"{r['deltas']['tuned']['mean_delta']:>11.4f}"
              f"{s['market']['brier'] - s['oracle']['brier']:>12.4f}")


def log_mlflow(results, cfg):
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(cfg.experiment_name)
    for r in results:
        with mlflow.start_run(run_name=f"{cfg.run_name}-{r['cell']['label']}"):
            mlflow.log_params({"structure": r["cell"]["structure"],
                               "n_agents": r["cell"]["n_agents"], "n": r["n"],
                               "models": ",".join(cfg.swarm.models)})
            for cond, s in r["scored"].items():
                for m in ("brier", "bss", "resolution", "ece"):
                    mlflow.log_metric(f"{cond}.{m}", float(s[m]))
            for other, d in r["deltas"].items():
                mlflow.log_metric(f"market_vs_{other}.delta", float(d["mean_delta"]))
            mlflow.log_metric("mean_disagreement", r["mean_disagreement"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infoagg.yaml")
    ap.add_argument("--n", type=int, default=24, help="instances per grid cell")
    ap.add_argument("--live", action="store_true", help="real models (default FakeLLM offline)")
    ap.add_argument("--preflight", action="store_true", help="2 live instances, verbose; verify + gauge cost")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8, help="instances run in parallel")
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    live = args.live or args.preflight
    client = get_client("litellm") if live else get_client("fake")
    log_path = ROOT / "data" / f"infoagg_{cfg.run_name}.jsonl"
    log_path.parent.mkdir(exist_ok=True)

    print(f"mode={'LIVE' if live else 'offline (FakeLLM)'} | models={cfg.swarm.models}")

    if args.preflight:
        cell = {"structure": "complementary", "n_agents": 3, "label": "preflight"}
        swarm = _swarm_for(cfg, client, 3)
        inst = generate_info_instances(n=2, n_agents=3, structure="complementary", seed=cfg.seed)
        for k, x in enumerate(inst):
            print(f"\n--- preflight instance {k} (truth={x.question.outcome:.0f}) ---")
            print(f"Q: {x.question.text}")
            private, probs = _conditions_for_instance(swarm, cfg, x, cfg.seed + k)
            for f in private:
                print(f"   private {f.model.split('/')[-1]}: p={f.probability:.2f} — {f.thesis[:90]}")
            print(f"   => average={probs['average']:.2f} market={probs['market']:.2f} "
                  f"debate={probs['debate']:.2f} oracle={probs['oracle']:.2f}  (truth {x.question.outcome:.0f})")
        print("\nPreflight done. If the oracle matches the truth and a slice-holding agent of a "
              "failed condition is confidently low, the apparatus is working. Then run the full grid.")
        return

    total_calls = sum(args.n * 3 * c["n_agents"] for c in DEFAULT_GRID)
    print(f"Grid: {len(DEFAULT_GRID)} cells x {args.n} instances. "
          f"~{total_calls} model calls (private+debate+oracle), concurrency={args.concurrency}.")
    if log_path.exists():
        log_path.unlink()

    results = [run_cell(c, cfg, client, args.n, log_path, args.concurrency) for c in DEFAULT_GRID]
    for r in results:
        print_cell(r)
    print_summary(results)
    if args.mlflow:
        log_mlflow(results, cfg)
    print(f"\nPer-instance log: {log_path}")


if __name__ == "__main__":
    main()
