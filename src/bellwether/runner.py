"""The benchmark runner: questions -> evidence -> swarm -> conditions -> scores.

This is the loop that answers the project's whole question. For each resolved
question it gathers (leakage-guarded) evidence, runs the swarm once, computes every
enabled condition's probability from that single set of forecasts (so any
difference between conditions is the *aggregation*, not different agents), scores
each condition, and computes the headline paired comparisons (D vs B, D vs C).

Everything is logged to MLflow when enabled, so each config becomes one comparable
row in the runs-table. Runs offline with the default FakeLLM — no API cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import scoring
from .agents.llm import FakeLLM, LLMClient
from .agents.swarm import Swarm
from .conditions import compute_conditions
from .config import BenchmarkConfig
from .evidence.base import apply_leakage_guard, get_evidence_source
from .questions.base import get_question_source


@dataclass
class BenchmarkResult:
    scores: dict[str, dict] = field(default_factory=dict)   # condition -> metrics
    deltas: dict[str, dict] = field(default_factory=dict)   # "D_vs_B" -> bootstrap CI
    per_question: list[dict] = field(default_factory=list)
    n_scored: int = 0


def _make_question_source(cfg: BenchmarkConfig):
    qc = cfg.questions
    if qc.source == "mock_internal":
        return get_question_source("mock_internal", seed=cfg.seed)
    if qc.source == "manifold":
        return get_question_source("manifold")
    if qc.source == "forecastbench":
        return get_question_source(
            "forecastbench",
            question_set=qc.name,
            resolution_set=qc.resolution_set,
            data_dir=qc.data_dir,
        )
    raise ValueError(f"unknown question source: {qc.source!r}")


def _make_evidence_source(cfg: BenchmarkConfig):
    if cfg.evidence.source == "mock_internal":
        return get_evidence_source("mock_internal", seed=cfg.seed)
    return get_evidence_source(cfg.evidence.source)


def run_benchmark(
    config: BenchmarkConfig,
    client: LLMClient | None = None,
    limit: int | None = None,
    mlflow_enabled: bool = False,
    log=print,
) -> BenchmarkResult:
    client = client or FakeLLM()  # offline default
    qsource = _make_question_source(config)
    esource = _make_evidence_source(config)
    swarm = Swarm(config.swarm, client)
    limit = limit or config.questions.limit

    questions = [q for q in qsource.fetch(limit=limit) if q.is_resolved]
    if not questions:
        raise RuntimeError("no resolved questions to score")

    rows: list[dict] = []
    for i, q in enumerate(questions):
        as_of = q.issue_date if config.questions.leakage_guard else None
        evidence = apply_leakage_guard(
            esource.gather(q, config.evidence.max_items, as_of=as_of), as_of
        )
        forecasts = swarm.run(q, evidence)
        conds = compute_conditions(forecasts, q, config, seed=config.seed + i)
        rows.append({"id": q.id, "outcome": float(q.outcome), **conds})
        log(
            f"[{i + 1}/{len(questions)}] {q.id} y={q.outcome:.0f} "
            + " ".join(f"{k}={v:.2f}" for k, v in conds.items())
        )

    result = _score(rows, config)
    if mlflow_enabled:
        _log_to_mlflow(config, result)
    return result


def _score(rows: list[dict], config: BenchmarkConfig) -> BenchmarkResult:
    result = BenchmarkResult(per_question=rows, n_scored=len(rows))

    for cond in config.conditions:
        probs = [r[cond] for r in rows if cond in r]
        ys = [r["outcome"] for r in rows if cond in r]
        if probs:
            result.scores[cond] = scoring.score_all(probs, ys)

    # Headline comparisons: is the market (D) better than naive (B) and tuned (C)?
    for other, name in (("B", "D_vs_B"), ("C", "D_vs_C")):
        if "D" in result.scores and other in result.scores:
            a = [r["D"] for r in rows if "D" in r and other in r]
            b = [r[other] for r in rows if "D" in r and other in r]
            ys = [r["outcome"] for r in rows if "D" in r and other in r]
            result.deltas[name] = scoring.paired_bootstrap_brier_delta(
                a, b, ys, seed=config.seed
            )
    return result


def _log_to_mlflow(config: BenchmarkConfig, result: BenchmarkResult) -> None:
    try:
        import os

        import mlflow
    except ImportError:
        return

    # Local-first SQLite backend (the file store is deprecated). Override with the
    # MLFLOW_TRACKING_URI env var. View with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params({k: str(v) for k, v in config.flat_params().items()})
        for cond, metrics in result.scores.items():
            for metric, val in metrics.items():
                mlflow.log_metric(f"{cond}.{metric}", float(val))
        for name, delta in result.deltas.items():
            for k, v in delta.items():
                mlflow.log_metric(f"{name}.{k}", float(v))
        _log_artifacts(result)


def _log_artifacts(result: BenchmarkResult) -> None:
    """Best-effort per-question CSV + reliability plot as MLflow artifacts."""
    import tempfile
    from pathlib import Path

    try:
        import mlflow
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as d:
        # per-question CSV
        try:
            import pandas as pd

            csv = Path(d) / "per_question.csv"
            pd.DataFrame(result.per_question).to_csv(csv, index=False)
            mlflow.log_artifact(str(csv))
        except ImportError:
            pass

        # reliability diagram for each condition
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from .scoring import calibration_curve

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
            ys = [r["outcome"] for r in result.per_question]
            for cond in result.scores:
                probs = [r[cond] for r in result.per_question if cond in r]
                yy = [r["outcome"] for r in result.per_question if cond in r]
                mp, of, _ = calibration_curve(probs, yy, n_bins=8)
                if len(mp):
                    ax.plot(mp, of, marker="o", label=cond)
            ax.set_xlabel("predicted probability")
            ax.set_ylabel("observed frequency")
            ax.set_title("Reliability")
            ax.legend()
            png = Path(d) / "reliability.png"
            fig.savefig(png, dpi=110, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(png))
        except ImportError:
            pass
