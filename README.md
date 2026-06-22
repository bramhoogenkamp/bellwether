# Bellwether

A research harness and prototype for prediction markets in which the traders are LLM agents rather than people.

## Overview

Bellwether studies whether a market of LLM agents produces better probabilities than simply averaging the
agents' individual forecasts. The motivation is internal forecasting. Companies have many questions (will this
feature ship, will this deal close) that never get a real prediction market because there are too few traders.
A swarm of agents can populate such a market, but it is not obvious that the market mechanism adds anything over
averaging the agents directly. That is what we test.

The setup is an LMSR market maker traded by a small swarm of agents, with the price read as a probability. We
compare the market price against several baselines: a single agent, a plain average, a confidence-weighted
aggregator, a one-round debate, and a fully informed oracle.

## The question

When a group of LLM agents each hold only part of the relevant information, can a market that lets them trade
reconstruct the full picture better than averaging their individual forecasts, and how does the answer depend
on how the information is split?

This is the long-standing argument for why markets exist (Hayek, rational expectations): prices aggregate
dispersed private knowledge. The closest prior work (Galanis 2026) checks whether an LLM market reaches the
truth; it does not compare the market against averaging or against a fully informed oracle, and it does not
vary the information structure. Those are the gaps we address. Background and citations are in
[`research/intro.md`](research/intro.md) and [`research/references.md`](research/references.md).

One risk runs through all of this: LLM agents are often highly correlated, so a swarm can collapse toward a
single shared answer. We measure decorrelated diversity rather than assuming it.

## Experiments

Two tracks, run independently.

**1. Information-aggregation study (the controlled experiment).**
Synthetic questions with known ground truth, where each agent privately sees one slice of the evidence. We
score six forecasters (single, average, tuned, market, debate, oracle) on identical instances, immediately and
without leakage. The variable we sweep is the information structure:

- substitutable: each slice is a redundant noisy estimate, so averaging should be near-optimal.
- complementary, AND: resolves yes only if every condition holds, so one agent can hold a decisive no.
- complementary, OR: resolves yes if any condition holds, so one agent can hold a decisive yes.
- complementary, threshold: resolves yes on a majority, so no single agent is decisive (the hardest case).

We expect the market to be close to averaging for substitutable signals, and to beat averaging and approach the
oracle for complementary signals. The comparison with debate tells us whether the market mechanism, or just the
exchange of reasoning, is doing the aggregating.

```bash
python scripts/run_infoagg.py --live --n 24 --mlflow      # offline (FakeLLM) without --live
```

**2. Real-world benchmark (external validity).**
Score the same swarm on ForecastBench, a large leak-free public benchmark with superforecaster and market
baselines, and forward-test it against live markets (Polymarket now, Kalshi planned). These questions resolve
in the future, so forecasts are logged and scored on resolution, which keeps retrieval honest.

```bash
python scripts/score_forecastbench.py --market-baseline-only   # the crowd baseline to beat
python scripts/forecast_open.py                                # forward-test on open markets
```

## Code: quickstart

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest                              # unit + offline tests

.venv/bin/python scripts/run_infoagg.py --n 24          # the experiment, offline (free)
mlflow ui --backend-store-uri sqlite:///mlflow.db       # browse results
```

Real models run through OpenRouter and need `OPENROUTER_API_KEY` in a `.env` file (gitignored).

### Code layout
```
src/bellwether/
  questions/    sources: synthetic (info-agg), forecastbench, polymarket, manifold, mock_internal
  evidence/     how agents get information: web (:online) research, mock_internal, leakage guard
  agents/       LLM clients (LiteLLM/OpenRouter + offline FakeLLM), Agent, Swarm (private + debate)
  market/       LMSR market maker + fractional-Kelly trading
  aggregate/    naive, tuned, market, ensemble
  scoring.py    Brier, Brier skill score, log loss, calibration/ECE, Murphy decomposition, bootstrap
  conditions.py conditions for the real-world benchmark
  runner.py     question -> evidence -> swarm -> conditions -> score -> MLflow
configs/        infoagg.yaml (the study), live_good.yaml, default.yaml
scripts/        run_infoagg.py, score_forecastbench.py, forecast_open.py, run_benchmark.py, demo_live.py
tests/          unit + offline end-to-end (pytest)
```

Design rationale and the literature behind each choice: [`research/intro.md`](research/intro.md),
[`research/design-questions.md`](research/design-questions.md).

## Status

The apparatus is built and tested (68 tests). The information-aggregation study has run live across all five
cells (gpt-5-mini, claude-sonnet-4.5, gemini-2.5-pro, deepseek-r1), and runs as a checkpointed loop over market
designs (`experiments/loop/live-experiments-ledger.md`). Findings so far, across three iterations:

- A market over the agents' private probabilities never beats a plain average on complementary information,
  whether capped, uncapped, or the closed-form equilibrium pool. The oracle is near-perfect, so the information
  is recoverable, but the price channel does not recover it
  ([01](experiments/info-aggregation/01-information-aggregation.md),
  [02](experiments/info-aggregation/02-market-design-probe.md)).
- Give the agents a language channel (shared rationales) and both the deliberated average and a market over
  those beliefs beat the plain average significantly on every complementary structure, but they tie each other
  ([03](experiments/info-aggregation/03-rationale-market.md)).
- Deliberation depth is the lever. Each round roughly halves the Brier toward the oracle, and about three
  rounds nearly recover the information (comp-AND 0.228 to 0.034 across rounds), generalizing across AND and OR
  ([04](experiments/info-aggregation/04-deliberation-depth.md)).
- On unstructured evidence (prose slices that require interpretation), the oracle is no longer perfect (0.021)
  and deliberation closes only about 35 to 50 percent of the gap, versus 83 to 85 percent on clean conditions.
  The bottleneck shifts from transmission to interpretation
  ([05](experiments/info-aggregation/05-unstructured-evidence.md)).

So the channel and the depth of deliberation do the aggregating, not the market mechanism: a scalar price
cannot carry a decisive fact, a sentence can. On clean conditions deliberation nearly recovers the answer; on
messy evidence it helps but stalls, because reading the evidence becomes the limit. The real-world benchmark
(ForecastBench, live markets) is wired but separate from this study.
