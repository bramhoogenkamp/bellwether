# Bellwether

*An agent-driven prediction market: AI agents trade an internal market whose price is a live probability.*


An exploration into a **company-run, internal prediction market populated and traded solely by AI agents** —
where you define an event, spin up agents, they take positions, and the market's trading price is the
live probability.

## The thesis

This exact product does not exist as of mid-2026. The pieces are split across vendors and nobody has fused
them into one internal, agent-driven, price-discovery market. That is the opportunity.

The four requirements, and who has which:

| Requirement | Who has it | What they're missing |
|---|---|---|
| Internal / self-hosted + create-your-own event + **real market price discovery** | Cultivate Labs | Built for human employees; agents only via REST API (DIY) |
| Internal + **ready-made AI bots** | Metaculus private instances + `forecasting-tools` | Aggregation/scoring, not a price-discovery market |
| **AI-agent-driven markets** | Olas / Rain | Public/crypto, not internal/enterprise |
| **Agent swarm for internal corporate decisions** | MiroFish | A simulation — no order book, no positions, no price = probability |

→ See [`research/landscape.md`](research/landscape.md) for the full map and sources.

## The core open problem

With AI agents as the *only* traders, the market price can collapse toward a single shared model output
("one model in a trenchcoat") instead of producing the diversity that makes markets smarter than any one
forecaster. **Engineering genuine agent diversity is the central technical challenge.**

→ See [`research/open-problems.md`](research/open-problems.md).

## What the research changed (2026-06-21)

Five parallel research threads (full synthesis in [`research/design-questions.md`](research/design-questions.md))
**reframed the project**:

- **The value is not the market mechanism** — it's (1) differentiated/internal information, (2) model
  diversity, (3) calibration, (4) accuracy-based reweighting. A naive average of ~12 LLMs already matches a
  925-human crowd, and a *tuned* aggregator beat a human market by ~13% (Atanasov 2017). The market must
  *earn* its complexity.
- **The moat is internal data, not the trading floor.** A public-info-only agent market collapses to "one
  model in a trenchcoat" (No-Trade Theorem + correlated LLM errors). The fix: wire agents to internal data
  (Slack/Jira/CRM via MCP), each reading a *different slice*.
- **The mechanism is settled: LMSR.** Binary market per question, liquidity set by parameter `b` (synthetic
  liquidity is literally `b`) — the textbook fix for thin markets. The swarm is settled too: 5–10 independent
  mixed-model agents → median → √3 calibration. No personas, no debate.
- **The make-or-break test**: does an AI-agent LMSR market beat both a naive mean *and* a tuned aggregator of
  the same agents? If only the former → product pivots to a **market + aggregator ensemble** (beats either
  alone). Run this comparison first.

**Decisions locked:** play money + bankroll (self-correcting reweight) · fractional-Kelly sizing ·
LMSR binary markets · 5–10 mixed-model agents · median + extremizing calibration · BSS + calibration curve as
headline metrics. See `research/design-questions.md` for the reasoning behind each.

## Layout

- `research/` — landscape, prior art, open problems, sources
- `experiments/` — prototypes (market engine, agent loop, diversity tests)
- `notes/` — running scratch notes, ideas, decisions

## Code: quickstart

```bash
# one-time setup (uv: https://docs.astral.sh/uv/)
uv venv .venv && uv pip install --python .venv/bin/python -e .

.venv/bin/python -m pytest                          # run the unit tests

# run the A-G benchmark offline (FakeLLM + mock questions — free, no API key):
.venv/bin/python scripts/run_benchmark.py --limit 120

# log runs to MLflow, then browse the runs-table to compare configs:
.venv/bin/python scripts/run_benchmark.py --mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db

# watch a market form on one question (offline mock):
.venv/bin/python scripts/demo_live.py

# real models via OpenRouter (needs OPENROUTER_API_KEY in .env):
.venv/bin/python scripts/run_benchmark.py --live
```

Design rationale for every choice: [`research/design-questions.md`](research/design-questions.md).

### Code layout
```
src/bellwether/
  questions/    where events come from: mock_internal (offline), manifold, forecastbench
  evidence/     how agents get information: mock_internal signal, web stub, MCP-ready seam
  agents/       LLM clients (FakeLLM + LiteLLM/OpenRouter), Agent, Swarm
  market/       LMSR market maker + fractional-Kelly trading
  aggregate/    naive (B), tuned (C), market (D), ensemble (E)
  conditions.py the A-G benchmark conditions
  scoring.py    Brier, BSS, log-loss, calibration/ECE, paired bootstrap
  calibrate.py  sqrt(3) extremizing transform (+ fit to own history)
  runner.py     questions -> evidence -> swarm -> conditions -> scores -> MLflow
  config.py     typed YAML config (logged to MLflow as run params)
configs/default.yaml  the knobs you sweep      scripts/  run_benchmark, demo_live, pull_questions
tests/                unit + offline end-to-end tests (run with pytest)
```

## Status

Building (2026-06-21). **Done:** Phases 0–5 — the full pipeline runs end-to-end **offline** (mock questions +
deterministic FakeLLM → swarm → LMSR market → conditions A–G → scoring → MLflow), 58 tests passing. On the
calibrated mock signal the swarm recovers the truth (≈ oracle), the ensemble (E) wins, and everything beats the
biased status-quo baseline — while the market (D) ≈ naive (B), i.e. no market edge yet on *already-calibrated*
inputs. **Next:** Phase 6 — the real benchmark on ForecastBench / Manifold with live models (needs
`OPENROUTER_API_KEY`), where under-confident real LLMs are where the market/ensemble should earn their keep.
The decisive test: does the market (D) beat the tuned aggregator (C)? See [`experiments/README.md`](experiments/README.md).
