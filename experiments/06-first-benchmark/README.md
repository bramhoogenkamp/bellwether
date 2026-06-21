# Experiment 06 — first real benchmark (live models)

**Date:** 2026-06-21 · **Config:** `configs/live.yaml` · logged to MLflow (`experiment=agent-market`,
`run_name=manifold-live`).

## Setup
- **Questions:** 25 resolved binary **Manifold** markets, quality-filtered (volume ≥ 250, lifetime ≥ 3 days).
  All are **June 2026 events** (silver price, US–Iran peace deal, World Cup line-ups, Strait of Hormuz, …),
  which **postdate the models' training cutoffs** — so this is a genuinely leak-free forecast.
- **Swarm:** 6 agents across three cheap families — `openai/gpt-4o-mini`, `google/gemini-2.5-flash-lite`,
  `meta-llama/llama-3.1-8b-instruct` — two lenses (base-rate, inside-view).
- **Evidence:** **none** (the web retrieval source is still a stub) → agents forecast **zero-shot** from the
  question text alone.

## Result

| cond | name | Brier | BSS | ECE |
|------|------|------:|----:|----:|
| D | market (LMSR) | 0.2495 | 0.000 | 0.051 |
| E | market+tuned | 0.2495 | 0.000 | 0.008 |
| C | tuned aggregator | 0.2496 | -0.000 | 0.004 |
| B | naive mean | 0.2497 | -0.000 | 0.011 |
| A | single LLM | 0.2506 | -0.004 | 0.032 |

Headline (paired bootstrap): D vs B ΔBrier = -0.0002 (n.s.); D vs C ΔBrier = -0.0001 (n.s.).

## Interpretation (the honest read)

**Brier ≈ 0.25, BSS ≈ 0 means no skill** — the swarm is forecasting at roughly the base rate. With cheap
models, **no retrieval**, and genuinely unknowable future events, the agents hedge toward ~0.5, so there is
**no differentiated signal for any aggregator to exploit**. That is exactly why every condition ties: you
cannot aggregate information that isn't there.

This is a *good* outcome for two reasons:
1. **The pipeline is faithful.** It did not manufacture fake skill — when the agents know nothing, the market,
   the naive mean, and the tuned aggregator all correctly collapse to "no better than chance."
2. **It confirms the project's central thesis empirically** ([`../../research/design-questions.md`](../../research/design-questions.md), Q2):
   *differentiated information is the binding constraint*, not the aggregation mechanism. The market can only
   beat averaging when there is real, miscalibrated signal to weight by — and here there is none.

Contrast with the **offline mock** (Experiment, calibrated signal present): there the swarm recovers the truth
(≈ oracle), the **ensemble (E) wins**, and everything beats the biased status-quo baseline. So the machinery
works when signal exists; the live run shows that *creating* the signal is the real job.

## Conclusion → next step

The market-vs-aggregation question **cannot be answered until the agents have real signal.** The highest-leverage
next move (per the research's #1 lever) is **retrieval / information**:
- implement the **web evidence source** (Tavily/Brave/SerpAPI) with a strict `as_of` date filter, and/or
- wire **internal data** (the MCP seam) for internal questions.

Then re-run this exact benchmark. Only with information in the agents' hands does testing whether the *market*
beats the *tuned aggregator* become meaningful. A secondary lever is stronger models, but without retrieval even
strong models score near chance on post-cutoff events.

## Engineering fixes made this phase
- Corrected an invalid OpenRouter model slug (`google/gemini-flash-1.5` → `google/gemini-2.5-flash-lite`);
  pull valid IDs from `https://openrouter.ai/api/v1/models`.
- Made the **swarm resilient**: a single failing model is logged and skipped instead of crashing the run.
- Added a **quality filter** to the Manifold source (min volume + min lifetime) to exclude spam/test markets.
