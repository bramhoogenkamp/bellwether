# Experiment 07 — Bellwether vs a real-money public market (Polymarket)

**Date:** 2026-06-21 · **Config:** `configs/polymarket.yaml` · MLflow `run_name=polymarket-vs-bellwether`.

## Setup
- **Questions:** 15 resolved binary **Polymarket** markets (real money), restricted to those that resolved
  **after 2025-07-01** (post the cheap models' cutoffs → leak-free), volume ≥ 5000.
- **Public-market baseline (condition M):** the market's **mid-life YES price**, reconstructed from the CLOB
  price history — i.e. the crowd's probability while the outcome was still genuinely uncertain (not the
  settled 0/1).
- **Swarm:** same cheap cross-family 6-agent swarm, **no retrieval** (zero-shot from the question text).

## Result

| cond | name | Brier | BSS |
|------|------|------:|----:|
| **M** | **public market (Polymarket)** | **0.229** | **+0.046** |
| A | single LLM | 0.242 | -0.010 |
| C | tuned aggregator | 0.247 | -0.030 |
| E | market+tuned | 0.247 | -0.031 |
| D | market (LMSR) | 0.248 | -0.032 |
| B | naive mean | 0.248 | -0.034 |

Headline (ΔBrier = ours − market; negative ⇒ we beat the market):
- D vs M: **+0.019** [-0.15, +0.17] — not significant
- E vs M: +0.018 [-0.15, +0.17] — not significant
- B vs M: +0.019 [-0.15, +0.17] — not significant

## Interpretation
- **The real-money market wins** — it's the only forecaster with positive skill (BSS +0.046); our no-retrieval
  swarm is at chance (negative BSS). Expected: the crowd has information our agents simply don't.
- **But not significantly, at n=15.** The CIs are huge because the mid-life prices on these obscure crypto
  "FDV at launch" markets are themselves noisy — in one case the market priced an event at ~4% that then
  happened. Real-money ≠ always sharp on thin/niche markets. A clean read needs many more questions.
- This **quantifies the bar**: retrieval has to move our swarm from ~chance (Brier ~0.245) to below the
  market's ~0.229 to claim Bellwether beats a real-money crowd. That's the target for the next phase.

## What this experiment delivers
The **"vs public market" machinery is now real and source-agnostic**: condition `M` scores any market's own
probability next to Bellwether, with paired-bootstrap deltas (D/E/B vs M). It works for Polymarket today and
for Manifold's `market_prob`; ForecastBench's superforecaster baseline plugs into condition `F` the same way.

## Caveats / honest limits
- **No retrieval** ⇒ our agents are near-chance; this is not yet a fair fight, it's a bar-setting run.
- **Point-in-time**: our agents forecast "now" (timeless), not strictly "as of" the mid-life date. For
  post-cutoff markets they're uninformed either way, so it's roughly fair, but a rigorous backtest needs
  as-of-dated retrieval aligned to the price snapshot.
- **n=15** ⇒ wide CIs; scale up once retrieval gives us signal worth testing.
