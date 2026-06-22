# Information Aggregation in Markets of LLM Agents

*Bellwether research introduction — v1 (2026-06-22)*

## The question

Prediction markets are supposed to do something that simple averaging cannot:
**aggregate information that is scattered across many participants, each of whom
knows only a piece.** This is the classic Hayekian argument for why markets produce
good probabilities. As large language models (LLMs) become competent forecasters,
a natural question follows:

> **If we give a swarm of LLM agents *different slices* of the evidence, can a market
> mechanism reconstruct the full-information forecast — and does it do so better than
> simply averaging the agents' individual forecasts?**

This is the question Bellwether is built to answer.

## Background: what is already known

**1. A market is, mathematically, a kind of weighted average.** Hanson's
Logarithmic Market Scoring Rule (LMSR) — the market maker we use — is provably a
*logarithmic opinion pool* of the traders' beliefs [Chakraborty & Das, NeurIPS 2015].
Under common assumptions, the price equals a wealth-weighted average belief
[Wolfers & Zitzewitz 2006]. So a market is not categorically different from
averaging; any advantage must come from *which* pool emerges (confidence-/skill-
weighted, extremized) and *when*.

**2. For humans, markets do not decisively beat good averaging.** Well-aggregated,
extremized, performance-weighted polls match or beat prediction markets
[Atanasov et al., *Management Science* 2017; Dana et al. 2019]. Markets aggregate
*dispersed private information* only **conditionally** — success depends on market
completeness, trader experience, and common knowledge of the payoff structure
[Plott & Sunder, *Econometrica* 1988; Forsythe & Lundholm 1990], and the effect is
fragile and institution-dependent [Corgnet et al. 2020].

**3. For LLMs specifically:**
- A simple **median/mean of several LLMs already rivals a human crowd** — averaging
  is a strong baseline [Schoenegger et al., *Science Advances* 2024].
- LLM **debate/discussion *fails* to pool distributed information**: groups given
  hidden-profile tasks reach ~30% accuracy vs ~81% for a single agent handed the
  pooled evidence [Li, Naito & Shirado, *HiddenBench*, 2025, arXiv:2505.11556].
- An LLM **LMSR market with dispersed private signals can approach the true value**
  [Galanis, "Information Aggregation with AI Agents," 2026, arXiv:2604.20050] — the
  closest predecessor to this work.
- LLM "diversity" is often an illusion: nominally different models make **highly
  correlated errors** (≈9 judges → ~2 effective votes) [Kohli, 2026, arXiv:2605.29800],
  and naively asking one LLM to synthesize others is *worse* than the plain mean
  [AIA Forecaster, 2025, arXiv:2511.07678].

## The gap

No study runs the comparison that actually decides whether a market *adds value*
over averaging for LLM agents holding dispersed information:

- **Market vs. average vs. pooled-oracle on the *same* private-information instances.**
  Galanis measures market-vs-*truth*; the silicon-crowd work measures *averaging*;
  HiddenBench measures *debate*. No one puts all three on the same instances.
- **Characterized by *signal structure*.** Theory says the answer depends on whether
  signals are *substitutable* (redundant) or *complementary* (each necessary). This
  axis has never been tested for LLM agents.
- **Market vs. debate.** Does the *market institution* (skin-in-the-game, price
  feedback) aggregate where mere discussion provably fails (HiddenBench)?

## What we will try (the novel contribution)

> **The first head-to-head, on identical dispersed-private-information instances, of
> a market (LMSR) vs. a naive average vs. a pooled "oracle" — identifying the
> signal-structure conditions under which a market of LLM agents beats averaging and
> approaches the oracle, and whether it succeeds where LLM debate fails.**

**Hypotheses** (theory-grounded, falsifiable):
- **H1 — Substitutable signals:** when each agent's slice is a redundant noisy
  estimate, averaging is near-optimal and **market ≈ average** (both approach the oracle).
- **H2 — Complementary signals:** when the outcome depends on *all* pieces (a
  hidden-profile / conjunction), averaging *cannot* recover the truth, but a market
  — where an agent holding a decisive piece can move the price — **beats the average
  and approaches the oracle.**
- **H3 — Mechanism matters:** the market aggregates complementary information that
  LLM *debate* does not (contra HiddenBench's failure result).

**Why this is the right design:**
- It is **decisive** — it isolates the market's marginal value over the strong
  averaging baseline, not against a strawman.
- It is **controlled and immediate** — instances are synthetic with *known ground
  truth*, so we score hundreds of them at once, with **no data leakage and no waiting
  for resolution** (the problem that plagues live forecasting benchmarks).
- It is **on-thesis** — "dispersed private information across agents" is exactly
  internal company data partitioned across agents, Bellwether's intended use.
- It is **edge-of-research** — information-aggregation theory applied to LLM agents,
  not another accuracy leaderboard.

## How it connects to public benchmarks

The synthetic experiment is the controlled core. To ground Bellwether against the
field, we also score it on **ForecastBench** [Karger et al., ICLR 2025,
arXiv:2409.19839] — a large, leak-free public benchmark with published
superforecaster and LLM baselines — and against live public markets (Polymarket,
Manifold). Those answer "is our forecaster any good in the wild"; the synthetic
experiment answers "does the *market mechanism* add value, and when."

## Honest caveats

- **Predecessor.** Galanis (2026) already builds an LLM+LMSR market with private
  signals; we differentiate by the market-vs-average-vs-oracle horse race, the
  signal-structure axis, and the debate ablation. We cite it prominently.
- **Effective diversity.** LLM agents are correlated; we measure decorrelated
  ("effective") diversity, not just model count [Kohli 2026].
- **Price ≠ belief, exactly.** An LMSR price confounds belief with the agents'
  prompted risk attitudes and the liquidity parameter [Manski 2006]; we check
  calibration and run no-information controls.
- **Strong LLMs may reason about what others know**, shrinking the market's edge —
  which is itself an interesting finding about where mechanism design still matters.

References: [`references.md`](references.md). Design details and metrics:
[`design-questions.md`](design-questions.md).
