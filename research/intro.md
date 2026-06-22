# Information Aggregation in Markets of LLM Agents

Bellwether research note, v2 (2026-06-22)

## The question

Prediction markets are valued for something averaging cannot do. They aggregate information that is scattered
across many participants, each of whom knows only a piece. This is the Hayekian argument for why markets
produce good probabilities. As language models become competent forecasters, a question follows: if we give a
swarm of LLM agents different slices of the evidence, can a market reconstruct the full-information forecast,
and does it do so better than averaging the agents' individual forecasts?

## Background

A market is, mathematically, a kind of weighted average. Hanson's Logarithmic Market Scoring Rule, the market
maker we use, is a logarithmic opinion pool of the traders' beliefs (Chakraborty and Das, NeurIPS 2015), and
under common assumptions the price equals a wealth-weighted average belief (Wolfers and Zitzewitz 2006). A
market is therefore not categorically different from averaging. Any advantage has to come from which pool
emerges, and when.

For humans, markets do not decisively beat good averaging. Well-aggregated, extremized, performance-weighted
polls match or beat prediction markets (Atanasov et al., Management Science 2017; Dana et al. 2019). Markets
aggregate dispersed private information only under conditions such as market completeness, trader experience,
and common knowledge of the payoff structure (Plott and Sunder, Econometrica 1988; Forsythe and Lundholm
1990), and the effect is fragile (Corgnet et al. 2020).

For LLMs:

- A simple median or mean of several models already rivals a human crowd, so averaging is a strong baseline
  (Schoenegger et al., Science Advances 2024).
- LLM debate fails to pool distributed information. On hidden-profile tasks, groups reach about 30 percent
  accuracy against about 81 percent for a single agent given the pooled evidence (Li, Naito and Shirado,
  HiddenBench, arXiv:2505.11556, 2025).
- An LLM LMSR market with dispersed private signals can approach the true value (Galanis, arXiv:2604.20050,
  2026). This is the closest prior work.
- LLM diversity is often an illusion. Nominally different models make highly correlated errors, so nine judges
  can carry about two independent votes (Kohli, arXiv:2605.29800, 2026), and asking one model to synthesize
  others is worse than the plain mean (AIA Forecaster, arXiv:2511.07678, 2025).

## The gap

No study runs the comparison that decides whether a market adds value over averaging for LLM agents holding
dispersed information. Galanis measures market against truth, the silicon-crowd work measures averaging, and
HiddenBench measures debate, but none puts market, average, and a fully informed oracle on the same instances,
and none varies the information structure.

## What this study does

We compare six forecasters on identical dispersed-private-information instances: a single agent, a naive
average, a confidence-weighted and extremized aggregator, an LMSR market, a one-round debate, and a fully
informed oracle. Each agent privately sees one slice of the evidence. Because the instances are synthetic with
known ground truth, every forecaster is scored immediately, with no leakage and no waiting for resolution.

The variable we sweep is the information structure:

- substitutable: each slice is a redundant noisy estimate of the same quantity. Averaging is near-optimal, so
  we expect the market to be close to the average, both approaching the oracle.
- complementary, AND: the outcome holds only if every condition holds. A single failed condition is decisive,
  and the agent who sees it knows the answer.
- complementary, OR: the outcome holds if any condition holds. A single satisfied condition is decisive.
- complementary, threshold: the outcome holds on a majority. No single agent is decisive, which is the hardest
  case for any mechanism.

Hypotheses:

- For substitutable signals, the market is close to the average.
- For complementary signals, the market beats the average and approaches the oracle, because an agent holding a
  decisive piece can move the price where averaging cannot.
- The comparison with debate separates two explanations: whether the market institution (skin in the game,
  price feedback) does the aggregating, or whether exchanging reasoning is enough on its own.

The instances are abstract by design, which is what isolates the mechanism. To show the forecaster is useful on
real events, we also score the same swarm on ForecastBench and forward-test it against live markets.

## Why this design

It is decisive, because it isolates the market's value over a strong averaging baseline rather than a strawman.
It is controlled and immediate, because the synthetic instances have known ground truth, so we can score many
at once with no leakage and no wait. It maps onto the product motivation, because dispersed private information
across agents is what internal company data partitioned across agents looks like.

## Caveats

- Galanis (2026) already builds an LLM LMSR market with private signals. We differ by comparing against
  averaging and the oracle, by varying the information structure, and by adding the debate condition, and we
  cite it directly.
- LLM agents are correlated, so we measure decorrelated diversity rather than assuming it (Kohli 2026).
- A market price confounds belief with the agents' prompted risk attitudes and the liquidity parameter (Manski
  2006). We check calibration and run no-information controls.
- Strong models may reason about what others know, which can shrink the market's edge. That is itself a useful
  finding about where mechanism design still matters.

References: [`references.md`](references.md). Design details and metrics: [`design-questions.md`](design-questions.md).
