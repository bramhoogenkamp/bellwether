# Information aggregation: market vs average vs deliberation vs oracle

First run of the information-aggregation study (see [`../../research/intro.md`](../../research/intro.md)).
Date 2026-06-22.

## What was run

Synthetic dispersed-private-information instances with known ground truth. Each agent privately sees one slice
of the evidence. On identical instances we score six forecasters: a single agent, a naive average of the
agents, a confidence-weighted and extremized aggregator (tuned), an LMSR market the agents trade, a one-round
debate, and a fully informed oracle that sees all slices.

Grid: five cells, 24 instances each.

- sub-lo, sub-hi: substitutable signals, where each slice is a redundant noisy estimate of the same quantity,
  at low and high noise. Controls.
- comp-AND: resolves yes only if all conditions hold, so a single failed condition is decisive.
- comp-OR: resolves yes if any condition holds, so a single satisfied condition is decisive.
- comp-THRESH: resolves yes on a majority, so no single agent is decisive.

Agents: four cross-family reasoning models (gpt-5-mini, claude-sonnet-4.5, gemini-2.5-pro, deepseek-r1), one
per slice. Config: `configs/infoagg.yaml`. Reproduce: `python scripts/run_infoagg.py --live --n 24 --mlflow`.

## Phase diagram (Brier, lower is better)

| cell | base rate | average | market | debate | oracle | market - average | debate - average | agent error corr |
|---|---|---|---|---|---|---|---|---|
| sub-lo (control) | 0.46 | 0.199 | 0.204 | 0.201 | 0.207 | +0.005 | +0.003 | 0.97 |
| sub-hi (control) | 0.46 | 0.221 | 0.230 | 0.229 | 0.241 | +0.009 | +0.009 | 0.88 |
| comp-AND | 0.29 | 0.228 | 0.228 | 0.180 | 0.000 | -0.000 | -0.047 | 0.98 |
| comp-OR | 0.46 | 0.386 | 0.385 | 0.282 | 0.001 | -0.000 | -0.103 | 0.95 |
| comp-THRESH | 0.38 | 0.189 | 0.202 | 0.178 | 0.000 | +0.013 | -0.012 | 0.82 |

Paired-bootstrap 95 percent intervals on the complementary cells:

- comp-AND: market vs average -0.000 [-0.001, +0.000]; market vs debate +0.047 [+0.014, +0.087].
- comp-OR: market vs average -0.000 [-0.007, +0.007]; market vs debate +0.103 [+0.043, +0.172].

## Findings

1. The market never beats the average. Market minus average is zero or slightly positive in every cell. On
   redundant signals it is marginally worse, on complementary signals it ties. The hypothesis that a market
   would beat averaging on complementary information is not supported.
2. The information is fully recoverable. The oracle is essentially perfect on every complementary cell (Brier
   near zero), so the answer is fully determined once the slices are pooled. Neither the average nor the
   market pools them, and both sit near chance.
3. Deliberation is what aggregates complementary information. Debate beats the average on all three
   complementary cells, significantly on AND and OR, and does nothing on the redundant controls, which is
   where pooling should not matter.
4. The swarm has very low effective diversity. Agent error correlations are 0.82 to 0.98, so the agents are
   near-clones in their numeric outputs. Their differing private information does not appear in their
   probabilities, only in their language.

## Interpretation

The result is about the bandwidth of the aggregation channel. A price and a probability are single scalars.
The agent who privately knows that one condition failed cannot put that fact into a price, and it does not
surface in a probability either, which is why both the market and the average stay near chance on
complementary signals while the oracle is perfect. A sentence in a debate carries the fact directly, so
deliberation recovers part of it. For LLM agents the high-bandwidth channel is language, not price.

This engages the two closest prior results. Galanis (2026) reports that an LLM market can reach the truth with
dispersed private signals; here the market does not beat plain averaging and falls far short of the oracle on
complementary structure. HiddenBench (2025) reports that LLM debate fails to pool hidden-profile information;
here, with strong reasoning models, debate does partially pool and beats the average, though it still trails
the oracle.

## Caveats

- 24 instances per cell. These are clear directional results, not tight estimates.
- One market design. The market is seeded at the consensus and uses capped fractional-Kelly sizing. A more
  aggressive design that let a decisive agent move the price further might change the market result. Testing
  that is the natural follow-up.
- The high agent error correlation is part of the finding, not a nuisance. The agents supply little
  decorrelated signal through their probabilities.

## Next

- Rerun comp-AND and comp-OR with an uncapped or all-in trading policy, to separate the mechanism from the
  parameters.
- Scale instances per cell for tighter intervals before any external sharing.
