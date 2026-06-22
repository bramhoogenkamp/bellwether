# Iteration 4: unstructured evidence

Follow-up to [`04-deliberation-depth.md`](04-deliberation-depth.md). Date 2026-06-22.

## Context

The earlier cells used labeled conditions (Condition A: COMPLETE), which a deterministic combiner could solve,
so the oracle was perfect and deliberation could nearly reach it. This iteration tests whether the result
survives when extracting the signal requires interpretation. Each condition's latent state is now rendered as
a messy natural-language snippet (for example "the checkout flow is still blocked on a dependency and got
pushed to next sprint") rather than a labeled field. The latent states still determine the outcome via the
rule, so there is ground truth, but no rule recovers the answer from the surface text.

## Setup

unstr-AND and unstr-OR, n=24, deliberation depths 1 to 3, same as iteration 3. Script: `scripts/run_depth.py`
with the unstructured cells. Reproduce:

```
python scripts/run_depth.py --cells unstr-AND,unstr-OR --n 24 --rounds 3 --live --mlflow
```

## Results (Brier, lower is better)

| | average | round 1 | round 2 | round 3 | oracle |
|---|---|---|---|---|---|
| unstr-AND | 0.171 | 0.138 | 0.124 | 0.119 | 0.021 |
| unstr-OR | 0.329 | 0.237 | 0.207 | 0.174 | 0.021 |

For comparison, the structured cells from iteration 3: comp-AND average 0.228 reaching 0.034 by round 3
(oracle 0.000), comp-OR average 0.386 reaching 0.068 (oracle 0.001). The market over deliberated beliefs again
tracked the deliberated average at every round.

Fraction of the gap from the average to the oracle that deliberation closed by round 3:

| | structured | unstructured |
|---|---|---|
| AND | about 85 percent | about 35 percent |
| OR | about 83 percent | about 50 percent |

## Findings

1. The oracle is no longer perfect (0.021). Once evidence is prose, even full information carries an
   irreducible interpretation cost.
2. Deliberation still helps, but recovers far less of the gap, roughly 35 to 50 percent on messy evidence
   versus 83 to 85 percent on clean conditions, and unstr-AND flattened by round 3.
3. So messy evidence shifts the bottleneck from information transmission to interpretation. Deliberation pools
   the easily extractable signal and then stalls; more rounds do not close the remaining gap.
4. The market continued to add nothing beyond the deliberated average at every depth.

The practical reading: deliberation among agents is genuinely useful for pooling messy dispersed evidence,
because it beats the average and a single read, and because no deterministic combiner exists in that setting.
But it is useful rather than complete, it leaves a real residual gap, and the limiting factor becomes how well
the models read and interpret the evidence, not the aggregation mechanism.

## Caveats

- n=24 per cell, two cells, one configuration.
- The prose is templated, so this is only semi-unstructured. Real evidence would be harder and the residual
  gap probably larger.
- unstr-OR was still descending at round 3, so its plateau, if any, is past three rounds.
