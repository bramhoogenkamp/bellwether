# Iteration 2: rationale-augmented market

Follow-up to [`01-information-aggregation.md`](01-information-aggregation.md) and
[`02-market-design-probe.md`](02-market-design-probe.md). Date 2026-06-22.

## Context

Iteration 0 (baseline market) found the market ties the plain average on the complementary cells and sits far
from the oracle, while deliberation pools the information. Iteration 1 (uncapped market) showed the failure was
not the cautious trading parameters. The replay search found the same for the closed-form equilibrium pool. So
the open question was whether the limit is the market mechanism or the channel a price provides.

This iteration gives the market a language channel. Agents share a one-line rationale in a single deliberation
round, which lets a decisive private fact enter the others' beliefs, and then the market trades over those
updated beliefs.

## Setup

Same as experiment 1 (capped market, four reasoning models, complementary cells, n=24), with one added
condition, market_debate, the market run over the post-deliberation beliefs. Because it uses the same market
settings as the market-over-private condition, the only difference between them is whether agents shared
rationales first. Config: `configs/infoagg_rationale.yaml`. Reproduce:

```
python scripts/run_infoagg.py --config configs/infoagg_rationale.yaml --cells comp --n 24 --live --mlflow
```

(Note: the first run of this iteration hung because the LLM client had no request timeout; a timeout was added
and the run repeated cleanly. See the commit history.)

## Results (Brier, lower is better; 24 instances per cell)

| cell | average | market (private) | debate | market_debate | oracle |
|---|---|---|---|---|---|
| comp-AND | 0.228 | 0.228 | 0.184 | 0.181 | 0.000 |
| comp-OR | 0.371 | 0.372 | 0.226 | 0.216 | 0.001 |
| comp-THRESH | 0.191 | 0.195 | 0.151 | 0.153 | 0.000 |

Paired-bootstrap 95 percent intervals:

| cell | market_debate vs average | market_debate vs debate |
|---|---|---|
| comp-AND | -0.047 [-0.108, -0.006] | -0.004 [-0.010, +0.002] |
| comp-OR | -0.155 [-0.223, -0.090] | -0.010 [-0.023, +0.003] |
| comp-THRESH | -0.038 [-0.068, -0.009] | +0.002 [-0.014, +0.017] |

## Findings

Consistent across all three complementary structures:

1. The rationale-augmented market beats the plain average, significantly, in every cell. This is the first
   market condition to beat the average on complementary information.
2. It ties the deliberated average in every cell. The market adds nothing beyond deliberation.
3. The market over private beliefs ties the plain average in every cell. Without the channel, no aggregation.

So the channel does the aggregating, not the market mechanism. The earlier failure was that a scalar price
cannot carry a decisive fact, and a one-line rationale can. Even the threshold case, where no single agent is
decisive, behaves the same way: deliberation lets the agents reconstruct the count.

A side observation on OR: the plain average there is 0.371, worse than chance, because each agent sees one
condition and is systematically biased, and averaging bakes the bias in. Deliberation corrects it.

## Caveats

- 24 instances per cell, synthetic abstract questions, one model and prompt configuration.
- A gap to the oracle remains (deliberation recovers much of the information but not all). This is one
  deliberation round; more rounds may close more of the gap, which is a natural next iteration.
- This says how aggregation works on a controlled task. It does not predict performance on real questions.
