# Market-design probe: can an uncapped market aggregate where the capped one did not

Follow-up to [`01-information-aggregation.md`](01-information-aggregation.md). Date 2026-06-22.

## Context from experiment 1

Experiment 1 compared six forecasters (single, average, tuned, market, debate, oracle) on synthetic
dispersed-private-information instances, across substitutable and complementary (AND, OR, threshold) signal
structures. The headline result: the market never beat a plain average, on any structure. On the complementary
cells the oracle was near-perfect, so the information was fully recoverable, but only deliberation pooled it.
The reading was that a price is a single scalar and cannot carry a decisive fact, while a sentence can.

That result came from one market design. Experiment 1's market is seeded at the agents' consensus and sizes
each bet with capped fractional Kelly (kelly_fraction 0.5, max bet 0.25 of bankroll, 3 rounds). So the agent
who privately knows that one condition failed, and is therefore near-certain, may simply have been prevented
from moving the price far enough. This experiment separates those two explanations.

## Question

If we remove the cap and let a near-certain agent trade at full Kelly over more rounds, does the market move
toward the truth on the complementary cells?

- If the market improves and approaches debate or the oracle, experiment 1's failure was parametric: the
  mechanism can aggregate, our settings held it back.
- If the market stays at the average even uncapped, the failure is closer to fundamental: a single price is
  too low-bandwidth to carry the decisive fact regardless of how hard the informed agent bets.

## Setup

Identical to experiment 1 in every respect (same synthetic generator and seed, same four reasoning models, the
same oracle and debate baselines, the same scoring) except the market trading policy:

| | experiment 1 (baseline) | experiment 2 (this) |
|---|---|---|
| kelly_fraction | 0.5 | 1.0 (full Kelly) |
| max bet fraction | 0.25 | 1.0 (no cap) |
| rounds | 3 | 5 |

Config: `configs/infoagg_uncapped.yaml`. Run on the three complementary cells only (comp-AND, comp-OR,
comp-THRESH). The substitutable controls are omitted because, with no edge to trade on, the policy cannot
change a no-trade consensus, so they would only reproduce experiment 1's null.

## Reproduce

Experiment 1 (baseline, unchanged):

```
python scripts/run_infoagg.py --live --n 24 --mlflow
```

Experiment 2 (this probe):

```
python scripts/run_infoagg.py --config configs/infoagg_uncapped.yaml --cells comp --live --n 24 --mlflow
```

## Results

Uncapped market, 24 instances per cell.

| cell | average | market (uncapped) | debate | oracle | market - average |
|---|---|---|---|---|---|
| comp-AND | 0.228 | 0.233 | 0.193 | 0.000 | +0.004 [+0.003, +0.006], significant (worse) |
| comp-OR | 0.401 | 0.392 | 0.260 | 0.001 | -0.009 [-0.021, +0.005], not significant |
| comp-THRESH | 0.193 | 0.187 | 0.167 | 0.001 | -0.005 [-0.014, +0.003], not significant |

Conclusion: removing the cap did not rescue the market. On comp-AND it is significantly, though tinily, worse
than the average; on comp-OR and comp-THRESH it is marginally better but not significant. The market still does
not beat the average on any complementary cell, still sits far from the oracle, and still trails debate. So
experiment 1's failure was not the capped parameters. Within the simulated-trading family the limit looks
fundamental, and the replay loop found the same for the closed-form equilibrium pool: no market design over the
agents' scalar beliefs recovers the complementary information.

Caveat: this is a separate live run, so the underlying beliefs differ from experiment 1 (for example the
comp-OR average is 0.401 here versus 0.386 in experiment 1). The valid comparison is market versus average
within this run, not market here versus market in experiment 1.
