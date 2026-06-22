# Iteration 5: agent-framing A/B

Follow-up to [`05-unstructured-evidence.md`](05-unstructured-evidence.md). Date 2026-06-22.

## Context

The design decision (see `research/self-improving-loop.md`) was to keep agents as neutral honest forecasters
with mechanical position sizing, rather than prompting them as profit-maximizing traders, and to test the
trader framing as a variable instead of assuming it. This iteration runs that A/B.

## Setup

comp-AND and comp-OR, n=24, paired: each instance is forecast under both framings (neutral honest forecaster
and profit-maximizing trader), and we compare the average, the market, and calibration (ECE). Script:
`scripts/run_framing.py`. Reproduce:

```
python scripts/run_framing.py --cells comp-AND,comp-OR --n 24 --live --mlflow
```

## Results

| cell | framing | avg Brier | mkt Brier | avg ECE | mkt ECE |
|---|---|---|---|---|---|
| comp-AND | neutral | 0.228 | 0.228 | 0.287 | 0.259 |
| comp-AND | trader | 0.229 | 0.229 | 0.312 | 0.286 |
| comp-OR | neutral | 0.372 | 0.367 | 0.389 | 0.390 |
| comp-OR | trader | 0.403 | 0.398 | 0.402 | 0.425 |

Paired bootstrap, trader minus neutral: comp-AND average +0.0009 [-0.001, +0.004], comp-OR average +0.0308
[-0.007, +0.067]. The market deltas track the average deltas.

## Findings

1. The trader framing does not help. comp-AND is a wash; comp-OR is worse by about 0.03.
2. Calibration is slightly worse under the trader framing on both cells (higher ECE).
3. The direction is consistent across both cells and both the average and the market, although the intervals
   include zero at n=24, so this is directional rather than significant.

This supports the original choice: keep agents as neutral honest forecasters with mechanical sizing, not
prompted gain-maximizers. Telling a model to maximize profit nudges it toward overconfidence without improving
accuracy, which is exactly the failure mode the neutral framing avoids.

## Caveats

- n=24 per cell, two cells, one wording of the trader prompt. A stronger or differently worded trader prompt
  could move this; the result is "no benefit and a mild cost," not "framing can never matter."
