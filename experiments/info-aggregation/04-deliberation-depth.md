# Iteration 3: deliberation depth

Follow-up to [`03-rationale-market.md`](03-rationale-market.md). Date 2026-06-22.

## Context

Iteration 2 showed that a single deliberation round lets agents pool dispersed information and beat the plain
average, but a real gap to the oracle remained. This iteration varies the number of deliberation rounds to see
whether more rounds keep closing that gap or plateau.

## Setup

comp-AND and comp-OR, n=24, capped market. In one nested pass per instance the agents deliberate up to three
rounds, and we record the deliberated average and a market over the deliberated beliefs after rounds 1, 2, and
3. Config: `configs/infoagg.yaml`. Script: `scripts/run_depth.py`. Reproduce:

```
python scripts/run_depth.py --cells comp-AND,comp-OR --n 24 --rounds 3 --live --mlflow
```

## Results (Brier, lower is better)

| | average | round 1 debate | round 2 debate | round 3 debate | oracle |
|---|---|---|---|---|---|
| comp-AND | 0.228 | 0.191 | 0.079 | 0.034 | 0.000 |
| comp-OR | 0.386 | 0.251 | 0.138 | 0.068 | 0.000 |

The market over the deliberated beliefs tracked the deliberated average at every round (comp-AND: 0.178,
0.071, 0.031; comp-OR: 0.243, 0.132, 0.068), so it is omitted from the table for clarity. The market over
private beliefs tied the plain average in both cells.

## Findings

1. Deliberation depth is the lever. Each round roughly halves the Brier toward the oracle, in both cells.
2. One round is not enough. It only partially pools the information (0.19 and 0.25). About three rounds nearly
   recover it: comp-AND reaches 0.034, essentially the oracle, and comp-OR reaches 0.068 and is still
   descending, so it would likely need a round or two more to fully converge.
3. The trajectory generalizes across AND and OR.
4. The market adds nothing beyond the deliberated average at any depth.

So the binding variables are the language channel and the depth of deliberation, not the market mechanism.
Given enough rounds, iterated deliberation among strong models nearly reconstructs information that no single
agent held.

This sharpens the contrast with HiddenBench, which reported that LLM debate fails to pool hidden-profile
information. With strong 2026 models and a few rounds, iterated deliberation essentially solves it here.

## Caveats

- This is the structured task, where a deterministic combiner exists and the oracle is that combiner. The
  result shows deliberation can route the pieces and recover the answer, but it does not yet show value over a
  deterministic process. The real test is unstructured evidence with no clean combination rule.
- n=24 per cell, two cells, one model and prompt configuration.
- comp-OR has not fully converged by round 3, so "three rounds is enough" is cell-dependent.
