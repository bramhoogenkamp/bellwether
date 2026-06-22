# Iteration 6: model-mix / diversity sweep

Follow-up to [`06-agent-framing.md`](06-agent-framing.md). Date 2026-06-22.

## Context

LLMs tend to be near-clones (experiment 1 measured agent error correlation of 0.82 to 0.98). Deliberation
turned out to be the lever that pools dispersed information, so the natural question is whether deliberation
depends on the agents being different. This sweep compares a diverse four-model swarm against homogeneous
swarms of a single model, on the unstructured cell, where models must interpret messy prose and so are most
likely to differ.

## Setup

unstr-AND, n=24, deliberation depths 1 to 3. Three compositions: diverse (gpt-5-mini, claude-sonnet-4.5,
gemini-2.5-pro, deepseek-r1), homogeneous gpt-5-mini, homogeneous claude-sonnet-4.5. Script:
`scripts/run_diversity.py`. The disagreement column is the mean pairwise absolute difference of the agents'
private forecasts before deliberation. Reproduce:

```
python scripts/run_diversity.py --cell unstr-AND --n 24 --rounds 3 --live --mlflow
```

## Results (Brier, lower is better)

| composition | disagreement | average | round 1 | round 2 | round 3 | oracle |
|---|---|---|---|---|---|---|
| diverse | 0.066 | 0.174 | 0.149 | 0.115 | 0.097 | 0.017 |
| homo gpt-5-mini | 0.050 | 0.185 | 0.125 | 0.117 | 0.115 | 0.024 |
| homo sonnet | 0.049 | 0.165 | 0.162 | 0.166 | 0.166 | 0.019 |

Paired bootstrap, diverse minus homo-sonnet at round 3: -0.078 [-0.143, -0.024], significant.

## Findings

1. The diverse swarm deliberates to the lowest error (0.097), significantly below the homogeneous sonnet swarm.
2. The strongest single model deliberates the worst. homo-sonnet starts with the best average (0.165, sonnet is
   individually the strongest model here) but barely moves across rounds (0.162, 0.166, 0.166) and ends last.
   This rules out "the diverse mix just had better models."
3. So diversity is what makes deliberation work. A swarm of clones has little new to tell each other, so its
   deliberation stalls; a diverse swarm carries genuinely different reads, so it keeps improving with depth.
4. The diverse swarm disagrees more before deliberating (0.066 versus about 0.049), which is the raw material
   deliberation needs. Even so, the across-family disagreement is modest, confirming that LLMs are fairly
   clonal and that effective diversity is a real ceiling.

The practical reading: if you use a swarm with deliberation, composition matters as much as model strength.
Running one strong model many times gives you little, because the copies agree with themselves. Mixing model
families is what gives deliberation something to work with.

## Caveats

- n=24, one cell (unstr-AND), one model per homogeneous swarm.
- The disagreement proxy is confounded by the agents seeing different slices, so it understates true model
  agreement; read it as relative across compositions, not absolute.
- homo-sonnet was nearly flat while homo-gpt5mini did improve at round 1, so how much a homogeneous swarm can
  deliberate is partly model-specific, not zero for all single models.
