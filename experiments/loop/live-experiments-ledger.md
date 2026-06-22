# Live experiment loop ledger

Mode: one iteration at a time, checkpoint between runs. Proposer: rule-based queue. Each iteration is a real
live experiment (calls OpenRouter), in contrast to the replay search in `ledger.md`, which only re-scores
designs over cached beliefs.

Objective: find a market design that beats the plain average and approaches the oracle on the complementary
cells, or establish that the limit is fundamental.

## Rule-based queue

1. Baseline market (capped fractional-Kelly). Done, iteration 0.
2. Uncapped market (full Kelly, no cap). Done, iteration 1.
3. Rationale-augmented market: agents share a rationale, then the market trades over the updated beliefs.
   Done, iteration 2.
4. Deliberation depth: vary the number of deliberation rounds, track how far the gap to the oracle closes.
   Done, iteration 3.
5. Unstructured evidence: messy natural-language slices with a known latent outcome, no deterministic combiner.
   Done, iteration 4.
6. Agent-framing A/B: neutral forecaster versus profit-maximizing trader. Done, iteration 5.
7. Model-mix and diversity sweep. Done, iteration 6.
8. Real-data forward test (outside the synthetic lab). Proposed, not yet run.

## Iterations

### Iteration 0, baseline market (capped)
Live run, experiment 1 ([../info-aggregation/01-information-aggregation.md](../info-aggregation/01-information-aggregation.md)).
Result: the market ties the plain average on every cell and sits far from the oracle on the complementary
cells; debate is the only mechanism that pools the complementary information.
Decision: the baseline does not aggregate. The market used cautious parameters, so propose testing whether
removing the cap changes it (iteration 1).

### Iteration 1, uncapped market
Live run, experiment 2 ([../info-aggregation/02-market-design-probe.md](../info-aggregation/02-market-design-probe.md)).
Result: uncapping did not rescue the market. comp-AND significantly but tinily worse than the average, comp-OR
and comp-THRESH marginally better but not significant, all still far from the oracle and behind debate. The
replay search found the same for the closed-form equilibrium pool.
Decision: reject the parametric explanation. Within the trading family over scalar beliefs the limit looks
fundamental. Next, test the bandwidth explanation directly by giving the market a language channel
(iteration 2).

### Iteration 2, rationale-augmented market
Live run, write-up in [../info-aggregation/03-rationale-market.md](../info-aggregation/03-rationale-market.md).
(The first attempt hung on a call with no timeout; fixed the timeout and reran clean.)
Result: across all three complementary cells, the market over post-deliberation beliefs beats the plain
average significantly (md vs avg: AND -0.047, OR -0.155, THRESH -0.038), ties the deliberated average, and the
market over private beliefs still ties the plain average.
Decision: the earlier failure was the channel, not the mechanism, confirmed across AND, OR, and threshold. A
language channel lets the decisive fact into the beliefs, and then averaging or a market over those beliefs
both work. The market still adds nothing beyond the deliberated average. A gap to the oracle remains.
Next: a gap to the oracle is still open after one deliberation round, so propose iteration 3 to vary the
number of deliberation rounds and see how much of that gap closes (does deliberation approach the oracle, or
plateau). The agent-framing A/B moves down the queue.

### Iteration 3, deliberation depth
Live run, write-up in [../info-aggregation/04-deliberation-depth.md](../info-aggregation/04-deliberation-depth.md).
Result: each deliberation round roughly halves the Brier toward the oracle, in both cells. comp-AND goes
0.228, 0.191, 0.079, 0.034 across rounds 0 to 3 (near the oracle); comp-OR goes 0.386, 0.251, 0.138, 0.068
(still descending). The market over deliberated beliefs tracks the deliberated average at every depth.
Decision: depth is the lever. One round under-pools, about three rounds nearly recover the information, and
the market adds nothing at any depth. This sharpens the contrast with HiddenBench. The mechanism question is
now well answered on the structured task, so the next iteration should attack external validity: the
structured task admits a deterministic combiner, so move to unstructured evidence with no clean combination
rule (iteration 4).

### Iteration 4, unstructured evidence
Live run, write-up in [../info-aggregation/05-unstructured-evidence.md](../info-aggregation/05-unstructured-evidence.md).
Result: with prose slices the oracle is no longer perfect (0.021), and deliberation closes only about 35 to 50
percent of the gap to the oracle, versus 83 to 85 percent on the structured cells. unstr-AND went 0.171,
0.138, 0.124, 0.119 across rounds 0 to 3 and flattened; unstr-OR went 0.329, 0.237, 0.207, 0.174 and was still
descending. The market again added nothing beyond the deliberated average.
Decision: the mechanism story holds (market adds nothing, deliberation does the work), but messy evidence
shifts the bottleneck from transmission to interpretation, so deliberation is useful but not complete and the
limit becomes how well the models read the evidence. This is the realistic regime. Natural next steps now move
outside the synthetic lab: a real-data forward test, and the agent-framing A/B. Pausing the synthetic loop here
is reasonable.

### Iteration 5, agent-framing A/B
Live run, write-up in [../info-aggregation/06-agent-framing.md](../info-aggregation/06-agent-framing.md).
Result: framing agents as profit-maximizing traders did not help and trended worse (comp-OR average +0.031,
slightly worse ECE on both cells), consistent in direction but not significant at n=24. The market tracked the
average under both framings.
Decision: keep agents as neutral honest forecasters; the trader framing buys overconfidence without accuracy.
The synthetic loop has now answered the mechanism, channel, depth, structure, and framing questions. One
synthetic question remained (does composition matter), run as iteration 6.

### Iteration 6, model-mix / diversity sweep
Live run, write-up in [../info-aggregation/07-diversity.md](../info-aggregation/07-diversity.md).
Result: on unstr-AND, the diverse four-model swarm deliberated to 0.097, significantly below a homogeneous
sonnet swarm at 0.166 (delta -0.078 [-0.143, -0.024]). The strongest single model started best (average 0.165)
but barely moved with deliberation, ruling out a model-quality explanation. The diverse swarm also disagreed
more before deliberating (0.066 vs ~0.049).
Decision: diversity is what makes deliberation work. A swarm of clones has little new to exchange, so its
deliberation stalls; mixing model families is what gives deliberation something to work with. The near-clone
concern is real and effective diversity is a genuine ceiling. The synthetic loop is now complete; the only
remaining queue item leaves the lab (a real-data forward test).
