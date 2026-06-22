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
   Proposed next.
5. Agent-framing A/B: neutral forecaster versus profit-maximizing trader.
6. Model-mix and forced-diversity variants.

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
Proposed. Awaiting go-ahead. Plan: run the complementary cells at one, two, and three deliberation rounds and
track how the deliberated average and market_debate move toward the oracle as rounds increase.
