# Live experiment loop ledger

Mode: one iteration at a time, checkpoint between runs. Proposer: rule-based queue. Each iteration is a real
live experiment (calls OpenRouter), in contrast to the replay search in `ledger.md`, which only re-scores
designs over cached beliefs.

Objective: find a market design that beats the plain average and approaches the oracle on the complementary
cells, or establish that the limit is fundamental.

## Rule-based queue

1. Baseline market (capped fractional-Kelly). Done, iteration 0.
2. Uncapped market (full Kelly, no cap). Done, iteration 1.
3. Rationale-augmented market: agents share a one-line rationale, then the market trades over the updated
   beliefs. Direct test of the bandwidth explanation. Proposed next.
4. Agent-framing A/B: neutral forecaster versus profit-maximizing trader.
5. Model-mix and forced-diversity variants.

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
Proposed. Awaiting go-ahead at the checkpoint. Plan: run one deliberation round so each agent's decisive fact
can enter the others' beliefs through a rationale, then run the market over those updated beliefs, and compare
the market price to the deliberated average, the plain average, and the oracle. If the market now moves toward
the oracle, it confirms that the earlier failure was the scalar-price channel, not the mechanism.
