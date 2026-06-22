# Running notes & ideas

## Decisions made (from research synthesis 2026-06-21, see research/design-questions.md)
- [x] **Mechanism:** LMSR, binary, one market per question. `b` set from max-loss budget.
- [x] **Money:** play money + per-agent bankroll (self-correcting reweight) + fractional-Kelly sizing. No real money.
- [x] **Swarm:** 5–10 independent agents, *mixed model families*, 2 lenses (base-rate + inside-view), median
      aggregation, √3 extremizing calibration. No personas, no debate.
- [x] **Context:** small & curated (~5–15 ranked items), partitioned across agents. Not "everything to everyone."
- [x] **Metrics:** Brier Skill Score + calibration curve (lead); raw Brier / log loss / Murphy (appendix).

## Still open
- [ ] **Build vs integrate?** Build a thin engine (full control, proves the thesis) vs integrate on
      Cultivate/Metaculus. Leaning build, the engine is only a few lines (LMSR) and we need control of the benchmark.
- [ ] **THE crux, Market or aggregation?** Does the LMSR market beat a *tuned* aggregator (not just a naive
      mean) of the same agents? Experiment 02. If no → product = market+aggregator ensemble. Run this FIRST.
- [ ] **Insider-knowledge problem:** internal events are often known to one person. Do we need a human insider
      in the loop, or pick only questions where no single person knows the answer? (Q1 risk.)

## Sharpest first test
Before building any product: run Experiment 02. The bar is NOT "beat one model" or "beat a naive average"
(both too easy / the naive average already matches a 925-human crowd). The real bar is **beat a *tuned*
aggregator** of the same agents (Atanasov: a tuned aggregator beat a human market by ~13%). If the market
can't clear that, the defensible product is the market+aggregator *ensemble*, not the market alone.

## Naming ideas
- (tbd)

## Parking lot
- Kalshi's internal "Harrison" agent handles contract *wording*, wording is a real sub-problem worth a
  dedicated agent.
- Diversity might come cheapest from information partitioning (different tools/sources) rather than
  different base models.
