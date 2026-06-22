# Experiments

Prototypes and tests. Keep each experiment self-contained in its own subfolder with a short README
stating the question it answers.

Design rationale for all of these lives in [`../research/design-questions.md`](../research/design-questions.md).

## The one experiment that decides everything

**`02-market-vs-aggregation`**, does an **AI-agent LMSR market** actually beat simpler aggregation of the
*same* agents? This is the make-or-break test; run it before building product. Score these conditions on
resolved questions (the same 10 agents feed every non-market condition, so any edge is the *mechanism*):

| # | Condition | Isolates |
|---|---|---|
| A | Single LLM, zero-shot, once | Floor |
| B | **Naive mean of 10 agents** | Wisdom-of-crowds baseline the market must beat |
| C | **Tuned aggregator** (accuracy- + recency-weighted, recalibrated, extremized) | Honest hard baseline (Atanasov) |
| D | **LMSR market price** (10 agents trade) | The mechanism under test |
| E | Market + tuned aggregator ensemble | Best achievable (AIA complementarity) |
| F | Superforecaster baseline (ForecastBench built-in) | Human gold standard |
| G | Company status-quo forecast (internal set only) | Incumbent to beat for the sale |

Metrics: **Brier Skill Score** (headline: "D is X% better than B") + calibration curve + Murphy
decomposition (does the market's edge come from *resolution*?). Paired per-question scoring, bootstrap 95% CIs.
- **Success:** D beats A and B significantly.
- **Stretch:** D or E beats C → strongest possible claim.
- **If D loses to C:** the product is E (market + aggregator ensemble). Pivot, don't abandon.

Test sets: (1) **ForecastBench** resolved questions (~1000, leak-free, superforecaster bar 0.093) for external
validity; (2) 100–300 **resolved internal company questions** for the real pitch.

## Build order (everything else supports Experiment 02)

1. **`01-lmsr-engine`**, minimal binary LMSR market maker (Python, ~a few lines):
   `C(q)=b·ln(Σexp(qᵢ/b))`, price `=softmax(q/b)`, trade cost `=C(q')−C(q)`, `b=L/ln(2)` from max-loss budget
   `L`. Use the **log-sum-exp trick** (subtract max before exp) to avoid overflow. Cap per-agent bankroll.
   Test: can N agents trade and move price toward a known outcome on resolved questions?
2. **`02-market-vs-aggregation`**, the decision experiment above.
3. **`03-agent-swarm`**, 5–10 independent agents, mixed models, short evidence brief (~5–15 ranked items),
   3–6 line thesis + probability each, median + √3 extremizing calibration. Fractional-Kelly (capped) sizing
   from confidence. Plug into 01.
4. **`04-internal-data`**, MCP connectors (Slack/Jira/CRM/Drive), partition slices across agents. This is the
   moat (Q2), but it needs a real company's data, so prototype with mock/seeded internal events first.

## Ground-truth data for backtests
- **ForecastBench** (~1000 leak-free resolved questions, built-in superforecaster baseline), best ready-made set.
- Resolved Polymarket / Metaculus questions (public APIs); Metaculus `forecasting-tools` for access + scaffolding.
- A hand-built set of resolved *internal-style* questions (ship dates, bookings hit/miss) to mock the enterprise case.
