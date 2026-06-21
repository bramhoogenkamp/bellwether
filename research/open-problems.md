# Open problems

## 1. Agent diversity — "one model in a trenchcoat" (the core risk)
A market beats a single forecaster only when traders disagree for *independent* reasons. If every agent is
the same base model reading the same data, the price collapses to that model's output with extra steps —
worse than just asking the model once (you add cost + noise, not signal).

Levers to engineer real diversity (to test):
- **Model diversity** — different base models (Opus, Sonnet, Haiku, non-Anthropic) per agent.
- **Information diversity** — partition data sources; give agents different evidence subsets / tools.
- **Prompt/persona diversity** — bull/bear, base-rate-first, inside vs outside view, contrarian.
- **Temporal diversity** — agents enter at different times / on different triggers.
- **Risk-appetite diversity** — different bankroll, Kelly fraction, conviction thresholds.

Open question: is engineered diversity *enough*, or does the absence of independent private information
(humans have private signals; cloned agents don't) fundamentally cap a closed agent market's edge?

## 2. Market mechanism
- **LMSR / automated market maker** vs **order book**. LMSR is simplest for thin, agent-only markets
  (always-on liquidity, bounded loss, well-understood). Likely the prototype choice.
- How is the bootstrap probability set before agents trade?

## 3. Resolution / oracle
- Internal events resolve from internal systems (Jira, CRM, finance, calendars). Need a clean oracle
  interface and a human override path for ambiguous resolutions.
- Question wording is itself hard (cf. Kalshi's internal "Harrison" agent for contract wording).

## 4. Incentives without real money
- Internal markets can't always use cash. What's the scoring/reward currency that keeps agents (and the
  humans tuning them) honest? Brier-scored reputation? Play-money bankroll that gates influence?

## 5. Evaluation — does it actually beat baselines?
Baselines any prototype must beat to justify existing:
- Single best model asked once.
- Naive average of N agents' point probabilities (no market).
- Current internal human estimate / status-quo process.
Metrics: Brier score, calibration curves, log loss, and decision-relevant lead time.

## 6. Why might this NOT work / why doesn't it exist yet
- Possibly substantive (diversity cap above), not just timing.
- Enterprise trust: will leaders act on a price set by bots?
- Governance: 40%+ of agentic AI projects projected to be canceled by 2027 (Gartner) — value/cost/governance.
