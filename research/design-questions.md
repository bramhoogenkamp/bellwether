# Design questions, research synthesis (2026-06-21)

Five parallel research threads, one per question. Each section: **verdict → why → minimal recommendation →
what to skip**. The overarching constraint ("don't overcomplicate") is applied throughout.

---

## The spine: what this research changed about the project

The biggest finding cuts across every thread and **reframes the product**:

> The value is **not** in the market mechanism. It's in **(1) differentiated/internal information,
> (2) genuine model diversity, (3) calibration, and (4) accuracy-based reweighting over time.**
> A market is just *one* way to get (3) and (4), and a contested one.

Three facts force this reframe:
1. A **naive average of ~12 LLMs already matches a 925-person human crowd** (Brier 0.20 vs 0.19, Schoenegger et al., *Science Advances* 2024). So averaging is a *strong* baseline, not a strawman.
2. A randomized arm of the Good Judgment Project (~37k forecasts) found **"just asking" for probabilities was at least as accurate as market prices**, and a **tuned aggregator beat the human market by ~13%** (Atanasov et al., *Management Science* 2017).
3. **LMSR is mathematically a sequential proper scoring rule ≈ regularized online averaging** (Abernethy et al. 2013). The market isn't a separate magic ingredient.

So the market has to *earn* its complexity. It earns it only where averaging can't compete:
- **Continuous updating** as news/evidence arrives (a price, not a one-shot poll).
- **Stake-weighting → resolution/discrimination** (confident, informed agents move price more), exactly the term where flat ensembles and single LLMs are weakest.
- **Automatic reweighting** of agents that prove accurate, via bankroll (no manual tuning).
- **Auditable price trajectory**, every move carries a rationale.

**The make-or-break empirical question** (this is Experiment 02): does an **AI-agent LMSR market** beat both a **naive mean** and a **tuned aggregator** of the *same* agents? If yes → defensible product. If it only beats the naive mean but loses to the tuned aggregator → the product is the **market + aggregator ensemble** (which beats either alone, Bridgewater AIA finding).

---

## Q1, Swarm design: hypothesis, cognitive focus, context

**Verdict: simple wins decisively. Median of N independent forecasts + calibration. Diversity from
different *models*, not personas. Never debate.**

Why:
- **Bridgewater AIA Forecaster** = 10 *identical*, independent agents → mean → supervisor → Platt calibration. Matches human superforecasters (Brier 0.108 vs 0.111). Diversity comes from independent *search paths*, not roles. [arXiv 2511.07678](https://arxiv.org/abs/2511.07678)
- **Metaculus template bot** = 1 research pass → same prompt run 5× → **median**. That's it. [metac-bot-template](https://github.com/Metaculus/metac-bot-template)
- **Personas don't help**: 162 personas × 2410 questions × 9 models → none beat baseline; can't auto-pick the good one. [arXiv 2311.10054](https://arxiv.org/html/2311.10054v3)
- **Debate hurts**: accuracy *drops* over rounds as models converge on agreement over correctness; majority voting already captures the gains. [arXiv 2509.05396](https://arxiv.org/abs/2509.05396)
- **Model diversity is the real lever**: 2 diverse agents can match 16 identical ones. [arXiv 2602.03794](https://arxiv.org/html/2602.03794)
- **Context: less is more.** Retrieval sweet spot ~5–15 relevance-ranked items; near-relevant clutter is the *worst* noise (one topically-similar irrelevant doc can drop accuracy ~25%). Partition context across agents; don't give everyone everything. [Halawi et al.](https://arxiv.org/html/2402.18563) · ["lost in the middle"](https://arxiv.org/abs/2307.03172)
- **Reasoning is double-edged**: a short evidence-grounded thesis improves calibration; open-ended reasoning *without new evidence* breeds overconfidence. Base rates are the one "lens" with positive evidence; explicit "Bayesian reasoning" prompts hurt.

**Minimal recommended swarm (per repricing tick):**
1. **One shared research pass** → rank → keep ~5–15 high-signal items → distill to a short evidence brief.
2. **N = 5–10 independent forecasts.** Each writes a 3–6 line thesis (drivers, base rate, what would change my mind) + a probability. Hypothesis-*lite*.
3. **Diversity, cheap & evidence-backed:** (a) different base models/vendors; (b) at most two lenses, an **outside-view/base-rate** agent and an **inside-view/current-evidence** agent; optional one **pre-mortem/contrarian** *voter* (never a debater).
4. **Aggregate by median** (binary).
5. **Calibrate** with a fixed extremizing transform (≈√3), later fit to your own Brier history.
6. Translate probability → LMSR trades.

**Skip in v1:** multi-round debate, large persona rosters, per-agent private research silos, long-context "give everyone everything," complex weighting.

**Key risks:** agents reading the same evidence aren't independent (median of near-clones ≈ 1 vote) → model diversity matters most; internal events are often *insider-known* (the tech lead just knows if it ships) → a swarm over stale artifacts may be beaten by asking the one person; **reflexivity**, keep agents blind to the live price or they herd.

---

## Q2, Public vs private information (the existential question)

**Verdict: a PUBLIC-info-only agent market largely collapses to "one model in a trenchcoat."
The value requires differentiated information, overwhelmingly your INTERNAL data.**

Why, two independent failure modes converge:
1. **Theory (No-Trade Theorem, Milgrom-Stokey / Aumann):** agents with shared information *and* shared priors won't trade on information alone; willingness to trade itself signals info and unwinds the trade. What actually generates informative trade is **heterogeneous priors/models + noise traders** (Grossman-Stiglitz: fully-revealing prices can't even exist). [No-trade theorem](https://en.wikipedia.org/wiki/No-trade_theorem)
2. **Empirics: LLM errors are strongly correlated, and the correlation is *worst for the best models*, even across providers** (Kim et al., ICML 2025). "Diverse reasoning" via personas/prompts/debate is an unreliable, often-null way to decorrelate them. [arXiv 2506.07962](https://arxiv.org/abs/2506.07962)
- Single LLMs trail human forecasters but become competitive **only inside retrieval-augmented pipelines**, information access, not reasoning variation, closes the gap. [Halawi et al.](https://arxiv.org/abs/2402.18563)

**Simplest ways to give agents differentiated information (ranked by leverage:effort):**
1. **Internal-data partitioning via MCP (highest leverage).** Wire connectors to Slack/Jira/CRM/Drive; give each agent a *different slice* (eng vs sales vs finance vs exec comms). This is the only step that produces *both* genuine information *and* genuine independence, and it's exactly what the base model and competitors can't have.
2. **Live web/tool access (easy, low differentiation).** Adds freshness; commodity signal shared by all.
3. **Genuinely different model families (cheap decorrelation).** Cross-family > persona/temperature.

**Implication for the project:** the defensible moat is **internal data plumbing**, not the trading floor.
A public-only market is not worth building as a *market*. Build the data plumbing first.

---

## Q3, Thin/small markets, mechanism, simulated liquidity

**Verdict: LMSR (Hanson's Logarithmic Market Scoring Rule), one binary market per question, with the
liquidity parameter `b` set explicitly from a max-loss budget. It's the textbook fix for thin markets.**

Why:
- Thin human markets have wide spreads, no counterparties, and are whale-manipulable. **LMSR's market maker *is* the counterparty** → always a price, always liquidity, even with a handful of agents. (Internal markets at HP/Siemens worked with 20–60 participants.)
- **"Simulating liquidity parameters" is literally what `b` is.** Liquidity is *set*, not emergent from volume. Cultivate Labs found `b ≈ 50` a "Goldilocks" value in practice. [Cultivate on liquidity](https://www.cultivatelabs.com/posts/what-is-liquidity-and-how-does-it-affect-prediction-markets)
- The log scoring rule is **strictly proper** → a risk-neutral agent's best move is to report its true belief.
- Plain LMSR's "flaw" (liquidity-*insensitivity*: a fixed bet moves price the same regardless of prior volume) is a **feature** for agent-only markets, depth is predictable and centrally controlled.

**The whole engine:**
- Cost: `C(q) = b · ln( Σᵢ exp(qᵢ / b) )`
- Price (= probability): `pᵢ = exp(qᵢ/b) / Σⱼ exp(qⱼ/b)` (softmax over `q/b`)
- Trade cost: `C(q') − C(q)`
- Max-loss / subsidy (binary): `b · ln(2) ≈ 0.693·b` → set `b = L / ln(2)` for budget `L`

**Skip:** order books, combinatorial/multi-outcome-correlated markets (`#P`-hard), LS-LMSR/Bayesian MMs (upgrade paths, not starting points). **Only real gotcha:** use the **log-sum-exp trick** to avoid `exp()` overflow; **cap per-agent budgets** so one rich agent can't steamroll the price.

---

## Q4, Play money & agent bankroll/behavior

**Verdict: play money is good enough. Keep a bankroll (it's the self-correcting reweighting mechanism).
Size by fractional Kelly. Light behavioral diversity only. No real money.**

Why:
- **Play ≈ real money accuracy** in the canonical field study (Servan-Schreiber/Wolfers/Pennock, TradeSports vs NewsFutures). Corporate internal markets that beat expert forecasts (Google, HP, Ford) ran on virtual currency. The one requirement: **a reason to be right** (a scored payout), which we control. [Does Money Matter?](http://users.nber.org/~jwolfers/Papers/DoesMoneyMatter.pdf)
- **Bankroll is the one feature worth keeping from "money":** accurate agents compound influence, bad ones fade, automatic reweighting a repeated poll can't do. [forecast convergence](https://arxiv.org/pdf/2402.16345)
- **Accuracy comes from a few skilled traders, not headcount** (~3% drive price discovery on Polymarket) → invest in **5–15 good agents**, not a big swarm. [Yale/LBS study](https://insights.som.yale.edu/insights/wisdom-of-the-few-prediction-markets-are-driven-by-small-number-of-skilled-traders)
- **Fractional Kelly** (~½ Kelly, capped) is the clean, few-lines way to size by confidence.
- Heterogeneous risk appetites help modestly but are a second-order knob; LLM agents skew "too rational," so inject diversity via prompts. Keep to 3–4 archetypes.

**Minimal money design:** LMSR pricing + bankroll-aware agents + fractional-Kelly (capped) sizing from each agent's confidence + resolve-and-pay-out + 3–4 prompt-level archetypes with varied starting bankrolls. **Skip:** real money, margin, leverage, portfolio optimization, personality engines.

---

## Q6, Benchmarking & value proposition

**Verdict: the headline metric is Brier Skill Score ("X% better than baseline") + a calibration curve.
The benchmark must prove the MARKET beats both a naive average AND a tuned aggregator of the same agents.**

State of the art (mid-2026):
- On clean benchmarks, the best multi-agent LLM systems (**AIA**) now **match superforecasters** (Brier 0.108 vs 0.111). [arXiv 2511.07678](https://arxiv.org/abs/2511.07678)
- On live head-to-head tournaments (**Metaculus AIB**), **human Pros still win**, via superior *resolution/discrimination*, and the gap isn't closing. [Q2 AIB results](https://www.lesswrong.com/posts/Surnjh8A4WjgtQTkZ/)
- **Markets + LLMs are complementary**: AIA + market consensus beats either alone (0.106 vs 0.126 / 0.111). This is the strongest tailwind for a market *of* AI agents.

**Metrics:** lead with **Brier Skill Score** (BSS = 1 − BS/BS_ref, reads as "% better than baseline") + calibration curve. Appendix: raw Brier, log loss, **Murphy decomposition** (Reliability − Resolution + Uncertainty), use it to diagnose whether the market's edge is calibration or **resolution** (expect resolution).

**Benchmark conditions, same 10 agents feed every non-market condition, so any market edge is the *mechanism*:**
| # | Condition | Isolates |
|---|---|---|
| A | Single LLM, zero-shot, once | Floor |
| B | **Naive mean of 10 agents** | Wisdom-of-crowds baseline the market must beat |
| C | **Tuned aggregator** (accuracy- + recency-weighted, recalibrated, extremized) | The honest hard baseline (Atanasov) |
| D | **LMSR market price** (10 agents trade) | The mechanism under test |
| E | Market + tuned aggregator ensemble | Best achievable (AIA complementarity) |
| F | Superforecaster baseline (ForecastBench built-in) | Human gold standard |
| G | Company status-quo forecast (internal set) | Incumbent to beat for the sale |

**Test sets:** (1) **ForecastBench** resolved questions (~1000, leak-free, built-in superforecaster bar 0.093) for external validity; (2) **100–300 resolved internal company questions** (past ship dates, bookings hit/miss, hire-by dates), this is what sells it.

**Stats:** paired per-question scoring, bootstrap 95% CIs on Brier deltas, p-values (mirror AIA's "statistically indistinguishable" test). **Success:** D beats A and B significantly (core proof). **Stretch:** D or E beats C. **Context:** D approaches F, clearly beats G.

**Value-prop (three beats):**
1. **As good as your best people, provably better than your current process**, AIA matches superforecasters; internal markets beat companies' own expert forecasts by up to **25% MSE reduction** (Cowgill-Zitzewitz, Google/Ford); measured on *your* resolved questions.
2. **The market beats just averaging the agents**, stake-weighting adds *resolution*, where ensembles are weakest. Our benchmark isolates this delta.
3. **Does what no human team will**, pennies per forecast vs $50–160/hr analysts; minutes not a 3–4-week S&OP cycle; always-on; scales to thousands of small questions (every project/region/SKU) nobody would staff; every price has an auditable rationale.

**Honest risk to flag:** Atanasov shows a tuned aggregator can beat a *human* market by ~13%. Whether an *AI-agent* market beats a tuned *AI* aggregator is open, which is exactly why C is in the benchmark, and why E (ensemble) is the fallback product.

---

## Net recommendation (the synthesis of all five)

Build, in order:
1. **Internal-data plumbing** (MCP connectors, partitioned per agent), the actual moat (Q2).
2. **A minimal swarm**, 5–10 independent agents, mixed models, 2 lenses, short evidence brief, median + √3 calibration (Q1).
3. **An LMSR market**, binary, one per question, `b` from a max-loss budget, capped bankrolls, fractional-Kelly sizing, resolve-and-pay (Q3, Q4).
4. **The benchmark harness**, conditions A–G on resolved questions, BSS + calibration + Murphy (Q6).

But **run the benchmark's core comparison (D vs B vs C) as early as possible** on cheap resolved data , 
because if the market doesn't beat a tuned aggregator, the product pivots to the market+aggregator
ensemble. Everything hinges on that one result.
