# Self-improving experiment loop (plan)

Date 2026-06-22. A plan for turning the ad hoc sequence of experiments into a loop that improves the market
design systematically, learns from each run, and proposes the next change, with guardrails so "self-improving"
does not become "overfitting".

## What we mean by self-improving

The primary sense here is a research loop that improves the design across iterations. We already started one by
hand: experiment 1 found the market ties a plain average, experiment 2 tests whether that was the capped
trading parameters. The loop formalizes that pattern so it is systematic and comparable rather than ad hoc.

There is a second, complementary sense worth noting: a deployed forecaster that improves online from resolved
outcomes (agents reweighted by track record, calibration fit to history). That is a real mechanism we can fold
in later, but it is not the focus of this plan.

## Goal of the loop

Answer, robustly, the question: under what conditions does a market of LLM agents aggregate dispersed
information better than averaging, and can any design push the mechanism toward the oracle on complementary
structure. Each iteration either moves the market toward that goal or sharpens the evidence that the limit is
fundamental.

## Objective the loop optimizes

- Primary: on the complementary cells, market Brier below the plain average, and the difference significant.
- Secondary: market Brier approaching the oracle (gap to oracle shrinking), and market not worse than debate.
- Constraints: stay calibrated (ECE), and do not regress the substitutable controls.

Every iteration keeps the same fixed reference conditions (single, average, tuned, debate, oracle) so runs are
directly comparable.

## Design space (the knobs the loop may change)

- Market mechanism: capped fractional-Kelly (the baseline), uncapped, trade-to-convergence, or the closed-form
  equilibrium pool (LMSR as a confidence and wealth weighted log pool, Chakraborty-Das). Moving to convergence
  or closed form removes the cap, rounds, ordering, and seed artifacts in one step.
- Channel: price only, or rationale-augmented, where each trade also carries a short reason other agents can
  read. This is the direct test of the bandwidth explanation from experiment 1.
- Agent framing: neutral honest forecaster (default) or profit-maximizing trader. Treated as a measured knob,
  see the decision below.
- Belief to stake: confidence-weighted mechanical sizing (default) or agent-decided stake.
- Diversity: model mix, forced decorrelation, granularity of the evidence partition.
- Structures: substitutable, AND, OR, threshold, plus new ones (weighted k-of-n, sequential arrival).
- Instances per cell.

## The loop

1. Propose. Pick the next design change. Start rule-based (a priority queue of the changes above); later an
   agent can read the latest results note and propose the highest-information next change.
2. Run. Generate fresh synthetic instances on a held-out seed, score all conditions. Ground truth is known, so
   scoring is immediate and cheap.
3. Diagnose. Compare to the hypotheses and to prior iterations in MLflow. Did the change move the market toward
   the oracle and past the average, did it regress the controls, what did it cost.
4. Record. Append a numbered results note and an MLflow run, and update a running summary table.
5. Decide. Continue, branch to a variant, or stop.

## Guardrails (so the loop does not just overfit)

- Train and validation split. Optimize design on one set of structures and seeds, then confirm any apparent win
  on held-out structures and seeds before believing it. This is the key safeguard, because a loop that keeps
  tuning against the same test set will manufacture a result that does not generalize.
- Pre-register the primary metric and comparison for each iteration before running it.
- Cost budget per iteration, reported each run.
- Stop conditions: either a design robustly beats the average on held-out complementary cells, or we have shown
  across several designs (including convergence and closed form) that the limit is fundamental.

## Automation

Drive it manually first, since each iteration teaches us where to look next and the synthetic runs are cheap.
A light automation is a runner that takes a design spec, runs it, scores it, and writes the note, plus an
optional proposer step. Keep a human in the loop on cost and on accepting a win, to avoid runaway spend and to
keep the train/validation discipline honest.

## Decision: do we tell agents they are rational gain-maximizing traders

No, not in the core aggregation study. By default each agent is a neutral, honest forecaster: it reads its
slice and reports a probability and a confidence, and a principled mechanical rule converts that belief into a
stake. We do not prompt it as a profit-maximizing trader. Three reasons:

1. It isolates the mechanism. We want to measure whether the market aggregates beliefs, not whether LLMs are
   skilled traders. A "maximize your gains" prompt mixes the two.
2. The incentive-compatibility argument does not clearly hold for LLMs. A proper scoring rule makes honesty
   optimal for an ideal rational agent, but an LLM's objective is unspecified, so a trader prompt can induce
   gaming, herding toward the current price, or risk role-play that biases the price rather than revealing
   belief.
3. It matches the theory. The clean reading of LMSR is a pool of the traders' beliefs, so eliciting beliefs and
   sizing them mechanically is the faithful version.

But the trader framing is interesting, so we make it a measured knob, not the baseline. One iteration runs an
A/B of neutral forecaster versus profit-maximizing trader and checks whether the framing changes aggregation or
calibration. The same framing is also the natural setup for a later, separate study on manipulation and
collusion, where strategic behavior is the object of study rather than a confound.

## First three iterations

1. Equilibrium pool. Replace capped simulated trading with trade-to-convergence or the closed-form pool. This
   is the rigorous version of experiment 2 and removes the parameter-dependence cautions.
2. Rationale-augmented market. Give the price market a language side-channel and test whether it recovers the
   complementary information. Direct test of the bandwidth explanation.
3. Agent-framing A/B. Neutral forecaster versus profit-maximizing trader, to settle the sub-question
   empirically.
