# Pre-registration: Manufactured Consensus in LLM Agent Collectives

Version 1, 2026-06-23. Registered before the confirmatory runs. Pilots are exploratory and are reported as
such; the confirmatory analysis below is fixed in advance. Design rationale is in [`study.md`](study.md).

## Models and versions (pinned)

Confirmatory swarm: openai/gpt-5-mini, anthropic/claude-sonnet-4.5, google/gemini-2.5-pro, deepseek/deepseek-r1,
queried through OpenRouter, temperature 1.0, with the model snapshots and dates recorded in each run's MLflow
parameters. The headline results are replicated on a second model generation before submission.

## Hypotheses, designs, and falsifiers

### H1. Confidence and agreement decouple from accuracy (the decoupling)
- Design. Deliberation over K=4 rounds across conditions: substitutable (redundant, nothing to pool),
  complementary AND and OR (a decisive dispersed piece exists), and consensus-trap (a shared misleading cue).
  Script: `scripts/run_decoupling.py`.
- Primary outcomes per round: swarm-mean Brier and its gap to the oracle (accuracy); calibration error of the
  swarm mean (the rigorous confidence measure); inter-agent agreement = 1 minus mean pairwise belief distance;
  mean extremity (a secondary confidence measure).
- Confirms H1 if, across rounds, agreement and confidence increase monotonically in every condition (Spearman
  rho > 0, bootstrap CI excluding 0), while accuracy improves only in the complementary conditions and is flat
  or worse in substitutable and consensus-trap. The decoupling is the divergence between the confidence and
  accuracy trajectories.
- Falsifies H1 if confidence rises only where accuracy rises, i.e., agreement tracks correctness.

### H2. Deliberation herds toward an injected confident view, regardless of truth (causal)
- Design. Per instance, deliberate three ways: no confederate (control), a confident wrong confederate, and a
  confident right confederate, injected as a peer each round. The confederate carries a confident position and
  a generic justification with no real information. Script: `scripts/run_confederate.py`.
- Primary outcomes: the herding coefficient (fraction of the way the swarm-mean moves from its control value
  toward the confederate's position), the change in confidence, and the change in Brier.
- Confirms H2 if the herding coefficient is positive with a bootstrap CI excluding 0, confidence rises under
  both confederates, and the wrong confederate raises Brier above control while the right one lowers it.
- Falsifies H2 if the swarm ignores the confederate (herding coefficient not different from 0).

### H3. Deliberation's sign is predictable, and the predictor transfers
- Design. Features from pre-deliberation beliefs only (dispersion, spread, extremity, largest dissent, mean).
  Label: whether deliberation reduced the swarm mean's squared error. Train with leave-one-condition-out
  cross-validation; then train on the full synthetic set and test on data we did not build (the forward test,
  and HiddenBench). Script: `scripts/predict_deliberation.py`.
- Primary outcomes: leave-one-condition-out AUC; transfer AUC; and the Brier of a gated policy (deliberate only
  when help is predicted) against always-deliberate and never-deliberate.
- Confirms H3 if leave-one-condition-out AUC is at least 0.70 with a bootstrap CI above 0.5, transfer AUC is
  above 0.5, and the gated policy's Brier is below both fixed policies.
- Falsifies H3 if AUC is at chance or the gated policy does not beat both fixed policies.

### H4. Frontier models have low effective diversity (monoculture)
- Design. At least eight models across families on a battery of judgment items with known answers. Estimate the
  effective number of independent agents N_eff as the participation ratio of the error-correlation matrix
  eigenvalues, (sum lambda_i)^2 / sum lambda_i^2. Regress aggregation accuracy gain on N_eff and on agent count.
- Confirms H4 if N_eff is far below the model count (single digits), and accuracy gain tracks N_eff rather than
  count (the N_eff coefficient is significant and the count coefficient is not, controlling for the other).
- Falsifies H4 if N_eff is close to the model count.

### H5. The consensus trap makes deliberation strictly harmful (controlled)
- Design. The consensus-trap condition above, with averaging and the no-deliberation prior as baselines.
- Confirms H5 if deliberated Brier exceeds both the averaged Brier and the no-information prior, with the gap
  growing across rounds, while the oracle remains accurate.
- Falsifies H5 if deliberation recovers and discounts the shared misleading cue.

## Sample sizes and power

- Synthetic confirmatory runs: 150 instances per condition, three seeds. This detects a Brier difference of
  about 0.03 at 80 percent power under paired bootstrap given the per-instance variance seen in the pilots.
- Confederate: 100 instances per condition.
- Effective diversity: at least 200 judgment items across at least eight models.
- Pilots (n=16 to 24) are exploratory and labeled as such; they do not enter the confirmatory tests.

## Analysis and corrections

- All point comparisons use the paired bootstrap over instances with 95 percent intervals.
- Monotonicity uses Spearman rho over rounds with bootstrap intervals.
- Across the five hypotheses we report Holm-corrected significance for the primary outcome of each.
- The forward-test analysis is scored only on the leakage-safe future-resolving subset, with the
  forecast-probability exclusion applied as in `scripts/score_forward.py`.

## Exclusions

- Instances where an agent call failed and fell back to the 0.5 default are flagged; primary analysis excludes
  any instance with more than one fallback among its agents.
- Forward-test questions whose retrieved evidence contained an external forecast probability are excluded
  (the Opta-style leakage filter).

## Stopping rules

- The synthetic confirmatory n is fixed at 150 per condition in advance; we do not peek and extend.
- The forward test is scored once at each market's resolution, not re-scored to fish for significance.

## Deviations

Any deviation from this plan is logged in this file with a date and reason before the affected analysis is run.
