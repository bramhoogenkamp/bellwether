# Agreement Is Not Evidence: Manufactured Consensus in LLM Agent Collectives

Bellwether study design, v1 (2026-06-23). Master plan for a publishable study. This reframes the project around
its strongest finding and keeps every experiment we have already run as the empirical base.

## Thesis

The methods we use to turn many opinions into one good probability, averaging, prediction markets, and
deliberation, were built for human crowds and rest on one assumption: that members make independent, diverse
errors. That independence is what lets a crowd beat its best member. Frontier language models violate it. They
are near-clones, with error correlation around 0.8 to 0.98 and an effective number of independent agents far
below their headcount. Two consequences follow, and they are the subject of this study. First, averaging a
swarm of LLMs gains little, because there is little independent error to cancel. Second, multi-round
deliberation does not pool dispersed information the way it does for people. It amplifies the shared prior,
raising inter-agent agreement and stated confidence monotonically with each round, whether or not it improves
accuracy. Deliberation helps only when a genuine, decisive piece of dispersed information is present for it to
surface, and it hurts otherwise, because it manufactures confident consensus around the agents' common bias.

The single sentence: in LLM collectives, agreement and confidence are produced by deliberation independently of
correctness, so consensus is not evidence, and we can predict from a swarm's pre-deliberation structure whether
deliberation will help or hurt.

## Why this is significant and not incremental

The prevailing narrative in multi-agent LLM work is that adding agents and letting them debate improves
answers. The defensible, contrarian claim available from our data is the opposite, with a mechanism and a
remedy attached. The contribution is not "debate can sometimes hurt," which is folklore. It is (1) a controlled
demonstration, against a known-answer oracle and across a parameterized space of information structures, that
deliberation's confidence and agreement decouple from its accuracy; (2) a measurement of the effective
diversity of frontier models that explains why; and (3) a predictor that decides ex ante when deliberation is
safe, recovering its upside while avoiding its failures. The societal corollary, that delegating many
independent decisions to a few correlated models removes the error-diversification independent humans
provided, connects the finding to the algorithmic-monoculture literature.

## Contributions

- C1. A controlled testbed for information aggregation in agent collectives, with synthetic
  dispersed-information tasks that have a known ground-truth oracle and are parameterized by information
  structure (redundant, complementary AND/OR/threshold) and by evidence encoding (clean labels vs noisy
  prose). This lets us measure how much recoverable information an aggregation method actually recovers, which
  is impossible without ground truth.
- C2. A measurement of the effective diversity of frontier LLMs (an effective-number-of-independent-agents
  statistic from the error-correlation structure), and evidence that aggregation gains scale with effective
  diversity, not agent count.
- C3. The manufactured-consensus result: across rounds of deliberation, inter-agent agreement and stated
  confidence rise monotonically and independently of accuracy; deliberation improves accuracy only when a
  decisive dispersed signal exists, and degrades it in the redundant and shared-bias regimes, including a
  controlled "consensus trap" and a resolved real-market case where it underperforms a uniform baseline.
- C4. A predictor of deliberation's sign from pre-deliberation features (belief dispersion, effective
  diversity, evidence overlap, confidence spread), validated out of sample, and a gating policy ("deliberate
  only when predicted to help") that beats both always-deliberate and never-deliberate.
- C5. External validity: a leakage-controlled forward test on live prediction markets, and a replication of
  the pattern on the third-party HiddenBench hidden-profile benchmark.

## Related work and positioning

The full annotated bibliography is in [`references.md`](references.md). The positioning:

- Wisdom of crowds and its assumptions. Condorcet's jury theorem and Hong and Page (2004) make collective
  accuracy depend on independent, diverse members. We test whether that premise holds for LLMs and show it
  largely does not.
- Markets as opinion pools. An LMSR market is a logarithmic opinion pool (Chakraborty and Das 2015) and the
  price is a weighted average belief (Wolfers and Zitzewitz 2006), so a market is not categorically different
  from averaging. Our market experiments confirm this empirically for LLM agents and let us set the market
  aside as a mechanism.
- Hidden profiles. Group decision-making famously fails to pool unshared information (Stasser and Titus 1985);
  HiddenBench (Li, Naito and Shirado 2025) reports the same failure for LLM debate. We extend this from a
  binary success/failure to a graded, predictable phenomenon with a confidence/accuracy decoupling and a
  controlled trap.
- Multi-agent debate and mixture-of-agents. The debate line (Du et al. 2023 and successors) argues interaction
  improves reasoning. We give controlled conditions under which it does the reverse and a rule to tell them
  apart.
- Effective diversity. Kohli (2026) shows nine LLM judges carry about two effective votes, and Schneider and
  Schramm (2025) find aggregation gains only for diverse models. We turn this into a measurement tied to a
  ground-truth aggregation outcome.
- Algorithmic monoculture. Kleinberg and Raghavan (2021) and the homogenization concern (Bommasani et al.
  2022) frame correlated algorithmic decisions as systemic risk. Our effective-diversity numbers quantify it
  for LLM judgment.
- Closest LLM-market predecessor. Galanis (2026) builds an LLM LMSR market with private signals and measures
  market-versus-truth, but not against averaging or an oracle and without varying signal structure or studying
  deliberation dynamics.

## The testbed (C1)

A task instance has a binary outcome y determined by k latent conditions and an aggregation rule. Each of n
agents privately sees one slice of the evidence. The oracle is an agent given all slices at once, and its score
is the achievable upper bound for that instance. We score every method by Brier loss against y and by gap to
the oracle. The controlled axes:

- Information structure. redundant/substitutable (each slice is a noisy estimate of the same quantity, so
  averaging is near-optimal and there is nothing to pool); complementary AND (the outcome needs every condition,
  so one failed condition is decisive); complementary OR (one satisfied condition is decisive); complementary
  threshold (a majority of conditions, so no single agent is decisive).
- Evidence encoding. clean labeled conditions vs messy natural-language snippets that must be interpreted. The
  prose version makes interpretation, not transmission, the binding step, and it lowers even the oracle below
  perfect.
- A new structure for this study, the consensus trap, defined under D2.

The generator and oracle already exist in `src/bellwether/questions/synthetic.py` and the swarm in
`src/bellwether/agents/`.

## Experiment suite

### Part A: what is already done, reframed

These ran during the project and form the base. They need rescaling and re-analysis under the new framing, not
re-invention. Mapping:

| Done | What it showed | Claim it supports |
|---|---|---|
| [01 information aggregation](../experiments/info-aggregation/01-information-aggregation.md) | market ties average across structures; oracle near-perfect on complementary; only deliberation pools | C1 testbed; deliberation, not the market, is the operative mechanism |
| [02 market-design probe](../experiments/info-aggregation/02-market-design-probe.md) | capped, uncapped, and equilibrium-pool markets all tie the average | market mechanism adds nothing; set it aside |
| [03 rationale-augmented market](../experiments/info-aggregation/03-rationale-market.md) | a language channel, not the price, is what lets the decisive fact in | reframes the result as being about deliberation |
| [04 deliberation depth](../experiments/info-aggregation/04-deliberation-depth.md) | each round halves error toward the oracle when a decisive signal exists | C3, the "helps when signal present" half |
| [05 agent framing](../experiments/info-aggregation/05-agent-framing.md) | trader framing does not help, slightly worsens calibration | robustness/ablation |
| [06 diversity sweep](../experiments/info-aggregation/07-diversity.md) | a diverse swarm deliberates to 0.097, a clone swarm of the strongest model stalls at 0.166 | C2, effective diversity is the lever |
| [07 forward test, resolved](loop/live-experiments-ledger.md) | crowd wins; deliberation worst and below the coin-flip line via overconfident herding | C3 in the wild; external validity |

### Part B: new experiments needed for the paper

Each lists the question, design, the metric, the result that would confirm it, and what would falsify it.

#### M1. Effective diversity of frontier models (C2)
- Question. How many effectively independent agents does a frontier-model ensemble contain, and does aggregation
  gain track that rather than headcount?
- Design. A zoo of at least eight models spanning families (GPT, Claude, Gemini, DeepSeek, Llama, Qwen,
  Mistral, and one or two more). Run them on a battery of judgment items with known answers. Form the
  per-item error vectors, estimate the error-correlation matrix, and compute the effective number of agents as
  the participation ratio of its eigenvalues, N_eff = (sum lambda_i)^2 / sum lambda_i^2. Then sweep ensembles of
  increasing nominal size and of increasing diversity and regress aggregation accuracy gain on N_eff vs on n.
- Metric. N_eff; slope of accuracy gain against N_eff vs against n.
- Confirms. N_eff is far below n (single digits even for large ensembles), and gain tracks N_eff, not n.
- Falsifies. N_eff close to n, or gain that tracks headcount, would mean LLMs are diverse enough that the
  classical results hold.

#### D1. The confidence-accuracy decoupling (C3, the headline)
- Question. Over rounds of deliberation, how do accuracy, inter-agent agreement, and stated confidence move,
  and do they move together?
- Design. For each information structure (redundant, complementary-decisive, consensus-trap) and each diversity
  level, run K deliberation rounds and log, per round: Brier against the outcome, agreement (one minus mean
  pairwise belief distance), confidence (mean distance of beliefs from 0.5), and calibration error.
- Metric. The three trajectories and their correlations across conditions.
- Confirms. Agreement and confidence rise monotonically in every condition; accuracy improves only in
  complementary-decisive; calibration worsens wherever confidence rises without accuracy. This decoupling is
  the central figure.
- Falsifies. Confidence that rises only when accuracy rises would mean agreement is a valid signal of
  correctness.

#### D2. The consensus trap, a controlled backfire (C3)
- Question. Can we make deliberation strictly worse than no deliberation under controlled conditions, matching
  the real-market backfire?
- Design. A new instance type where each agent gets a weak, independent, correct private signal plus a salient
  shared misleading cue, while the corrective information needed to discount the cue is genuinely dispersed.
  Averaging preserves the independent signal; deliberation should converge on the shared cue.
- Metric. Brier of average vs deliberated vs the no-information prior, and the rate at which the swarm converges
  to the shared cue.
- Confirms. Deliberation underperforms both averaging and the prior, and convergence to the cue increases with
  rounds.
- Falsifies. If deliberation recovers and discounts the shared cue, the manufactured-consensus claim is too
  strong.

#### P1. Predicting deliberation's sign (C4, the method)
- Question. From features available before deliberating, can we predict whether deliberation will raise or lower
  accuracy?
- Design. Features per instance: pre-deliberation belief dispersion, swarm N_eff, evidence overlap between
  agents, confidence spread, and a cheap "is there a dissenting minority" indicator. Label: the sign of the
  change in Brier from deliberation. Train an interpretable classifier with leave-one-structure-out and
  leave-one-dataset-out validation. Then evaluate a gating policy that deliberates only when help is predicted.
- Metric. Out-of-sample AUC and the Brier of the gated policy vs always-deliberate and never-deliberate.
- Confirms. AUC well above chance and a gated policy that dominates both fixed policies.
- Falsifies. Near-chance AUC would mean the failure is not predictable from pre-deliberation structure.

#### X. External validity (C5)
- X1, forward test. The live-market forward test, leakage-controlled and pre-registered, scored on resolution.
  The sports subset (resolved) already shows the manufactured-consensus pattern; the analyzable non-sports
  subset is logged and pre-registered for scoring as it resolves. See the ledger and `scripts/run_forward.py`,
  `scripts/score_forward.py`.
- X2, HiddenBench replication. Run our swarm, deliberation, and the predictor on the third-party HiddenBench
  hidden-profile tasks, to connect to prior work and show the predictor transfers to a dataset we did not build.

### Scaling and robustness (required for review)
- Sample size. Raise synthetic instances per cell from 24 to at least 150, with multiple seeds, and report
  paired-bootstrap confidence intervals on every comparison.
- Ablations. Temperature; prompt-induced diversity vs model-family diversity (does varying personas of one
  model substitute for varying models, given M1); number of agents; number of rounds; the extremizing
  coefficient.
- Model drift. Pin model versions and dates; rerun the headline on a second model generation to show the
  finding is not version-specific.

## Metrics and statistics

- Accuracy: Brier loss and Brier skill score against the uniform and base-rate baselines; gap to the oracle.
- Calibration: expected calibration error and a reliability curve; overconfidence as mean confidence minus
  accuracy.
- Agreement: one minus mean pairwise absolute belief distance, per round.
- Effective diversity: N_eff as the participation ratio defined in M1.
- Inference: paired bootstrap over instances for all deltas; pre-registration of the forward-test analysis;
  a power calculation to justify the per-cell sample size.

## Threats to validity and mitigations

- Prompt sensitivity. Report the ablation over prompt-induced diversity and fix prompts across conditions so
  only the studied variable moves.
- Model versions and drift. Pin versions; replicate the headline on a second generation.
- Synthetic artificiality. The unstructured-prose encoding and the forward test bridge to realistic evidence;
  the synthetic control is what buys the ground-truth oracle.
- Oracle as upper bound. The oracle is the same model class given pooled evidence, so it bounds what the swarm
  could in principle reach; we report gap-to-oracle, not just raw accuracy.
- Leakage. Forward questions resolve in the future, retrieval strips odds and external forecast probabilities,
  and retrieved evidence is logged and audited. The future-resolving subset is the leak-safe headline.

## Reproducibility

Every run is a typed config logged to MLflow, with seeds. The generator, swarm, market, scoring, and the
forward-test and scoring scripts are in the repo. Each experiment note carries its exact reproduce command.

## Target venues and framing

- Primary: COLM or a main ML venue (NeurIPS, ICLR). Contribution type is an empirical finding plus a controlled
  testbed plus a predictor method, which fits a main-track empirical paper.
- Strong alternative: FAccT or AIES for the monoculture and delegated-decision framing, where the correlated-
  judgment risk is the lead.
- The Collective Intelligence conference is the thematic home if we lead with the social-epistemology angle.

## Roadmap

1. Done: the testbed, the seven base experiments, the forward test and its first resolution.
2. Next, cheap and on the existing harness: D1 (decoupling trajectories), D2 (consensus trap), P1 (predictor).
   These need no new infrastructure, only the new conditions and logging, and they carry the paper.
3. Then, needs a model zoo and budget: M1 (effective diversity at scale) and the scaling/ablations.
4. Ongoing: X1 forward-test resolutions; X2 HiddenBench replication.
