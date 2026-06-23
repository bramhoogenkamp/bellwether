# Manufactured Consensus in LLM Agent Collectives

Bellwether research note, v3 (2026-06-23). This supersedes the earlier market-centric framing. The full
experimental design is in [`study.md`](study.md); the bibliography is in [`references.md`](references.md).

## The question

We have well-studied ways to turn many opinions into one good probability: average them, run a prediction
market, or let the group deliberate. All of them inherit a premise from the wisdom-of-crowds literature, that
the members make independent and diverse errors. Independence is what lets a crowd beat its best member, lets a
market aggregate scattered private knowledge, and lets deliberation surface what no single person knew. The
question this study asks is whether that premise survives when the members are language models, and what
happens to the methods when it does not.

## Why it is in doubt

Language models are not independent. Nominally different frontier models make highly correlated errors, so a
panel of nine can carry about two effective votes (Kohli 2026), and aggregation gains show up only when the
models are genuinely different (Schneider and Schramm 2025). A market does not rescue this, because an LMSR
market is a weighted average of its traders' beliefs (Chakraborty and Das 2015; Wolfers and Zitzewitz 2006), so
it cannot add diversity that is not there. And deliberation, the remaining hope, has its own failure mode:
groups of people fail to pool unshared information and instead converge on what they already share (Stasser and
Titus 1985), and LLM debate reproduces this on hidden-profile tasks (HiddenBench, Li, Naito and Shirado 2025).

## What we find, and the claim

Across a controlled testbed with a known-answer oracle, and on a live-market forward test, the same picture
holds. Averaging a swarm of LLMs gains little over a single capable model, because there is little independent
error to cancel. A market adds nothing over the average. Deliberation is the one thing that moves the needle,
but it is double-edged. Round by round it raises inter-agent agreement and stated confidence monotonically,
regardless of whether it is becoming more accurate. When a decisive piece of dispersed information is present,
deliberation surfaces it and accuracy follows. When it is not, deliberation amplifies the agents' shared bias
into confident consensus, and accuracy stalls or falls. On the resolved real-market questions, where the agents
held no decisive private edge, deliberation was the worst aggregator and scored below a coin flip by herding
into overconfident agreement.

The claim, stated plainly: in LLM collectives, deliberation produces agreement and confidence independently of
correctness, so consensus is not a signal of truth. The effect is not random, it tracks the agents' effective
diversity and the presence of a decisive dispersed signal, and it is predictable from a swarm's
pre-deliberation structure.

## The testbed

Each instance has a binary outcome fixed by latent conditions, and each agent privately sees one slice of the
evidence, so the swarm jointly holds the answer but no single agent does. An oracle given all slices sets the
achievable bound. We vary the information structure (redundant, complementary AND/OR/threshold) and the
evidence encoding (clean labels vs messy prose), which lets us place the swarm in regimes where pooling should
help, should not matter, or should backfire, and measure what each method actually recovers. Because the
instances are synthetic, every method is scored immediately against ground truth, with no leakage. A
leakage-controlled forward test on live prediction markets carries the finding to real events.

## Contributions

- A controlled testbed for measuring information aggregation in agent collectives against a ground-truth
  oracle, parameterized by information structure and evidence encoding.
- A measurement of the effective diversity of frontier models, and evidence that aggregation gains scale with
  it rather than with agent count.
- The manufactured-consensus result: deliberation's agreement and confidence decouple from its accuracy, with
  a controlled regime where deliberation strictly hurts.
- A predictor that decides, before deliberating, whether deliberation will help, and a gating policy that beats
  always-deliberate and never-deliberate.

## What this reframes

The project began as a test of whether a market of LLM agents beats averaging. The answer, that it does not,
is a clean negative result and it is now one experiment in service of the larger point: the machinery of
collective intelligence behaves differently for correlated agents, and the interesting object is not the
market but the way deliberation manufactures consensus. The earlier experiments are the base of that argument;
[`study.md`](study.md) lays out how they fit and what remains to run.
