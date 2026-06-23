"""Synthetic dispersed-private-information instances (the novel experiment's core).

Each instance has a KNOWN ground-truth outcome and splits the evidence into private
"slices", one per agent. This lets us score market vs. average vs. pooled-oracle on
identical instances, immediately, with no leakage and no waiting (see research/intro.md).

Two signal structures — the key independent variable:

* SUBSTITUTABLE: each agent's slice is a redundant noisy estimate of the same latent
  probability. Averaging the agents' forecasts is near-optimal, so we expect
  market ≈ average (both approach the oracle).

* COMPLEMENTARY (hidden-profile / conjunction): the event resolves YES only if ALL of
  k conditions hold, and each agent privately knows the status of just one condition.
  No single agent can know the outcome from its slice; an agent who sees a FAILED
  condition knows the answer is NO with certainty. Averaging individual forecasts
  cannot recover this; a market — where the agent holding the decisive piece can move
  the price — should. This is where we expect market >> average, toward the oracle.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass, field

import numpy as np

from ..evidence.base import EvidenceItem
from .base import Question


@dataclass
class InfoInstance:
    question: Question                       # carries the known ground-truth outcome
    slices: list[list[EvidenceItem]]         # private evidence, one list per agent
    pooled: list[EvidenceItem]               # all slices combined (for the oracle)
    structure: str                           # "substitutable" | "complementary"
    meta: dict = field(default_factory=dict)


def _substitutable(i: int, n_agents: int, rng, noise: float) -> InfoInstance:
    p_true = float(rng.uniform(0.2, 0.8))
    outcome = 1.0 if rng.random() < p_true else 0.0
    slices, pooled = [], []
    for j in range(n_agents):
        est = float(np.clip(p_true + rng.normal(0, noise), 0.02, 0.98))
        item = EvidenceItem(
            text=f"Independent indicator {j + 1} suggests about {est:.2f} likelihood.",
            source=f"indicator_{j + 1}",
        )
        slices.append([item])
        pooled.append(item)
    q = Question(
        id=f"syn-sub-{i:04d}",
        text="Will the event occur? Several independent indicators each give a noisy "
        "estimate of its likelihood.",
        outcome=outcome,
        source="synthetic",
        category="substitutable",
        metadata={"p_true": p_true},
    )
    return InfoInstance(q, slices, pooled, "substitutable", {"p_true": p_true})


def _complementary(
    i: int, n_agents: int, rng, target_base_rate: float, rule: str = "and"
) -> InfoInstance:
    """Hidden-profile instance. ``rule`` sets the aggregation function — the axis that
    determines whether any single agent holds decisive information:

      * "and"       — YES iff ALL conditions hold (decisive piece = a single NO)
      * "or"        — YES iff ANY condition holds (decisive piece = a single YES)
      * "threshold" — YES iff a majority hold (NO single agent is decisive — hardest)
    """
    k = n_agents
    letters = string.ascii_uppercase[:k]
    if rule == "and":
        q = target_base_rate ** (1.0 / k)
        threshold = k
        rule_text = f"resolves YES only if ALL {k} required conditions are met"
    elif rule == "or":
        q = 1.0 - (1.0 - target_base_rate) ** (1.0 / k)
        threshold = 1
        rule_text = f"resolves YES if AT LEAST ONE of the {k} conditions is met"
    elif rule == "threshold":
        q = 0.5
        threshold = k // 2 + 1
        rule_text = f"resolves YES if AT LEAST {threshold} of the {k} conditions are met"
    else:
        raise ValueError(f"unknown rule {rule!r}")

    statuses = [rng.random() < q for _ in range(k)]
    outcome = 1.0 if sum(statuses) >= threshold else 0.0

    text = (
        f"This {rule_text}: {', '.join(letters)}. You have been privately told the "
        "status of only some conditions; treat the others as unknown."
    )
    slices, pooled = [], []
    for j in range(k):
        state = "COMPLETE" if statuses[j] else "NOT complete"
        item = EvidenceItem(text=f"Condition {letters[j]}: {state}.", source=f"cond_{letters[j]}")
        slices.append([item])
        pooled.append(item)
    qn = Question(
        id=f"syn-{rule}-{i:04d}",
        text=text,
        outcome=outcome,
        source="synthetic",
        category=f"complementary-{rule}",
        metadata={"statuses": statuses, "k": k, "rule": rule, "threshold": threshold},
    )
    return InfoInstance(qn, slices, pooled, "complementary", {"statuses": statuses, "rule": rule})


# Unstructured variant: each condition's latent state is rendered as a messy
# natural-language snippet that implies "ready" or "not ready" indirectly, so the
# agent must interpret prose rather than read a labeled field. The latent states still
# determine the outcome via the rule, so ground truth and the oracle are preserved.
_COMPONENTS = [
    "the payments service", "the checkout flow", "the data pipeline",
    "the auth migration", "the onboarding redesign", "the search index", "the billing module",
]
_READY = [
    "Heard from the {c} team in standup, they wrapped it up and QA signed off.",
    "{c} shipped to production last week and has been stable since.",
    "Caught the {c} lead at lunch, she said it is finished and running clean.",
    "Saw the release notes, {c} made it in and the smoke tests passed.",
]
_NOT_READY = [
    "{c} is still blocked on a dependency and got pushed to next sprint.",
    "There was a thread about {c}; it is only half built and the owner is out sick.",
    "{c} failed its last review, so the team is reworking a big chunk of it.",
    "Heard {c} is behind; the demo last week did not work.",
]


def _unstructured(i: int, n_agents: int, rng, target_base_rate: float, rule: str = "and") -> InfoInstance:
    k = n_agents
    comps = list(rng.choice(_COMPONENTS, size=k, replace=False))
    if rule == "and":
        q = target_base_rate ** (1.0 / k)
        threshold = k
        rule_text = f"ships only if ALL {k} of these are production-ready"
    elif rule == "or":
        q = 1.0 - (1.0 - target_base_rate) ** (1.0 / k)
        threshold = 1
        rule_text = "ships if AT LEAST ONE of these is production-ready"
    elif rule == "threshold":
        q = 0.5
        threshold = k // 2 + 1
        rule_text = f"ships if at least {threshold} of these are production-ready"
    else:
        raise ValueError(f"unknown rule {rule!r}")

    statuses = [rng.random() < q for _ in range(k)]
    outcome = 1.0 if sum(statuses) >= threshold else 0.0
    text = (
        f"A launch {rule_text}: {', '.join(comps)}. You have each heard about some of them "
        "through different channels; judge from what you heard."
    )
    slices, pooled = [], []
    for j in range(k):
        pool = _READY if statuses[j] else _NOT_READY
        snippet = pool[int(rng.integers(len(pool)))].format(c=comps[j])
        item = EvidenceItem(text=snippet, source="hearsay")
        slices.append([item])
        pooled.append(item)
    qn = Question(
        id=f"syn-unstr-{rule}-{i:04d}",
        text=text,
        outcome=outcome,
        source="synthetic",
        category=f"unstructured-{rule}",
        metadata={"statuses": statuses, "k": k, "rule": rule, "threshold": threshold, "components": comps},
    )
    return InfoInstance(qn, slices, pooled, "unstructured", {"statuses": statuses, "rule": rule})


# Consensus trap (the controlled backfire). Every agent sees the same loud, salient,
# misleading headline pointing the wrong way, plus one quiet private detail pointing the
# right way. A single agent anchored on the shared headline leans wrong; the pooled
# private details outweigh the headline, so the oracle is right. Deliberation converges
# on the common ground (the shared headline), so it should herd the swarm toward the
# wrong answer. This isolates manufactured consensus from genuine pooling.
_TRAP_DETAILS = [
    "a line item in the appendix",
    "a timestamp on the internal log",
    "a footnote in the supplier contract",
    "a quiet remark from the team lead",
    "a small discrepancy in the figures",
    "an offhand note in the changelog",
    "a single dissenting reviewer comment",
]


def _consensus_trap(i: int, n_agents: int, rng) -> InfoInstance:
    k = n_agents
    y = 1.0 if rng.random() < 0.5 else 0.0
    correct = "YES" if y == 1.0 else "NO"
    wrong = "NO" if y == 1.0 else "YES"
    text = (
        "Will the project clear its review? You have each seen the same public summary "
        "plus one private detail of your own. Weigh them and give your probability."
    )
    headline = EvidenceItem(
        text=(f"PUBLIC SUMMARY (widely circulated, prominent): essentially every headline "
              f"indicator points to the answer being {wrong}."),
        source="shared-headline",
    )
    details = list(rng.choice(_TRAP_DETAILS, size=k, replace=False))
    slices, pooled = [], [headline]
    for j in range(k):
        hint = EvidenceItem(
            text=f"Your private detail: {details[j]} quietly but specifically indicates {correct}.",
            source="private-detail",
        )
        slices.append([headline, hint])
        pooled.append(hint)
    qn = Question(
        id=f"syn-trap-{i:04d}", text=text, outcome=y, source="synthetic",
        category="consensus-trap",
        metadata={"k": k, "shared_points": wrong, "private_points": correct},
    )
    return InfoInstance(qn, slices, pooled, "consensus-trap", {"y": y})


def generate_info_instances(
    n: int = 50,
    n_agents: int = 4,
    structure: str = "complementary",
    seed: int = 0,
    noise: float = 0.08,
    target_base_rate: float = 0.4,
    rule: str = "and",
) -> list[InfoInstance]:
    """Generate ``n`` synthetic instances with one private slice per agent.

    ``structure`` = "substitutable" | "complementary"; for complementary, ``rule`` =
    "and" | "or" | "threshold" sets the aggregation function (the key categorization).
    """
    if structure not in ("substitutable", "complementary", "unstructured", "consensus-trap"):
        raise ValueError("structure must be 'substitutable', 'complementary', 'unstructured', or 'consensus-trap'")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if structure == "substitutable":
            out.append(_substitutable(i, n_agents, rng, noise))
        elif structure == "unstructured":
            out.append(_unstructured(i, n_agents, rng, target_base_rate, rule))
        elif structure == "consensus-trap":
            out.append(_consensus_trap(i, n_agents, rng))
        else:
            out.append(_complementary(i, n_agents, rng, target_base_rate, rule))
    return out
