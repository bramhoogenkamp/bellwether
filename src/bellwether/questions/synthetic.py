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
    if structure not in ("substitutable", "complementary"):
        raise ValueError("structure must be 'substitutable' or 'complementary'")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if structure == "substitutable":
            out.append(_substitutable(i, n_agents, rng, noise))
        else:
            out.append(_complementary(i, n_agents, rng, target_base_rate, rule))
    return out
