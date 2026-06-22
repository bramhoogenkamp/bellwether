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


def _complementary(i: int, n_agents: int, rng, target_base_rate: float) -> InfoInstance:
    k = n_agents
    # P(all true) = q^k -> set q for a chosen base rate so the set isn't all-NO.
    q = target_base_rate ** (1.0 / k)
    letters = string.ascii_uppercase[:k]
    statuses = [rng.random() < q for _ in range(k)]
    outcome = 1.0 if all(statuses) else 0.0

    conditions = ", ".join(letters)
    text = (
        f"This resolves YES only if ALL {k} required conditions are met: {conditions}. "
        "You have been privately told the status of only some conditions; treat the "
        "others as unknown."
    )
    slices, pooled = [], []
    for j in range(k):
        state = "COMPLETE" if statuses[j] else "NOT complete"
        item = EvidenceItem(text=f"Condition {letters[j]}: {state}.", source=f"cond_{letters[j]}")
        slices.append([item])
        pooled.append(item)
    qn = Question(
        id=f"syn-comp-{i:04d}",
        text=text,
        outcome=outcome,
        source="synthetic",
        category="complementary",
        metadata={"statuses": statuses, "k": k, "q": q},
    )
    return InfoInstance(qn, slices, pooled, "complementary", {"statuses": statuses})


def generate_info_instances(
    n: int = 50,
    n_agents: int = 4,
    structure: str = "complementary",
    seed: int = 0,
    noise: float = 0.08,
    target_base_rate: float = 0.4,
) -> list[InfoInstance]:
    """Generate ``n`` synthetic instances with one private slice per agent."""
    if structure not in ("substitutable", "complementary"):
        raise ValueError("structure must be 'substitutable' or 'complementary'")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        if structure == "substitutable":
            out.append(_substitutable(i, n_agents, rng, noise))
        else:
            out.append(_complementary(i, n_agents, rng, target_base_rate))
    return out
