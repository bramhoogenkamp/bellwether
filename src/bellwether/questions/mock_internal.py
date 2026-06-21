"""A deterministic, offline set of internal-style questions with known outcomes.

This is our test fixture for the whole pipeline: it lets us run end-to-end (swarm
-> market -> scoring) with no API calls and no network, and it gives us a clean
*ground truth* to score against.

How it works: each question gets a hidden "true probability" ``p_true``. The
outcome is drawn once from Bernoulli(p_true) with a fixed seed, so the data is
reproducible and *calibrated by construction* — meaning the best achievable
forecast is p_true itself. The matching evidence source (evidence/mock_internal)
reveals noisy reads of p_true, so a good swarm should recover something close to
p_true and beat the deliberately-biased ``status_quo_prob``.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .base import Question

# (topic, question template) — kept generic so the set reads like a real backlog.
_TEMPLATES = [
    ("engineering", "Will feature {n} ship by the end of the sprint?"),
    ("sales", "Will deal {n} close this quarter?"),
    ("hiring", "Will the role {n} be filled within 30 days?"),
    ("reliability", "Will service {n} stay above its SLA this month?"),
    ("product", "Will launch {n} hit its target adoption in week one?"),
    ("finance", "Will cost center {n} come in under budget this quarter?"),
]


class MockInternalQuestionSource:
    """Generates a fixed, reproducible set of resolved internal questions."""

    def __init__(self, n: int = 24, seed: int = 0):
        self.n = n
        self.seed = seed

    def fetch(self, limit: int = 20) -> list[Question]:
        rng = np.random.default_rng(self.seed)
        questions: list[Question] = []
        for i in range(min(limit, self.n)):
            topic, template = _TEMPLATES[i % len(_TEMPLATES)]
            p_true = float(rng.uniform(0.15, 0.85))
            outcome = 1.0 if rng.random() < p_true else 0.0

            # Status quo (condition G): a coarse, optimism-biased human estimate —
            # rounded to the nearest 25% and nudged up. This is what the swarm/market
            # should be able to beat.
            status_quo = float(np.clip(round(p_true * 4) / 4 + 0.1, 0.05, 0.95))

            questions.append(
                Question(
                    id=f"mock-{i:03d}",
                    text=template.format(n=i + 1),
                    background=f"Internal {topic} question generated for offline testing.",
                    resolution_criteria="Resolves YES if the stated outcome occurs by the resolution date.",
                    issue_date=date(2026, 1, 1),
                    resolution_date=date(2026, 3, 31),
                    outcome=outcome,
                    category=topic,
                    source="mock_internal",
                    status_quo_prob=status_quo,
                    metadata={"signal_truth": p_true},
                )
            )
        return questions
