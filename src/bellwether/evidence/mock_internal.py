"""Deterministic mock internal evidence — the offline stand-in for a real connector.

For a mock question carrying a hidden ``signal_truth`` (its true probability), this
produces a handful of internal-style snippets that each reveal a *noisy read* of
that signal, with the number embedded in the text (e.g. "...estimate ~0.72..."). A
real LLM reads the sentences and reasons; our offline ``FakeLLM`` parses the
numbers. Either way the agents get a genuine-but-noisy signal to forecast from.

This deliberately mimics how internal data behaves: scattered, individually noisy
observations of an outcome that is mostly already determined but not yet visible.
All items are dated on/before the question's issue date so the leakage guard is a
no-op here (nothing leaks from the future).
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Optional

import numpy as np

from ..questions.base import Question
from .base import EvidenceItem

_SOURCES = [
    ("eng lead standup note", "On-time confidence sits around ~{p:.2f}."),
    ("Jira burndown", "Remaining-work trend implies roughly {p:.2f} chance of hitting the date."),
    ("Slack #status", "Team sentiment this week reads about {p:.2f}."),
    ("manager 1:1 note", "PM puts the odds near {p:.2f}."),
    ("dashboard snapshot", "Leading indicator currently maps to ~{p:.2f}."),
    ("retro action item", "Historical base rate for this kind of item is about {p:.2f}."),
]


def _seed_for(question: Question) -> int:
    h = hashlib.md5(question.id.encode()).hexdigest()
    return int(h[:8], 16)


class MockInternalEvidenceSource:
    """Generates noisy, dated internal snippets from a question's hidden signal."""

    def __init__(self, noise: float = 0.08, seed: int = 0):
        self.noise = noise
        self.seed = seed

    def gather(
        self, question: Question, max_items: int = 10, as_of: Optional[date] = None
    ) -> list[EvidenceItem]:
        truth = float(question.metadata.get("signal_truth", 0.5))
        rng = np.random.default_rng(_seed_for(question) ^ self.seed)
        issued = question.issue_date or date(2026, 1, 1)

        items: list[EvidenceItem] = []
        for i in range(min(max_items, len(_SOURCES))):
            src, template = _SOURCES[i]
            p = float(np.clip(truth + rng.normal(0, self.noise), 0.02, 0.98))
            items.append(
                EvidenceItem(
                    text=template.format(p=p),
                    source=src,
                    # spread the snippets over the two weeks before issue date
                    dated=issued - timedelta(days=int(rng.integers(1, 14))),
                )
            )
        return items
