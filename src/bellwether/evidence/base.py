"""The ``EvidenceItem`` type, the ``EvidenceSource`` interface, and leakage control."""

from __future__ import annotations

from datetime import date
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from ..questions.base import Question


class EvidenceItem(BaseModel):
    """One piece of evidence shown to the agents (a Slack line, a doc snippet, ...)."""

    text: str
    source: str = "internal"
    dated: Optional[date] = None  # when this evidence existed; used by the leakage guard


@runtime_checkable
class EvidenceSource(Protocol):
    def gather(
        self, question: Question, max_items: int = 10, as_of: Optional[date] = None
    ) -> list[EvidenceItem]: ...


def apply_leakage_guard(
    items: list[EvidenceItem], as_of: Optional[date]
) -> list[EvidenceItem]:
    """Drop any evidence dated after ``as_of`` (the question's issue date).

    This is the single most important guard against fooling ourselves: without it,
    an agent forecasting a past event could "retrieve" the answer. Items with no
    date are kept (assumed background knowledge), so date your time-sensitive
    evidence.
    """
    if as_of is None:
        return items
    return [it for it in items if it.dated is None or it.dated <= as_of]


def get_evidence_source(name: str, **kwargs) -> EvidenceSource:
    """Factory mapping a config string to a concrete evidence source (lazy)."""
    if name in ("none", "", None):
        from .base import _EmptyEvidenceSource

        return _EmptyEvidenceSource()
    if name == "mock_internal":
        from .mock_internal import MockInternalEvidenceSource

        return MockInternalEvidenceSource(**kwargs)
    if name == "web":
        from .web import WebEvidenceSource

        return WebEvidenceSource(**kwargs)
    raise ValueError(f"unknown evidence source: {name!r}")


class _EmptyEvidenceSource:
    """Returns no evidence — useful for a 'reason from priors only' baseline."""

    def gather(self, question, max_items=10, as_of=None) -> list[EvidenceItem]:
        return []
