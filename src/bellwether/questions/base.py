"""The ``Question`` type and the ``QuestionSource`` interface.

A Question is deliberately source-agnostic: a mock internal question, a
ForecastBench item, and a Manifold market all become the same object. Only binary
(YES/NO) questions are supported in v1 — that keeps the market and scoring simple,
and covers the large majority of useful forecasting questions.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Question(BaseModel):
    """A single binary forecasting question."""

    id: str
    text: str
    background: str = ""
    resolution_criteria: str = ""

    # Dates matter for leakage control: evidence must be restricted to on/before
    # ``issue_date`` so an agent can't "forecast" using information from after the
    # question was asked.
    issue_date: Optional[date] = None
    resolution_date: Optional[date] = None

    # Ground truth: 1.0 = YES happened, 0.0 = NO. None while the question is open.
    outcome: Optional[float] = None

    category: str = "general"
    source: str = "unknown"

    # Optional baselines carried alongside the question, used by conditions F/G:
    market_prob: Optional[float] = None          # a public market's probability
    superforecaster_prob: Optional[float] = None # ForecastBench human baseline (F)
    status_quo_prob: Optional[float] = None       # company's current process (G)

    # Free-form extras (e.g. hidden signal for mock questions).
    metadata: dict = Field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.outcome is not None


@runtime_checkable
class QuestionSource(Protocol):
    """Anything that can hand us a batch of questions."""

    def fetch(self, limit: int = 20) -> list[Question]: ...


def get_question_source(name: str, **kwargs) -> QuestionSource:
    """Factory: map a config string to a concrete source (lazy imports).

    Network-backed sources are imported only when asked for, so the offline path
    (mock_internal) never needs httpx / forecasting-tools installed.
    """
    if name == "mock_internal":
        from .mock_internal import MockInternalQuestionSource

        return MockInternalQuestionSource(**kwargs)
    if name == "manifold":
        from .manifold import ManifoldQuestionSource

        return ManifoldQuestionSource(**kwargs)
    if name == "polymarket":
        from .polymarket import PolymarketQuestionSource

        return PolymarketQuestionSource(**kwargs)
    if name == "forecastbench":
        from .forecastbench import ForecastBenchQuestionSource

        return ForecastBenchQuestionSource(**kwargs)
    raise ValueError(f"unknown question source: {name!r}")
