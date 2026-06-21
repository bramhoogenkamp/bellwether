"""Public web-search evidence (stub).

Wire this to a search API (Tavily, SerpAPI, Brave, or a provider's web tool) when
you want agents to pull public information. Two things matter here:

  * it's a *commodity* signal — every agent (and competitor) can get it, so it adds
    freshness but little differentiation; the real edge is internal data.
  * the leakage guard MUST be honoured: when forecasting a past event for a
    backtest, restrict results to ``as_of`` (results dated on/before issue date),
    otherwise you retrieve the answer. Date-filtered web search is known to leak,
    so prefer a frozen corpus for offline scoring.

Left as a stub so the offline path never depends on a search provider.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from ..questions.base import Question
from .base import EvidenceItem


class WebEvidenceSource:
    def __init__(self, provider: str = "none"):
        self.provider = provider

    def gather(
        self, question: Question, max_items: int = 10, as_of: Optional[date] = None
    ) -> list[EvidenceItem]:
        raise NotImplementedError(
            "WebEvidenceSource is a stub. Wire a search provider and enforce the "
            "as_of date filter before using it for offline backtests."
        )
