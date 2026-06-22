"""Web-research evidence via OpenRouter's built-in web search (`:online` models).

This gives the swarm *real, current information* with no extra API key — it reuses
the OpenRouter key. We do ONE research call per question (a shared brief), rather
than letting every agent search independently: cheaper, and it matches the
research finding that a single good research pass + independent reasoning beats
per-agent search.

HONESTY — two distinct leaks to avoid:
  * Outcome leak: `:online` cannot be restricted to a past date, so it WILL find the
    answer to an already-resolved question. Only use this for OPEN questions.
  * Market-price leak (the subtle one): for a question that already has a public
    market / betting line, search returns the odds and the agents just parrot them.
    "Matching the market" then proves nothing. Set ``exclude_markets=True`` to make
    the research gather only PRIMARY signal (form, news, fundamentals) and to filter
    out any odds / bookmaker / prediction-market figures — so a match with the crowd
    reflects real reasoning, not retrieval of the answer's price.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from ..questions.base import Question
from .base import EvidenceItem

_SYSTEM = (
    "You are a research assistant. Use web search to gather current, factual, dated "
    "information that bears on a forecasting question. Return a concise bullet list of "
    "findings only — do NOT state a probability or a final answer."
)

# Stricter brief: primary fundamentals only, no market/odds figures.
_SYSTEM_PRIMARY = (
    "You are a research assistant. Use web search to gather current, factual, dated "
    "PRIMARY information for a forecasting question: recent results/events, form, "
    "injuries or availability, schedule/draw/structure, and expert analysis of "
    "fundamentals. STRICTLY EXCLUDE betting odds, bookmaker prices, prediction-market "
    "prices (e.g. Polymarket, Kalshi), and any implied-probability or odds-aggregator "
    "figures. Do NOT state any probability. Return a concise bullet list of findings only."
)

# Backstop filter: drop any finding that smuggles in a market/odds figure.
_MARKET_RE = re.compile(
    r"\bodds\b|implied|bookmaker|book\s?maker|polymarket|kalshi|betting|oddschecker|"
    r"moneyline|vegas|\+\d{3,}|\b\d{1,3}/\d{1,2}\b",
    re.I,
)


class WebEvidenceSource:
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini:online",
        client=None,
        max_items: int = 6,
        exclude_markets: bool = False,
    ):
        self.model = model
        self._client = client
        self.max_items = max_items
        self.exclude_markets = exclude_markets

    def _client_or_default(self):
        if self._client is None:
            from ..agents.llm import LiteLLMClient

            self._client = LiteLLMClient()
        return self._client

    def gather(
        self, question: Question, max_items: Optional[int] = None, as_of: Optional[date] = None
    ) -> list[EvidenceItem]:
        n = max_items or self.max_items
        system = _SYSTEM_PRIMARY if self.exclude_markets else _SYSTEM
        user = (
            f"Question: {question.text}\n\n"
            f"List up to {n} concise, factual findings (include a source or date where "
            f"possible) that help estimate whether this resolves YES. One finding per "
            f"line, each starting with '- '."
        )
        raw = self._client_or_default().complete(
            model=self.model, system=system, user=user, temperature=0
        )

        items: list[EvidenceItem] = []
        for line in (raw or "").splitlines():
            line = line.strip().lstrip("-*•").strip()
            if len(line) <= 8:
                continue
            if self.exclude_markets and _MARKET_RE.search(line):
                continue  # backstop: discard any odds/market figure that slipped through
            items.append(EvidenceItem(text=line[:400], source="web", dated=as_of))
            if len(items) >= n:
                break
        return items
