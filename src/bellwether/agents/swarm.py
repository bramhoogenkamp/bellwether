"""The swarm: a small set of independent agents whose forecasts we later aggregate.

Diversity is engineered the way the research says actually works: by mixing model
families and a couple of evidence-backed lenses — NOT by role-play personas or
debate. Agents run independently (no debate, no seeing each other's answers); the
aggregation step is where their views combine.

We deliberately keep the swarm small (5-10). Accuracy comes from a few good,
diverse forecasters, not from headcount.
"""

from __future__ import annotations

import sys

from ..config import Lens, SwarmConfig
from ..evidence.base import EvidenceItem
from ..questions.base import Question
from .agent import Agent, Forecast
from .llm import LLMClient


class Swarm:
    def __init__(self, cfg: SwarmConfig, client: LLMClient):
        self.cfg = cfg
        self.client = client
        self.agents = self._build_agents()

    def _build_agents(self) -> list[Agent]:
        models = self.cfg.models or ["openai/gpt-4o-mini"]
        lenses = self.cfg.lenses or [Lens.neutral]
        agents: list[Agent] = []
        for i in range(self.cfg.n_agents):
            # Cycle so each agent gets a distinct (model, lens) pairing where possible.
            model = models[i % len(models)]
            lens = lenses[i % len(lenses)]
            agents.append(Agent(model, lens, self.client, self.cfg.temperature,
                                getattr(self.cfg, "framing", "neutral")))
        return agents

    def run(self, question: Question, evidence: list[EvidenceItem]) -> list[Forecast]:
        forecasts: list[Forecast] = []
        for agent in self.agents:
            for sample in range(self.cfg.forecasts_per_agent):
                try:
                    forecasts.append(agent.forecast(question, evidence, sample=sample))
                except Exception as exc:  # one flaky model shouldn't kill the run
                    print(
                        f"[swarm] {agent.model}/{agent.lens.value} failed: {exc}",
                        file=sys.stderr,
                    )
        return forecasts

    def run_private(self, question: Question, slices: list[list[EvidenceItem]]) -> list[Forecast]:
        """Dispersed-private-information mode: agent i sees ONLY ``slices[i]``.

        Failed agents are skipped (length may be < n_agents). For experiments that need
        one forecast per agent (aligned), use ``forecast_each_private``.
        """
        forecasts: list[Forecast] = []
        for i, agent in enumerate(self.agents):
            evidence = slices[i % len(slices)] if slices else []
            try:
                forecasts.append(agent.forecast(question, evidence))
            except Exception as exc:
                print(
                    f"[swarm] {agent.model}/{agent.lens.value} failed: {exc}",
                    file=sys.stderr,
                )
        return forecasts

    def _private_evidence(self, slices, i):
        return slices[i % len(slices)] if slices else []

    def forecast_each_private(
        self, question: Question, slices: list[list[EvidenceItem]]
    ) -> list[Forecast]:
        """One forecast per agent, aligned to agents (0.5 fallback on failure)."""
        out: list[Forecast] = []
        for i, agent in enumerate(self.agents):
            try:
                out.append(agent.forecast(question, self._private_evidence(slices, i)))
            except Exception as exc:
                print(f"[swarm] {agent.model}/{agent.lens.value} failed: {exc}", file=sys.stderr)
                out.append(
                    Forecast(probability=0.5, confidence=0.3, thesis="(failed)",
                             model=agent.model, lens=agent.lens.value)
                )
        return out

    def run_debate_round(
        self, question: Question, slices: list[list[EvidenceItem]], prior: list[Forecast],
        extra_peers: list[Forecast] | None = None,
    ) -> list[Forecast]:
        """One deliberation round: each agent revises after seeing peers' forecasts +
        theses (its private slice still attached). The comparator for 'does a market
        beat mere talk?' — peers' theses carry their private information, as in debate.

        ``extra_peers`` are injected, shown to every agent as additional peers but NOT
        part of the returned swarm. This is the confederate manipulation: pass a single
        confident (often wrong) Forecast and measure whether the real agents herd toward
        it and grow more confident, which separates manufactured consensus from genuine
        pooling causally.
        """
        injected = extra_peers or []
        revised: list[Forecast] = []
        for i, agent in enumerate(self.agents):
            peers = [
                EvidenceItem(text=f"Another forecaster says p={f.probability:.2f}: {f.thesis}",
                             source="peer")
                for j, f in enumerate(prior) if j != i
            ] + [
                EvidenceItem(text=f"Another forecaster says p={f.probability:.2f}: {f.thesis}",
                             source="peer")
                for f in injected
            ]
            evidence = self._private_evidence(slices, i) + peers
            try:
                revised.append(agent.forecast(question, evidence))
            except Exception:
                revised.append(prior[i])
        return revised
