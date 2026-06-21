"""The swarm: a small set of independent agents whose forecasts we later aggregate.

Diversity is engineered the way the research says actually works: by mixing model
families and a couple of evidence-backed lenses — NOT by role-play personas or
debate. Agents run independently (no debate, no seeing each other's answers); the
aggregation step is where their views combine.

We deliberately keep the swarm small (5-10). Accuracy comes from a few good,
diverse forecasters, not from headcount.
"""

from __future__ import annotations

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
            agents.append(Agent(model, lens, self.client, self.cfg.temperature))
        return agents

    def run(self, question: Question, evidence: list[EvidenceItem]) -> list[Forecast]:
        forecasts: list[Forecast] = []
        for agent in self.agents:
            for sample in range(self.cfg.forecasts_per_agent):
                forecasts.append(agent.forecast(question, evidence, sample=sample))
        return forecasts
