"""The agent swarm: LLM-backed forecasters that read evidence and output a probability."""

from .agent import Agent, Forecast
from .llm import FakeLLM, LiteLLMClient, LLMClient, get_client
from .swarm import Swarm

__all__ = [
    "Agent",
    "Forecast",
    "LLMClient",
    "FakeLLM",
    "LiteLLMClient",
    "get_client",
    "Swarm",
]
