"""Typed configuration for benchmark runs.

Every knob we sweep over lives here as a plain pydantic model, loaded from a YAML
file. The runner logs the whole config to MLflow as run parameters, so the MLflow
runs-table becomes a side-by-side comparison of "config -> results". Secrets (API
keys) are kept separate in ``Settings``, read from the environment / .env, and are
never logged.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets / environment. Never logged to MLflow."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""


class Lens(str, Enum):
    """An agent's reasoning angle.

    Only ``base_rate`` and ``inside_view`` have positive evidence in the research;
    ``contrarian`` is an optional independent *voter* (a pre-mortem), never a debater.
    """

    neutral = "neutral"
    base_rate = "base_rate"      # outside view / reference class
    inside_view = "inside_view"  # current specific evidence
    contrarian = "contrarian"    # argue why the consensus is wrong


class EvidenceConfig(BaseModel):
    source: str = "mock_internal"   # mock_internal | web | none
    max_items: int = 10             # context sweet spot is ~5-15 items


class SwarmConfig(BaseModel):
    # Model diversity is the #1 lever against correlated errors -> list several.
    models: list[str] = Field(default_factory=lambda: ["openai/gpt-4o-mini"])
    n_agents: int = 6
    lenses: list[Lens] = Field(
        default_factory=lambda: [Lens.base_rate, Lens.inside_view]
    )
    forecasts_per_agent: int = 1
    temperature: float = 0.7


class MarketConfig(BaseModel):
    max_loss_budget: float = 100.0  # b is derived as budget / ln(N)
    kelly_fraction: float = 0.5
    max_bet_fraction: float = 0.25
    rounds: int = 2                 # passes over the agents (lets price converge)
    starting_bankroll: float = 100.0


class CalibrationConfig(BaseModel):
    extremize_coef: float = 1.7320508  # sqrt(3); set to 1.0 to disable


class QuestionSetConfig(BaseModel):
    source: str = "mock_internal"   # forecastbench | manifold | mock_internal
    name: str = "demo"              # for forecastbench: the question_set filename
    resolution_set: Optional[str] = None  # forecastbench: the matching resolution file
    data_dir: str = "data/forecastbench"
    limit: int = 20
    leakage_guard: bool = True      # forbid retrieval after a question's issue date


class BenchmarkConfig(BaseModel):
    experiment_name: str = "agent-market"
    run_name: str = "default"
    conditions: list[str] = Field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    seed: int = 0

    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    questions: QuestionSetConfig = Field(default_factory=QuestionSetConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)

    def flat_params(self) -> dict:
        """Flatten to scalar params for MLflow logging (one column per knob)."""
        out: dict = {}
        for section, val in self.model_dump(mode="json").items():
            if isinstance(val, dict):
                for k, v in val.items():
                    out[f"{section}.{k}"] = v
            else:
                out[section] = val
        return out
