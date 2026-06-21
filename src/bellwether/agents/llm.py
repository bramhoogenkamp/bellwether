"""LLM clients: one real (LiteLLM -> OpenRouter), one fake (deterministic, offline).

Both implement the same tiny ``LLMClient`` interface, so the entire pipeline runs
identically whether we're spending money on real models or running free offline
tests. ``FakeLLM`` is what makes the benchmark testable with zero API cost: it
reads the numeric signals embedded in the evidence, adds a deterministic
per-(model, prompt) wobble, and returns a forecast — so a swarm of "different
models" produces genuinely different-but-correlated numbers, just like the real
thing.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol, runtime_checkable

_FLOAT = re.compile(r"\d*\.\d+")


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, model: str, system: str, user: str, temperature: float = 0.7) -> str: ...


def _unit_hash(text: str) -> float:
    """Map a string to a deterministic value in [-1, 1] (stable across processes)."""
    h = hashlib.md5(text.encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1


class FakeLLM:
    """Deterministic offline stand-in for a real model.

    Belief = (mean of the probabilities mentioned in the evidence) + a small
    deterministic wobble keyed by the model and prompt. Different models / lenses
    therefore disagree a little — exactly the diversity a market needs — while all
    tracking the same underlying signal.
    """

    def __init__(self, noise: float = 0.08):
        self.noise = noise

    def complete(self, *, model: str, system: str, user: str, temperature: float = 0.7) -> str:
        signals = [float(x) for x in _FLOAT.findall(user) if 0.0 < float(x) < 1.0]
        base = sum(signals) / len(signals) if signals else 0.5

        wobble = self.noise * _unit_hash(f"{model}|{system}")
        belief = min(max(base + wobble, 0.02), 0.98)

        # More decisive beliefs come with higher stated confidence.
        confidence = min(max(0.5 + abs(belief - 0.5), 0.3), 0.95)
        thesis = f"Evidence centres near {base:.2f}; adjusted to {belief:.2f}."
        return json.dumps(
            {"probability": belief, "confidence": confidence, "thesis": thesis}
        )


class LiteLLMClient:
    """Real client via LiteLLM, routed through OpenRouter (one key, many models)."""

    def __init__(self, api_key: str | None = None, prefix: str = "openrouter/"):
        if api_key is None:
            from ..config import Settings

            api_key = Settings().openrouter_api_key
        self.api_key = api_key
        self.prefix = prefix

    def complete(self, *, model: str, system: str, user: str, temperature: float = 0.7) -> str:
        import litellm  # lazy: only needed for real runs

        m = model if model.startswith(self.prefix) else self.prefix + model
        resp = litellm.completion(
            model=m,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            api_key=self.api_key,
        )
        return resp.choices[0].message.content or ""


def get_client(name: str = "fake", **kwargs) -> LLMClient:
    """Factory: 'fake' (offline, default) or 'litellm' (real, needs an API key)."""
    if name == "fake":
        return FakeLLM(**kwargs)
    if name in ("litellm", "openrouter", "live"):
        return LiteLLMClient(**kwargs)
    raise ValueError(f"unknown LLM client: {name!r}")
