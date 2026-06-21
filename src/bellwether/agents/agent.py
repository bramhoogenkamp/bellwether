"""A single forecasting agent: turn (question + evidence) into a probability.

Each agent reads a short evidence brief, writes a one-line thesis, and outputs a
probability + a confidence. The reasoning is deliberately *light* — enough to
ground the number and improve calibration, but not an open-ended chain that breeds
overconfidence. An agent's "lens" tweaks only its system prompt; the only lenses
with positive evidence are base-rate (outside view) and inside-view, plus an
optional contrarian *voter*.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from ..config import Lens
from ..evidence.base import EvidenceItem
from ..questions.base import Question
from .llm import LLMClient

_BASE_INSTRUCTION = (
    "You forecast whether a binary question resolves YES. Weigh the evidence, then "
    'give a probability in [0,1]. Reply with ONLY JSON: '
    '{"probability": <float 0..1>, "confidence": <float 0..1>, "thesis": "<one line>"}.'
)

_LENS_PROMPT = {
    Lens.neutral: "You are a careful, calibrated forecaster.",
    Lens.base_rate: (
        "You are a forecaster who thinks reference-class first: estimate the base "
        "rate for events like this, anchor on it, then adjust for the specifics."
    ),
    Lens.inside_view: (
        "You are a forecaster who weighs the latest specific evidence and current "
        "trajectory most heavily."
    ),
    Lens.contrarian: (
        "You are a pre-mortem forecaster: actively look for why the obvious read "
        "might be wrong, and price in what others are missing."
    ),
}

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_FLOAT = re.compile(r"\d*\.\d+|\d+")


class Forecast(BaseModel):
    probability: float
    confidence: float = 0.5
    thesis: str = ""
    model: str = ""
    lens: str = Lens.neutral.value


def _clip01(x: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, float(x)))


def build_system(lens: Lens, sample: int = 0) -> str:
    system = f"{_LENS_PROMPT.get(lens, _LENS_PROMPT[Lens.neutral])}\n{_BASE_INSTRUCTION}"
    if sample > 0:  # nudge repeated draws to differ (real: temperature; fake: hash)
        system += f"\n[sample {sample}]"
    return system


def build_user(question: Question, evidence: list[EvidenceItem]) -> str:
    lines = [
        f"Question: {question.text}",
    ]
    if question.resolution_criteria:
        lines.append(f"Resolution: {question.resolution_criteria}")
    if question.background:
        lines.append(f"Background: {question.background}")
    if evidence:
        lines.append("\nEvidence:")
        lines += [f"- [{it.source}] {it.text}" for it in evidence]
    else:
        lines.append("\nNo evidence available; reason from base rates.")
    lines.append("\nReturn your forecast as JSON.")
    return "\n".join(lines)


def parse_forecast(raw: str) -> tuple[float, float, str]:
    """Best-effort parse of the model's reply into (probability, confidence, thesis)."""
    match = _JSON_OBJ.search(raw or "")
    if match:
        try:
            data = json.loads(match.group(0))
            return (
                _clip01(data["probability"]),
                _clip01(float(data.get("confidence", 0.5)), 0.0, 1.0),
                str(data.get("thesis", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    # Fallback: grab the first plausible probability from the text.
    nums = [float(x) for x in _FLOAT.findall(raw or "") if 0.0 <= float(x) <= 1.0]
    prob = _clip01(nums[0]) if nums else 0.5
    return prob, 0.5, "(unparsed reply)"


class Agent:
    def __init__(self, model: str, lens: Lens, client: LLMClient, temperature: float = 0.7):
        self.model = model
        self.lens = lens
        self.client = client
        self.temperature = temperature

    def forecast(
        self, question: Question, evidence: list[EvidenceItem], sample: int = 0
    ) -> Forecast:
        system = build_system(self.lens, sample)
        user = build_user(question, evidence)
        raw = self.client.complete(
            model=self.model, system=system, user=user, temperature=self.temperature
        )
        prob, conf, thesis = parse_forecast(raw)
        return Forecast(
            probability=prob,
            confidence=conf,
            thesis=thesis,
            model=self.model,
            lens=self.lens.value,
        )
