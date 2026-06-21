"""Tests for the FakeLLM, agent parsing, and the swarm (all offline)."""

from bellwether.agents.agent import Agent, parse_forecast
from bellwether.agents.llm import FakeLLM, get_client
from bellwether.agents.swarm import Swarm
from bellwether.config import Lens, SwarmConfig
from bellwether.evidence.mock_internal import MockInternalEvidenceSource
from bellwether.questions.mock_internal import MockInternalQuestionSource


def _qe():
    q = MockInternalQuestionSource(seed=0).fetch(limit=1)[0]
    ev = MockInternalEvidenceSource(seed=0).gather(q, max_items=6, as_of=q.issue_date)
    return q, ev


def test_fake_llm_is_deterministic():
    f = FakeLLM()
    kw = dict(model="m", system="s", user="signal 0.7 and 0.6")
    assert f.complete(**kw) == f.complete(**kw)


def test_fake_llm_tracks_evidence_signal():
    f = FakeLLM(noise=0.0)  # no wobble -> belief equals the mean signal
    out = f.complete(model="m", system="s", user="reads 0.80 and 0.80")
    import json

    assert abs(json.loads(out)["probability"] - 0.80) < 1e-6


def test_parse_forecast_handles_clean_and_messy():
    p, c, t = parse_forecast('{"probability": 0.73, "confidence": 0.6, "thesis": "x"}')
    assert p == 0.73 and c == 0.6
    # messy: prose around the number, no valid JSON
    p2, _, _ = parse_forecast("I think it's about 0.42 likely.")
    assert abs(p2 - 0.42) < 1e-9


def test_agent_returns_forecast_in_range():
    q, ev = _qe()
    agent = Agent("openai/gpt-4o-mini", Lens.base_rate, FakeLLM())
    fc = agent.forecast(q, ev)
    assert 0.0 < fc.probability < 1.0
    assert fc.model == "openai/gpt-4o-mini"
    assert fc.lens == "base_rate"


def test_swarm_produces_one_forecast_per_agent_with_diversity():
    q, ev = _qe()
    cfg = SwarmConfig(
        models=["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-flash-1.5"],
        n_agents=6,
        lenses=[Lens.base_rate, Lens.inside_view],
    )
    swarm = Swarm(cfg, get_client("fake"))
    forecasts = swarm.run(q, ev)
    assert len(forecasts) == 6
    probs = [f.probability for f in forecasts]
    assert len(set(probs)) > 1  # genuine disagreement, not clones


def test_swarm_forecasts_per_agent_multiplies_count():
    q, ev = _qe()
    cfg = SwarmConfig(models=["m1"], n_agents=2, lenses=[Lens.neutral], forecasts_per_agent=3)
    swarm = Swarm(cfg, get_client("fake"))
    assert len(swarm.run(q, ev)) == 6
