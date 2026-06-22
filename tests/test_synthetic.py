"""Tests for the synthetic dispersed-private-information generator + private-evidence swarm."""

import re

from bellwether.agents.llm import get_client
from bellwether.agents.swarm import Swarm
from bellwether.config import Lens, SwarmConfig
from bellwether.questions.synthetic import generate_info_instances


def test_substitutable_structure():
    inst = generate_info_instances(n=5, n_agents=4, structure="substitutable", seed=0)
    assert len(inst) == 5
    for x in inst:
        assert x.question.outcome in (0.0, 1.0)
        assert len(x.slices) == 4 and len(x.pooled) == 4
        assert re.search(r"\d\.\d", x.slices[0][0].text)  # a parseable estimate


def test_complementary_outcome_is_AND_of_conditions():
    inst = generate_info_instances(n=30, n_agents=3, structure="complementary", seed=1)
    for x in inst:
        statuses = x.question.metadata["statuses"]
        assert x.question.outcome == (1.0 if all(statuses) else 0.0)
        assert len(x.slices) == 3
        assert "COMPLETE" in x.slices[0][0].text or "NOT complete" in x.slices[0][0].text


def test_complementary_has_both_outcomes():
    inst = generate_info_instances(n=80, n_agents=3, structure="complementary", seed=2)
    ys = {x.question.outcome for x in inst}
    assert ys == {0.0, 1.0}


def test_generator_is_deterministic():
    a = generate_info_instances(n=10, structure="complementary", seed=3)
    b = generate_info_instances(n=10, structure="complementary", seed=3)
    assert [x.question.outcome for x in a] == [x.question.outcome for x in b]


def test_swarm_run_private_gives_each_agent_its_own_slice():
    inst = generate_info_instances(n=1, n_agents=4, structure="substitutable", seed=0)[0]
    cfg = SwarmConfig(models=["m1"], n_agents=4, lenses=[Lens.neutral])
    swarm = Swarm(cfg, get_client("fake"))
    forecasts = swarm.run_private(inst.question, inst.slices)
    assert len(forecasts) == 4
    # different private slices -> different parsed beliefs (not clones)
    assert len({round(f.probability, 3) for f in forecasts}) > 1
