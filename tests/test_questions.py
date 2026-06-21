"""Tests for question sources (the offline mock + the factory)."""

import pytest

from bellwether.questions import get_question_source
from bellwether.questions.mock_internal import MockInternalQuestionSource


def test_mock_questions_are_resolved_and_well_formed():
    qs = MockInternalQuestionSource(seed=0).fetch(limit=20)
    assert len(qs) == 20
    for q in qs:
        assert q.is_resolved
        assert q.outcome in (0.0, 1.0)
        assert 0.0 <= q.metadata["signal_truth"] <= 1.0
        assert 0.0 < q.status_quo_prob < 1.0
        assert q.issue_date is not None


def test_mock_questions_are_deterministic():
    a = MockInternalQuestionSource(seed=0).fetch(limit=10)
    b = MockInternalQuestionSource(seed=0).fetch(limit=10)
    assert [q.outcome for q in a] == [q.outcome for q in b]
    assert [q.metadata["signal_truth"] for q in a] == [q.metadata["signal_truth"] for q in b]


def test_outcomes_track_signal_on_average():
    # Calibrated-by-construction: high-signal questions resolve YES more often.
    qs = MockInternalQuestionSource(seed=1).fetch(limit=24)
    hi = [q.outcome for q in qs if q.metadata["signal_truth"] > 0.5]
    lo = [q.outcome for q in qs if q.metadata["signal_truth"] <= 0.5]
    assert sum(hi) / len(hi) > sum(lo) / len(lo)


def test_factory_returns_mock_source():
    src = get_question_source("mock_internal", seed=0)
    assert len(src.fetch(limit=5)) == 5


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        get_question_source("nope")
