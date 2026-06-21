"""Tests for evidence sources and the leakage guard."""

from datetime import date

from bellwether.evidence import apply_leakage_guard, get_evidence_source
from bellwether.evidence.base import EvidenceItem
from bellwether.evidence.mock_internal import MockInternalEvidenceSource
from bellwether.questions.mock_internal import MockInternalQuestionSource


def _question():
    return MockInternalQuestionSource(seed=0).fetch(limit=1)[0]


def test_mock_evidence_is_dated_before_issue_and_mentions_numbers():
    q = _question()
    items = MockInternalEvidenceSource().gather(q, max_items=5, as_of=q.issue_date)
    assert len(items) == 5
    for it in items:
        assert it.dated <= q.issue_date
        # each snippet embeds a parseable probability
        assert any(ch.isdigit() for ch in it.text)


def test_mock_evidence_is_deterministic():
    q = _question()
    a = MockInternalEvidenceSource(seed=3).gather(q, max_items=5)
    b = MockInternalEvidenceSource(seed=3).gather(q, max_items=5)
    assert [i.text for i in a] == [i.text for i in b]


def test_leakage_guard_drops_future_items():
    items = [
        EvidenceItem(text="past", dated=date(2026, 1, 1)),
        EvidenceItem(text="future", dated=date(2026, 6, 1)),
        EvidenceItem(text="undated", dated=None),
    ]
    kept = apply_leakage_guard(items, as_of=date(2026, 1, 31))
    texts = {i.text for i in kept}
    assert "past" in texts and "undated" in texts and "future" not in texts


def test_empty_source_returns_nothing():
    src = get_evidence_source("none")
    assert src.gather(_question()) == []
