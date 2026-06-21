"""Evidence sources: how an agent gets information about a question.

This is the seam where a real internal-data connector (Slack/Jira/CRM via MCP)
will plug in later. For now we ship a deterministic mock source (the agents'
"private signal" stand-in) and a web stub. Everything goes through the same
``EvidenceSource`` interface, and all retrieval is passed through a leakage guard
so an agent can never see information dated after the question was asked.
"""

from .base import EvidenceItem, EvidenceSource, apply_leakage_guard, get_evidence_source

__all__ = [
    "EvidenceItem",
    "EvidenceSource",
    "apply_leakage_guard",
    "get_evidence_source",
]
