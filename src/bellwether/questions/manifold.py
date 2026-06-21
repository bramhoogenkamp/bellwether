"""Manifold Markets question source (resolved binary markets).

Manifold is free, bot-friendly, and needs no auth for reads, which makes it the
easiest public market to benchmark against. We pull *resolved* binary markets so
each one has a ground-truth outcome, and we keep the market's own probability as
``market_prob`` so we can compare our swarm's forecast against the crowd's price.

Note on point-in-time correctness: for a rigorous market-vs-model backtest you
want the market price *at the moment your forecast was made* (reconstructed from
the bet stream's ``probAfter``), not the final pre-resolution price. v1 stores the
lite-market probability as a starting point; Phase 6 refines this. Network only.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from .base import Question

_API = "https://api.manifold.markets/v0"


def _to_date(ms: int | None):
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()


class ManifoldQuestionSource:
    def __init__(self, limit_scan: int = 500, timeout: float = 20.0):
        self.limit_scan = limit_scan
        self.timeout = timeout

    def fetch(self, limit: int = 20) -> list[Question]:
        import httpx  # lazy: only needed for this network source

        out: list[Question] = []
        before: str | None = None
        with httpx.Client(timeout=self.timeout) as client:
            while len(out) < limit:
                params = {"limit": min(1000, self.limit_scan)}
                if before:
                    params["before"] = before
                resp = client.get(f"{_API}/markets", params=params)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                for m in batch:
                    before = m["id"]
                    if m.get("outcomeType") != "BINARY" or not m.get("isResolved"):
                        continue
                    res = m.get("resolution")
                    if res not in ("YES", "NO"):
                        continue  # skip MKT/CANCEL/ambiguous resolutions
                    out.append(
                        Question(
                            id=f"manifold-{m['id']}",
                            text=m.get("question", ""),
                            issue_date=_to_date(m.get("createdTime")),
                            resolution_date=_to_date(m.get("resolutionTime")),
                            outcome=1.0 if res == "YES" else 0.0,
                            category="manifold",
                            source="manifold",
                            market_prob=m.get("probability"),
                            metadata={"url": m.get("url", "")},
                        )
                    )
                    if len(out) >= limit:
                        break
        return out
