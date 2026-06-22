"""Polymarket question source (resolved binary markets + a real market probability).

Polymarket is a real-money market, which makes it a *sharp* external benchmark. Two
quirks the research flagged:
  * resolved markets expose only the SETTLED price (0/1) — that's the outcome, not a
    forecast. To get the market's actual probability we pull the CLOB price history
    and take the price at the MIDPOINT of the market's trading life (a genuine
    "while still uncertain" forecast).
  * the Gamma ``endDate`` is nominal and unreliable, so we derive timing from the
    price history itself.

Leakage warning: famous past markets (e.g. the 2024 election) are exactly the ones
a model already "knows" the answer to. Use ``resolved_after`` to restrict to markets
that resolved after the models' knowledge cutoff for a leak-free comparison.
Network only.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from .base import Question

_GAMMA = "https://gamma-api.polymarket.com/markets"
_CLOB_HISTORY = "https://clob.polymarket.com/prices-history"


def _epoch_to_date(ts: int) -> date:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date()


def _parse_iso(s) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


class PolymarketQuestionSource:
    def __init__(
        self,
        min_volume: float = 5000.0,
        resolved_after: date | None = date(2025, 7, 1),
        price_fraction: float = 0.5,  # where in the market's life to read its forecast
        scan_limit: int = 250,
        timeout: float = 25.0,
    ):
        self.min_volume = min_volume
        self.resolved_after = resolved_after
        self.price_fraction = price_fraction
        self.scan_limit = scan_limit
        self.timeout = timeout

    def fetch(self, limit: int = 20) -> list[Question]:
        import httpx  # lazy: only for this network source

        out: list[Question] = []
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                _GAMMA,
                params={
                    "closed": "true",
                    "order": "volumeNum",
                    "ascending": "false",
                    "limit": self.scan_limit,
                },
            )
            resp.raise_for_status()
            for m in resp.json():
                if len(out) >= limit:
                    break
                q = self._to_question(client, m)
                if q is not None:
                    out.append(q)
        return out

    def _to_question(self, client, m) -> Question | None:
        if m.get("outcomes") != '["Yes", "No"]' or not m.get("clobTokenIds"):
            return None
        if (m.get("volumeNum") or 0) < self.min_volume:
            return None
        try:
            settled = json.loads(m["outcomePrices"])
            outcome = 1.0 if float(settled[0]) >= 0.5 else 0.0
            yes_token = json.loads(m["clobTokenIds"])[0]
        except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
            return None

        # Pull the YES price history and read the market's mid-life probability.
        try:
            h = client.get(
                _CLOB_HISTORY,
                params={"market": yes_token, "interval": "max", "fidelity": "1440"},
            )
            history = h.json().get("history", [])
        except Exception:
            return None
        if len(history) < 8:  # need a real trading life, not a flash market
            return None

        resolved_on = _epoch_to_date(history[-1]["t"])
        if self.resolved_after and resolved_on < self.resolved_after:
            return None  # leak control: skip pre-cutoff markets

        idx = int(self.price_fraction * (len(history) - 1))
        market_prob = float(history[idx]["p"])

        return Question(
            id=f"poly-{m.get('id', yes_token[:10])}",
            text=m.get("question", ""),
            issue_date=_epoch_to_date(history[idx]["t"]),
            resolution_date=resolved_on,
            outcome=outcome,
            category="polymarket",
            source="polymarket",
            market_prob=market_prob,
            metadata={"slug": m.get("slug", "")},
        )

    def fetch_open_within(self, days: int = 7, limit: int = 40) -> list[Question]:
        """Open binary markets resolving within ``days`` (short-horizon forward test).

        Uses the Gamma end-date window so we target soon-resolving markets directly,
        rather than the highest-volume markets (which are long-horizon). Same filters as
        ``fetch_open``: binary Yes/No, above ``min_volume``, and a genuinely uncertain
        price. The endDate window is reliable here because we read it as a server filter.
        """
        import httpx

        today = datetime.now(timezone.utc).date()
        cutoff = today + timedelta(days=days)
        out: list[Question] = []
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                _GAMMA,
                params={
                    "closed": "false",
                    "active": "true",
                    "order": "volumeNum",
                    "ascending": "false",
                    "limit": self.scan_limit,
                    "end_date_min": today.isoformat(),
                    "end_date_max": cutoff.isoformat(),
                },
            )
            resp.raise_for_status()
            for m in resp.json():
                if len(out) >= limit:
                    break
                if m.get("outcomes") != '["Yes", "No"]':
                    continue
                if (m.get("volumeNum") or 0) < self.min_volume:
                    continue
                try:
                    yes = float(json.loads(m["outcomePrices"])[0])
                except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
                    continue
                if not (0.05 < yes < 0.95):
                    continue
                rdate = _parse_iso(m.get("endDate"))
                if rdate is None or not (today <= rdate <= cutoff):
                    continue
                out.append(
                    Question(
                        id=f"poly-open-{m.get('id', '')}",
                        text=m.get("question", ""),
                        issue_date=today,
                        resolution_date=rdate,
                        outcome=None,
                        category="polymarket-open",
                        source="polymarket",
                        market_prob=yes,
                        metadata={"slug": m.get("slug", "")},
                    )
                )
        return out

    def fetch_open(self, limit: int = 10) -> list[Question]:
        """Currently-OPEN binary markets (outcome unknown -> honest forward-test).

        ``market_prob`` is the live YES price (the crowd's current probability), and
        ``outcome`` is None until the market resolves. Use this with web retrieval:
        there is no answer to leak because the event hasn't happened yet.
        """
        import httpx

        out: list[Question] = []
        today = datetime.now(timezone.utc).date()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                _GAMMA,
                params={
                    "closed": "false",
                    "active": "true",
                    "order": "volumeNum",
                    "ascending": "false",
                    "limit": self.scan_limit,
                },
            )
            resp.raise_for_status()
            for m in resp.json():
                if len(out) >= limit:
                    break
                if m.get("outcomes") != '["Yes", "No"]':
                    continue
                if (m.get("volumeNum") or 0) < self.min_volume:
                    continue
                try:
                    yes = float(json.loads(m["outcomePrices"])[0])
                except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
                    continue
                if not (0.05 < yes < 0.95):
                    continue  # skip near-settled / dead markets -> keep genuinely uncertain ones
                out.append(
                    Question(
                        id=f"poly-open-{m.get('id', '')}",
                        text=m.get("question", ""),
                        issue_date=today,
                        resolution_date=_parse_iso(m.get("endDate")),
                        outcome=None,
                        category="polymarket-open",
                        source="polymarket",
                        market_prob=yes,
                        metadata={"slug": m.get("slug", "")},
                    )
                )
        return out
