"""ForecastBench question source (offline, from the cached datasets repo).

ForecastBench (forecastingresearch/forecastbench-datasets, CC BY-SA) is a large,
professionally-built, leak-free-by-design benchmark. We load a resolution set
(ground-truth outcomes) and join it to its question set by id.

Two question types matter for honesty:
  * MARKET questions (manifold / metaculus / polymarket / infer): carry the crowd's
    frozen probability in ``freeze_datetime_value`` — we keep it as ``market_prob``
    (a baseline to beat) and must NOT show it to the agents.
  * DATASET questions (yfinance / fred / dbnomics / wikipedia): auto-resolving from
    data feeds, with no market. These are the cleanest honest targets — there is no
    price to look up.

Run ``scripts/pull_questions.py forecastbench`` (or git clone the datasets repo into
data/forecastbench) first.

LEAKAGE: these questions have already resolved, so live web retrieval (`:online`)
will find the answer. Score them with NO live retrieval (parametric reasoning only),
and prefer questions that resolved after the model's cutoff. For an *informed* honest
number, forecast an OPEN question set instead and score it on a future resolution set.
"""

from __future__ import annotations

import glob
import json
from datetime import date
from pathlib import Path

from .base import Question

_MARKET_SOURCES = {"manifold", "metaculus", "polymarket", "infer"}


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


class ForecastBenchQuestionSource:
    def __init__(
        self,
        resolution_set: str | None = None,
        question_set: str | None = None,
        data_dir: str = "data/forecastbench/datasets",
        sources: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.res_dir = self.data_dir / "resolution_sets"
        self.qs_dir = self.data_dir / "question_sets"
        self.resolution_set = resolution_set
        self.question_set = question_set
        self.sources = set(sources) if sources else None

    def _resolution_path(self) -> Path:
        if self.resolution_set:
            return self.res_dir / self.resolution_set
        files = sorted(glob.glob(str(self.res_dir / "*_resolution_set.json")))
        if not files:
            raise FileNotFoundError(f"no resolution sets in {self.res_dir}")
        return Path(files[-1])  # latest

    def fetch(self, limit: int = 100) -> list[Question]:
        res = json.loads(self._resolution_path().read_text())
        resolutions = res.get("resolutions", [])

        qs_name = self.question_set or res.get("question_set")
        qpath = self.qs_dir / qs_name if qs_name else None
        if not qpath or not qpath.exists():
            raise FileNotFoundError(
                f"question set {qs_name!r} not found in {self.qs_dir}; "
                "pass question_set explicitly"
            )
        qmap = {q["id"]: q for q in json.loads(qpath.read_text()).get("questions", [])}

        out: list[Question] = []
        for r in resolutions:
            if len(out) >= limit:
                break
            if not r.get("resolved"):
                continue
            outcome = r.get("resolved_to")
            if outcome not in (0, 1, 0.0, 1.0):
                continue  # keep only clean binary resolutions
            if self.sources and r.get("source") not in self.sources:
                continue
            q = qmap.get(r["id"])
            if not q:
                continue

            market_prob = None
            if r.get("source") in _MARKET_SOURCES:
                try:
                    mp = float(q.get("freeze_datetime_value"))
                    market_prob = mp if 0.0 <= mp <= 1.0 else None
                except (TypeError, ValueError):
                    market_prob = None

            out.append(
                Question(
                    id=f"fb-{r['id']}@{r.get('resolution_date', '')}",
                    text=q.get("question", ""),
                    background=q.get("background", ""),
                    resolution_criteria=q.get("resolution_criteria", ""),
                    resolution_date=_parse_date(r.get("resolution_date")),
                    outcome=float(outcome),
                    category=r.get("source", "forecastbench"),
                    source="forecastbench",
                    market_prob=market_prob,
                )
            )
        return out
