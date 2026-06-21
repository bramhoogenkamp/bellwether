"""ForecastBench question source (offline, from the cached datasets repo).

ForecastBench (forecastingresearch/forecastbench-datasets, CC BY-SA) publishes
question sets plus nightly ``resolution_set`` files with ground-truth outcomes, and
leaderboards with published superforecaster + LLM-crowd baselines. We load a
question set, join it to its resolution set for outcomes, and (optionally) attach
the superforecaster baseline as ``superforecaster_prob`` (condition F).

Run ``scripts/pull_questions.py`` first to clone the datasets repo into ./data.
This loader is intentionally defensive about the JSON schema (the live format
evolves); adjust the field names in ``_parse`` if the upstream shape changes.

IMPORTANT (leakage): only score offline questions whose resolution is after the
model's knowledge cutoff, and keep evidence retrieval restricted to issue_date.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .base import Question


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
        question_set: str,
        resolution_set: str | None = None,
        data_dir: str = "data/forecastbench",
    ):
        self.qs_path = Path(data_dir) / question_set
        self.res_path = Path(data_dir) / resolution_set if resolution_set else None

    def fetch(self, limit: int = 20) -> list[Question]:
        qs = json.loads(self.qs_path.read_text())
        questions = qs.get("questions", qs if isinstance(qs, list) else [])

        resolutions: dict[str, float] = {}
        if self.res_path and self.res_path.exists():
            res = json.loads(self.res_path.read_text())
            for r in res.get("resolutions", res if isinstance(res, list) else []):
                qid = str(r.get("id"))
                val = r.get("resolved_to", r.get("resolution"))
                if val in (0, 1, 0.0, 1.0, "yes", "no", "YES", "NO"):
                    resolutions[qid] = (
                        1.0 if str(val).lower() in ("1", "1.0", "yes") else 0.0
                    )

        out: list[Question] = []
        for q in questions[:limit]:
            qid = str(q.get("id"))
            out.append(
                Question(
                    id=f"fb-{qid}",
                    text=q.get("question", ""),
                    background=q.get("background", ""),
                    resolution_criteria=q.get("resolution_criteria", ""),
                    issue_date=_parse_date(q.get("issue_date") or q.get("date")),
                    resolution_date=_parse_date(q.get("resolution_date")),
                    outcome=resolutions.get(qid),
                    category=q.get("source", "forecastbench"),
                    source="forecastbench",
                    superforecaster_prob=q.get("superforecaster_prob"),
                )
            )
        return out
