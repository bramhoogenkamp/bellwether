"""Question sources: where the events we forecast come from.

Every source returns a list of ``Question`` objects with the same shape, so the
rest of the pipeline doesn't care whether a question came from a mock internal
dataset, ForecastBench, or a live Manifold market.
"""

from .base import Question, QuestionSource, get_question_source

__all__ = ["Question", "QuestionSource", "get_question_source"]
