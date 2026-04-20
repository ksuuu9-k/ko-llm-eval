"""Base judge contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ko_llm_eval.schemas import EvaluationInput, JudgeResult


class BaseJudge(ABC):
    """Abstract base class for all judges."""

    name: str
    weight: float

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, metric: str, evaluation_input: EvaluationInput) -> JudgeResult:
        """Return a normalized score for the given metric."""


class UnsupportedMetricError(ValueError):
    """Raised when a judge receives a metric it does not support."""
