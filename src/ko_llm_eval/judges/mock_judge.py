"""Deterministic mock judges for local development and tests."""

from __future__ import annotations

from ko_llm_eval.judges.base import BaseJudge, UnsupportedMetricError
from ko_llm_eval.schemas import EvaluationInput, JudgeResult


class MockJudge(BaseJudge):
    """Return predefined scores without calling external services."""

    def __init__(
        self,
        name: str,
        score_map: dict[str, float],
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.score_map = score_map

    def evaluate(self, metric: str, evaluation_input: EvaluationInput) -> JudgeResult:
        if metric not in self.score_map:
            raise UnsupportedMetricError(f"{self.name} does not support metric '{metric}'.")

        score = self.score_map[metric]
        return JudgeResult(
            judge_name=self.name,
            metric=metric,
            score=score,
            confidence=1.0,
            reasoning=f"Mock evaluation for metric '{metric}'.",
            raw={
                "prompt_length": len(evaluation_input.prompt),
                "answer_length": len(evaluation_input.answer),
            },
        )
