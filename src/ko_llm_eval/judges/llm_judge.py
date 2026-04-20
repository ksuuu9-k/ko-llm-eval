"""LLM-backed general semantic judge."""

from __future__ import annotations

from ko_llm_eval.judges.base import BaseJudge, UnsupportedMetricError
from ko_llm_eval.judges.llm_client import BaseLLMClient
from ko_llm_eval.judges.prompting import LLM_JUDGE_SYSTEM_PROMPT, METRIC_INSTRUCTIONS, build_llm_user_prompt
from ko_llm_eval.schemas import EvaluationInput, JudgeResult


class LLMJudge(BaseJudge):
    """General-purpose semantic judge using an external model."""

    def __init__(
        self,
        client: BaseLLMClient,
        name: str = "llm",
        weight: float = 0.45,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.client = client

    def evaluate(self, metric: str, evaluation_input: EvaluationInput) -> JudgeResult:
        if metric not in METRIC_INSTRUCTIONS:
            raise UnsupportedMetricError(f"{self.name} does not support metric '{metric}'.")

        response = self.client.complete_json(
            system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
            user_prompt=build_llm_user_prompt(metric, evaluation_input),
        )
        return _build_judge_result(self.name, metric, response)


def _build_judge_result(judge_name: str, metric: str, response: dict) -> JudgeResult:
    return JudgeResult(
        judge_name=judge_name,
        metric=metric,
        score=float(response["score"]),
        confidence=float(response["confidence"]),
        reasoning=str(response["reasoning"]),
        tags=[str(tag) for tag in response["tags"]],
        raw=response,
    )
