"""Korean rubric-focused judge."""

from __future__ import annotations

from ko_llm_eval.judges.base import BaseJudge, UnsupportedMetricError
from ko_llm_eval.judges.llm_client import BaseLLMClient
from ko_llm_eval.judges.llm_judge import _build_judge_result
from ko_llm_eval.judges.prompting import KOREAN_RUBRICS, RUBRIC_SYSTEM_PROMPT, build_korean_rubric_prompt
from ko_llm_eval.schemas import EvaluationInput, JudgeResult


class KoreanRubricJudge(BaseJudge):
    """LLM-backed judge specialized for Korean rubric evaluation."""

    def __init__(
        self,
        client: BaseLLMClient,
        name: str = "ko_rubric",
        weight: float = 0.35,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.client = client

    def evaluate(self, metric: str, evaluation_input: EvaluationInput) -> JudgeResult:
        if metric not in KOREAN_RUBRICS:
            raise UnsupportedMetricError(f"{self.name} does not support metric '{metric}'.")

        response = self.client.complete_json(
            system_prompt=RUBRIC_SYSTEM_PROMPT,
            user_prompt=build_korean_rubric_prompt(metric, evaluation_input),
        )
        return _build_judge_result(self.name, metric, response)
