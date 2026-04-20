from ko_llm_eval.judges import KoreanRubricJudge, LLMJudge
from ko_llm_eval.judges.llm_client import BaseLLMClient
from ko_llm_eval.schemas import EvaluationInput


class FakeLLMClient(BaseLLMClient):
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def test_llm_judge_builds_prompt_and_maps_json_response() -> None:
    client = FakeLLMClient(
        {
            "score": 0.77,
            "confidence": 0.66,
            "reasoning": "Grounded enough.",
            "tags": ["partially_grounded"],
        }
    )
    judge = LLMJudge(client=client)

    result = judge.evaluate(
        "grounding",
        EvaluationInput(prompt="환불 정책 알려줘", context="7일 이내 환불 가능", answer="7일 이내 가능합니다."),
    )

    assert result.score == 0.77
    assert result.confidence == 0.66
    assert result.tags == ["partially_grounded"]
    assert "Metric: grounding" in client.calls[0]["user_prompt"]


def test_korean_rubric_judge_uses_rubric_prompt() -> None:
    client = FakeLLMClient(
        {
            "score": 0.91,
            "confidence": 0.8,
            "reasoning": "존댓말이 일관적입니다.",
            "tags": [],
        }
    )
    judge = KoreanRubricJudge(client=client)

    result = judge.evaluate(
        "tone",
        EvaluationInput(prompt="환불 안내", context=None, answer="환불은 구매 후 7일 이내에 가능합니다."),
    )

    assert result.score == 0.91
    assert "Korean Rubric" in client.calls[0]["user_prompt"]
    assert "consistent honorific" in client.calls[0]["user_prompt"]
