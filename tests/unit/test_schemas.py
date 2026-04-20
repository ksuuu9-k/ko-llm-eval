from pydantic import ValidationError

from ko_llm_eval.schemas.evaluation import EvaluationInput, EvaluationResult, JudgeResult


def test_evaluation_input_requires_prompt_and_answer() -> None:
    payload = EvaluationInput(prompt="질문", answer="답변")
    assert payload.prompt == "질문"
    assert payload.answer == "답변"
    assert payload.metadata == {}


def test_judge_result_score_must_be_normalized() -> None:
    try:
        JudgeResult(judge_name="rule", metric="tone", score=1.5)
    except ValidationError as exc:
        assert "score" in str(exc)
    else:
        raise AssertionError("JudgeResult should reject scores above 1.0")


def test_evaluation_result_supports_nested_metric_outputs() -> None:
    result = EvaluationResult(
        overall_score=0.8,
        confidence=0.7,
        agreement="medium",
        metrics={},
        failures=[],
    )
    assert result.overall_score == 0.8
