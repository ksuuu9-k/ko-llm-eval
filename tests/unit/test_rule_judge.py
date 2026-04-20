from ko_llm_eval.judges import RuleJudge
from ko_llm_eval.schemas import EvaluationInput


def test_rule_judge_detects_mixed_tone() -> None:
    judge = RuleJudge()
    payload = EvaluationInput(
        prompt="환불 안내",
        answer="환불 가능합니다. 지금 바로 해.",
    )

    result = judge.evaluate("tone", payload)

    assert result.score == 0.35
    assert "tone_mixed" in result.tags


def test_rule_judge_uses_context_overlap_for_grounding() -> None:
    judge = RuleJudge()
    payload = EvaluationInput(
        prompt="환불 안내",
        context="환불은 구매 후 7일 이내 가능합니다",
        answer="환불은 구매 후 7일 이내에 가능합니다.",
    )

    result = judge.evaluate("grounding", payload)

    assert result.score >= 0.7
