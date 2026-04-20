import pytest

from ko_llm_eval.judges import MockJudge, UnsupportedMetricError
from ko_llm_eval.schemas import EvaluationInput


def test_mock_judge_returns_deterministic_score() -> None:
    judge = MockJudge(name="mock", score_map={"tone": 0.9}, weight=0.3)
    payload = EvaluationInput(prompt="안내해줘", answer="안내드리겠습니다.")

    result = judge.evaluate("tone", payload)

    assert result.judge_name == "mock"
    assert result.metric == "tone"
    assert result.score == 0.9
    assert result.confidence == 1.0


def test_mock_judge_rejects_unknown_metric() -> None:
    judge = MockJudge(name="mock", score_map={"tone": 0.9})
    payload = EvaluationInput(prompt="안내해줘", answer="안내드리겠습니다.")

    with pytest.raises(UnsupportedMetricError):
        judge.evaluate("grounding", payload)
