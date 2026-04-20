from ko_llm_eval.judges import MockJudge
from ko_llm_eval.metrics.registry import MetricRegistry
from ko_llm_eval.orchestrator.evaluator import Evaluator
from ko_llm_eval.schemas import EvaluationInput


def test_evaluator_aggregates_metric_results() -> None:
    registry = MetricRegistry()
    registry.register(
        "tone",
        [
            MockJudge(name="judge_a", score_map={"tone": 0.9}, weight=0.7),
            MockJudge(name="judge_b", score_map={"tone": 0.6}, weight=0.3),
        ],
    )
    registry.register(
        "grounding",
        [
            MockJudge(name="judge_a", score_map={"grounding": 0.8}, weight=0.7),
            MockJudge(name="judge_b", score_map={"grounding": 0.7}, weight=0.3),
        ],
    )

    evaluator = Evaluator(registry=registry)
    result = evaluator.evaluate(EvaluationInput(prompt="질문", context="문맥", answer="답변"))

    assert result.overall_score > 0.0
    assert "tone" in result.metrics
    assert len(result.metrics["tone"].judges) == 2
    assert result.metrics["tone"].aggregated_score == 0.81
