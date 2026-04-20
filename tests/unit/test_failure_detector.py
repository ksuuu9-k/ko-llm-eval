from ko_llm_eval.failure_detection.detector import detect_failures
from ko_llm_eval.schemas import JudgeResult, MetricResult


def _metric_result(
    *,
    aggregated_score: float,
    confidence: float,
    agreement: str,
    judges: list[JudgeResult],
) -> MetricResult:
    return MetricResult(
        aggregated_score=aggregated_score,
        confidence=confidence,
        agreement=agreement,
        judges=judges,
    )


def _judge_result(
    *,
    metric: str,
    score: float,
    reasoning: str | None = None,
    tags: list[str] | None = None,
) -> JudgeResult:
    return JudgeResult(
        judge_name="test_judge",
        metric=metric,
        score=score,
        confidence=0.9,
        reasoning=reasoning,
        tags=tags or [],
    )


def test_detect_failures_uses_tone_tags_and_grounding_reasoning() -> None:
    metrics = {
        "tone": _metric_result(
            aggregated_score=0.8,
            confidence=0.8,
            agreement="medium",
            judges=[
                _judge_result(metric="tone", score=0.8, tags=["tone_mixed"]),
            ],
        ),
        "grounding": _metric_result(
            aggregated_score=0.76,
            confidence=0.8,
            agreement="medium",
            judges=[
                _judge_result(
                    metric="grounding",
                    score=0.76,
                    reasoning="The answer adds unsupported details beyond the context.",
                ),
            ],
        ),
        "fluency": _metric_result(
            aggregated_score=0.85,
            confidence=0.7,
            agreement="high",
            judges=[
                _judge_result(metric="fluency", score=0.85, reasoning="Fluent enough."),
            ],
        ),
    }

    failures = detect_failures(metrics)

    assert "tone_inconsistency" in failures
    assert "partial_hallucination" in failures
    assert "instruction_violation" not in failures


def test_detect_failures_uses_fluency_format_tags() -> None:
    metrics = {
        "fluency": _metric_result(
            aggregated_score=0.73,
            confidence=0.92,
            agreement="high",
            judges=[
                _judge_result(
                    metric="fluency",
                    score=0.73,
                    tags=["missing_sentence_ending", "too_short"],
                    reasoning="형식과 길이 제약을 충분히 지키지 못했습니다.",
                ),
            ],
        )
    }

    failures = detect_failures(metrics)

    assert failures == ["instruction_violation"]


def test_detect_failures_can_return_empty_list() -> None:
    metrics = {
        "tone": _metric_result(
            aggregated_score=0.91,
            confidence=0.84,
            agreement="high",
            judges=[_judge_result(metric="tone", score=0.91, reasoning="존댓말이 일관적입니다.")],
        ),
        "grounding": _metric_result(
            aggregated_score=0.88,
            confidence=0.9,
            agreement="high",
            judges=[_judge_result(metric="grounding", score=0.88, reasoning="Context support is clear.")],
        ),
        "fluency": _metric_result(
            aggregated_score=0.9,
            confidence=0.88,
            agreement="high",
            judges=[_judge_result(metric="fluency", score=0.9, reasoning="자연스럽고 매끄럽습니다.")],
        ),
    }

    assert detect_failures(metrics) == []
