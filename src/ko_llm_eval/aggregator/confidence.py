"""Confidence calculation helpers."""

from __future__ import annotations

from ko_llm_eval.schemas import JudgeResult


def calculate_confidence(results: list[JudgeResult]) -> float:
    """Map lower score variance to higher confidence in the 0-1 range."""

    if not results:
        return 0.0
    if len(results) == 1:
        return 1.0

    scores = [result.score for result in results]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)

    return max(0.0, min(1.0, 1.0 - variance * 4.0))
