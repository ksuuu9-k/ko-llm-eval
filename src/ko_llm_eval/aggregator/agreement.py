"""Agreement level helpers."""

from __future__ import annotations

from ko_llm_eval.schemas import AgreementLevel, JudgeResult


def calculate_agreement(results: list[JudgeResult]) -> AgreementLevel:
    """Classify agreement using score spread."""

    if len(results) <= 1:
        return "high"

    scores = [result.score for result in results]
    spread = max(scores) - min(scores)

    if spread <= 0.15:
        return "high"
    if spread <= 0.35:
        return "medium"
    return "low"
