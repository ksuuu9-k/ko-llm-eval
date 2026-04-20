"""Weighted score aggregation utilities."""

from __future__ import annotations

from ko_llm_eval.judges import BaseJudge
from ko_llm_eval.schemas import JudgeResult


def weighted_mean(results: list[JudgeResult], judges: list[BaseJudge]) -> float:
    """Compute a weighted mean using the configured judge weights."""

    weight_map = {judge.name: judge.weight for judge in judges}
    weighted_sum = 0.0
    total_weight = 0.0

    for result in results:
        weight = weight_map.get(result.judge_name, 1.0)
        weighted_sum += result.score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight
