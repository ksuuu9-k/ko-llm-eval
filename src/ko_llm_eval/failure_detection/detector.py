"""Failure tag extraction using multiple judge signals."""

from __future__ import annotations

from ko_llm_eval.schemas import JudgeResult, MetricResult


def detect_failures(metrics: dict[str, MetricResult]) -> list[str]:
    """Derive failure tags from score, agreement, tags, and reasoning."""

    failures: list[str] = []

    tone = metrics.get("tone")
    if tone and _is_tone_inconsistency(tone):
        failures.append("tone_inconsistency")

    grounding = metrics.get("grounding")
    if grounding and _is_partial_hallucination(grounding):
        failures.append("partial_hallucination")

    fluency = metrics.get("fluency")
    if fluency and _is_instruction_violation(fluency):
        failures.append("instruction_violation")

    return failures


def _is_tone_inconsistency(metric: MetricResult) -> bool:
    tags = _collect_tags(metric)
    reasons = _collect_reasoning(metric)

    if "tone_mixed" in tags or "tone_unclear" in tags:
        return True
    if metric.aggregated_score < 0.55:
        return True
    if metric.aggregated_score < 0.7 and metric.agreement == "low":
        return True
    return any(
        keyword in reasons
        for keyword in ("tone mixed", "inconsistent tone", "speech level mix", "반말", "혼용", "어색한 톤")
    )


def _is_partial_hallucination(metric: MetricResult) -> bool:
    tags = _collect_tags(metric)
    reasons = _collect_reasoning(metric)

    if {"low_grounding", "missing_context"} & tags:
        return True
    if "partial_grounding" in tags and metric.aggregated_score < 0.75:
        return True
    if metric.aggregated_score < 0.6:
        return True
    return any(
        keyword in reasons
        for keyword in (
            "hallucination",
            "unsupported",
            "not supported",
            "beyond the context",
            "contradict",
            "근거 부족",
            "문맥에 없는",
        )
    )


def _is_instruction_violation(metric: MetricResult) -> bool:
    tags = _collect_tags(metric)
    reasons = _collect_reasoning(metric)

    format_tags = {
        "missing_sentence_ending",
        "fragmented_sentence",
        "too_short",
        "empty_answer",
    }
    if format_tags & tags:
        return True
    if metric.aggregated_score < 0.45:
        return True
    if metric.aggregated_score < 0.6 and metric.confidence > 0.75:
        return True
    return any(
        keyword in reasons
        for keyword in ("instruction", "format", "length", "empty", "형식", "지시", "길이")
    )


def _collect_tags(metric: MetricResult) -> set[str]:
    tags: set[str] = set()
    for judge in metric.judges:
        tags.update(tag.lower() for tag in judge.tags)
    return tags


def _collect_reasoning(metric: MetricResult) -> str:
    return " ".join(_normalize_reasoning(judge) for judge in metric.judges)


def _normalize_reasoning(judge: JudgeResult) -> str:
    return (judge.reasoning or "").lower()
