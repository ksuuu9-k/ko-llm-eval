"""Default metric-to-judge wiring for the MVP skeleton."""

from __future__ import annotations

from ko_llm_eval.config import JudgeModelConfig, Settings, load_settings
from ko_llm_eval.judges import (
    KoreanRubricJudge,
    LLMJudge,
    MockJudge,
    RuleJudge,
    build_llm_client,
)
from ko_llm_eval.metrics.registry import MetricRegistry


def _distributed_weight(configs: list[JudgeModelConfig], total_weight: float) -> float:
    if not configs:
        return total_weight
    return total_weight / len(configs)


def build_semantic_judges(settings: Settings, metric: str) -> list:
    if settings.semantic_judges:
        default_weight = _distributed_weight(settings.semantic_judges, 0.45)
        return [
            LLMJudge(
                client=build_llm_client(config),
                name=config.name or f"semantic_{index}",
                weight=config.weight if config.weight is not None else default_weight,
            )
            for index, config in enumerate(settings.semantic_judges, start=1)
        ]

    return [
        MockJudge(
            name="mock_llm",
            score_map={metric: 0.85 if metric == "tone" else 0.82 if metric == "fluency" else 0.78},
            weight=0.45,
        )
    ]


def build_rubric_judges(settings: Settings, metric: str) -> list:
    if settings.rubric_judges:
        default_weight = _distributed_weight(settings.rubric_judges, 0.35)
        return [
            KoreanRubricJudge(
                client=build_llm_client(config),
                name=config.name or f"rubric_{index}",
                weight=config.weight if config.weight is not None else default_weight,
            )
            for index, config in enumerate(settings.rubric_judges, start=1)
        ]

    return [
        MockJudge(
            name="mock_ko_rubric",
            score_map={metric: 0.9 if metric == "tone" else 0.88 if metric == "fluency" else 0.74},
            weight=0.35,
        )
    ]


def build_default_registry(settings: Settings | None = None) -> MetricRegistry:
    """Return a registry that prefers real judges when configured."""

    settings = settings or load_settings()

    registry = MetricRegistry()

    registry.register(
        "tone",
        [
            RuleJudge(weight=0.2),
            *build_semantic_judges(settings, "tone"),
            *build_rubric_judges(settings, "tone"),
        ],
    )
    registry.register(
        "fluency",
        [
            RuleJudge(weight=0.2),
            *build_semantic_judges(settings, "fluency"),
            *build_rubric_judges(settings, "fluency"),
        ],
    )
    registry.register(
        "grounding",
        [
            RuleJudge(weight=0.2),
            *build_semantic_judges(settings, "grounding"),
            *build_rubric_judges(settings, "grounding"),
        ],
    )

    return registry
