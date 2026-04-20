from ko_llm_eval.config import JudgeModelConfig, Settings
from ko_llm_eval.metrics.defaults import build_default_registry, build_rubric_judges, build_semantic_judges


def test_build_semantic_and_rubric_judges_are_separated() -> None:
    settings = Settings(
        semantic_judges=[
            JudgeModelConfig(provider="openai", model="gpt-a", api_key="a", name="sem_a"),
            JudgeModelConfig(provider="anthropic", model="claude-b", api_key="b", name="sem_b"),
        ],
        rubric_judges=[
            JudgeModelConfig(provider="gemini", model="gemini-c", api_key="c", name="rub_c"),
        ],
    )

    semantic = build_semantic_judges(settings, "tone")
    rubric = build_rubric_judges(settings, "tone")

    assert [judge.name for judge in semantic] == ["sem_a", "sem_b"]
    assert [judge.name for judge in rubric] == ["rub_c"]


def test_build_default_registry_includes_all_configured_judges() -> None:
    settings = Settings(
        semantic_judges=[
            JudgeModelConfig(provider="openai", model="gpt-a", api_key="a", name="sem_a"),
            JudgeModelConfig(provider="anthropic", model="claude-b", api_key="b", name="sem_b"),
        ],
        rubric_judges=[
            JudgeModelConfig(provider="gemini", model="gemini-c", api_key="c", name="rub_c"),
            JudgeModelConfig(provider="custom", model="internal-d", api_key="d", name="rub_d"),
        ],
    )

    registry = build_default_registry(settings)
    tone_judges = registry.get_judges("tone")

    assert [judge.name for judge in tone_judges] == ["rule", "sem_a", "sem_b", "rub_c", "rub_d"]
