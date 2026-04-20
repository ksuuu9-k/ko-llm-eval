from ko_llm_eval.metrics.defaults import build_default_registry


def test_default_registry_uses_fallback_judges_without_env() -> None:
    registry = build_default_registry()

    tone_judges = registry.get_judges("tone")

    assert len(tone_judges) == 3
    assert tone_judges[0].name == "rule"
