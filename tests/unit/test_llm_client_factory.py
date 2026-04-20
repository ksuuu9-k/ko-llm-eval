from ko_llm_eval.config import JudgeModelConfig
from ko_llm_eval.judges import (
    AnthropicClient,
    CustomAPIClient,
    GeminiClient,
    OpenAICompatibleClient,
    build_llm_client,
)


def _config(provider: str) -> JudgeModelConfig:
    return JudgeModelConfig(
        provider=provider,
        model="test-model",
        api_key="test-key",
        base_url="https://example.com",
    )


def test_factory_builds_openai_compatible_clients() -> None:
    assert isinstance(build_llm_client(_config("openai")), OpenAICompatibleClient)
    assert isinstance(build_llm_client(_config("openai_compatible")), OpenAICompatibleClient)


def test_factory_builds_anthropic_client() -> None:
    assert isinstance(build_llm_client(_config("anthropic")), AnthropicClient)


def test_factory_builds_gemini_client() -> None:
    assert isinstance(build_llm_client(_config("gemini")), GeminiClient)


def test_factory_builds_custom_client() -> None:
    assert isinstance(build_llm_client(_config("custom")), CustomAPIClient)
