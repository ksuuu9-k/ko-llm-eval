from ko_llm_eval.config import load_settings


def test_load_settings_supports_provider_fields(monkeypatch) -> None:
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_PROVIDER", "anthropic")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_MODEL", "claude-test")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_API_KEY", "secret")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_EXTRA_HEADERS", '{"x-test":"1"}')
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_EXTRA_BODY", '{"max_tokens":512}')

    settings = load_settings()

    assert settings.semantic_judges is not None
    assert settings.semantic_judges[0].provider == "anthropic"
    assert settings.semantic_judges[0].extra_headers == {"x-test": "1"}
    assert settings.semantic_judges[0].extra_body == {"max_tokens": 512}


def test_load_settings_supports_multiple_semantic_and_rubric_judges(monkeypatch) -> None:
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_PROVIDER", "openai")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_MODEL", "gpt-test")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_API_KEY", "a")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_1_NAME", "semantic_openai")

    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_2_PROVIDER", "anthropic")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_2_MODEL", "claude-test")
    monkeypatch.setenv("KO_LLM_EVAL_SEMANTIC_JUDGE_2_API_KEY", "b")

    monkeypatch.setenv("KO_LLM_EVAL_RUBRIC_JUDGE_1_PROVIDER", "gemini")
    monkeypatch.setenv("KO_LLM_EVAL_RUBRIC_JUDGE_1_MODEL", "gemini-test")
    monkeypatch.setenv("KO_LLM_EVAL_RUBRIC_JUDGE_1_API_KEY", "c")

    settings = load_settings()

    assert settings.semantic_judges is not None
    assert len(settings.semantic_judges) == 2
    assert settings.semantic_judges[0].name == "semantic_openai"
    assert settings.rubric_judges is not None
    assert len(settings.rubric_judges) == 1
