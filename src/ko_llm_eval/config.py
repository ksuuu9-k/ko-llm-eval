"""Project-wide configuration defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_METRICS = ("tone", "fluency", "grounding")

ProviderType = Literal["openai", "anthropic", "gemini", "openai_compatible", "custom"]


@dataclass(slots=True)
class JudgeModelConfig:
    """Provider configuration for externally hosted judge models."""

    provider: ProviderType
    model: str
    api_key: str
    name: str | None = None
    weight: float | None = None
    base_url: str = "https://api.openai.com/v1"
    api_path: str | None = None
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    extra_headers: dict[str, str] | None = None
    extra_body: dict | None = None


@dataclass(slots=True)
class Settings:
    """Runtime settings used to wire the default registry."""

    semantic_judges: list[JudgeModelConfig] | None = None
    rubric_judges: list[JudgeModelConfig] | None = None


def _load_judge_model_config(prefix: str) -> JudgeModelConfig | None:
    model = os.getenv(f"{prefix}_MODEL")
    api_key = os.getenv(f"{prefix}_API_KEY")

    if not model or not api_key:
        return None

    provider = os.getenv(f"{prefix}_PROVIDER", "openai_compatible").lower()
    name = os.getenv(f"{prefix}_NAME")
    weight_raw = os.getenv(f"{prefix}_WEIGHT")
    base_url = os.getenv(f"{prefix}_BASE_URL", "https://api.openai.com/v1")
    api_path = os.getenv(f"{prefix}_API_PATH")
    timeout_seconds = float(os.getenv(f"{prefix}_TIMEOUT_SECONDS", "30"))
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0"))
    extra_headers_raw = os.getenv(f"{prefix}_EXTRA_HEADERS")
    extra_body_raw = os.getenv(f"{prefix}_EXTRA_BODY")

    extra_headers = json.loads(extra_headers_raw) if extra_headers_raw else None
    extra_body = json.loads(extra_body_raw) if extra_body_raw else None

    return JudgeModelConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        name=name,
        weight=float(weight_raw) if weight_raw is not None else None,
        base_url=base_url,
        api_path=api_path,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )


def _load_judge_model_configs(prefix: str) -> list[JudgeModelConfig]:
    indices: set[str] = set()

    for key in os.environ:
        if key.startswith(prefix) and key.endswith("_MODEL"):
            suffix = key[len(prefix) : -len("_MODEL")]
            if suffix.startswith("_"):
                indices.add(suffix[1:])

    configs = [
        config
        for suffix in sorted(indices)
        if (config := _load_judge_model_config(f"{prefix}_{suffix}")) is not None
    ]
    return configs


def load_settings() -> Settings:
    """Load provider configuration from environment variables."""

    semantic_judges = _load_judge_model_configs("KO_LLM_EVAL_SEMANTIC_JUDGE")
    rubric_judges = _load_judge_model_configs("KO_LLM_EVAL_RUBRIC_JUDGE")

    legacy_semantic = _load_judge_model_config("KO_LLM_EVAL_LLM_JUDGE")
    legacy_rubric = _load_judge_model_config("KO_LLM_EVAL_KO_RUBRIC_JUDGE")

    if legacy_semantic is not None:
        semantic_judges.append(legacy_semantic)
    if legacy_rubric is not None:
        rubric_judges.append(legacy_rubric)

    return Settings(
        semantic_judges=semantic_judges,
        rubric_judges=rubric_judges,
    )
