"""Client abstractions for external LLM judges."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from urllib.parse import urlencode
from urllib import error, request

from ko_llm_eval.config import JudgeModelConfig
from ko_llm_eval.judges.prompting import REPAIR_SYSTEM_PROMPT, build_repair_user_prompt


class LLMClientError(RuntimeError):
    """Raised when an upstream judge model call fails."""


RESPONSE_KEY_ALIASES = {
    "score": ("score", "overall_score", "rating", "value"),
    "confidence": ("confidence", "confidence_score", "certainty"),
    "reasoning": ("reasoning", "reason", "rationale", "explanation", "comment"),
    "tags": ("tags", "labels", "issues", "failure_tags"),
}


class BaseLLMClient(ABC):
    """Minimal interface for chat-completion style judge calls."""

    @abstractmethod
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        """Return a parsed JSON response from the upstream model."""


class HTTPJSONClient(BaseLLMClient):
    """Shared HTTP helpers for provider-specific clients."""

    def __init__(self, config: JudgeModelConfig) -> None:
        self.config = config

    def _post_json(self, *, url: str, headers: dict[str, str], payload: dict) -> dict:
        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise LLMClientError(f"Judge request failed with HTTP {exc.code}: {message}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"Judge request failed: {exc.reason}") from exc

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        raw_response = self._request_response_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        try:
            return self._coerce_payload_to_result(raw_response)
        except LLMClientError as first_error:
            repaired_payload = self._request_response_payload(
                system_prompt=REPAIR_SYSTEM_PROMPT,
                user_prompt=build_repair_user_prompt(_stringify_payload(raw_response)),
            )
            try:
                return self._coerce_payload_to_result(repaired_payload)
            except LLMClientError as repair_error:
                raise LLMClientError(
                    f"{first_error} Repair attempt failed: {repair_error}"
                ) from repair_error

    @abstractmethod
    def _request_response_payload(self, *, system_prompt: str, user_prompt: str):
        """Return the provider-specific raw response payload."""

    @abstractmethod
    def _coerce_payload_to_result(self, payload) -> dict:
        """Normalize a provider response payload into the shared judge format."""


class OpenAICompatibleClient(HTTPJSONClient):
    """Simple OpenAI-compatible chat completion client."""

    def _request_response_payload(self, *, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.config.extra_body:
            payload.update(self.config.extra_body)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        body = self._post_json(
            url=f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload,
        )
        return body

    def _coerce_payload_to_result(self, payload) -> dict:
        return _parse_openai_content(payload)


class AnthropicClient(HTTPJSONClient):
    """Anthropic Messages API client."""

    def _request_response_payload(self, *, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.config.model,
            "system": system_prompt,
            "temperature": self.config.temperature,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.config.extra_body:
            payload.update(self.config.extra_body)

        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        body = self._post_json(
            url=f"{self.config.base_url.rstrip('/')}/v1/messages",
            headers=headers,
            payload=payload,
        )
        return body

    def _coerce_payload_to_result(self, payload) -> dict:
        return _parse_json_text(_extract_anthropic_text(payload))


class GeminiClient(HTTPJSONClient):
    """Google Gemini generateContent API client."""

    def _request_response_payload(self, *, system_prompt: str, user_prompt: str) -> dict:
        model = self.config.model
        query = urlencode({"key": self.config.api_key})
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.config.temperature,
                "responseMimeType": "application/json",
            },
        }
        if self.config.extra_body:
            payload.update(self.config.extra_body)

        headers = {
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        body = self._post_json(
            url=f"{self.config.base_url.rstrip('/')}/v1beta/models/{model}:generateContent?{query}",
            headers=headers,
            payload=payload,
        )
        return body

    def _coerce_payload_to_result(self, payload) -> dict:
        return _parse_json_text(_extract_gemini_text(payload))


class CustomAPIClient(HTTPJSONClient):
    """Generic custom HTTP endpoint client for internal models."""

    def _request_response_payload(self, *, system_prompt: str, user_prompt: str) -> dict:
        api_path = self.config.api_path or "/judge"
        payload = {
            "model": self.config.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": self.config.temperature,
        }
        if self.config.extra_body:
            payload.update(self.config.extra_body)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)

        body = self._post_json(
            url=f"{self.config.base_url.rstrip('/')}/{api_path.lstrip('/')}",
            headers=headers,
            payload=payload,
        )
        return body

    def _coerce_payload_to_result(self, payload) -> dict:
        return _parse_custom_body(payload)


def build_llm_client(config: JudgeModelConfig) -> BaseLLMClient:
    """Instantiate the appropriate client for the configured provider."""

    provider = config.provider
    if provider == "openai":
        return OpenAICompatibleClient(config)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    if provider == "anthropic":
        return AnthropicClient(config)
    if provider == "gemini":
        return GeminiClient(config)
    if provider == "custom":
        return CustomAPIClient(config)
    raise LLMClientError(f"Unsupported provider: {provider}")


def _parse_openai_content(body: dict) -> dict:
    try:
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return _parse_json_text(content)
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("Judge response was not valid OpenAI-compatible content.") from exc


def _extract_anthropic_text(body: dict) -> str:
    try:
        blocks = body["content"]
        return "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
    except (KeyError, TypeError) as exc:
        raise LLMClientError("Judge response was not valid Anthropic content.") from exc


def _extract_gemini_text(body: dict) -> str:
    try:
        parts = body["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("Judge response was not valid Gemini content.") from exc


def _parse_custom_body(body: dict) -> dict:
    if isinstance(body.get("result"), dict):
        return _normalize_judge_payload(body["result"])
    if isinstance(body.get("output"), dict):
        return _normalize_judge_payload(body["output"])
    if isinstance(body.get("data"), dict):
        return _normalize_judge_payload(body["data"])
    if {"score", "confidence", "reasoning", "tags"} & set(body.keys()):
        return _normalize_judge_payload(body)
    raise LLMClientError("Judge response was not valid custom provider content.")


def _parse_json_text(text: str) -> dict:
    try:
        return _normalize_judge_payload(json.loads(text))
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        if extracted is None:
            raise LLMClientError("Judge response was not valid JSON content.") from None
        try:
            return _normalize_judge_payload(json.loads(extracted))
        except json.JSONDecodeError as exc:
            raise LLMClientError("Judge response was not valid JSON content.") from exc


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _normalize_judge_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise LLMClientError("Judge response payload must be a JSON object.")

    return {
        "score": _normalize_score(_get_first_alias(payload, "score"), field_name="score"),
        "confidence": _normalize_score(_get_first_alias(payload, "confidence"), field_name="confidence"),
        "reasoning": _normalize_reasoning(_get_first_alias(payload, "reasoning")),
        "tags": _normalize_tags(_get_first_alias(payload, "tags")),
    }


def _get_first_alias(payload: dict, field_name: str):
    for alias in RESPONSE_KEY_ALIASES[field_name]:
        if alias in payload:
            return payload[alias]
    return None


def _normalize_score(value, *, field_name: str) -> float:
    if value is None:
        return 0.0

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMClientError(f"Judge response field '{field_name}' must be numeric.") from exc

    if 0.0 <= numeric <= 1.0:
        return numeric
    if 1.0 < numeric <= 5.0:
        return numeric / 5.0
    if 5.0 < numeric <= 10.0:
        return numeric / 10.0
    return max(0.0, min(1.0, numeric))


def _normalize_reasoning(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _normalize_tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = re.split(r"[,/\n|]+", value)
    elif isinstance(value, list):
        candidates = [str(item) for item in value]
    else:
        candidates = [str(value)]

    tags: list[str] = []
    for candidate in candidates:
        cleaned = _slugify_tag(candidate)
        if cleaned and cleaned not in tags:
            tags.append(cleaned)
    return tags


def _slugify_tag(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _stringify_payload(payload) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)
