from ko_llm_eval.config import JudgeModelConfig
from ko_llm_eval.judges.llm_client import HTTPJSONClient


class RepairableTestClient(HTTPJSONClient):
    def __init__(self, responses: list[str | dict]) -> None:
        super().__init__(
            JudgeModelConfig(
                provider="custom",
                model="test-model",
                api_key="test-key",
                base_url="https://example.com",
            )
        )
        self.responses = responses
        self.calls: list[dict[str, str]] = []

    def _request_response_payload(self, *, system_prompt: str, user_prompt: str):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.responses.pop(0)

    def _coerce_payload_to_result(self, payload) -> dict:
        if isinstance(payload, str):
            from ko_llm_eval.judges.llm_client import _parse_json_text

            return _parse_json_text(payload)
        raise TypeError("Unexpected payload type for test client.")


def test_http_client_repairs_malformed_response_once() -> None:
    client = RepairableTestClient(
        responses=[
            "I think the answer is good. score=0.8 confidence=0.7",
            '{"score":0.8,"confidence":0.7,"reasoning":"Reformatted.","tags":["ok"]}',
        ]
    )

    result = client.complete_json(
        system_prompt="judge",
        user_prompt="evaluate",
    )

    assert result["score"] == 0.8
    assert result["confidence"] == 0.7
    assert len(client.calls) == 2
    assert "Rewrite it as exactly one JSON object" in client.calls[1]["user_prompt"]
