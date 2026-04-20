from ko_llm_eval.judges.llm_client import _parse_json_text


def test_parse_json_text_extracts_fenced_json_and_normalizes_tags() -> None:
    payload = _parse_json_text(
        """```json
        {
          "score": "0.8",
          "confidence": "0.7",
          "reasoning": ["Grounded enough."],
          "tags": "Partial Grounding, Tone-Mixed"
        }
        ```"""
    )

    assert payload == {
        "score": 0.8,
        "confidence": 0.7,
        "reasoning": "Grounded enough.",
        "tags": ["partial_grounding", "tone_mixed"],
    }


def test_parse_json_text_accepts_alias_fields_and_common_scales() -> None:
    payload = _parse_json_text(
        """
        Judge output:
        {
          "overall_score": 4,
          "certainty": 8,
          "explanation": "Mostly grounded.",
          "labels": ["Needs Citation"]
        }
        """
    )

    assert payload == {
        "score": 0.8,
        "confidence": 0.8,
        "reasoning": "Mostly grounded.",
        "tags": ["needs_citation"],
    }
