"""Prompt builders shared by LLM-based judges."""

from __future__ import annotations

from ko_llm_eval.schemas import EvaluationInput


LLM_JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for Korean-language LLM application responses.
Return strict JSON with keys: score, confidence, reasoning, tags.
Scores must be normalized between 0 and 1.
Reasoning should be concise and evidence-based.
Tags must be a JSON array of short snake_case strings.
Do not wrap the JSON in markdown fences.
Do not include any preamble, explanation, or trailing text.
Return exactly one JSON object and nothing else.
""".strip()


RUBRIC_SYSTEM_PROMPT = """
You are a Korean-native quality judge focused on service-response quality.
Evaluate only the requested metric using the provided Korean rubric.
Return strict JSON with keys: score, confidence, reasoning, tags.
Scores must be normalized between 0 and 1.
Do not wrap the JSON in markdown fences.
Do not include any preamble, explanation, or trailing text.
Return exactly one JSON object and nothing else.
""".strip()


REPAIR_SYSTEM_PROMPT = """
You convert malformed judge outputs into strict JSON.
Return exactly one JSON object with keys: score, confidence, reasoning, tags.
Do not include markdown fences.
Do not include any extra text.
If a field is missing, infer the safest possible value.
""".strip()


METRIC_INSTRUCTIONS = {
    "tone": "Check whether the answer keeps a consistent and appropriate Korean speech level and tone.",
    "fluency": "Check whether the Korean phrasing sounds natural, fluent, and native-like.",
    "grounding": "Check whether the answer is supported by the provided context and avoids unsupported claims.",
}


KOREAN_RUBRICS = {
    "tone": """
- High score: consistent honorific/casual style, no mixing between sentence endings, service-appropriate tone.
- Medium score: mostly consistent but slightly awkward or uneven tone.
- Low score: clear speech-level mixing, abrupt style shifts, impolite or inconsistent service tone.
""".strip(),
    "fluency": """
- High score: natural Korean wording, idiomatic phrasing, smooth sentence flow.
- Medium score: understandable but slightly translated or stiff phrasing.
- Low score: awkward, unnatural, fragmented, or clearly non-native Korean output.
""".strip(),
    "grounding": """
- High score: claims are traceable to context, no meaningful unsupported additions.
- Medium score: mostly grounded but includes small extrapolations or imprecise phrasing.
- Low score: unsupported claims, contradictions with context, or hallucinated details.
""".strip(),
}


def build_llm_user_prompt(metric: str, evaluation_input: EvaluationInput) -> str:
    """Build a provider-neutral prompt for general semantic judges."""

    metric_instruction = METRIC_INSTRUCTIONS[metric]
    context = evaluation_input.context or "(no context provided)"
    return f"""
Metric: {metric}
Instruction: {metric_instruction}

Prompt:
{evaluation_input.prompt}

Context:
{context}

Answer:
{evaluation_input.answer}

Respond as JSON:
{{
  "score": 0.0,
  "confidence": 0.0,
  "reasoning": "short explanation",
  "tags": ["tag_if_needed"]
}}

Constraints:
- score and confidence must be numbers between 0 and 1
- reasoning must be a single string
- tags must be an array of strings
- do not output markdown or any extra text
""".strip()


def build_korean_rubric_prompt(metric: str, evaluation_input: EvaluationInput) -> str:
    """Build a rubric-heavy prompt for Korean quality judging."""

    rubric = KOREAN_RUBRICS[metric]
    context = evaluation_input.context or "(no context provided)"
    return f"""
Metric: {metric}
Korean Rubric:
{rubric}

User Prompt:
{evaluation_input.prompt}

Reference Context:
{context}

Model Answer:
{evaluation_input.answer}

Evaluate according to the Korean rubric only and return JSON:
{{
  "score": 0.0,
  "confidence": 0.0,
  "reasoning": "short explanation in Korean or English",
  "tags": ["tag_if_needed"]
}}

Constraints:
- score and confidence must be numbers between 0 and 1
- reasoning must be a single string
- tags must be an array of strings
- do not output markdown or any extra text
""".strip()


def build_repair_user_prompt(raw_response: str) -> str:
    """Ask the model to reformat a malformed judge response into the required schema."""

    return f"""
The following judge output did not match the required response format.
Rewrite it as exactly one JSON object with this schema:
{{
  "score": 0.0,
  "confidence": 0.0,
  "reasoning": "short explanation",
  "tags": ["tag_if_needed"]
}}

Constraints:
- score and confidence must be numeric values between 0 and 1 when possible
- reasoning must be a single string
- tags must be an array of strings
- do not include markdown fences
- do not include any text before or after the JSON

Original output:
{raw_response}
""".strip()
