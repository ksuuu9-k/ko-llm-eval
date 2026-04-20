"""Deterministic rule-based judge for MVP metrics."""

from __future__ import annotations

import re

from ko_llm_eval.judges.base import BaseJudge, UnsupportedMetricError
from ko_llm_eval.schemas import EvaluationInput, JudgeResult

HONORIFIC_ENDINGS = ("습니다", "세요", "드립니다", "가능합니다", "됩니다", "하십시오")
CASUAL_ENDINGS = ("야", "해", "했어", "할게", "같아", "줘")
SENTENCE_ENDINGS = ("다.", "요.", "니다.", "까?", "요?", "다?", "!" , ".")
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]{2,}")


class RuleJudge(BaseJudge):
    """Simple heuristic judge for tone, fluency, and grounding."""

    def __init__(self, name: str = "rule", weight: float = 0.2) -> None:
        super().__init__(name=name, weight=weight)

    def evaluate(self, metric: str, evaluation_input: EvaluationInput) -> JudgeResult:
        if metric == "tone":
            score, reasoning, tags = self._evaluate_tone(evaluation_input.answer)
        elif metric == "fluency":
            score, reasoning, tags = self._evaluate_fluency(evaluation_input.answer)
        elif metric == "grounding":
            score, reasoning, tags = self._evaluate_grounding(
                evaluation_input.context,
                evaluation_input.answer,
            )
        else:
            raise UnsupportedMetricError(f"{self.name} does not support metric '{metric}'.")

        return JudgeResult(
            judge_name=self.name,
            metric=metric,
            score=score,
            confidence=0.9,
            reasoning=reasoning,
            tags=tags,
        )

    def _evaluate_tone(self, answer: str) -> tuple[float, str, list[str]]:
        honorific_hits = sum(ending in answer for ending in HONORIFIC_ENDINGS)
        casual_hits = sum(ending in answer for ending in CASUAL_ENDINGS)
        tags: list[str] = []

        if honorific_hits and casual_hits:
            tags.append("tone_mixed")
            return 0.35, "존댓말과 반말 표현이 함께 감지되었습니다.", tags
        if honorific_hits:
            return 0.95, "존댓말 톤이 비교적 일관적으로 유지되었습니다.", tags
        if casual_hits:
            return 0.7, "반말 표현이 감지되었지만 혼용은 확인되지 않았습니다.", tags

        tags.append("tone_unclear")
        return 0.6, "명확한 톤 신호가 적어 일관성을 확신하기 어렵습니다.", tags

    def _evaluate_fluency(self, answer: str) -> tuple[float, str, list[str]]:
        tags: list[str] = []
        score = 1.0

        if not answer.strip():
            return 0.0, "응답이 비어 있습니다.", ["empty_answer"]

        if not answer.endswith(SENTENCE_ENDINGS):
            score -= 0.15
            tags.append("missing_sentence_ending")

        if "  " in answer:
            score -= 0.05
            tags.append("double_space")

        short_sentences = [segment for segment in re.split(r"[.!?]\s*", answer) if segment]
        if any(len(segment.strip()) < 2 for segment in short_sentences):
            score -= 0.1
            tags.append("fragmented_sentence")

        if len(answer) < 5:
            score -= 0.2
            tags.append("too_short")

        score = max(0.0, min(1.0, score))
        reasoning = "기본 문장 종결과 길이 기준으로 자연스러움을 평가했습니다."
        if tags:
            reasoning = f"{reasoning} 감지된 이슈: {', '.join(tags)}."
        return score, reasoning, tags

    def _evaluate_grounding(
        self,
        context: str | None,
        answer: str,
    ) -> tuple[float, str, list[str]]:
        if not context:
            return 0.5, "참조 context가 없어 grounding을 제한적으로 평가했습니다.", ["missing_context"]

        context_tokens = set(TOKEN_PATTERN.findall(context))
        answer_tokens = set(TOKEN_PATTERN.findall(answer))

        if not context_tokens or not answer_tokens:
            return 0.5, "유효한 토큰이 부족해 grounding을 제한적으로 평가했습니다.", ["insufficient_tokens"]

        overlap = context_tokens & answer_tokens
        overlap_ratio = len(overlap) / len(answer_tokens)

        if overlap_ratio >= 0.5:
            return 0.9, "응답이 context와 충분히 겹치는 표현을 포함합니다.", []
        if overlap_ratio >= 0.25:
            return 0.7, "응답이 context를 부분적으로 반영합니다.", ["partial_grounding"]
        return 0.35, "응답과 context 간 표현 겹침이 낮습니다.", ["low_grounding"]
