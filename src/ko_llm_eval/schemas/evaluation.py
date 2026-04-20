"""Pydantic models for evaluation inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AgreementLevel = Literal["high", "medium", "low"]


class EvaluationInput(BaseModel):
    """Single evaluation payload for a service response."""

    prompt: str = Field(min_length=1)
    context: str | None = None
    answer: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeResult(BaseModel):
    """Per-judge score and reasoning for a single metric."""

    judge_name: str
    metric: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = None
    tags: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Aggregated output for one metric."""

    aggregated_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    agreement: AgreementLevel
    judges: list[JudgeResult] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Final response returned by the evaluator."""

    overall_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    agreement: AgreementLevel
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
