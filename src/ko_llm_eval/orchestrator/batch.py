"""Batch evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ko_llm_eval.orchestrator.evaluator import Evaluator
from ko_llm_eval.schemas import EvaluationInput, EvaluationResult
from ko_llm_eval.storage.base import ResultWriter


def load_jsonl_inputs(path: Path) -> list[EvaluationInput]:
    """Load a JSONL batch input file into evaluation payloads."""

    inputs: list[EvaluationInput] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            try:
                inputs.append(EvaluationInput.model_validate(payload))
            except Exception as exc:
                raise ValueError(f"Invalid evaluation input on line {line_number}: {exc}") from exc
    return inputs


def evaluate_batch(
    evaluator: Evaluator,
    inputs: list[EvaluationInput],
    writer: ResultWriter | None = None,
) -> list[EvaluationResult]:
    """Run the evaluator across a batch and optionally stream results to disk."""

    results: list[EvaluationResult] = []
    for evaluation_input in inputs:
        result = evaluator.evaluate(evaluation_input)
        results.append(result)
        if writer is not None:
            writer.write(result)
    return results
