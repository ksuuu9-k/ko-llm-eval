"""High-level evaluation entrypoint."""

from __future__ import annotations

from ko_llm_eval.aggregator.agreement import calculate_agreement
from ko_llm_eval.aggregator.confidence import calculate_confidence
from ko_llm_eval.aggregator.weighted_mean import weighted_mean
from ko_llm_eval.failure_detection.detector import detect_failures
from ko_llm_eval.metrics.registry import MetricRegistry
from ko_llm_eval.schemas import EvaluationInput, EvaluationResult, MetricResult


class Evaluator:
    """Run metric-level evaluations and return an aggregated result."""

    def __init__(self, registry: MetricRegistry) -> None:
        self.registry = registry

    def evaluate(self, evaluation_input: EvaluationInput) -> EvaluationResult:
        metric_outputs: dict[str, MetricResult] = {}

        for metric in self.registry.metrics():
            judges = self.registry.get_judges(metric)
            judge_results = [judge.evaluate(metric, evaluation_input) for judge in judges]

            metric_outputs[metric] = MetricResult(
                aggregated_score=weighted_mean(judge_results, judges),
                confidence=calculate_confidence(judge_results),
                agreement=calculate_agreement(judge_results),
                judges=judge_results,
            )

        overall_score = (
            sum(metric.aggregated_score for metric in metric_outputs.values()) / len(metric_outputs)
            if metric_outputs
            else 0.0
        )
        overall_confidence = (
            sum(metric.confidence for metric in metric_outputs.values()) / len(metric_outputs)
            if metric_outputs
            else 0.0
        )
        overall_agreement = calculate_agreement(
            [
                result
                for metric_result in metric_outputs.values()
                for result in metric_result.judges
            ]
        )

        return EvaluationResult(
            overall_score=overall_score,
            confidence=overall_confidence,
            agreement=overall_agreement,
            metrics=metric_outputs,
            failures=detect_failures(metric_outputs),
        )
