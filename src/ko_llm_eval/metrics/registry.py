"""Metric registry for mapping metrics to judges."""

from __future__ import annotations

from collections.abc import Iterable

from ko_llm_eval.judges import BaseJudge


class MetricRegistry:
    """A lightweight registry mapping metrics to judge lists."""

    def __init__(self) -> None:
        self._registry: dict[str, list[BaseJudge]] = {}

    def register(self, metric: str, judges: Iterable[BaseJudge]) -> None:
        self._registry[metric] = list(judges)

    def get_judges(self, metric: str) -> list[BaseJudge]:
        return self._registry.get(metric, [])

    def metrics(self) -> list[str]:
        return list(self._registry.keys())
