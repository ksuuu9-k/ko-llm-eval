"""Storage backend interfaces for evaluation results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ko_llm_eval.schemas import EvaluationResult


class ResultWriter(ABC):
    """Abstract writer for serialized evaluation outputs."""

    @abstractmethod
    def write(self, result: EvaluationResult) -> None:
        """Persist a single evaluation result."""

    @abstractmethod
    def close(self) -> None:
        """Release any open file handles."""


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an output path when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
