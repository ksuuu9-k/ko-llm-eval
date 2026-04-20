"""JSONL result writer."""

from __future__ import annotations

from pathlib import Path

from ko_llm_eval.schemas import EvaluationResult
from ko_llm_eval.storage.base import ResultWriter, ensure_parent_dir


class JsonlResultWriter(ResultWriter):
    """Append serialized evaluation results to a JSONL file."""

    def __init__(self, path: Path) -> None:
        ensure_parent_dir(path)
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def write(self, result: EvaluationResult) -> None:
        self._handle.write(result.model_dump_json())
        self._handle.write("\n")

    def close(self) -> None:
        self._handle.close()
