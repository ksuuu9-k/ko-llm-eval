"""Storage backends for evaluation results."""

from ko_llm_eval.storage.base import ResultWriter
from ko_llm_eval.storage.jsonl import JsonlResultWriter

__all__ = ["JsonlResultWriter", "ResultWriter"]
