"""CLI entrypoint for ko-llm-eval."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ko_llm_eval.metrics.defaults import build_default_registry
from ko_llm_eval.orchestrator.batch import evaluate_batch, load_jsonl_inputs
from ko_llm_eval.orchestrator.evaluator import Evaluator
from ko_llm_eval.schemas import EvaluationInput
from ko_llm_eval.storage import JsonlResultWriter

app = typer.Typer(help="Korean LLM evaluation orchestration CLI.")


@app.command()
def version() -> None:
    """Print the current package version placeholder."""
    typer.echo("ko-llm-eval 0.1.0")


@app.command()
def run(input: Path) -> None:
    """Run a single evaluation from a JSON file."""
    payload = json.loads(input.read_text(encoding="utf-8"))
    evaluation_input = EvaluationInput.model_validate(payload)

    evaluator = Evaluator(registry=build_default_registry())
    result = evaluator.evaluate(evaluation_input)

    typer.echo(result.model_dump_json(indent=2))


@app.command()
def batch(
    input: Path = typer.Option(..., "--input", help="Input JSONL file."),
    output: Path = typer.Option(..., "--output", help="Output JSONL file."),
    failures_output: Path | None = typer.Option(
        None,
        "--failures-output",
        help="Optional JSONL file that stores only failed evaluations.",
    ),
) -> None:
    """Run a batch evaluation from a JSONL file and store results."""
    evaluator = Evaluator(registry=build_default_registry())
    batch_inputs = load_jsonl_inputs(input)

    writer = JsonlResultWriter(output)
    failure_writer = JsonlResultWriter(failures_output) if failures_output is not None else None

    try:
        results = evaluate_batch(evaluator, batch_inputs, writer=writer)
        if failure_writer is not None:
            for result in results:
                if result.failures:
                    failure_writer.write(result)
    finally:
        writer.close()
        if failure_writer is not None:
            failure_writer.close()

    typer.echo(
        json.dumps(
            {
                "processed": len(results),
                "output": str(output),
                "failures_output": str(failures_output) if failures_output is not None else None,
            },
            ensure_ascii=False,
        )
    )
