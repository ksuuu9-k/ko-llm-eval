import json
from pathlib import Path
from uuid import uuid4

from typer.testing import CliRunner

from ko_llm_eval.cli.main import app
from ko_llm_eval.metrics.defaults import build_default_registry
from ko_llm_eval.orchestrator.batch import evaluate_batch, load_jsonl_inputs
from ko_llm_eval.orchestrator.evaluator import Evaluator
from ko_llm_eval.storage import JsonlResultWriter


def test_load_jsonl_inputs_reads_multiple_records() -> None:
    inputs = load_jsonl_inputs(Path("examples/sample_batch.jsonl"))

    assert len(inputs) == 2
    assert inputs[0].prompt == "환불 절차 안내"


def test_evaluate_batch_writes_jsonl_output() -> None:
    output_dir = Path(".test_artifacts") / uuid4().hex
    output = output_dir / "results.jsonl"
    inputs = load_jsonl_inputs(Path("examples/sample_batch.jsonl"))
    evaluator = Evaluator(registry=build_default_registry())
    writer = JsonlResultWriter(output)

    try:
        results = evaluate_batch(evaluator, inputs, writer=writer)
    finally:
        writer.close()

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(results) == 2
    assert len(lines) == 2


def test_cli_batch_command_writes_outputs() -> None:
    runner = CliRunner()
    output_dir = Path(".test_artifacts") / uuid4().hex
    output = output_dir / "results.jsonl"
    failures_output = output_dir / "failures.jsonl"

    result = runner.invoke(
        app,
        [
            "batch",
            "--input",
            "examples/sample_batch.jsonl",
            "--output",
            str(output),
            "--failures-output",
            str(failures_output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["processed"] == 2
    assert output.exists()
