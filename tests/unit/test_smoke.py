from typer.testing import CliRunner
from pathlib import Path

from ko_llm_eval.cli.main import app


def test_cli_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_run_command_uses_sample_input() -> None:
    runner = CliRunner()
    sample = Path("examples/sample_request.json")

    result = runner.invoke(app, ["run", str(sample)])

    assert result.exit_code == 0
    assert '"overall_score"' in result.stdout
