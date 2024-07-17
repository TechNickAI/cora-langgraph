from cli.app import cli
from click.testing import CliRunner
from heart.si import SI
import os, pytest


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize("llm_provider", SI.LLM_PROVIDERS)
def test_cli_help(runner, llm_provider):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, f"CLI help command failed with exit code {result.exit_code}"
    assert "Usage:" in result.output, "CLI help output is missing 'Usage:' section"
    assert "Cora: Heart Centered AI Assistant" in result.output, "CLI help output is missing the application name"
    assert llm_provider in result.output, f"CLI help output is missing the LLM provider '{llm_provider}'"


@pytest.mark.parametrize("llm_provider", SI.LLM_PROVIDERS)
def test_cli_missing_api_key(runner, monkeypatch, llm_provider):
    api_key_env = SI.LLM_CONFIG[llm_provider]["api_key_env"]
    monkeypatch.delenv(api_key_env, raising=False)
    result = runner.invoke(cli, ["-q", "Test query", "-p", llm_provider])
    assert result.exit_code != 0, f"CLI did not fail when API key for {llm_provider} was missing"


@pytest.mark.parametrize("llm_provider", SI.LLM_PROVIDERS)
def test_cli_simple_query(runner, monkeypatch, llm_provider):
    api_key_env = SI.LLM_CONFIG[llm_provider]["api_key_env"]
    api_key = os.environ.get(api_key_env)

    if not api_key:
        pytest.skip(f"API key not set for {llm_provider}")

    result = runner.invoke(cli, ["-q", "Tell me a joke", "-p", llm_provider])

    error_message = f"""
    CLI query failed for {llm_provider}
    Exit code: {result.exit_code}
    Output:
    {result.output}
    Exception:
    {result.exception}
    """

    assert result.exit_code == 0, error_message
    assert (
        "Response:" in result.output
    ), f"CLI query output is missing 'Response:' for {llm_provider}\nOutput:\n{result.output}"
    assert (
        len(result.output) > 20
    ), f"CLI query response for {llm_provider} is too short (less than 20 characters)\nOutput:\n{result.output}"
