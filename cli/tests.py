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
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Cora: Heart Centered AI Assistant" in result.output
    assert llm_provider in result.output


@pytest.mark.parametrize("llm_provider", SI.LLM_PROVIDERS)
def test_cli_missing_api_key(runner, monkeypatch, llm_provider):
    api_key_env = SI.LLM_CONFIG[llm_provider]["api_key_env"]
    monkeypatch.delenv(api_key_env, raising=False)
    result = runner.invoke(cli, ["-q", "Test query", "-p", llm_provider])
    assert result.exit_code != 0


@pytest.mark.parametrize("llm_provider", SI.LLM_PROVIDERS)
def test_cli_simple_query(runner, monkeypatch, llm_provider):
    api_key_env = SI.LLM_CONFIG[llm_provider]["api_key_env"]
    api_key = os.environ.get(api_key_env)

    if not api_key:
        pytest.skip(f"API key not set for {llm_provider}")

    result = runner.invoke(cli, ["-q", "Tell me a joke", "-p", llm_provider])
    assert result.exit_code == 0
    assert "Response:" in result.output
    assert len(result.output) > 20  # Ensure we got a substantial response
