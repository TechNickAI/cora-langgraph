from heart.si import SI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os, pytest


@pytest.fixture
def mock_env_vars(monkeypatch):
    for provider, config in SI.LLM_CONFIG.items():
        monkeypatch.setenv(config["api_key_env"], f"mock_{provider}_key")


@pytest.mark.parametrize("provider", SI.LLM_PROVIDERS)
def test_check_api_key_valid(mock_env_vars, provider):
    assert SI.check_api_key(provider), f"API key check failed for provider: {provider}"


def test_check_api_key_invalid():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        SI.check_api_key("invalid_provider")


@pytest.mark.parametrize("provider", SI.LLM_PROVIDERS)
def test_check_api_key_missing(monkeypatch, provider):
    api_key_env = SI.LLM_CONFIG[provider]["api_key_env"]
    monkeypatch.delenv(api_key_env, raising=False)
    with pytest.raises(ValueError, match=f"API key for {provider} is not set"):
        SI.check_api_key(provider)


@pytest.mark.parametrize(
    "provider,expected_class",
    [
        ("openai", ChatOpenAI),
        ("anthropic", ChatAnthropic),
        ("groq", ChatGroq),
    ],
)
def test_get_chat_model(mock_env_vars, provider, expected_class):
    model = SI.get_chat_model(provider)
    assert isinstance(
        model, expected_class
    ), f"Expected {expected_class.__name__} for provider {provider}, but got {type(model).__name__}"


def test_get_chat_model_invalid():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        SI.get_chat_model("invalid_provider")


def test_create_tools():
    tools_with_search = SI.create_tools({})
    assert len(tools_with_search) >= 1, f"Expected at least 1 tool, but got {len(tools_with_search)}"


@pytest.mark.parametrize("provider", SI.LLM_PROVIDERS)
def test_create_agent_graph(mock_env_vars, provider):
    api_key_env = SI.LLM_CONFIG[provider]["api_key_env"]
    if not os.environ.get(api_key_env):
        pytest.skip(f"API key for {provider} is not set")

    settings = {"llm_provider": provider, "search_web": True}
    graph = SI.create_agent_graph(settings)

    assert graph is not None, f"Agent graph creation failed for provider: {provider}"
