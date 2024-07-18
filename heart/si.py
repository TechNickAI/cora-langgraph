from heart.prompts import assistant_prompt_text, prompt_engineer_prompt_text
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os


class SI:
    # LLM providers
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"

    # LLM provider configuration
    LLM_CONFIG = {
        OPENAI: {
            "model": "gpt-4",
            "api_key_env": "OPENAI_API_KEY",
            "default_temperature": 0.7,
        },
        ANTHROPIC: {
            "model": "claude-3-sonnet-20240229",
            "api_key_env": "ANTHROPIC_API_KEY",
            "default_temperature": 0.7,
        },
        GROQ: {
            "model": "mixtral-8x7b-32768",
            "api_key_env": "GROQ_API_KEY",
            "default_temperature": 0.7,
        },
    }
    # Groq isn't fully supported yet because it handles streaming differently
    LLM_PROVIDERS = [OPENAI, ANTHROPIC]

    @staticmethod
    def check_api_key(llm_provider):
        llm_provider = llm_provider.lower()
        if llm_provider not in SI.LLM_CONFIG:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

        api_key_env = SI.LLM_CONFIG[llm_provider]["api_key_env"]
        if os.getenv(api_key_env) is None:
            raise ValueError(
                f"API key for {llm_provider} is not set. Please set the {api_key_env} environment variable."
            )
        return True

    @staticmethod
    def create_agent_graph(settings):
        llm = SI.get_chat_model(settings["llm_provider"])
        # Set up a memory saver
        memory = MemorySaver()

        tools = SI.create_tools(settings)

        return create_react_agent(model=llm, tools=tools, checkpointer=memory, messages_modifier=assistant_prompt_text)

    @staticmethod
    def create_tools(settings):
        tools = []

        tools.append(TavilySearchResults())

        return tools

    @staticmethod
    def get_chat_model(llm_provider, temperature=None, streaming=True):
        llm_provider = llm_provider.lower()
        SI.check_api_key(llm_provider)

        config = SI.LLM_CONFIG[llm_provider]
        temp = temperature if temperature is not None else config["default_temperature"]

        if llm_provider == SI.OPENAI:
            return ChatOpenAI(model=config["model"], temperature=temp, streaming=streaming)
        elif llm_provider == SI.ANTHROPIC:
            return ChatAnthropic(model=config["model"], temperature=temp, streaming=streaming)
        elif llm_provider == SI.GROQ:
            return ChatGroq(model=config["model"], temperature=temp, streaming=streaming)
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

    @staticmethod
    def parse_response(response):
        if "messages" not in response:
            return response

        messages = response["messages"]
        if len(messages) < 2:
            return response

        ai_message = messages[-1]  # Get the last message, which should be the AI's response

        if isinstance(ai_message.content, str):
            # OpenAI format
            return ai_message.content
        elif isinstance(ai_message.content, list):
            # Anthropic format
            return ai_message.content[0]["text"]
        else:
            # Unknown format, return as is
            return response

    @staticmethod
    def prompt_engineer(user_request):
        # Take a user request, and make it better (prompt engineer it) using groq
        chat = ChatGroq(temperature=0.5, streaming=False)

        human = "{user_request}"
        prompt = ChatPromptTemplate.from_messages([("system", prompt_engineer_prompt_text), ("human", human)])

        chain = prompt | chat
        return chain.invoke({"user_request": user_request})
