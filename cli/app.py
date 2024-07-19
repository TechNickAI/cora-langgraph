from heart.si import SI
from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_core._api.beta_decorator import LangChainBetaWarning
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from zep_cloud.client import Zep
from zep_cloud.errors import NotFoundError
from zep_cloud.langchain import ZepChatMessageHistory
import asyncio, click, uuid, warnings

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# Initialize console for rich output
console = Console()

# Define style for prompt
style = Style.from_dict(
    {
        "prompt": "#ansibrightcyan",
    }
)

# Create PromptSession with history
session = PromptSession(
    history=FileHistory(str(Path.home() / ".cora_history")),
    style=style,
    multiline=False,  # Set to False for single-line input by default
)

thread_id = str(uuid.uuid4())

zep_client = Zep()


def setup_user(email):
    # Extract username from email
    username = email.split("@")[0] if "@" in email else email

    # Try to get the user
    try:
        zep_user = zep_client.user.get(email)
    except NotFoundError:
        zep_user = zep_client.user.add(user_id=email, email=email)

    # Set up session
    zep_client.memory.add_session(
        session_id=thread_id,
        user_id=email,
    )

    # Create avatar
    avatar = username[0].upper()

    return zep_user, username, avatar


# Set up user and avatar
zep_user, username, avatar = setup_user("gorillamania+test@gmail.com")
zep_chat_history = ZepChatMessageHistory(
    thread_id,
    zep_client,
)
zep_memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=zep_chat_history)


def process_event(event, live):
    output = ""

    if event["event"] == "on_chat_model_start":
        output = "\nThinking..."
    elif event["event"] == "on_chat_model_stream":
        if "chunk" in event["data"]:
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content"):
                if isinstance(chunk.content, str):
                    # Anthropic format
                    output = chunk.content
                elif isinstance(chunk.content, list) and chunk.content:
                    # OpenAI format
                    output = chunk.content[0].get("text", "")
    elif event["event"] == "on_tool_start":
        tool_name = event.get("name", "Unknown tool")
        output = f"🔍 Searching using {tool_name}..."
    elif event["event"] == "on_tool_end":
        output = "✅ Search complete"
    elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
        final_message = event["data"]["output"]["messages"][-1]
        if hasattr(final_message, "content"):
            if isinstance(final_message.content, str):
                # Anthropic format
                output = final_message.content
            elif isinstance(final_message.content, list) and final_message.content:
                # OpenAI format
                output = final_message.content[0].get("text", "")
            live.update(Markdown(output))
            return

    if output:
        current_content = live.renderable.markup
        updated_content = current_content + output
        live.update(Markdown(updated_content))


async def process_query(query, config):
    # Pre-process query with Groq
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description="Pondering your thoughts... 🤔", total=None)
        processed_query = SI.prompt_engineer(query)
        llm_provider = processed_query.llm_provider
        llm_provider = "groq"
        enhanced_query = processed_query.enhanced_query
        progress.update(task, completed=True)

    console.print(Markdown(f"**I understand you're asking about:** {enhanced_query}"))

    # Create agent graph with updated llm_provider
    settings = {
        "llm_provider": llm_provider,
        "format_instructions": "The response will be displayed in a terminal using markdown via the rich library",
        "memory": zep_memory,
        "history": zep_chat_history,
    }
    agent_graph = SI.create_agent_graph(settings)

    # Execute query with streaming
    with Live(Markdown("")) as live:
        async for event in agent_graph.astream_events(
            {"messages": [HumanMessage(content=processed_query.enhanced_query)]},
            config=config,
            version="v2",
        ):
            process_event(event, live)


@click.command()
@click.option("-q", "--query", help="What would you like to chat about? 😊")
@click.option("-v", "--verbosity", count=True, help="Let's get more detailed! 🔍")
def cli(query, verbosity):
    """Cora: Your Heart-Centered AI Companion 💙"""

    config = RunnableConfig(configurable={"thread_id": thread_id})

    if query:
        # Execute the query provided via command line
        asyncio.run(process_query(query, config))
    else:
        # Interactive mode
        while True:
            try:
                # Prompt for user input with heart emoji and LLM provider
                prompt = "💙 Cora> "
                human_input = session.prompt(prompt, default="")

                if human_input.lower() == "\\q":
                    break

                if human_input.lower().endswith(r"\e"):
                    # Open editor for extended input
                    initial_text = human_input.replace("\\e", "").strip()
                    edited_input = click.edit(initial_text)
                    if edited_input is not None:
                        human_input = edited_input.strip()
                    else:
                        # User closed editor without saving, keep original input
                        human_input = initial_text

                # Process the query
                asyncio.run(process_query(human_input, config))

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

    click.echo(click.style("It was a joy chatting with you! 💙", fg="cyan", bold=True))


if __name__ == "__main__":
    cli()
