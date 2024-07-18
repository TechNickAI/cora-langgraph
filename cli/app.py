from heart import SI
from langchain.schema import HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
import click, uuid

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


def process_query(query, config):
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
        enhanced_query = processed_query.enhanced_query
        progress.update(task, completed=True)
    console.print(Markdown(f"**I understand you're asking about:** {enhanced_query}"))

    # Create agent graph with updated llm_provider
    settings = {
        "llm_provider": llm_provider,
        "format_instructions": "The response will be displayed in a terminal using markdown via the rich library",
    }
    agent_graph = SI.create_agent_graph(settings)

    # Execute query with progress spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description=f"Crafting a heartfelt response using {llm_provider}... 💭", total=None)
        response = agent_graph.invoke({"messages": [HumanMessage(content=processed_query.enhanced_query)]}, config)
        parsed_response = SI.parse_response(response)
        progress.update(task, completed=True)

    # Display response
    console.print(Markdown(f"**Here's what I think:**\n\n{parsed_response}"))


@click.command()
@click.option("-q", "--query", help="What would you like to chat about? 😊")
@click.option("-v", "--verbosity", count=True, help="Let's get more detailed! 🔍")
def cli(query, verbosity):
    """Cora: Your Heart-Centered AI Companion 💙"""

    thread_id = str(uuid.uuid4())
    config = RunnableConfig(configurable={"thread_id": thread_id})

    if query:
        # Execute the query provided via command line
        process_query(query, config)
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
                process_query(human_input, config)

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

    click.echo(click.style("It was a joy chatting with you! 💙", fg="cyan", bold=True))


if __name__ == "__main__":
    cli()
