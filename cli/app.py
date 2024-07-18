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
import click, tempfile, uuid

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


@click.command()
@click.option(
    "-p",
    "--llm-provider",
    type=click.Choice(SI.LLM_PROVIDERS),
    default=SI.ANTHROPIC,
    help="The language model provider to use.",
)
@click.option("-q", "--query", help="The query to ask the AI assistant.")
@click.option("-v", "--verbosity", count=True, help="Increase output verbosity.")
def cli(llm_provider, query, verbosity):
    """Cora: Heart Centered AI Assistant"""

    try:
        SI.check_api_key(llm_provider)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    # Set up settings for the agent
    settings = {
        "llm_provider": llm_provider,
    }

    # Create agent graph
    agent_graph = SI.create_agent_graph(settings)

    thread_id = str(uuid.uuid4())
    config = RunnableConfig(configurable={"thread_id": thread_id})

    if query:
        # Execute the query provided via command line
        process_query(query, agent_graph, config, llm_provider)
    else:
        # Interactive mode
        while True:
            try:
                # Prompt for user input with heart emoji and LLM provider
                prompt = f"🤖 Cora 💙 ({llm_provider})> "
                human_input = session.prompt(prompt, default="")

                if human_input.lower() == "\\q":
                    break

                if human_input.lower().endswith(r"\e"):
                    # Open Vim for extended editing
                    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt") as tf:
                        tf.write(human_input.replace("\\e", ""))
                        tf.flush()
                        human_input = click.edit(filename=tf.name)
                        if human_input is None:
                            human_input = ""

                # Pre-process query with Groq API
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(description="Pre-processing query...", total=None)
                    enhanced_query = SI.prompt_engineer(human_input)
                    progress.update(task, completed=True)
                console.print(Markdown(f"**Enhanced query:** {enhanced_query.content}"))

                # Execute query with progress spinner
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task(description="Cora is thinking...", total=None)
                    response = agent_graph.invoke({"messages": [HumanMessage(content=enhanced_query.content)]}, config)
                    parsed_response = SI.parse_response(response)
                    progress.update(task, completed=True)

                # Display response
                console.print(Markdown(f"**Response:**\n\n{parsed_response}"))

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

    click.echo(click.style("Thank you for using Cora! Have a heart-centered day! 💙", fg="cyan", bold=True))


def process_query(query, agent_graph, config, llm_provider):
    # Pre-process query with Groq API
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description="Pre-processing query...", total=None)
        enhanced_query = SI.prompt_engineer(query)
        progress.update(task, completed=True)
    console.print(Markdown(f"**Enhanced query:** {enhanced_query.content}"))

    # Execute query with progress spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description="Cora is thinking...", total=None)
        response = agent_graph.invoke({"messages": [HumanMessage(content=enhanced_query.content)]}, config)
        progress.update(task, completed=True)

    # Display response
    console.print(Markdown(f"**Response:**\n\n{response['messages'][-1].content}"))


if __name__ == "__main__":
    cli()
