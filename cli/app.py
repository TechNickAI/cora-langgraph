from heart.si import SI
from langchain.schema import HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
import click, uuid

# Initialize console for rich output
console = Console()


@click.command()
@click.option("-q", "--query", prompt="Your query", help="The query to ask the AI assistant.")
@click.option(
    "-p",
    "--llm-provider",
    type=click.Choice(SI.LLM_PROVIDERS),
    default=SI.ANTHROPIC,
    help="The language model provider to use.",
)
@click.option("-v", "--verbosity", count=True, help="Increase output verbosity.")
def cli(query, llm_provider, verbosity):
    """Cora: Heart Centered AI Assistant"""

    try:
        SI.check_api_key(llm_provider)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    # Pre-process query with Groq API
    with Progress() as progress:
        task = progress.add_task("[green]Pre-processing query...", total=100)
        enhanced_query = SI.prompt_engineer(query)
        progress.update(task, completed=100)
    console.print(Markdown(f"**Enhanced query:** {enhanced_query.content}"))

    # Set up settings for the agent
    settings = {
        "llm_provider": llm_provider,
    }

    # Create agent graph
    agent_graph = SI.create_agent_graph(settings)

    thread_id = str(uuid.uuid4())
    config = RunnableConfig(configurable={"thread_id": thread_id})

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
