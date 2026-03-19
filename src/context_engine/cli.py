"""Context Engine CLI - Main entry point.

Commands:
    build        Build context from query (main pipeline)
    build-graph  Build Neo4j knowledge graph (optional)
    synthesize   Generate outputs from context (optional)
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# =============================================================================
# MAIN CLI GROUP
# =============================================================================


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.version_option(version="0.1.0", prog_name="context-engine")
def cli(debug: bool) -> None:
    """Context Engine - Build and synthesize unlimited context."""
    # Configure logging
    log_level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


# =============================================================================
# BUILD COMMAND (Main Pipeline)
# =============================================================================


@cli.command()
@click.argument("query")
@click.option("--session-id", "-s", help="Session ID (auto-generated if not provided)")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    help="Input file or directory to ingest",
)
@click.option("--max-iterations", default=3, help="Maximum critic loop iterations")
@click.option("--pass-threshold", default=0.7, help="Minimum score to pass (0.0-1.0)")
@click.option("--no-langfuse", is_flag=True, help="Disable LangFuse prompts")
def build(
    query: str,
    session_id: Optional[str],
    input_path: Optional[str],
    max_iterations: int,
    pass_threshold: float,
    no_langfuse: bool,
) -> None:
    """Build context from a research query.

    This is the main pipeline that:
    1. Ingests input files (if provided)
    2. Runs main agent to gather context
    3. Evaluates with critic
    4. Loops until quality threshold met

    Example:
        uv run context-engine build "What is quantum computing?"
        uv run context-engine build "Analyze these docs" -i ./documents/
    """
    from context_engine.pipeline.runner import run_pipeline

    console.print(
        Panel(
            f"[bold blue]Building Context[/bold blue]\n\nQuery: {query[:100]}...",
            title="Context Engine",
            border_style="blue",
        )
    )

    # Load input files if provided
    input_files = None
    if input_path:
        input_files = _load_input_files(input_path)
        console.print(f"[green]Loaded {len(input_files)} input files[/green]")

    # Run pipeline
    result = asyncio.run(
        run_pipeline(
            query=query,
            session_id=session_id,
            input_files=input_files,
            max_iterations=max_iterations,
            pass_threshold=pass_threshold,
            use_langfuse_prompts=not no_langfuse,
        )
    )

    # Display results
    _display_build_result(result)


def _load_input_files(input_path: str) -> dict[str, str]:
    """Load input files from path."""
    path = Path(input_path)
    files = {}

    if path.is_file():
        files[path.name] = path.read_text(encoding="utf-8")
    elif path.is_dir():
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [
                ".txt",
                ".md",
                ".json",
                ".yaml",
                ".yml",
            ]:
                try:
                    files[str(file_path.relative_to(path))] = file_path.read_text(
                        encoding="utf-8"
                    )
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")

    return files


def _display_build_result(result: dict) -> None:
    """Display build result in a nice format."""
    console.print()

    status = "[green]PASSED[/green]" if result["passed"] else "[red]FAILED[/red]"

    console.print(
        Panel(
            f"Status: {status}\n"
            f"Score: {result['score']:.2f}\n"
            f"Iterations: {result['iterations']}\n"
            f"Session: {result['session_id']}",
            title="[bold]Build Complete[/bold]",
            border_style="green" if result["passed"] else "red",
        )
    )

    if result.get("issues"):
        console.print("\n[yellow]Issues:[/yellow]")
        for issue in result["issues"]:
            console.print(f"  - {issue}")

    if result.get("suggestions"):
        console.print("\n[blue]Suggestions:[/blue]")
        for suggestion in result["suggestions"]:
            console.print(f"  - {suggestion}")


# =============================================================================
# BUILD-GRAPH COMMAND (Optional Neo4j)
# =============================================================================


@cli.command("build-graph")
@click.option("--session-id", "-s", required=True, help="Session ID to process")
@click.option("--clear", is_flag=True, help="Clear existing graph first")
def build_graph(session_id: str, clear: bool) -> None:
    """Build Neo4j knowledge graph from session context.

    This is an OPTIONAL post-processing step that:
    1. Reads all content from LangGraph Store
    2. Extracts entities and relationships using LLM
    3. Stores in Neo4j for graph queries

    Requires Neo4j to be running.

    Example:
        uv run context-engine build-graph --session-id abc123
    """
    from context_engine.graph_builder import Neo4jBuilder

    console.print(
        Panel(
            f"[bold blue]Building Knowledge Graph[/bold blue]\n\nSession: {session_id}",
            title="Neo4j Graph Builder",
            border_style="blue",
        )
    )

    async def _run():
        builder = Neo4jBuilder()

        # Check availability
        if not builder.is_available():
            console.print("[red]Error: Neo4j is not available[/red]")
            console.print("Make sure Neo4j is running and credentials are configured.")
            sys.exit(1)

        # Clear if requested
        if clear:
            deleted = builder.clear_session(session_id)
            console.print(f"[yellow]Cleared {deleted} existing nodes[/yellow]")

        # Build graph
        result = await builder.build_graph(session_id)

        builder.close()

        return result

    result = asyncio.run(_run())

    # Display results
    console.print(
        Panel(
            f"Entities Created: {result['entities_created']}\n"
            f"Relationships Created: {result['relationships_created']}\n"
            f"Items Processed: {result.get('total_items_processed', 0)}",
            title="[bold]Graph Build Complete[/bold]",
            border_style="green",
        )
    )

    if result.get("errors"):
        console.print("\n[yellow]Warnings:[/yellow]")
        for error in result["errors"]:
            console.print(f"  - {error}")


# =============================================================================
# SYNTHESIZE COMMAND (Optional Output Generation)
# =============================================================================


@cli.command()
@click.option("--session-id", "-s", required=True, help="Session ID to synthesize")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["twitter", "podcast", "report"]),
    required=True,
    help="Output format",
)
@click.option("--output", "-o", "output_path", help="Output file path")
def synthesize(
    session_id: str,
    output_format: str,
    output_path: Optional[str],
) -> None:
    """Generate formatted output from session context.

    This is an OPTIONAL post-processing step that:
    1. Reads all content from LangGraph Store
    2. Generates formatted output (twitter, podcast, report)

    Example:
        uv run context-engine synthesize --session-id abc123 --format podcast
        uv run context-engine synthesize -s abc123 -f report -o report.md
    """
    console.print(
        Panel(
            f"[bold blue]Synthesizing Output[/bold blue]\n\n"
            f"Session: {session_id}\n"
            f"Format: {output_format}",
            title="Context Synthesis",
            border_style="blue",
        )
    )

    async def _run():
        if output_format == "twitter":
            from context_engine.synthesis import (
                synthesize_twitter_thread,
                format_thread_output,
            )

            thread = await synthesize_twitter_thread(session_id)
            return format_thread_output(thread)

        elif output_format == "podcast":
            from context_engine.synthesis import (
                synthesize_podcast,
                format_podcast_output,
            )

            script = await synthesize_podcast(session_id)
            return format_podcast_output(script)

        elif output_format == "report":
            from context_engine.synthesis import (
                synthesize_report,
                format_report_output,
            )

            report = await synthesize_report(session_id)
            return format_report_output(report)

        else:
            raise ValueError(f"Unknown format: {output_format}")

    try:
        output = asyncio.run(_run())
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    # Output to file or stdout
    if output_path:
        Path(output_path).write_text(output, encoding="utf-8")
        console.print(f"[green]Output written to: {output_path}[/green]")
    else:
        console.print()
        console.print(output)


# =============================================================================
# QUERY COMMAND (Optional - Query Neo4j)
# =============================================================================


@cli.command()
@click.option("--session-id", "-s", required=True, help="Session ID to query")
@click.option("--entity", "-e", help="Search for entity by name")
@click.option(
    "--type",
    "-t",
    "entity_type",
    type=click.Choice(["Person", "Organization", "Location", "Concept", "Technology"]),
    help="Filter by entity type",
)
@click.option("--limit", default=20, help="Max results")
def query(
    session_id: str,
    entity: Optional[str],
    entity_type: Optional[str],
    limit: int,
) -> None:
    """Query the Neo4j knowledge graph.

    Example:
        uv run context-engine query --session-id abc123 --entity "Python"
        uv run context-engine query -s abc123 --type Person
    """
    from context_engine.graph_builder import Neo4jBuilder, EntityType

    builder = Neo4jBuilder()

    if not builder.is_available():
        console.print("[red]Error: Neo4j is not available[/red]")
        sys.exit(1)

    etype = EntityType(entity_type) if entity_type else None

    results = builder.query_graph(
        session_id=session_id,
        entity_name=entity,
        entity_type=etype,
        limit=limit,
    )

    builder.close()

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Display as table
    table = Table(title="Query Results")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description", style="white")
    table.add_column("Relationships", style="yellow")

    for row in results:
        rels = row.get("relationships", [])
        rel_count = len([r for r in rels if r.get("target")])
        table.add_row(
            row.get("name", ""),
            row.get("type", ""),
            (row.get("description", "") or "")[:50] + "...",
            str(rel_count),
        )

    console.print(table)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
