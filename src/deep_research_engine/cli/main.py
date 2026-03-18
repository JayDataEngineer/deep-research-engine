"""CLI entry point."""

import asyncio
import argparse
from pathlib import Path

from ..engine import run_workflow
from ..config import load_config


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deep Research Engine - Config-driven workflow engine"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Run RAG research")
    research_parser.add_argument("query", help="Research query")
    research_parser.add_argument("--config", default="rag.yaml", help="Config file")
    
    # PDF command
    pdf_parser = subparsers.add_parser("pdf", help="Run PDF summarizer")
    pdf_parser.add_argument("pdf_path", help="Path to PDF file")
    pdf_parser.add_argument("--config", default="pdf_summarizer.yaml", help="Config file")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run Telegram ingestion")
    ingest_parser.add_argument("export_path", help="Path to Telegram export")
    ingest_parser.add_argument("--config", default="telegram_ingest.yaml", help="Config file")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Run skill generator")
    scrape_parser.add_argument("url", help="URL to scrape")
    scrape_parser.add_argument("--config", default="skill_generator.yaml", help="Config file")
    
    # Common arguments
    for subparser in subparsers.choices.values():
        subparser.add_argument("--stream", action="store_true", default=True, help="Stream output")
        subparser.add_argument("--output", "-o", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Look in configs directory
        configs_dir = Path(__file__).parent.parent.parent / "configs"
        config_path = configs_dir / args.config
    
    config = load_config(str(config_path))
    
    # Get input based on command
    if args.command == "research":
        input_data = args.query
    elif args.command == "pdf":
        input_data = {"pdf_path": args.pdf_path}
    elif args.command == "ingest":
        input_data = {"export_path": args.export_path}
    elif args.command == "scrape":
        input_data = {"url": args.url}
    
    # Run workflow
    async def run():
        async for event in run_workflow(
            input_data,
            config,
            stream=args.stream,
        ):
            if args.stream:
                print_event(event)
            else:
                result = event
        
        if not args.stream and args.output:
            write_output(result, args.output)
    
    asyncio.run(run())


def print_event(event: dict) -> None:
    """Print event to stdout."""
    event_type = event.get("type", "unknown")
    
    if event_type == "phase_start":
        print(f"\n▶ Starting phase: {event['phase']}")
    elif event_type == "phase_complete":
        print(f"✓ Phase complete: {event['phase']} ({event.get('duration', 0):.2f}s)")
    elif event_type == "worker_complete":
        status = "✓" if event.get("success") else "✗"
        print(f"  {status} Worker {event.get('worker_id', 'unknown')} complete")
    elif event_type == "synthesis_complete":
        print(f"\n✓ Synthesis complete")
        if event.get("output"):
            print(f"\n{event['output']}")
    elif event_type == "error":
        print(f"✗ ERROR: {event.get('error')}")
    elif event_type == "complete":
        print(f"\n✓ Workflow complete")


def write_output(result: dict, path: str) -> None:
    """Write output to file."""
    from pathlib import Path
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(result.get("output", str(result)))
