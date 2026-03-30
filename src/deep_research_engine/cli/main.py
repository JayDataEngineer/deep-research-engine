"""CLI entry point."""

import argparse
import asyncio
from pathlib import Path

from ..config import load_config
from ..engine import run_workflow


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

    # Resume command
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a workflow from checkpoint",
    )
    resume_parser.add_argument(
        "thread_id",
        help="Thread ID to resume from",
    )
    resume_parser.add_argument(
        "--config",
        default="rag.yaml",
        help="Config file",
    )

    # Common arguments for all subparsers
    for subparser in subparsers.choices.values():
        subparser.add_argument("--stream", action="store_true", default=True, help="Stream output")
        subparser.add_argument("--no-stream", action="store_false", dest="stream", help="Disable streaming")
        subparser.add_argument("--output", "-o", help="Output file path")
        subparser.add_argument("--thread-id", help="Thread ID for checkpointing")
        subparser.add_argument(
            "--use-stategraph",
            action="store_true",
            default=True,
            help="Use StateGraph instead of legacy loop",
        )
        subparser.add_argument(
            "--legacy",
            action="store_true",
            default=False,
            help="Use legacy manual loop instead of StateGraph",
        )
        subparser.add_argument(
            "--storage-backends",
            nargs="+",
            help="Storage backends to enable (e.g., postgres neo4j)",
        )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Look in project root configs directory
        project_root = Path(__file__).parent.parent.parent.parent
        configs_dir = project_root / "configs"
        config_path = configs_dir / args.config

    config = load_config(str(config_path))

    # Get input based on command
    if args.command == "resume":
        # Resume mode
        input_data = {}  # No input data for resume
        session_id = args.thread_id
        resume_from = args.thread_id
    elif args.command == "research":
        input_data = args.query
        session_id = args.thread_id
        resume_from = None
    elif args.command == "pdf":
        input_data = {"pdf_path": args.pdf_path}
        session_id = args.thread_id
        resume_from = None
    elif args.command == "ingest":
        export_path = Path(args.export_path)
        if not export_path.is_absolute():
            export_path = Path.cwd() / export_path
        input_data = {"export_path": str(export_path.resolve())}
        session_id = args.thread_id
        resume_from = None
    elif args.command == "scrape":
        input_data = {"url": args.url}
        session_id = args.thread_id
        resume_from = None
    else:
        print(f"Unknown command: {args.command}")
        return

    # Determine use_stategraph flag
    use_stategraph = args.use_stategraph and not args.legacy

    # Handle storage backends
    storage_backends = args.storage_backends if hasattr(args, "storage_backends") else None

    # Run workflow
    async def run():
        async for event in run_workflow(
            input_data,
            config,
            stream=args.stream,
            session_id=session_id,
            resume_from=resume_from,
            storage_backends=storage_backends,
            use_stategraph=use_stategraph,
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
        print(f"\n[Phase] Starting: {event['phase']}")
    elif event_type == "phase_complete":
        print(f"[Complete] Phase: {event['phase']} ({event.get('duration', 0):.2f}s)")
    elif event_type == "phase_retry":
        print(f"[Retry] Phase: {event['phase']} (reason: {event.get('reason', 'unknown')})")
    elif event_type == "phase_update":
        print(f"[Update] Phase update from {event.get('node', 'unknown')}")
    elif event_type == "worker_complete":
        status = "[OK]" if event.get("success") else "[FAIL]"
        print(f"  {status} Worker {event.get('worker_id', 'unknown')} complete")
    elif event_type == "artifact_maker_complete":
        print("\n[Complete] Artifact maker complete")
        if event.get("output"):
            print(f"\n{event['output']}")
    elif event_type == "error":
        print(f"[ERROR] {event.get('error')}")
    elif event_type == "complete":
        print("\n[Complete] Workflow complete")
        if event.get("output"):
            print(f"\n{event['output'][:500]}...")


def write_output(result: dict, path: str) -> None:
    """Write output to file.

    Args:
        result: Result dictionary to write
        path: Path to output file
    """
    import json
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = result.get("output", "")
    
    # Handle non-string outputs
    if not isinstance(output, str):
        if isinstance(output, (dict, list)):
            output = json.dumps(output, indent=2)
        elif output:
            output = str(output)
        else:
            output = ""
    
    output_path.write_text(output)
