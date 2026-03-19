"""Deep Research CLI Plugin

Provides commands for deep research functionality.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


def cmd_research(args):
    """Perform deep research on a topic."""
    from src.extensions.content.deep_research.manager import DeepResearchManager
    from src.extensions.content.deep_research.utils import ensure_lang_tools_sync

    try:
        # Handle PDF file argument for RAG
        pdf_file = getattr(args, 'pdf_file', None)
        if pdf_file:
            pdf_path = Path(pdf_file)
            if not pdf_path.exists():
                print(f"Error: PDF file not found: {pdf_file}")
                return {"error": f"PDF file not found: {pdf_file}"}
            print(f"PDF file provided for RAG: {pdf_file}")

        # Ensure MCP server is accessible
        print("Checking MCP server accessibility...")
        docker_result = ensure_lang_tools_sync()

        if docker_result["success"]:
            if docker_result["started"]:
                print("✓ lang-tools started successfully")
            elif docker_result["already_running"]:
                print("✓ lang-tools already running")
        else:
            # CRASH-ONLY: Docker services are REQUIRED - no fallbacks
            # Exception: PDF-only mode doesn't need web services
            pdf_only = getattr(args, 'pdf_only', False)
            if not pdf_only:
                print(f"\n╔═══════════════════════════════════════════════════════════════════╗")
                print(f"║           MCP WEB SERVER REQUIRED - CRITICAL FAILURE                 ║")
                print(f"╚═══════════════════════════════════════════════════════════════════╝\n")
                print(f"MCP web server is REQUIRED for research.\n")
                print(f"Error: {docker_result.get('error', 'Unknown error')}\n")
                print(f"Please ensure the MCP web server is running and accessible.\n")
                return {"error": "Docker services required"}
            else:
                print("  PDF-only mode: web services not required, proceeding...")

        manager = DeepResearchManager()

        print(f"Starting deep research on: {args.topic}")
        if args.use_case:
            print(f"Use case: {args.use_case}")
        if hasattr(args, 'variant') and args.variant:
            print(f"Variant: {args.variant}")
        if hasattr(args, 'overdrive') and args.overdrive:
            print("Overdrive mode: ENABLED")
        if pdf_file:
            print(f"PDF RAG: ENABLED ({pdf_path.name})")
        if getattr(args, 'pdf_only', False):
            print("PDF-only mode: ENABLED (skipping web research)")

        # Perform research
        if args.refine:
            result = manager.refine_research(args.topic, args.refine)
        else:
            use_case = getattr(args, 'use_case', 'base')
            variant = getattr(args, 'variant', None)
            overdrive = getattr(args, 'overdrive', False)

            # PDF file support - pass to manager
            pdf_only = getattr(args, 'pdf_only', False)

            # Create Store - REQUIRED for InformationFunnel
            # Using InMemoryStore for CLI operations
            from langgraph.store.memory import InMemoryStore
            store = InMemoryStore()

            result = manager.perform_research(
                args.topic,
                use_case=use_case,
                variant=variant,
                overdrive=overdrive,
                pdf_file=pdf_path if pdf_file else None,
                pdf_only=pdf_only,
                store=store  # REQUIRED - crash if None
            )

        print(f"\n{'='*60}")
        print(f"RESEARCH RESULTS:")
        print(f"{'='*60}\n")
        print(result)

        return {"result": result}
    except Exception as e:
        print(f"Error performing research: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def register(parser):
    """Register deep research commands with the parser."""
    subparsers = parser.add_subparsers(dest="research_subcommand", help="Deep research commands")

    # Research command
    research_parser = subparsers.add_parser("research", help="Perform deep research on a topic")
    research_parser.add_argument("topic", help="Topic to research")
    research_parser.add_argument("--refine", help="Refinement query for existing research")
    research_parser.add_argument("--use-case", default="base", help="Use case (base, rag, podcast, news, etc.)")
    research_parser.add_argument("--variant", default=None, help="Variant (default, accelerated, etc.)")
    research_parser.add_argument("--overdrive", action="store_true", help="Enable overdrive mode (max quality, max context, deeper research)")
    research_parser.add_argument("--pdf-file", help="Path to PDF file for RAG analysis (ingests and queries PDF)")
    research_parser.add_argument("--pdf-only", action="store_true", help="Skip web research, focus only on PDF analysis")
    research_parser.set_defaults(func=cmd_research)


if __name__ == "__main__":
    # For testing
    parser = argparse.ArgumentParser(description="Deep Research CLI")
    register(parser)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        result = args.func(args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    else:
        parser.print_help()