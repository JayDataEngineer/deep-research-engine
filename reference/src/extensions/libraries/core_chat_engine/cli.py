#!/usr/bin/env python3
"""CLI entry point for core_chat_engine.

Run companion/roleplay chat directly from command line without FastAPI server.

Usage:
    # Companion mode
    python -m src.extensions.engines.core_chat_engine.cli --mode companion

    # Roleplay mode
    python -m src.extensions.engines.core_chat_engine.cli --mode roleplay --character "My Character"

    # Standard mode
    python -m src.extensions.engines.core_chat_engine.cli --mode standard
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
# cli.py is in src/extensions/libraries/core_chat_engine/
# Go up 4 levels: core_chat_engine -> libraries -> extensions -> src
script_dir = Path(__file__).parent.resolve()
src_root = script_dir.parent.parent.parent.parent
for p in [str(src_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Try rich for nicer output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

from src.config import settings
from src.shared.chat_llm import get_chat_llm


async def run_chat(
    mode: str = "companion",
    personality: str = "friendly",
    character_name: Optional[str] = None,
    character_backstory: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    """Run interactive chat loop."""
    from src.extensions.libraries.core_chat_engine.brain_graph import CompanionBrainGraphV2
    from src.extensions.libraries.core_chat_engine.config.personality_config import get_preset
    from src.shared.checkpointer import get_shared_checkpointer

    if RICH_AVAILABLE:
        console.print(Panel.fit(
            f"[bold cyan]Core Chat Engine CLI[/bold cyan]\n"
            f"Mode: {mode}\n"
            f"Personality: {personality}\n"
            f"Character: {character_name or 'N/A'}",
            border_style="cyan"
        ))

    # Initialize LLM with thinking mode (using shared chat LLM)
    llm = get_chat_llm(enable_thinking=True)

    # Load personality config
    personality_config = get_preset(personality) if personality else None

    # Override with character backstory for roleplay
    if mode == "roleplay" and character_name and character_backstory:
        if personality_config:
            personality_config.system_prompt = f"""You are {character_name}.

{character_backstory}

Stay in character at all times. Respond as {character_name} would."""

    # Get checkpointer for conversation persistence
    checkpointer = await get_shared_checkpointer()

    # Create brain
    brain = CompanionBrainGraphV2(
        llm=llm,
        config=None,
        checkpointer=checkpointer,
        thread_id=thread_id,
        personality_enabled=True,
        mode=mode,
        personality=personality_config,
    )

    if RICH_AVAILABLE:
        console.print("[green]✓[/green] Chat engine initialized\n")
    else:
        print("Chat engine initialized\n")

    # Chat loop
    from langchain_core.messages import HumanMessage, AIMessage

    while True:
        try:
            if RICH_AVAILABLE:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            else:
                user_input = input("You: ")

            if user_input.lower() in {"exit", "quit", "q"}:
                if RICH_AVAILABLE:
                    console.print("[yellow]Goodbye![/yellow]")
                else:
                    print("Goodbye!")
                break

            if user_input.lower() == "reset":
                brain.reset_conversation()
                if RICH_AVAILABLE:
                    console.print("[yellow]Conversation reset[/yellow]\n")
                else:
                    print("Conversation reset\n")
                continue

            if not user_input.strip():
                continue

            # Process with streaming
            if RICH_AVAILABLE:
                with console.status("[bold green]Thinking...", spinner="dots"):
                    response = await brain.process(user_input)
            else:
                print("Thinking...")
                response = await brain.process(user_input)

            # Display response
            response_text = response.text or ""

            if RICH_AVAILABLE:
                console.print(Panel(
                    Markdown(response_text),
                    title="[bold magenta]Assistant[/bold magenta]",
                    border_style="magenta",
                ))
            else:
                print(f"\nAssistant: {response_text}\n")

        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow]Interrupted[/yellow]")
            else:
                print("\nInterrupted")
            break
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Error: {e}[/red]")
            else:
                print(f"Error: {e}")
            import traceback
            traceback.print_exc()


async def run_single_turn(
    message: str,
    mode: str = "companion",
    personality: str = "friendly",
    character_name: Optional[str] = None,
    character_backstory: Optional[str] = None,
):
    """Run a single turn and exit."""
    from src.extensions.libraries.core_chat_engine.brain_graph import CompanionBrainGraphV2
    from src.extensions.libraries.core_chat_engine.config.personality_config import get_preset
    from src.shared.checkpointer import get_shared_checkpointer

    llm = get_chat_llm(enable_thinking=True)

    personality_config = get_preset(personality) if personality else None

    if mode == "roleplay" and character_name and character_backstory:
        if personality_config:
            personality_config.system_prompt = f"""You are {character_name}.

{character_backstory}

Stay in character at all times. Respond as {character_name} would."""

    checkpointer = await get_shared_checkpointer()

    brain = CompanionBrainGraphV2(
        llm=llm,
        config=None,
        checkpointer=checkpointer,
        personality_enabled=True,
        mode=mode,
        personality=personality_config,
    )

    response = await brain.process(message)
    response_text = response.text or ""
    print(response_text)


async def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nOptions:")
        print("  --mode <mode>          companion, roleplay, standard (default: companion)")
        print("  --personality <name>   Personality preset (default: friendly)")
        print("  --character <name>     Character name (for roleplay)")
        print("  --backstory <text>     Character backstory (for roleplay)")
        print("  --single               Single-turn mode, don't enter interactive loop")
        print("  --message <text>       Message for single-turn mode")
        print("  --thread-id <id>       Thread ID for conversation persistence")
        print("\nExamples:")
        print("  # Interactive companion mode")
        print("  python -m src.extensions.engines.core_chat_engine.cli --mode companion")
        print("\n  # Interactive roleplay mode")
        print('  python -m src.extensions.engines.core_chat_engine.cli --mode roleplay --character "AI Assistant" --backstory "A helpful AI."')
        print("\n  # Single-turn query")
        print('  python -m src.extensions.engines.core_chat_engine.cli --mode companion --single --message "Hello!"')
        return

    # Parse arguments
    args = sys.argv[1:]
    mode = "companion"
    personality = "friendly"
    character_name = None
    character_backstory = None
    single = False
    message = None
    thread_id = None

    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif args[i] == "--personality" and i + 1 < len(args):
            personality = args[i + 1]
            i += 2
        elif args[i] == "--character" and i + 1 < len(args):
            character_name = args[i + 1]
            i += 2
        elif args[i] == "--backstory" and i + 1 < len(args):
            character_backstory = args[i + 1]
            i += 2
        elif args[i] == "--single":
            single = True
            i += 1
        elif args[i] == "--message" and i + 1 < len(args):
            message = args[i + 1]
            i += 2
        elif args[i] == "--thread-id" and i + 1 < len(args):
            thread_id = args[i + 1]
            i += 2
        else:
            i += 1

    # Run
    if single:
        await run_single_turn(
            message=message or "Hello!",
            mode=mode,
            personality=personality,
            character_name=character_name,
            character_backstory=character_backstory,
        )
    else:
        await run_chat(
            mode=mode,
            personality=personality,
            character_name=character_name,
            character_backstory=character_backstory,
            thread_id=thread_id,
        )


if __name__ == "__main__":
    asyncio.run(main())
