"""
Deep Agents CLI

Command-line interface for the deep agents extension.
"""

import click
from .....shared.logging_utils import get_extension_logger

logger = get_extension_logger("deep_agents")


def register(parser):
    """
    Register CLI commands for the deep agents extension.

    This function is called by the extension manager to discover commands.
    """
    # Create parent command group
    deep_agents_parser = parser.add_parser(
        "deep_agents",
        help="Deep Agents mode - infinite conversation with router and subgraphs"
    )

    subparsers = deep_agents_parser.add_subparsers(dest="command")
    subparsers.required = True  # Require a subcommand

    # Chat command - start interactive deep agents chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start interactive deep agents chat session"
    )
    chat_parser.add_argument(
        "--user-id",
        default="default",
        help="User ID for memory isolation"
    )
    chat_parser.add_argument(
        "--agent-id",
        default="deep_agents",
        help="Agent ID for procedural memory"
    )
    chat_parser.add_argument(
        "--router-type",
        choices=["llm_based", "rule_based"],
        default="llm_based",
        help="Routing strategy"
    )
    chat_parser.set_defaults(func=chat_command)

    # Memory management commands
    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage deep agents memory"
    )
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command")
    memory_subparsers.required = True

    # List memories
    list_parser = memory_subparsers.add_parser(
        "list",
        help="List stored memories for a user"
    )
    list_parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to list memories for"
    )
    list_parser.add_argument(
        "--memory-type",
        choices=["semantic", "episodic", "procedural", "all"],
        default="all",
        help="Type of memory to list"
    )
    list_parser.set_defaults(func=list_memories_command)

    # Clear memories
    clear_parser = memory_subparsers.add_parser(
        "clear",
        help="Clear memories for a user"
    )
    clear_parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to clear memories for"
    )
    clear_parser.add_argument(
        "--memory-type",
        choices=["semantic", "episodic", "procedural", "all"],
        default="all",
        help="Type of memory to clear"
    )
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    clear_parser.set_defaults(func=clear_memories_command)

    # Export memories
    export_parser = memory_subparsers.add_parser(
        "export",
        help="Export memories to JSON file"
    )
    export_parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to export memories for"
    )
    export_parser.add_argument(
        "--output",
        default="deep_agents_memories_export.json",
        help="Output file path"
    )
    export_parser.set_defaults(func=export_memories_command)

    # Import memories
    import_parser = memory_subparsers.add_parser(
        "import",
        help="Import memories from JSON file"
    )
    import_parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file path"
    )
    import_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    import_parser.set_defaults(func=import_memories_command)

    # Instructions management
    instructions_parser = subparsers.add_parser(
        "instructions",
        help="Manage procedural memory (instructions)"
    )
    instructions_subparsers = instructions_parser.add_subparsers(dest="instructions_command")
    instructions_subparsers.required = True

    # Show instructions
    show_parser = instructions_subparsers.add_parser(
        "show",
        help="Show current instructions for an agent"
    )
    show_parser.add_argument(
        "--agent-id",
        default="deep_agents",
        help="Agent ID"
    )
    show_parser.add_argument(
        "--instruction-type",
        choices=["system_prompt", "guidelines", "reflections"],
        default="system_prompt",
        help="Type of instructions to show"
    )
    show_parser.set_defaults(func=show_instructions_command)

    # Update instructions
    update_parser = instructions_subparsers.add_parser(
        "update",
        help="Update instructions for an agent"
    )
    update_parser.add_argument(
        "--agent-id",
        default="deep_agents",
        help="Agent ID"
    )
    update_parser.add_argument(
        "--instruction-type",
        choices=["system_prompt", "guidelines", "reflections"],
        default="system_prompt",
        help="Type of instructions to update"
    )
    update_parser.add_argument(
        "--content",
        required=True,
        help="New instruction content"
    )
    update_parser.add_argument(
        "--feedback",
        help="Optional feedback/reasoning for the change"
    )
    update_parser.set_defaults(func=update_instructions_command)

    # Reset instructions
    reset_parser = instructions_subparsers.add_parser(
        "reset",
        help="Reset instructions to default"
    )
    reset_parser.add_argument(
        "--agent-id",
        default="deep_agents",
        help="Agent ID"
    )
    reset_parser.add_argument(
        "--instruction-type",
        choices=["system_prompt", "guidelines", "reflections"],
        default="system_prompt",
        help="Type of instructions to reset"
    )
    reset_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    reset_parser.set_defaults(func=reset_instructions_command)

    # Statistics
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show deep agents statistics"
    )
    stats_parser.add_argument(
        "--user-id",
        help="Filter by user ID"
    )
    stats_parser.add_argument(
        "--agent-id",
        help="Filter by agent ID"
    )
    stats_parser.set_defaults(func=stats_command)

    return deep_agents_parser


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def chat_command(args):
    """Start interactive deep agents chat session."""
    import asyncio
    from .graph import build_deep_agents_graph
    from langchain_core.messages import HumanMessage

    async def chat_loop():
        # Build graph
        graph = build_deep_agents_graph()

        # Config for infinite conversation
        config = {
            "configurable": {
                "thread_id": f"{args.user_id}_conversation",
                "user_id": args.user_id,
                "agent_id": args.agent_id
            }
        }

        click.echo(f"🤖 Deep Agents Chat")
        click.echo(f"User ID: {args.user_id}")
        click.echo(f"Agent ID: {args.agent_id}")
        click.echo(f"Router: {args.router_type}")
        click.echo(f"Type 'quit' or 'exit' to end")
        click.echo("")

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    click.echo("Goodbye!")
                    break

                # Invoke graph
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )

                # Print response
                if result.get("messages"):
                    last_message = result["messages"][-1]
                    if hasattr(last_message, 'content'):
                        click.echo(f"AI: {last_message.content}")

            except KeyboardInterrupt:
                click.echo("\nGoodbye!")
                break
            except Exception as e:
                click.echo(f"Error: {e}")
                logger.exception("Error in chat loop")

    asyncio.run(chat_loop())


def list_memories_command(args):
    """List stored memories for a user."""
    import json
    from .graph.memory import MemoryManager

    try:
        # Initialize memory manager
        manager = MemoryManager(db_uri="postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable")

        click.echo(f"\n📝 Memories for user: {args.user_id}")
        click.echo("=" * 60)

        # For now, this is a placeholder
        # In production, you would query the store and display results

        click.echo(f"\nSemantic memories: (query not implemented yet)")
        click.echo(f"Episodic memories: (query not implemented yet)")
        click.echo(f"Procedural memories: (query not implemented yet)")

        click.echo("\n💡 Use the full API to query specific memory types")

    except Exception as e:
        click.echo(f"Error: {e}")


def clear_memories_command(args):
    """Clear memories for a user."""
    if not args.confirm:
        click.confirm(
            f"Are you sure you want to clear all {args.memory_type} memories for user {args.user_id}?",
            abort=True
        )

    click.echo(f"Clearing {args.memory_type} memories for user {args.user_id}...")
    # Implementation would go here
    click.echo("Done! (Placeholder - not yet implemented)")


def export_memories_command(args):
    """Export memories to JSON file."""
    import json

    click.echo(f"Exporting memories for user {args.user_id} to {args.output}...")
    # Implementation would go here
    click.echo("Export complete! (Placeholder - not yet implemented)")


def import_memories_command(args):
    """Import memories from JSON file."""
    if not args.confirm:
        click.confirm(f"Import memories from {args.input}?", abort=True)

    click.echo(f"Importing memories from {args.input}...")
    # Implementation would go here
    click.echo("Import complete! (Placeholder - not yet implemented)")


def show_instructions_command(args):
    """Show current instructions for an agent."""
    import asyncio
    from .graph.memory import MemoryManager

    async def show():
        manager = MemoryManager(db_uri="postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable")

        instructions = await manager.get_instructions(
            agent_id=args.agent_id,
            instruction_type=args.instruction_type
        )

        click.echo(f"\n📋 Instructions for agent: {args.agent_id}")
        click.echo(f"Type: {args.instruction_type}")
        click.echo("=" * 60)
        click.echo(instructions)
        click.echo("=" * 60)

    asyncio.run(show())


def update_instructions_command(args):
    """Update instructions for an agent."""
    import asyncio
    from .graph.memory import MemoryManager

    async def update():
        manager = MemoryManager(db_uri="postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable")

        await manager.update_instructions(
            agent_id=args.agent_id,
            instruction_type=args.instruction_type,
            new_content=args.content,
            feedback=args.feedback
        )

        click.echo(f"✓ Updated {args.instruction_type} for agent {args.agent_id}")

    asyncio.run(update())


def reset_instructions_command(args):
    """Reset instructions to default."""
    if not args.confirm:
        click.confirm(
            f"Reset {args.instruction_type} for agent {args.agent_id} to default?",
            abort=True
        )

    click.echo(f"Resetting {args.instruction_type} for agent {args.agent_id}...")
    # Implementation would go here
    click.echo("Done! (Placeholder - not yet implemented)")


def stats_command(args):
    """Show deep agents statistics."""
    click.echo(f"\n📊 Deep Agents Statistics")
    click.echo("=" * 60)

    if args.user_id:
        click.echo(f"User ID: {args.user_id}")

    if args.agent_id:
        click.echo(f"Agent ID: {args.agent_id}")

    click.echo("\n(More statistics would be displayed here)")
    click.echo("(Placeholder - not yet implemented)")
