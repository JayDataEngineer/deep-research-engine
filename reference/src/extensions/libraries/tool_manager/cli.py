"""Tool Manager CLI Plugin

Provides CLI commands for managing tools.

Usage:
    uv run backend/manage.py tools list
    uv run backend/manage.py tools list <mode>
    uv run backend/manage.py tools info <tool>
    uv run backend/manage.py tools enable <mode> <tool>
    uv run backend/manage.py tools disable <mode> <tool>
    uv run backend/manage.py tools categories
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def cmd_list(args):
    """List all tools or tools for a specific mode."""
    from src.extensions.libraries.tool_manager import ToolManager

    manager = ToolManager()

    if args.mode:
        tools = manager.list_tools_for_mode(args.mode)
        print(f"\nTools for mode '{args.mode}':")
        for tool in tools:
            info = manager.get_tool_info(tool)
            desc = info["description"] if info else "No description"
            print(f"  {tool:<30} - {desc}")
    else:
        tools = manager.list_all_tools()
        print(f"\nAll Available Tools ({len(tools)} total):")

        # Group by category
        categories = ["web", "image", "memory", "vision", "osint", "agent"]
        for cat in categories:
            cat_tools = manager.get_tools_by_category(cat)
            if cat_tools:
                print(f"\n{cat.upper()}:")
                for tool in cat_tools:
                    if tool in tools:
                        info = manager.get_tool_info(tool)
                        desc = info["description"] if info else "No description"
                        print(f"  {tool:<30} - {desc[:50]}...")

    return {"tools": tools if not args.mode else manager.list_tools_for_mode(args.mode)}


def cmd_info(args):
    """Show information about a specific tool."""
    from src.extensions.libraries.tool_manager import ToolManager

    manager = ToolManager()
    info = manager.get_tool_info(args.tool)

    if not info:
        print(f"Tool not found: {args.tool}")
        return {"error": "Tool not found"}

    print(f"\nTool: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Function: {info['function']}")

    # Show which modes have this tool enabled
    print("\nEnabled in modes:")
    from src.extensions.libraries.core_chat_engine.brain_graph import _MODE_TOOLS
    for mode, tools in _MODE_TOOLS.items():
        if args.tool in tools:
            print(f"  ✓ {mode}")

    return {"tool": info}


def cmd_enable(args):
    """Enable a tool for a mode."""
    from src.extensions.libraries.tool_manager import ToolManager

    manager = ToolManager()
    success = manager.enable_tool_for_mode(args.mode, args.tool)

    if success:
        print(f"Enabled tool '{args.tool}' for mode '{args.mode}'")
        return {"success": True, "mode": args.mode, "tool": args.tool}
    else:
        print(f"Failed to enable tool '{args.tool}' for mode '{args.mode}'")
        return {"error": "Failed to enable tool"}


def cmd_disable(args):
    """Disable a tool for a mode."""
    from src.extensions.libraries.tool_manager import ToolManager

    manager = ToolManager()
    success = manager.disable_tool_for_mode(args.mode, args.tool)

    if success:
        print(f"Disabled tool '{args.tool}' for mode '{args.mode}'")
        return {"success": True, "mode": args.mode, "tool": args.tool}
    else:
        print(f"Failed to disable tool '{args.tool}' for mode '{args.mode}'")
        return {"error": "Failed to disable tool"}


def cmd_categories(args):
    """List tool categories."""
    from src.extensions.libraries.tool_manager import ToolManager

    manager = ToolManager()
    categories = ["web", "image", "memory", "vision", "osint", "agent"]

    print("\nTool Categories:")
    for cat in categories:
        tools = manager.get_tools_by_category(cat)
        print(f"  {cat:<10} - {len(tools)} tools")

    return {"categories": categories}


def register(parser):
    """Register CLI commands with the parser.

    Called by manage.py when discovering plugins.

    Args:
        parser: An ArgumentParser that we add subparsers to
    """
    # Add subparsers to the parser
    subparsers = parser.add_subparsers(
        dest="tools_subcommand",
        help="Tool management commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List tools")
    list_parser.add_argument("--mode", "-m", help="Filter by mode")
    list_parser.set_defaults(func=cmd_list)

    # info command
    info_parser = subparsers.add_parser("info", help="Show tool information")
    info_parser.add_argument("tool", help="Tool name")
    info_parser.set_defaults(func=cmd_info)

    # enable command
    enable_parser = subparsers.add_parser("enable", help="Enable tool for mode")
    enable_parser.add_argument("mode", help="Mode name")
    enable_parser.add_argument("tool", help="Tool name")
    enable_parser.set_defaults(func=cmd_enable)

    # disable command
    disable_parser = subparsers.add_parser("disable", help="Disable tool for mode")
    disable_parser.add_argument("mode", help="Mode name")
    disable_parser.add_argument("tool", help="Tool name")
    disable_parser.set_defaults(func=cmd_disable)

    # categories command
    cat_parser = subparsers.add_parser("categories", help="List tool categories")
    cat_parser.set_defaults(func=cmd_categories)
