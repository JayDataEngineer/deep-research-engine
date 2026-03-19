"""Long-Term Memory CLI Plugin

Provides commands for managing long-term memories.

Usage:
    uv run backend/manage.py memory list
    uv run backend/manage.py memory add <content>
    uv run backend/manage.py memory search <query>
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the LongTermMemoryManager instance."""
    from src.extensions.features.assistant.longterm_memory.manager import LongTermMemoryManager
    return LongTermMemoryManager()


def cmd_list(args):
    """List memories."""
    manager = get_manager()

    memories = manager.list_memories(
        category=args.category,
        character=args.character,
        tags=args.tags,
        min_importance=args.min_importance or 0.0,
        limit=args.limit
    )

    if not memories:
        print("\nNo memories found.")
        return {"memories": []}

    print(f"\nMemories ({len(memories)} shown):")
    print("-" * 80)

    for i, memory in enumerate(memories, 1):
        print(f"\n  [{i}] [{memory.category.upper()}] {memory.content[:60]}...")
        print(f"      ID: {memory.id}")
        print(f"      Importance: {memory.importance:.2f}")
        if memory.character_name:
            print(f"      Character: {memory.character_name}")
        if memory.tags:
            print(f"      Tags: {', '.join(memory.tags)}")

    return {"memories": [m.to_dict() for m in memories]}


def cmd_add(args):
    """Add a memory."""
    manager = get_manager()

    # Get content from prompt or file
    if args.file:
        with open(args.file, 'r') as f:
            content = f.read()
    else:
        content = args.content or input("Enter memory content: ")

    memory = manager.add_memory(
        content=content,
        category=args.category,
        character_name=args.character or "",
        importance=args.importance or 0.5,
        tags=args.tags or []
    )

    print(f"\n✓ Memory added!")
    print(f"  ID: {memory.id}")
    print(f"  Category: {memory.category}")

    return {"memory": memory.to_dict()}


def cmd_search(args):
    """Search memories."""
    manager = get_manager()

    memories = manager.search_memories(args.query, limit=args.limit)

    if not memories:
        print(f"\nNo memories found for: {args.query}")
        return {"memories": []}

    print(f"\nSearch results for '{args.query}' ({len(memories)} found):")
    print("-" * 80)

    for i, memory in enumerate(memories, 1):
        print(f"\n  [{i}] {memory.content[:80]}...")
        print(f"      Category: {memory.category} | Importance: {memory.importance:.2f}")

    return {"memories": [m.to_dict() for m in memories]}


def cmd_delete(args):
    """Delete a memory."""
    manager = get_manager()

    memory = manager.get_memory(args.id)
    if not memory:
        print(f"\n✗ Memory not found: {args.id}")
        return {"error": "Not found"}

    print(f"\nDeleting memory:")
    print(f"  Content: {memory.content[:60]}...")

    if args.force or input("  Confirm delete? (yes/no): ").lower() == "yes":
        if manager.delete_memory(args.id):
            print("\n✓ Memory deleted")
            return {"deleted": True}
        else:
            print("\n✗ Failed to delete")
            return {"error": "Delete failed"}
    else:
        print("\nCancelled")
        return {"cancelled": True}


def cmd_categories(args):
    """List memory categories."""
    manager = get_manager()
    categories = manager.get_categories()

    print("\nMemory Categories:")
    print("-" * 40)
    for cat in categories:
        count = len(manager.list_memories(category=cat))
        print(f"  {cat:20} ({count} memories)")

    return {"categories": categories}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="memory_subcommand", help="Memory commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--character", help="Filter by character")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    list_parser.add_argument("--min-importance", "-m", type=float, help="Minimum importance")
    list_parser.add_argument("--limit", "-l", type=int, default=50, help="Max items")
    list_parser.set_defaults(func=cmd_list)

    # add command
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("content", nargs="?", help="Memory content")
    add_parser.add_argument("--file", "-f", help="Read content from file")
    add_parser.add_argument("--category", "-c", default="fact", help="Memory category")
    add_parser.add_argument("--character", help="Associated character")
    add_parser.add_argument("--importance", "-i", type=float, default=0.5, help="Importance (0.0-1.0)")
    add_parser.add_argument("--tags", nargs="+", help="Tags")
    add_parser.set_defaults(func=cmd_add)

    # search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    search_parser.set_defaults(func=cmd_search)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory")
    delete_parser.add_argument("id", help="Memory ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)

    # categories command
    cat_parser = subparsers.add_parser("categories", help="List categories")
    cat_parser.set_defaults(func=cmd_categories)
