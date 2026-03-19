"""Reasoning CLI Plugin

Provides commands for managing chain-of-thought reasoning and logprobs data.

Usage:
    uv run backend/manage.py reasoning list
    uv run backend/manage.py reasoning show <id>
    uv run backend/manage.py reasoning search "<query>"
    uv run backend/manage.py reasoning stats
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the ReasoningManager instance."""
    from src.extensions.features.text.reasoning.manager import ReasoningManager
    return ReasoningManager()


def cmd_list(args):
    """List stored reasonings."""
    manager = get_manager()

    reasonings = manager.list_reasonings(
        limit=args.limit,
        model=args.model,
        character_id=args.character,
        chat_id=args.chat,
        has_reasoning=args.reasoning_only,
    )

    if not reasonings:
        print("\nNo reasoning data found.")
        return {"reasonings": []}

    print(f"\nStored Reasonings ({len(reasonings)}):")
    print("-" * 60)

    for r in reasonings:
        r_marker = " [R]" if r.get("has_reasoning") else ""
        print(f"\n  {r['id']}{r_marker}")
        print(f"    Time: {r['timestamp']}")
        print(f"    Model: {r['model']}")
        print(f"    Tokens: {r['tokens']:,}")
        if r.get("character_id"):
            print(f"    Character: {r['character_id']}")

    return {"reasonings": reasonings}


def cmd_show(args):
    """Show reasoning details."""
    manager = get_manager()

    data = manager.get_full_response(args.id)
    if not data:
        print(f"\nReasoning not found: {args.id}")
        return {"error": "Not found"}

    reasoning = data["reasoning"]

    print(f"\nReasoning Data: {reasoning['id']}")
    print("-" * 60)
    print(f"  Time: {reasoning['timestamp']}")
    print(f"  Model: {reasoning['model']}")
    print(f"  Temperature: {reasoning['temperature']}")
    print(f"  Tokens: {reasoning['total_tokens']:,} (reasoning: {reasoning['reasoning_tokens']}, response: {reasoning['response_tokens']})")

    if reasoning.get('character_id'):
        print(f"  Character: {reasoning['character_id']}")
    if reasoning.get('chat_id'):
        print(f"  Chat: {reasoning['chat_id']}")

    if reasoning.get('reasoning'):
        print(f"\n  Chain-of-Thought:")
        print(f"    {reasoning['reasoning'][:500]}")
        if len(reasoning['reasoning']) > 500:
            print(f"    ... ({len(reasoning['reasoning'])} chars total)")

    if args.show_response:
        print(f"\n  Response:")
        print(f"    {reasoning['response'][:500]}")
        if len(reasoning['response']) > 500:
            print(f"    ... ({len(reasoning['response'])} chars total)")

    if data.get("logprobs"):
        logprobs = data["logprobs"]
        print(f"\n  Logprobs: {len(logprobs)} tokens")
        if args.show_logprobs:
            print(f"    First 10 tokens:")
            for lp in logprobs[:10]:
                top = lp.get("top_logprobs", [])
                top_str = ", ".join(f"{k}:{v:.2f}" for k, v in top[:3])
                print(f"      '{lp['token']}' ({lp['logprob']:.2f}) [{top_str}]")

    return data


def cmd_search(args):
    """Search reasoning content."""
    manager = get_manager()

    results = manager.search_reasonings(args.query, limit=args.limit)

    if not results:
        print(f"\nNo results for: {args.query}")
        return {"results": []}

    print(f"\nSearch Results for '{args.query}' ({len(results)} found):")
    print("-" * 60)

    for r in results:
        print(f"\n  [{r['id']}] {r['timestamp']}")
        print(f"    Model: {r['model']}")
        print(f"    Preview: {r['preview'][:200]}...")

    return {"results": results}


def cmd_stats(args):
    """Show reasoning statistics."""
    manager = get_manager()

    stats = manager.get_stats()

    print("\nReasoning Storage Statistics:")
    print("-" * 60)
    print(f"  Total reasonings: {stats['total_reasonings']:,}")
    print(f"  With reasoning: {stats['with_reasoning']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Logprobs stored: {stats['logprobs_stored']:,}")

    if stats['models']:
        print(f"\n  By model:")
        for model, count in stats['models'].items():
            print(f"    {model}: {count:,}")

    return stats


def cmd_delete(args):
    """Delete reasoning data."""
    manager = get_manager()

    if args.force:
        if manager.delete_reasoning(args.id):
            print(f"\nDeleted: {args.id}")
            return {"deleted": True}
        else:
            print(f"\nNot found: {args.id}")
            return {"deleted": False}
    else:
        print(f"\nUse --force to confirm deletion")
        return {"deleted": False}


def cmd_clear_old(args):
    """Clear old reasoning data."""
    manager = get_manager()

    if args.force:
        count = manager.clear_old(days=args.days)
        print(f"\nCleared {count} entries older than {args.days} days")
        return {"cleared": count}
    else:
        print(f"\nUse --force to confirm (will delete entries older than {args.days} days)")
        return {"cleared": 0}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="reasoning_subcommand", help="Reasoning commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List stored reasonings")
    list_parser.add_argument("--limit", "-l", type=int, default=50, help="Max results")
    list_parser.add_argument("--model", "-m", help="Filter by model")
    list_parser.add_argument("--character", "-c", help="Filter by character ID")
    list_parser.add_argument("--chat", help="Filter by chat ID")
    list_parser.add_argument("--reasoning-only", "-r", action="store_true", help="Only entries with reasoning")
    list_parser.set_defaults(func=cmd_list)

    # show command
    show_parser = subparsers.add_parser("show", help="Show reasoning details")
    show_parser.add_argument("id", help="Reasoning ID")
    show_parser.add_argument("--show-response", "-R", action="store_true", help="Show full response")
    show_parser.add_argument("--show-logprobs", "-L", action="store_true", help="Show logprobs")
    show_parser.set_defaults(func=cmd_show)

    # search command
    search_parser = subparsers.add_parser("search", help="Search reasoning content")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    search_parser.set_defaults(func=cmd_search)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete reasoning data")
    delete_parser.add_argument("id", help="Reasoning ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Confirm deletion")
    delete_parser.set_defaults(func=cmd_delete)

    # clear-old command
    clear_parser = subparsers.add_parser("clear-old", help="Clear old reasoning data")
    clear_parser.add_argument("--days", "-d", type=int, default=30, help="Days threshold")
    clear_parser.add_argument("--force", "-f", action="store_true", help="Confirm")
    clear_parser.set_defaults(func=cmd_clear_old)
