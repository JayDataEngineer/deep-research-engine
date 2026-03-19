"""Bookmarks CLI Plugin

Provides commands for managing chat bookmarks and snapshots.

Usage:
    uv run backend/manage.py bookmarks list
    uv run backend/manage.py bookmarks create <chat> <index> [name]
    uv run backend/manage.py bookmarks restore <id>
    uv run backend/manage.py bookmarks export <id> [path]
    uv run backend/manage.py bookmarks delete <id>
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the BookmarkManager instance."""
    from src.extensions.features.bookmarks.manager import BookmarkManager
    return BookmarkManager()


def cmd_list(args):
    """List all bookmarks."""
    manager = get_manager()

    bookmarks = manager.list_bookmarks(chat_name=args.chat)

    if not bookmarks:
        print("\nNo bookmarks found.")
        if args.chat:
            print(f"  Chat: {args.chat}")
        return {"bookmarks": []}

    print(f"\nBookmarks ({len(bookmarks)} found):")
    print("-" * 80)

    for i, bm in enumerate(bookmarks, 1):
        snapshot_indicator = " [SNAPSHOT]" if bm.snapshot_data else ""
        print(f"\n  [{i}] {bm.name}{snapshot_indicator}")
        print(f"      ID: {bm.id}")
        print(f"      Chat: {bm.chat_name} (message {bm.message_index})")
        print(f"      Character: {bm.character_name or 'N/A'}")
        print(f"      Created: {bm.created_at}")
        if bm.notes:
            print(f"      Notes: {bm.notes}")

    return {"bookmarks": [b.to_dict() for b in bookmarks]}


def cmd_create(args):
    """Create a new bookmark."""
    manager = get_manager()

    print(f"\nCreating bookmark for: {args.chat}")
    print(f"  Message index: {args.index}")
    if args.name:
        print(f"  Name: {args.name}")
    if args.notes:
        print(f"  Notes: {args.notes}")
    if args.snapshot:
        print(f"  Creating full snapshot: YES")

    try:
        bookmark = manager.create_bookmark(
            chat_name=args.chat,
            message_index=args.index,
            name=args.name or "",
            notes=args.notes or "",
            create_snapshot=args.snapshot
        )

        print(f"\n✓ Bookmark created successfully!")
        print(f"  ID: {bookmark.id}")
        print(f"  Name: {bookmark.name}")
        if bookmark.snapshot_data:
            print(f"  Snapshot: Created")

        return {"bookmark": bookmark.to_dict()}

    except Exception as e:
        logger.error(f"Error creating bookmark: {e}")
        print(f"\n✗ Error: {e}")
        return {"error": str(e)}


def cmd_restore(args):
    """Restore a bookmark."""
    manager = get_manager()

    print(f"\nRestoring bookmark: {args.id}")

    restore_data = manager.restore_bookmark(args.id)

    if not restore_data:
        print(f"✗ Bookmark not found: {args.id}")
        return {"error": "Not found"}

    bookmark = restore_data["bookmark"]
    print(f"\n✓ Bookmark restored: {bookmark['name']}")
    print(f"  Chat: {bookmark['chat_name']}")
    print(f"  Original message count: {restore_data['original_message_count']}")
    print(f"  Restored message count: {restore_data['restored_message_count']}")

    if args.export:
        output_path = f"{bookmark['chat_name']}_restored.jsonl"
        from pathlib import Path
        import json

        with open(output_path, 'w') as f:
            for msg in restore_data["messages"]:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        print(f"  Exported to: {output_path}")

    return {"restore_data": restore_data}


def cmd_export(args):
    """Export a bookmark to a file."""
    manager = get_manager()

    print(f"\nExporting bookmark: {args.id}")

    output_path = manager.export_bookmark(args.id, args.path)

    if not output_path:
        print(f"✗ Bookmark not found: {args.id}")
        return {"error": "Not found"}

    print(f"\n✓ Bookmark exported to: {output_path}")
    return {"export_path": str(output_path)}


def cmd_delete(args):
    """Delete a bookmark."""
    manager = get_manager()

    bookmark = manager.get_bookmark(args.id)
    if not bookmark:
        print(f"\n✗ Bookmark not found: {args.id}")
        return {"error": "Not found"}

    print(f"\nDeleting bookmark: {bookmark.name}")
    print(f"  ID: {bookmark.id}")
    print(f"  Chat: {bookmark.chat_name}")

    if args.force or input("  Confirm delete? (yes/no): ").lower() == "yes":
        if manager.delete_bookmark(args.id):
            print("\n✓ Bookmark deleted")
            return {"deleted": True}
        else:
            print("\n✗ Failed to delete bookmark")
            return {"error": "Delete failed"}
    else:
        print("\nCancelled")
        return {"cancelled": True}


def cmd_stats(args):
    """Show bookmark statistics."""
    manager = get_manager()
    stats = manager.get_stats()

    print("\nBookmark Statistics:")
    print("-" * 40)
    print(f"  Total bookmarks: {stats['total_bookmarks']}")
    print(f"  Chats with bookmarks: {stats['chats_with_bookmarks']}")
    print(f"  Snapshots: {stats['snapshots_count']}")
    print(f"  Storage: {stats['storage_path']}")

    return {"stats": stats}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(
        dest="bookmarks_subcommand",
        help="Bookmark commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List bookmarks")
    list_parser.add_argument("--chat", "-c", help="Filter by chat name")
    list_parser.set_defaults(func=cmd_list)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a bookmark")
    create_parser.add_argument("chat", help="Chat name")
    create_parser.add_argument("index", type=int, help="Message index to bookmark")
    create_parser.add_argument("name", nargs="?", help="Optional bookmark name")
    create_parser.add_argument("--notes", "-n", help="Optional notes")
    create_parser.add_argument("--snapshot", "-s", action="store_true", help="Create full snapshot")
    create_parser.set_defaults(func=cmd_create)

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a bookmark")
    restore_parser.add_argument("id", help="Bookmark ID")
    restore_parser.add_argument("--export", "-e", action="store_true", help="Export restored chat")
    restore_parser.set_defaults(func=cmd_restore)

    # export command
    export_parser = subparsers.add_parser("export", help="Export a bookmark")
    export_parser.add_argument("id", help="Bookmark ID")
    export_parser.add_argument("path", nargs="?", help="Output path (optional)")
    export_parser.set_defaults(func=cmd_export)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a bookmark")
    delete_parser.add_argument("id", help="Bookmark ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show bookmark statistics")
    stats_parser.set_defaults(func=cmd_stats)
