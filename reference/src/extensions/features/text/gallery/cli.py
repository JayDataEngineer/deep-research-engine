"""Gallery CLI Plugin

Provides commands for managing the image gallery.

Usage:
    uv run backend/manage.py gallery list
    uv run backend/manage.py gallery add <image> [chat_id]
    uv run backend/manage.py gallery caption <id> <text>
    uv run backend/manage.py gallery export
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the GalleryManager instance."""
    from src.extensions.features.text.gallery.manager import GalleryManager
    return GalleryManager()


def cmd_list(args):
    """List gallery items."""
    manager = get_manager()

    items = manager.list_items(
        chat_id=args.chat,
        source=args.source,
        tags=args.tags,
        limit=args.limit
    )

    if not items:
        print("\nNo images found in gallery.")
        return {"items": []}

    print(f"\nGallery ({len(items)} images):")
    print("-" * 80)

    for i, item in enumerate(items, 1):
        source_tag = f"[{item.source.upper()}]" if item.source else ""
        print(f"\n  [{i}] {item.filename}{source_tag}")
        print(f"      ID: {item.id}")
        if item.caption:
            print(f"      Caption: {item.caption[:60]}...")
        if item.tags:
            print(f"      Tags: {', '.join(item.tags)}")
        if item.chat_id:
            print(f"      Chat: {item.chat_id}")

    return {"items": [item.to_dict() for item in items]}


def cmd_add(args):
    """Add an image to the gallery."""
    manager = get_manager()

    print(f"\nAdding image to gallery:")
    print(f"  Path: {args.image}")
    print(f"  Source: {args.source}")
    if args.chat:
        print(f"  Chat ID: {args.chat}")

    try:
        item = manager.add_image(
            image_path=args.image,
            chat_id=args.chat or "",
            source=args.source,
            caption=args.caption or "",
            tags=args.tags or []
        )

        print(f"\n✓ Image added to gallery!")
        print(f"  ID: {item.id}")
        print(f"  Filename: {item.filename}")

        return {"item": item.to_dict()}

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error adding image: {e}")
        print(f"\n✗ Error: {e}")
        return {"error": str(e)}


def cmd_caption(args):
    """Update image caption."""
    manager = get_manager()

    print(f"\nUpdating caption for: {args.id}")
    print(f"  New caption: {args.caption}")

    item = manager.update_caption(args.id, args.caption)

    if not item:
        print(f"\n✗ Image not found: {args.id}")
        return {"error": "Not found"}

    print(f"\n✓ Caption updated")
    return {"item": item.to_dict()}


def cmd_tags(args):
    """Add tags to an image."""
    manager = get_manager()

    print(f"\nAdding tags to: {args.id}")
    print(f"  Tags: {', '.join(args.tags)}")

    item = manager.add_tags(args.id, args.tags)

    if not item:
        print(f"\n✗ Image not found: {args.id}")
        return {"error": "Not found"}

    print(f"\n✓ Tags added")
    print(f"  All tags: {', '.join(item.tags)}")
    return {"item": item.to_dict()}


def cmd_remove(args):
    """Remove an image from the gallery."""
    manager = get_manager()

    item = manager.get_item(args.id)
    if not item:
        print(f"\n✗ Image not found: {args.id}")
        return {"error": "Not found"}

    print(f"\nRemoving image: {item.filename}")
    print(f"  ID: {item.id}")

    if args.force or input("  Confirm remove? (yes/no): ").lower() == "yes":
        if manager.remove_item(args.id):
            print("\n✓ Image removed")
            return {"removed": True}
        else:
            print("\n✗ Failed to remove image")
            return {"error": "Remove failed"}
    else:
        print("\nCancelled")
        return {"cancelled": True}


def cmd_export(args):
    """Export the gallery."""
    manager = get_manager()

    print(f"\nExporting gallery...")
    print(f"  Include images: {args.images}")

    output_path = manager.export_gallery(
        output_path=args.path,
        include_images=args.images
    )

    if not output_path:
        print("\n✗ Failed to export gallery")
        return {"error": "Export failed"}

    print(f"\n✓ Gallery exported to: {output_path}")
    return {"export_path": str(output_path)}


def cmd_stats(args):
    """Show gallery statistics."""
    manager = get_manager()
    stats = manager.get_stats()

    print("\nGallery Statistics:")
    print("-" * 40)
    print(f"  Total images: {stats['total_images']}")
    print(f"  Chats with images: {stats['chats_with_images']}")
    print("\n  By source:")
    for source, count in stats['by_source'].items():
        print(f"    {source}: {count}")
    print(f"\n  Storage: {stats['storage_path']}")

    return {"stats": stats}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(
        dest="gallery_subcommand",
        help="Gallery commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List gallery images")
    list_parser.add_argument("--chat", "-c", help="Filter by chat ID")
    list_parser.add_argument("--source", "-s", help="Filter by source")
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    list_parser.add_argument("--limit", "-l", type=int, default=100, help="Max items to show")
    list_parser.set_defaults(func=cmd_list)

    # add command
    add_parser = subparsers.add_parser("add", help="Add an image to gallery")
    add_parser.add_argument("image", help="Path to image file")
    add_parser.add_argument("--chat", "-c", help="Associated chat ID")
    add_parser.add_argument("--source", "-s", default="upload", help="Image source")
    add_parser.add_argument("--caption", help="Image caption")
    add_parser.add_argument("--tags", nargs="+", help="Image tags")
    add_parser.set_defaults(func=cmd_add)

    # caption command
    caption_parser = subparsers.add_parser("caption", help="Update image caption")
    caption_parser.add_argument("id", help="Image ID")
    caption_parser.add_argument("caption", help="New caption text")
    caption_parser.set_defaults(func=cmd_caption)

    # tags command
    tags_parser = subparsers.add_parser("tags", help="Add tags to an image")
    tags_parser.add_argument("id", help="Image ID")
    tags_parser.add_argument("tags", nargs="+", help="Tags to add")
    tags_parser.set_defaults(func=cmd_tags)

    # remove command
    remove_parser = subparsers.add_parser("remove", help="Remove an image")
    remove_parser.add_argument("id", help="Image ID")
    remove_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    remove_parser.set_defaults(func=cmd_remove)

    # export command
    export_parser = subparsers.add_parser("export", help="Export gallery")
    export_parser.add_argument("--path", "-p", help="Output path (optional)")
    export_parser.add_argument("--images", action="store_true", default=True, help="Include image files")
    export_parser.set_defaults(func=cmd_export)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show gallery statistics")
    stats_parser.set_defaults(func=cmd_stats)
