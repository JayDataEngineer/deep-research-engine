"""Expressions CLI Plugin

Provides commands for managing character expressions and sprites.

Usage:
    uv run backend/manage.py expressions list
    uv run backend/manage.py expressions add <character> <image> [emotion]
    uv run backend/manage.py expressions import <character> <folder>
    uv run backend/manage.py expressions remove <id>
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the ExpressionManager instance."""
    from src.extensions.features.expressions.manager import ExpressionManager
    return ExpressionManager()


def cmd_list(args):
    """List all sprites."""
    manager = get_manager()

    sprites = manager.list_sprites(
        character_name=args.character,
        emotion=args.emotion
    )

    if not sprites:
        print("\nNo sprites found.")
        if args.character:
            print(f"  Character: {args.character}")
        return {"sprites": []}

    # Group by character
    by_character = {}
    for sprite in sprites:
        if sprite.character_name not in by_character:
            by_character[sprite.character_name] = []
        by_character[sprite.character_name].append(sprite)

    print(f"\nSprites ({len(sprites)} found):")
    print("-" * 80)

    for character, char_sprites in sorted(by_character.items()):
        print(f"\n  {character}:")
        for sprite in char_sprites:
            print(f"    [{sprite.id}] {sprite.emotion:15} - {sprite.name}")
            print(f"         Path: {sprite.image_path}")

    return {"sprites": [s.to_dict() for s in sprites]}


def cmd_add(args):
    """Add a new sprite."""
    manager = get_manager()

    print(f"\nAdding sprite:")
    print(f"  Character: {args.character}")
    print(f"  Image: {args.image}")
    print(f"  Emotion: {args.emotion or 'neutral'}")

    try:
        sprite = manager.add_sprite(
            character_name=args.character,
            image_path=args.image,
            emotion=args.emotion or "neutral",
            name=args.name or "",
            notes=args.notes or ""
        )

        print(f"\n✓ Sprite added successfully!")
        print(f"  ID: {sprite.id}")
        print(f"  Stored at: {sprite.image_path}")

        return {"sprite": sprite.to_dict()}

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error adding sprite: {e}")
        print(f"\n✗ Error: {e}")
        return {"error": str(e)}


def cmd_import(args):
    """Import sprites from a folder."""
    manager = get_manager()

    print(f"\nImporting sprites from: {args.folder}")
    print(f"  Character: {args.character}")
    print(f"  Naming: {args.naming}")

    try:
        imported = manager.import_from_folder(
            character_name=args.character,
            folder_path=args.folder,
            naming_convention=args.naming
        )

        print(f"\n✓ Imported {len(imported)} sprites:")
        for sprite in imported:
            print(f"  - {sprite.emotion:15} ({sprite.id})")

        return {"imported": [s.to_dict() for s in imported]}

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error importing sprites: {e}")
        print(f"\n✗ Error: {e}")
        return {"error": str(e)}


def cmd_remove(args):
    """Remove a sprite."""
    manager = get_manager()

    sprite = manager.get_sprite(args.id)
    if not sprite:
        print(f"\n✗ Sprite not found: {args.id}")
        return {"error": "Not found"}

    print(f"\nRemoving sprite:")
    print(f"  Character: {sprite.character_name}")
    print(f"  Emotion: {sprite.emotion}")
    print(f"  Path: {sprite.image_path}")

    if args.force or input("  Confirm remove? (yes/no): ").lower() == "yes":
        if manager.remove_sprite(args.id):
            print("\n✓ Sprite removed")
            return {"removed": True}
        else:
            print("\n✗ Failed to remove sprite")
            return {"error": "Remove failed"}
    else:
        print("\nCancelled")
        return {"cancelled": True}


def cmd_emotions(args):
    """List available emotion categories."""
    manager = get_manager()
    emotions = manager.get_emotion_list()

    print("\nAvailable Emotion Categories:")
    print("-" * 40)
    for i, emotion in enumerate(emotions, 1):
        print(f"  {emotion}")

    return {"emotions": emotions}


def cmd_stats(args):
    """Show expression statistics."""
    manager = get_manager()
    stats = manager.get_stats()

    print("\nExpression Statistics:")
    print("-" * 40)
    print(f"  Total sprites: {stats['total_sprites']}")
    print(f"  Characters: {stats['characters']}")
    print(f"  Emotions used: {', '.join(stats['emotions_used']) or 'None'}")
    print(f"  Storage: {stats['storage_path']}")

    return {"stats": stats}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(
        dest="expressions_subcommand",
        help="Expression commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List sprites")
    list_parser.add_argument("--character", "-c", help="Filter by character")
    list_parser.add_argument("--emotion", "-e", help="Filter by emotion")
    list_parser.set_defaults(func=cmd_list)

    # add command
    add_parser = subparsers.add_parser("add", help="Add a sprite")
    add_parser.add_argument("character", help="Character name")
    add_parser.add_argument("image", help="Path to sprite image")
    add_parser.add_argument("emotion", nargs="?", help="Emotion category (default: neutral)")
    add_parser.add_argument("--name", "-n", help="Optional sprite name")
    add_parser.add_argument("--notes", help="Optional notes")
    add_parser.set_defaults(func=cmd_add)

    # import command
    import_parser = subparsers.add_parser("import", help="Import sprites from folder")
    import_parser.add_argument("character", help="Character name")
    import_parser.add_argument("folder", help="Path to folder with sprite images")
    import_parser.add_argument("--naming", choices=["prefix", "suffix"], default="prefix",
                             help="Filename naming convention (default: prefix)")
    import_parser.set_defaults(func=cmd_import)

    # remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a sprite")
    remove_parser.add_argument("id", help="Sprite ID")
    remove_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    remove_parser.set_defaults(func=cmd_remove)

    # emotions command
    emotions_parser = subparsers.add_parser("emotions", help="List emotion categories")
    emotions_parser.set_defaults(func=cmd_emotions)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show expression statistics")
    stats_parser.set_defaults(func=cmd_stats)
