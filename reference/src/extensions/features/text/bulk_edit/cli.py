"""Bulk Edit CLI Plugin

Provides commands for bulk editing messages and content.

Usage:
    uv run backend/manage.py bulk_edit replace "find" "replace" --file messages.json
    uv run backend/manage.py bulk_edit regex "s/foo/bar/g" "replacement" --file messages.json
    uv run backend/manage.py bulk_edit prepend "Note: " --file messages.json
    uv run backend/manage.py bulk_edit templates
"""

import argparse
import json
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the BulkEditManager instance."""
    from src.extensions.features.text.bulk_edit.manager import BulkEditManager
    return BulkEditManager()


def _load_items_from_file(filepath: str) -> list:
    """Load items from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def cmd_replace(args):
    """Bulk replace text."""
    manager = get_manager()

    # Load items
    items = _load_items_from_file(args.file)

    print(f"\nBulk Replace:")
    print(f"  File: {args.file}")
    print(f"  Items loaded: {len(items)}")
    print(f"  Find: {args.target}")
    print(f"  Replace: {args.replacement}")

    result = manager.bulk_replace(
        items=items,
        target=args.target,
        replacement=args.replacement,
        case_sensitive=args.case_sensitive,
        field=args.field or "content",
        preview=args.preview,
    )

    _print_result(result)

    # Save if not preview
    if not args.preview and result.items_modified > 0:
        with open(args.file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"\n  Saved {len(items)} items to {args.file}")

    return result.to_dict()


def cmd_regex(args):
    """Bulk regex replace."""
    import re

    manager = get_manager()

    # Load items
    items = _load_items_from_file(args.file)

    # Build flags
    flags = 0
    if args.ignore_case:
        flags |= re.IGNORECASE
    if args.multiline:
        flags |= re.MULTILINE
    if args.dotall:
        flags |= re.DOTALL

    print(f"\nBulk Regex Replace:")
    print(f"  File: {args.file}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Replacement: {args.replacement}")

    result = manager.bulk_regex_replace(
        items=items,
        pattern=args.pattern,
        replacement=args.replacement,
        flags=flags,
        field=args.field or "content",
        preview=args.preview,
    )

    _print_result(result)

    if not args.preview and result.items_modified > 0:
        with open(args.file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"\n  Saved {len(items)} items to {args.file}")

    return result.to_dict()


def cmd_prepend(args):
    """Prepend text to items."""
    manager = get_manager()

    items = _load_items_from_file(args.file)

    print(f"\nBulk Prepend:")
    print(f"  File: {args.file}")
    print(f"  Text: {args.text}")

    result = manager.bulk_prepend(
        items=items,
        text=args.text,
        field=args.field or "content",
        preview=args.preview,
    )

    _print_result(result)

    if not args.preview:
        with open(args.file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"\n  Saved {len(items)} items to {args.file}")

    return result.to_dict()


def cmd_append(args):
    """Append text to items."""
    manager = get_manager()

    items = _load_items_from_file(args.file)

    print(f"\nBulk Append:")
    print(f"  File: {args.file}")
    print(f"  Text: {args.text}")

    result = manager.bulk_append(
        items=items,
        text=args.text,
        field=args.field or "content",
        preview=args.preview,
    )

    _print_result(result)

    if not args.preview:
        with open(args.file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"\n  Saved {len(items)} items to {args.file}")

    return result.to_dict()


def cmd_templates(args):
    """List saved templates."""
    manager = get_manager()

    templates = manager.list_templates()

    if not templates:
        print("\nNo templates found.")
        return {"templates": []}

    print(f"\nBulk Edit Templates ({len(templates)}):")
    print("-" * 60)

    for tmpl in templates:
        print(f"\n  {tmpl['name']}")
        print(f"    Operation: {tmpl['operation']}")
        if tmpl.get('target_pattern'):
            print(f"    Pattern: {tmpl['target_pattern']}")
        if tmpl.get('replacement'):
            print(f"    Replacement: {tmpl['replacement']}")

    return {"templates": templates}


def cmd_save_template(args):
    """Save an operation as a template."""
    manager = get_manager()

    from src.extensions.features.text.bulk_edit.manager import BulkEditOperation
    operation = BulkEditOperation(
        operation=args.operation,
        target_pattern=args.pattern or "",
        replacement=args.replacement or "",
    )

    manager.save_template(args.name, operation)

    print(f"\nTemplate saved: {args.name}")
    return {"name": args.name, "operation": operation.to_dict()}


def cmd_history(args):
    """Show operation history."""
    manager = get_manager()

    history = manager.get_history(limit=args.limit)

    if not history:
        print("\nNo operation history found.")
        return {"history": []}

    print(f"\nRecent Operations ({len(history)}):")
    print("-" * 60)

    for entry in history:
        print(f"\n  {entry['timestamp']}")
        print(f"    Operation: {entry['operation']['operation']}")
        print(f"    Processed: {entry['items_processed']}")
        print(f"    Modified: {entry['items_modified']}")

    return {"history": history}


def _print_result(result):
    """Print operation result."""
    print(f"\n  Processed: {result.items_processed} items")
    print(f"  Modified: {result.items_modified} items")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"    - {error}")
        if len(result.errors) > 5:
            print(f"    ... and {len(result.errors) - 5} more")

    if result.preview_changes and len(result.preview_changes) <= 5:
        print(f"\n  Preview changes:")
        for change in result.preview_changes:
            print(f"    [{change['id']}]")
            if 'before' in change:
                print(f"      Before: {change['before'][:100]}...")
                print(f"      After:  {change['after'][:100]}...")


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="bulk_edit_subcommand", help="Bulk Edit commands")

    # replace command
    replace_parser = subparsers.add_parser("replace", help="Bulk replace text")
    replace_parser.add_argument("target", help="Text to find")
    replace_parser.add_argument("replacement", help="Text to replace with")
    replace_parser.add_argument("--file", "-f", required=True, help="JSON file with items")
    replace_parser.add_argument("--field", help="Field name (default: content)")
    replace_parser.add_argument("--case-sensitive", action="store_true", help="Case-sensitive search")
    replace_parser.add_argument("--preview", "-p", action="store_true", help="Preview without saving")
    replace_parser.set_defaults(func=cmd_replace)

    # regex command
    regex_parser = subparsers.add_parser("regex", help="Bulk regex replace")
    regex_parser.add_argument("pattern", help="Regex pattern")
    regex_parser.add_argument("replacement", help="Replacement string")
    regex_parser.add_argument("--file", "-f", required=True, help="JSON file with items")
    regex_parser.add_argument("--field", help="Field name (default: content)")
    regex_parser.add_argument("--ignore-case", "-i", action="store_true", help="Case-insensitive")
    regex_parser.add_argument("--multiline", "-m", action="store_true", help="Multiline mode")
    regex_parser.add_argument("--dotall", "-s", action="store_true", help="Dot matches newline")
    regex_parser.add_argument("--preview", "-p", action="store_true", help="Preview without saving")
    regex_parser.set_defaults(func=cmd_regex)

    # prepend command
    prepend_parser = subparsers.add_parser("prepend", help="Prepend text to items")
    prepend_parser.add_argument("text", help="Text to prepend")
    prepend_parser.add_argument("--file", "-f", required=True, help="JSON file with items")
    prepend_parser.add_argument("--field", help="Field name (default: content)")
    prepend_parser.add_argument("--preview", "-p", action="store_true", help="Preview without saving")
    prepend_parser.set_defaults(func=cmd_prepend)

    # append command
    append_parser = subparsers.add_parser("append", help="Append text to items")
    append_parser.add_argument("text", help="Text to append")
    append_parser.add_argument("--file", "-f", required=True, help="JSON file with items")
    append_parser.add_argument("--field", help="Field name (default: content)")
    append_parser.add_argument("--preview", "-p", action="store_true", help="Preview without saving")
    append_parser.set_defaults(func=cmd_append)

    # templates command
    templates_parser = subparsers.add_parser("templates", help="List saved templates")
    templates_parser.set_defaults(func=cmd_templates)

    # save-template command
    save_tmpl_parser = subparsers.add_parser("save-template", help="Save operation as template")
    save_tmpl_parser.add_argument("name", help="Template name")
    save_tmpl_parser.add_argument("--operation", "-o", default="replace", help="Operation type")
    save_tmpl_parser.add_argument("--pattern", "-p", help="Search pattern")
    save_tmpl_parser.add_argument("--replacement", "-r", help="Replacement text")
    save_tmpl_parser.set_defaults(func=cmd_save_template)

    # history command
    history_parser = subparsers.add_parser("history", help="Show operation history")
    history_parser.add_argument("--limit", "-l", type=int, default=20, help="Max entries")
    history_parser.set_defaults(func=cmd_history)
