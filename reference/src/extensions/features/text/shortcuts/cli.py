"""Keyboard Shortcuts CLI Plugin

Provides commands for managing keyboard shortcuts.

Usage:
    uv run backend/manage.py shortcuts list
    uv run backend/manage.py shortcuts create my-shortcut --keys "ctrl+k" --command "my.command"
    uv run backend/manage.py shortcuts update chat.send --keys "ctrl+enter"
    uv run backend/manage.py shortcuts export shortcuts.json
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the ShortcutsManager instance."""
    from src.extensions.features.text.shortcuts.manager import ShortcutsManager
    return ShortcutsManager()


def _parse_keys(keys_str: str) -> list:
    """Parse key combination string."""
    return [k.strip().lower() for k in keys_str.split("+")]


def cmd_list(args):
    """List all shortcuts."""
    manager = get_manager()

    shortcuts = manager.list_shortcuts(context=args.context)

    print(f"\nKeyboard Shortcuts ({len(shortcuts)}):")
    if args.context:
        print(f"  Context: {args.context}")
    print("-" * 60)

    current_context = None
    for shortcut in shortcuts:
        if shortcut.context != current_context:
            current_context = shortcut.context
            print(f"\n  [{current_context.upper()}]")

        enabled = "" if shortcut.enabled else " [disabled]"
        print(f"    {shortcut.get_key_combo():20} → {shortcut.command}{enabled}")
        if shortcut.description:
            print(f"      ({shortcut.description})")

    return {"shortcuts": [s.to_dict() for s in shortcuts]}


def cmd_show(args):
    """Show shortcut details."""
    manager = get_manager()

    shortcut = manager.get_shortcut(args.shortcut_id)
    if not shortcut:
        print(f"\nShortcut not found: {args.shortcut_id}")
        return {"error": "Not found"}

    print(f"\n{shortcut.id}")
    print("-" * 40)
    print(f"  Keys: {shortcut.get_key_combo()}")
    print(f"  Command: {shortcut.command}")
    print(f"  Context: {shortcut.context}")
    print(f"  Enabled: {shortcut.enabled}")
    if shortcut.description:
        print(f"  Description: {shortcut.description}")

    return {"shortcut": shortcut.to_dict()}


def cmd_create(args):
    """Create a new shortcut."""
    manager = get_manager()

    keys = _parse_keys(args.keys)

    # Check for conflicts
    conflicts = manager.get_conflicts(keys, args.context or "global")
    if conflicts:
        print(f"\nWarning: Key combination conflicts with:")
        for c in conflicts:
            print(f"  - {c.id}: {c.get_key_combo()} ({c.command})")

    try:
        shortcut = manager.create_shortcut(
            id=args.id,
            keys=keys,
            command=args.command,
            description=args.description or "",
            context=args.context or "global",
        )

        print(f"\nShortcut created: {shortcut.id}")
        print(f"  Keys: {shortcut.get_key_combo()}")
        print(f"  Command: {shortcut.command}")

        return {"shortcut": shortcut.to_dict()}

    except ValueError as e:
        print(f"\nError: {e}")
        return {"error": str(e)}


def cmd_update(args):
    """Update a shortcut."""
    manager = get_manager()

    keys = _parse_keys(args.keys) if args.keys else None

    shortcut = manager.update_shortcut(
        shortcut_id=args.shortcut_id,
        keys=keys,
        command=args.command,
        description=args.description,
        enabled=None if args.enable is None else args.enable,
    )

    if not shortcut:
        print(f"\nShortcut not found: {args.shortcut_id}")
        return {"error": "Not found"}

    print(f"\nShortcut updated: {shortcut.id}")
    print(f"  Keys: {shortcut.get_key_combo()}")
    print(f"  Command: {shortcut.command}")

    return {"shortcut": shortcut.to_dict()}


def cmd_enable(args):
    """Enable a shortcut."""
    manager = get_manager()

    if manager.enable_shortcut(args.shortcut_id):
        print(f"\nShortcut enabled: {args.shortcut_id}")
        return {"enabled": True}
    else:
        print(f"\nShortcut not found: {args.shortcut_id}")
        return {"enabled": False}


def cmd_disable(args):
    """Disable a shortcut."""
    manager = get_manager()

    if manager.disable_shortcut(args.shortcut_id):
        print(f"\nShortcut disabled: {args.shortcut_id}")
        return {"disabled": True}
    else:
        print(f"\nShortcut not found: {args.shortcut_id}")
        return {"disabled": False}


def cmd_delete(args):
    """Delete a shortcut."""
    manager = get_manager()

    if manager.delete_shortcut(args.shortcut_id):
        print(f"\nShortcut deleted: {args.shortcut_id}")
        return {"deleted": True}
    else:
        print(f"\nShortcut not found: {args.shortcut_id}")
        return {"deleted": False}


def cmd_export(args):
    """Export shortcuts to a file."""
    manager = get_manager()

    manager.export_shortcuts(args.output)
    print(f"\nShortcuts exported to: {args.output}")

    return {"exported": args.output}


def cmd_import(args):
    """Import shortcuts from a file."""
    manager = get_manager()

    manager.import_shortcuts(args.file, merge=not args.replace)
    action = "merged" if not args.replace else "replaced"
    print(f"\nShortcuts {action} from: {args.file}")

    return {"imported": args.file}


def cmd_reset(args):
    """Reset shortcuts to defaults."""
    manager = get_manager()

    if args.force:
        manager.reset_to_defaults()
        print("\nShortcuts reset to defaults")
        return {"reset": True}
    else:
        print("\nUse --force to confirm reset")
        return {"reset": False}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="shortcuts_subcommand", help="Shortcuts commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List all shortcuts")
    list_parser.add_argument("--context", "-c", help="Filter by context")
    list_parser.set_defaults(func=cmd_list)

    # show command
    show_parser = subparsers.add_parser("show", help="Show shortcut details")
    show_parser.add_argument("shortcut_id", help="Shortcut ID")
    show_parser.set_defaults(func=cmd_show)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a shortcut")
    create_parser.add_argument("id", help="Shortcut ID")
    create_parser.add_argument("--keys", "-k", required=True, help="Key combination (e.g., ctrl+k)")
    create_parser.add_argument("--command", "-c", required=True, help="Command to execute")
    create_parser.add_argument("--description", "-d", help="Description")
    create_parser.add_argument("--context", help="Shortcut context (default: global)")
    create_parser.set_defaults(func=cmd_create)

    # update command
    update_parser = subparsers.add_parser("update", help="Update a shortcut")
    update_parser.add_argument("shortcut_id", help="Shortcut ID")
    update_parser.add_argument("--keys", "-k", help="New key combination")
    update_parser.add_argument("--command", "-c", help="New command")
    update_parser.add_argument("--description", "-d", help="New description")
    update_parser.add_argument("--enable", action="store_true", help="Enable shortcut")
    update_parser.add_argument("--disable", action="store_false", dest="enable", help="Disable shortcut")
    update_parser.set_defaults(func=cmd_update)

    # enable command
    enable_parser = subparsers.add_parser("enable", help="Enable a shortcut")
    enable_parser.add_argument("shortcut_id", help="Shortcut ID")
    enable_parser.set_defaults(func=cmd_enable)

    # disable command
    disable_parser = subparsers.add_parser("disable", help="Disable a shortcut")
    disable_parser.add_argument("shortcut_id", help="Shortcut ID")
    disable_parser.set_defaults(func=cmd_disable)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a shortcut")
    delete_parser.add_argument("shortcut_id", help="Shortcut ID")
    delete_parser.set_defaults(func=cmd_delete)

    # export command
    export_parser = subparsers.add_parser("export", help="Export shortcuts")
    export_parser.add_argument("-o", "--output", required=True, help="Output file")
    export_parser.set_defaults(func=cmd_export)

    # import command
    import_parser = subparsers.add_parser("import", help="Import shortcuts")
    import_parser.add_argument("file", help="Shortcuts JSON file")
    import_parser.add_argument("--replace", "-r", action="store_true", help="Replace existing shortcuts")
    import_parser.set_defaults(func=cmd_import)

    # reset command
    reset_parser = subparsers.add_parser("reset", help="Reset to defaults")
    reset_parser.add_argument("--force", "-f", action="store_true", help="Confirm reset")
    reset_parser.set_defaults(func=cmd_reset)
