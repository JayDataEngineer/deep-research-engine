"""Themes CLI Plugin

Provides commands for managing UI themes.

Usage:
    uv run backend/manage.py themes list
    uv run backend/manage.py themes create my-theme --name "My Theme" --dark
    uv run backend/manage.py themes set dark
    uv run backend/manage.py themes export dark theme.json
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the ThemesManager instance."""
    from src.extensions.features.themes.manager import ThemesManager
    return ThemesManager()


def cmd_list(args):
    """List all themes."""
    manager = get_manager()

    themes = manager.list_themes()

    active = manager.get_active_theme()
    active_id = active.id if active else None

    print(f"\nAvailable Themes ({len(themes)}):")
    print("-" * 60)

    for theme in themes:
        marker = " [ACTIVE]" if theme.id == active_id else ""
        builtin = " [built-in]" if theme.is_builtin else ""
        dark = " [dark]" if theme.is_dark else " [light]"

        print(f"\n  {theme.display_name}{marker}{builtin}{dark}")
        print(f"    ID: {theme.id}")
        print(f"    Colors: {len(theme.colors)} defined")

    return {"themes": [t.to_dict() for t in themes], "active": active_id}


def cmd_create(args):
    """Create a new theme."""
    manager = get_manager()

    # Parse colors if provided
    colors = {}
    if args.colors:
        for color_spec in args.colors:
            if "=" in color_spec:
                key, value = color_spec.split("=", 1)
                colors[key.strip()] = value.strip()

    try:
        theme = manager.create_theme(
            id=args.id,
            name=args.id,
            display_name=args.name or args.id.title(),
            colors=colors,
            is_dark=args.dark,
        )

        print(f"\nTheme created: {theme.display_name}")
        print(f"  ID: {theme.id}")
        print(f"  Colors: {len(theme.colors)}")

        return {"theme": theme.to_dict()}

    except ValueError as e:
        print(f"\nError: {e}")
        return {"error": str(e)}


def cmd_set(args):
    """Set the active theme."""
    manager = get_manager()

    if manager.set_active_theme(args.theme_id):
        theme = manager.get_theme(args.theme_id)
        print(f"\nActive theme set to: {theme.display_name if theme else args.theme_id}")
        return {"active": args.theme_id}
    else:
        print(f"\nTheme not found: {args.theme_id}")
        return {"error": "Not found"}


def cmd_show(args):
    """Show theme details."""
    manager = get_manager()

    theme = manager.get_theme(args.theme_id)
    if not theme:
        print(f"\nTheme not found: {args.theme_id}")
        return {"error": "Not found"}

    print(f"\n{theme.display_name}")
    print("-" * 40)
    print(f"  ID: {theme.id}")
    print(f"  Type: {'Dark' if theme.is_dark else 'Light'}")
    print(f"  Built-in: {theme.is_builtin}")

    print(f"\n  Colors:")
    for key, value in theme.colors.items():
        print(f"    {key}: {value}")

    if theme.fonts:
        print(f"\n  Fonts:")
        for key, value in theme.fonts.items():
            print(f"    {key}: {value}")

    if theme.custom_css:
        print(f"\n  Custom CSS: {len(theme.custom_css)} chars")

    return {"theme": theme.to_dict()}


def cmd_css(args):
    """Get theme as CSS."""
    manager = get_manager()

    css = manager.to_css(args.theme_id)

    if not css:
        print(f"\nTheme not found: {args.theme_id}")
        return {"error": "Not found"}

    if args.output:
        with open(args.output, "w") as f:
            f.write(css)
        print(f"\nCSS exported to: {args.output}")
    else:
        print(f"\n{css}")

    return {"css": css}


def cmd_export(args):
    """Export a theme."""
    manager = get_manager()

    if manager.export_theme(args.theme_id, args.output):
        print(f"\nTheme exported to: {args.output}")
        return {"exported": args.output}
    else:
        print(f"\nTheme not found: {args.theme_id}")
        return {"error": "Not found"}


def cmd_import(args):
    """Import a theme."""
    manager = get_manager()

    theme = manager.import_theme(args.file)
    if theme:
        print(f"\nTheme imported: {theme.display_name}")
        print(f"  ID: {theme.id}")
        return {"theme": theme.to_dict()}
    else:
        print(f"\nFailed to import theme")
        return {"error": "Import failed"}


def cmd_delete(args):
    """Delete a custom theme."""
    manager = get_manager()

    if manager.delete_theme(args.theme_id):
        print(f"\nTheme deleted: {args.theme_id}")
        return {"deleted": True}
    else:
        print(f"\nCannot delete (built-in or not found): {args.theme_id}")
        return {"deleted": False}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(dest="themes_subcommand", help="Themes commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List all themes")
    list_parser.set_defaults(func=cmd_list)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a new theme")
    create_parser.add_argument("id", help="Theme ID (slug)")
    create_parser.add_argument("--name", help="Display name")
    create_parser.add_argument("--colors", "-c", nargs="+", help="Color specs (key=value)")
    create_parser.add_argument("--dark", action="store_true", help="Dark theme")
    create_parser.set_defaults(func=cmd_create)

    # set command
    set_parser = subparsers.add_parser("set", help="Set active theme")
    set_parser.add_argument("theme_id", help="Theme ID")
    set_parser.set_defaults(func=cmd_set)

    # show command
    show_parser = subparsers.add_parser("show", help="Show theme details")
    show_parser.add_argument("theme_id", help="Theme ID")
    show_parser.set_defaults(func=cmd_show)

    # css command
    css_parser = subparsers.add_parser("css", help="Get theme as CSS")
    css_parser.add_argument("theme_id", help="Theme ID")
    css_parser.add_argument("--output", "-o", help="Output file")
    css_parser.set_defaults(func=cmd_css)

    # export command
    export_parser = subparsers.add_parser("export", help="Export theme")
    export_parser.add_argument("theme_id", help="Theme ID")
    export_parser.add_argument("-o", "--output", required=True, help="Output file")
    export_parser.set_defaults(func=cmd_export)

    # import command
    import_parser = subparsers.add_parser("import", help="Import theme")
    import_parser.add_argument("file", help="Theme JSON file")
    import_parser.set_defaults(func=cmd_import)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a custom theme")
    delete_parser.add_argument("theme_id", help="Theme ID")
    delete_parser.set_defaults(func=cmd_delete)
