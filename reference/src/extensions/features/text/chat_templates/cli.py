"""Chat Templates CLI Plugin

Provides commands for managing chat templates.

Usage:
    uv run backend/manage.py templates list
    uv run backend/manage.py templates create <name>
    uv run backend/manage.py templates apply <id> --character <name>
    uv run backend/manage.py templates delete <id>
"""

import argparse
import json
import logging

logger = logging.getLogger(__name__)


def get_manager():
    """Get the ChatTemplateManager instance."""
    from src.extensions.features.text.chat_templates.manager import ChatTemplateManager
    return ChatTemplateManager()


def cmd_list(args):
    """List all templates."""
    manager = get_manager()

    templates = manager.list_templates(
        category=args.category,
        tags=args.tags
    )

    if not templates:
        print("\nNo templates found.")
        return {"templates": []}

    # Group by category
    by_category = {}
    for template in templates:
        if template.category not in by_category:
            by_category[template.category] = []
        by_category[template.category].append(template)

    print(f"\nChat Templates ({len(templates)} found):")
    print("-" * 80)

    for category, cat_templates in sorted(by_category.items()):
        print(f"\n  {category.upper()}:")
        for template in cat_templates:
            builtin = " [BUILT-IN]" if template.author == "System" else ""
            print(f"    [{template.id}] {template.name}{builtin}")
            print(f"         {template.description}")
            if template.tags:
                print(f"         Tags: {', '.join(template.tags)}")

    return {"templates": [t.to_dict() for t in templates]}


def cmd_create(args):
    """Create a new template."""
    manager = get_manager()

    print(f"\nCreating template: {args.name}")
    print(f"  Category: {args.category}")
    print(f"  Description: {args.description}")

    # For now, templates need to be created via JSON file or interactive
    # This is a simplified version
    if args.file:
        try:
            with open(args.file, 'r') as f:
                messages = json.load(f)

            template = manager.create_template(
                name=args.name,
                description=args.description,
                category=args.category,
                messages=messages,
                character_name=args.character or "",
                tags=args.tags or [],
                author="User"
            )

            print(f"\n✓ Template created successfully!")
            print(f"  ID: {template.id}")
            return {"template": template.to_dict()}

        except Exception as e:
            print(f"\n✗ Error: {e}")
            return {"error": str(e)}
    else:
        print("\n✗ Please provide a JSON file with messages using --file")
        print("  Message format: [{\"name\": \"char\", \"is_user\": false, \"mes\": \"Hello\"}]")
        return {"error": "No messages file provided"}


def cmd_apply(args):
    """Apply a template to generate chat messages."""
    manager = get_manager()

    print(f"\nApplying template: {args.id}")
    print(f"  Character: {args.character or 'default'}")
    print(f"  User: {args.user}")

    messages = manager.apply_template(
        template_id=args.id,
        character_name=args.character or "",
        user_name=args.user
    )

    if not messages:
        print(f"\n✗ Template not found: {args.id}")
        return {"error": "Not found"}

    print(f"\n✓ Generated {len(messages)} messages:")
    for i, msg in enumerate(messages, 1):
        sender = "You" if msg.is_user else msg.name
        print(f"\n  [{i}] {sender}:")
        print(f"      {msg.mes[:100]}...")

    if args.export:
        output_path = f"chat_from_template.jsonl"
        with open(output_path, 'w') as f:
            for msg in messages:
                f.write(json.dumps(msg.to_dict(), ensure_ascii=False) + "\n")
        print(f"\n  Exported to: {output_path}")

    return {"messages": [m.to_dict() for m in messages]}


def cmd_delete(args):
    """Delete a template."""
    manager = get_manager()

    template = manager.get_template(args.id)
    if not template:
        print(f"\n✗ Template not found: {args.id}")
        return {"error": "Not found"}

    if template.author == "System":
        print(f"\n✗ Cannot delete built-in template: {args.id}")
        return {"error": "Cannot delete built-in"}

    print(f"\nDeleting template: {template.name}")
    print(f"  ID: {template.id}")

    if args.force or input("  Confirm delete? (yes/no): ").lower() == "yes":
        if manager.delete_template(args.id):
            print("\n✓ Template deleted")
            return {"deleted": True}
        else:
            print("\n✗ Failed to delete template")
            return {"error": "Delete failed"}
    else:
        print("\nCancelled")
        return {"cancelled": True}


def cmd_export(args):
    """Export a template to a file."""
    manager = get_manager()

    print(f"\nExporting template: {args.id}")

    output_path = manager.export_template(args.id, args.path)

    if not output_path:
        print(f"✗ Template not found: {args.id}")
        return {"error": "Not found"}

    print(f"\n✓ Template exported to: {output_path}")
    return {"export_path": str(output_path)}


def cmd_import(args):
    """Import a template from a file."""
    manager = get_manager()

    print(f"\nImporting template from: {args.file}")

    template = manager.import_template(args.file)

    if not template:
        print(f"\n✗ Failed to import template")
        return {"error": "Import failed"}

    print(f"\n✓ Template imported successfully!")
    print(f"  ID: {template.id}")
    print(f"  Name: {template.name}")
    return {"template": template.to_dict()}


def cmd_categories(args):
    """List template categories."""
    manager = get_manager()
    categories = manager.get_categories()

    print("\nTemplate Categories:")
    print("-" * 40)
    for category in categories:
        count = len(manager.list_templates(category=category))
        print(f"  {category:20} ({count} templates)")

    return {"categories": categories}


def register(parser):
    """Register CLI commands with the parser."""
    subparsers = parser.add_subparsers(
        dest="templates_subcommand",
        help="Template commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List templates")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--tags", "-t", nargs="+", help="Filter by tags")
    list_parser.set_defaults(func=cmd_list)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a template")
    create_parser.add_argument("name", help="Template name")
    create_parser.add_argument("category", help="Template category")
    create_parser.add_argument("description", help="Template description")
    create_parser.add_argument("--file", "-f", required=True, help="JSON file with messages")
    create_parser.add_argument("--character", help="Default character name")
    create_parser.add_argument("--tags", nargs="+", help="Template tags")
    create_parser.set_defaults(func=cmd_create)

    # apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a template")
    apply_parser.add_argument("id", help="Template ID")
    apply_parser.add_argument("--character", "-c", help="Character name")
    apply_parser.add_argument("--user", "-u", default="You", help="User name")
    apply_parser.add_argument("--export", "-e", action="store_true", help="Export to file")
    apply_parser.set_defaults(func=cmd_apply)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a template")
    delete_parser.add_argument("id", help="Template ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)

    # export command
    export_parser = subparsers.add_parser("export", help="Export a template")
    export_parser.add_argument("id", help="Template ID")
    export_parser.add_argument("path", nargs="?", help="Output path (optional)")
    export_parser.set_defaults(func=cmd_export)

    # import command
    import_parser = subparsers.add_parser("import", help="Import a template")
    import_parser.add_argument("file", help="Path to template JSON file")
    import_parser.set_defaults(func=cmd_import)

    # categories command
    cat_parser = subparsers.add_parser("categories", help="List categories")
    cat_parser.set_defaults(func=cmd_categories)
