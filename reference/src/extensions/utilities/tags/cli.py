"""Tags CLI Plugin

Provides commands for managing tags on characters, messages, and content.

Usage:
    uv run backend/manage.py tags list
    uv run backend/manage.py tags create "My Tag" --color "#ff0000"
    uv run backend/manage.py tags tag <item_id> <tag_id>
    uv run backend/manage.py tags search <query>

Refactored to use base infrastructure.
"""

import argparse
import logging

from src.extensions.libraries.base.cli_base import BaseCLI, ResponseFormatter
from src.extensions.features.tags.manager import TagsManager

logger = logging.getLogger(__name__)


class TagsCLI(BaseCLI):
    """CLI implementation for Tags feature."""

    def __init__(self):
        super().__init__("tags")

    def get_manager(self):
        """Get the TagsManager instance."""
        return TagsManager()

    @property
    def item_name(self) -> str:
        return "Tag"

    @property
    def item_name_plural(self) -> str:
        return "Tags"

    # Custom display implementation

    def display_item(self, tag, indent: str = "  ") -> None:
        """Display a single tag with custom formatting.

        Args:
            tag: Tag to display
            indent: Indentation string
        """
        print(f"{indent}[{tag.id}]")
        print(f"{indent}  Name: {tag.name}")
        print(f"{indent}  Color: {tag.color}")
        if tag.description:
            print(f"{indent}  Description: {tag.description}")
        print(f"{indent}  Used: {tag.count} items")

    # Custom commands

    def cmd_create(self, args) -> dict:
        """Create a new tag.

        Args:
            args: Command arguments

        Returns:
            Result dict
        """
        manager = self.get_manager()

        try:
            tag = manager.create_tag(
                name=args.name,
                color=args.color or "#3b82f6",
                description=args.description or "",
            )

            return ResponseFormatter.success(
                f"Tag created!",
                {"tag": tag.id, "name": tag.name, "color": tag.color}
            )

        except ValueError as e:
            return ResponseFormatter.error("Failed to create tag", str(e))

    def cmd_tag_item(self, args) -> dict:
        """Apply a tag to an item.

        Args:
            args: Command arguments

        Returns:
            Result dict
        """
        manager = self.get_manager()

        if manager.tag_item(args.item_id, args.tag_id):
            return ResponseFormatter.success(f"Tagged {args.item_id} with {args.tag_id}")
        else:
            return ResponseFormatter.error(
                "Failed to tag",
                "Tag may not exist or already applied"
            )

    def cmd_untag_item(self, args) -> dict:
        """Remove a tag from an item.

        Args:
            args: Command arguments

        Returns:
            Result dict
        """
        manager = self.get_manager()

        if manager.untag_item(args.item_id, args.tag_id):
            return ResponseFormatter.success(f"Removed tag {args.tag_id} from {args.item_id}")
        else:
            return ResponseFormatter.error("Failed to untag")

    def cmd_item_tags(self, args) -> dict:
        """Show tags for an item.

        Args:
            args: Command arguments

        Returns:
            Result dict
        """
        manager = self.get_manager()
        tags = manager.get_item_tags(args.item_id)

        if not tags:
            print(f"\nNo tags found for {args.item_id}")
            return {"tags": []}

        print(f"\nTags for {args.item_id} ({len(tags)}):")
        for tag in tags:
            print(f"  - {tag.name} ({tag.color})")

        return {"tags": [t.to_dict() for t in tags]}

    def cmd_find_by_tag(self, args) -> dict:
        """Find items with a specific tag.

        Args:
            args: Command arguments

        Returns:
            Result dict
        """
        manager = self.get_manager()
        items = manager.find_items_by_tag(args.tag_id)

        if not items:
            print(f"\nNo items found with tag: {args.tag_id}")
            return {"items": []}

        print(f"\nItems with tag '{args.tag_id}' ({len(items)}):")
        for item_id in items:
            print(f"  - {item_id}")

        return {"items": items}

    def cmd_search(self, args) -> dict:
        """Search tags by name or description.

        Args:
            args: Command arguments

        Returns:
            Result dict
        """
        manager = self.get_manager()
        tags = manager.search_tags(args.query)

        if not tags:
            print(f"\nNo tags matching: {args.query}")
            return {"tags": []}

        print(f"\nTags matching '{args.query}' ({len(tags)}):")
        for tag in tags:
            print(f"  - {tag.name} ({tag.id})")

        return {"tags": [t.to_dict() for t in tags]}

    def cmd_stats(self, args) -> dict:
        """Show tag statistics.

        Args:
            args: Command arguments

        Returns:
            Stats dict
        """
        manager = self.get_manager()
        stats = manager.get_stats()

        print("\nTag Statistics:")
        print("-" * 40)
        print(f"  Total tags: {stats['total_tags']}")
        print(f"  Total assignments: {stats['total_assignments']}")
        print(f"  Untagged items: {stats['untagged_items']}")

        if stats['most_used_tags']:
            print("\n  Most used tags:")
            for tag in stats['most_used_tags']:
                print(f"    - {tag.name}: {tag.count} items")

        return stats

    # Override list_items for custom sorting

    def list_items(self, manager, args) -> list:
        """Get tags to list.

        Args:
            manager: Manager instance
            args: Command arguments

        Returns:
            List of tags
        """
        return manager.list_tags()

    # Register subparsers

    def register_subparsers(self, parser) -> None:
        """Register all subcommands for tags.

        Args:
            parser: Argument parser
        """
        subparsers = parser.add_subparsers(dest="tags_subcommand", help="Tags commands")

        # Standard commands from base
        self.register_list_command(subparsers)
        self.register_delete_command(subparsers)

        # Custom commands
        self._register_create(subparsers)
        self._register_tag(subparsers)
        self._register_untag(subparsers)
        self._register_item_tags(subparsers)
        self._register_find(subparsers)
        self._register_search(subparsers)
        self._register_update(subparsers)
        self._register_stats(subparsers)

    def _register_create(self, subparsers) -> None:
        """Register create command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("create", help="Create a new tag")
        parser.add_argument("name", help="Tag name")
        parser.add_argument("--color", "-c", help="Hex color code (default: #3b82f6)")
        parser.add_argument("--description", "-d", help="Tag description")
        parser.set_defaults(func=self.cmd_create)

    def _register_tag(self, subparsers) -> None:
        """Register tag command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("tag", help="Apply a tag to an item")
        parser.add_argument("item_id", help="ID of the item to tag")
        parser.add_argument("tag_id", help="Tag ID to apply")
        parser.set_defaults(func=self.cmd_tag_item)

    def _register_untag(self, subparsers) -> None:
        """Register untag command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("untag", help="Remove a tag from an item")
        parser.add_argument("item_id", help="ID of the item")
        parser.add_argument("tag_id", help="Tag ID to remove")
        parser.set_defaults(func=self.cmd_untag_item)

    def _register_item_tags(self, subparsers) -> None:
        """Register item-tags command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("item-tags", help="Show tags for an item")
        parser.add_argument("item_id", help="ID of the item")
        parser.set_defaults(func=self.cmd_item_tags)

    def _register_find(self, subparsers) -> None:
        """Register find command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("find", help="Find items with a tag")
        parser.add_argument("tag_id", help="Tag ID to search")
        parser.set_defaults(func=self.cmd_find_by_tag)

    def _register_search(self, subparsers) -> None:
        """Register search command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("search", help="Search tags")
        parser.add_argument("query", help="Search query")
        parser.set_defaults(func=self.cmd_search)

    def _register_update(self, subparsers) -> None:
        """Register update command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("update", help="Update a tag")
        parser.add_argument("tag_id", help="Tag ID to update")
        parser.add_argument("--name", "-n", help="New name")
        parser.add_argument("--color", "-c", help="New color")
        parser.add_argument("--description", "-d", help="New description")
        parser.set_defaults(func=lambda args: self._cmd_update_wrapper(args))

    def _register_stats(self, subparsers) -> None:
        """Register stats command.

        Args:
            subparsers: Subparsers object
        """
        parser = subparsers.add_parser("stats", help="Show tag statistics")
        parser.set_defaults(func=self.cmd_stats)

    def _cmd_update_wrapper(self, args) -> dict:
        """Wrapper for update command to use base show functionality.

        Args:
            args: Command arguments

        Returns:
            Result dict
        """
        manager = self.get_manager()
        tag = manager.update_tag(
            tag_id=args.tag_id,
            name=args.name,
            color=args.color,
            description=args.description,
        )

        if not tag:
            return ResponseFormatter.not_found("Tag", args.tag_id)

        return ResponseFormatter.success(f"Tag updated: {tag.name}", {"tag": tag.to_dict()})


# Create CLI instance
_cli = TagsCLI()


def register(parser):
    """Register CLI commands with the parser.

    Args:
        parser: Argument parser
    """
    _cli.register_subparsers(parser)
