"""Base CLI Infrastructure

Provides common CLI patterns and utilities for feature implementations.
"""

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Standardized response formatting for CLI commands."""

    @staticmethod
    def success(message: str, data: Dict = None) -> Dict:
        """Format success response.

        Args:
            message: Success message to print
            data: Optional data to include in response

        Returns:
            Response dict
        """
        if message:
            print(f"\n{message}")
        result = {"success": True}
        if data:
            result.update(data)
        return result

    @staticmethod
    def error(message: str, details: str = None) -> Dict:
        """Format error response.

        Args:
            message: Error message
            details: Optional error details

        Returns:
            Error response dict
        """
        if message:
            print(f"\nError: {message}")
        if details:
            print(f"  Details: {details}")
        return {"success": False, "error": details or message}

    @staticmethod
    def not_found(item_type: str, identifier: str) -> Dict:
        """Format not found response.

        Args:
            item_type: Type of item (e.g., "Tag", "Theme")
            identifier: Item identifier

        Returns:
            Error response dict
        """
        print(f"\n{item_type} not found: {identifier}")
        return {"success": False, "error": "Not found"}

    @staticmethod
    def list_header(item_type: str, count: int, width: int = 60) -> None:
        """Print standardized list header.

        Args:
            item_type: Type of items being listed
            count: Number of items
            width: Width of separator line
        """
        print(f"\n{item_type} ({count}):")
        print("-" * width)

    @staticmethod
    def item_display(
        name: str,
        item_id: str,
        details: Dict[str, Any] = None,
        indent: str = "  "
    ) -> None:
        """Print standardized item display.

        Args:
            name: Item name
            item_id: Item ID
            details: Optional details dict
            indent: Indentation string
        """
        print(f"{indent}[{item_id}] {name}")
        if details:
            for key, value in details.items():
                print(f"{indent}  {key}: {value}")


class BaseCLI(ABC):
    """Base class for feature CLI implementations.

    Provides common patterns for:
    - Manager instantiation
    - Standard commands (list, show, create, delete)
    - Response formatting
    """

    def __init__(self, feature_name: str):
        """Initialize the base CLI.

        Args:
            feature_name: Name of the feature (e.g., "tags", "themes")
        """
        self.feature_name = feature_name
        self.manager: Any = None

    @abstractmethod
    def get_manager(self):
        """Get the manager instance for this feature.

        Subclasses must implement this to return their specific manager.

        Returns:
            Manager instance
        """
        pass

    @property
    @abstractmethod
    def item_name(self) -> str:
        """Get the name of items this CLI manages (e.g., "Tag", "Theme").

        Returns:
            Item name for display
        """
        pass

    @property
    @abstractmethod
    def item_name_plural(self) -> str:
        """Get the plural name of items.

        Returns:
            Plural item name
        """
        pass

    # Generic command implementations

    def cmd_list(self, args) -> Dict:
        """Generic list command.

        Args:
            args: Command arguments

        Returns:
            Dict with items list
        """
        manager = self.get_manager()
        items = self.list_items(manager, args)

        if not items:
            print(f"\nNo {self.item_name_plural.lower()} found.")
            return {self.item_name_plural.lower(): []}

        ResponseFormatter.list_header(
            f"{self.item_name_plural}",
            len(items)
        )

        for item in items:
            self.display_item(item)

        return {
            self.item_name_plural.lower(): [
                item.to_dict() if hasattr(item, 'to_dict') else item
                for item in items
            ]
        }

    def list_items(self, manager, args) -> List[Any]:
        """Get items to list. Override for custom filtering.

        Args:
            manager: Manager instance
            args: Command arguments

        Returns:
            List of items
        """
        return getattr(manager, f"list_{self.item_name_plural.lower()}")()

    def display_item(self, item: Any, indent: str = "  ") -> None:
        """Display a single item. Override for custom display.

        Args:
            item: Item to display
            indent: Indentation
        """
        if hasattr(item, 'to_dict'):
            data = item.to_dict()
            name = data.get('name', data.get('id', ''))
            item_id = data.get('id', '')
            ResponseFormatter.item_display(name, item_id, indent=indent)
        else:
            print(f"{indent}{item}")

    def cmd_show(self, args) -> Dict:
        """Generic show command.

        Args:
            args: Command arguments with 'id' attribute

        Returns:
            Dict with item data
        """
        manager = self.get_manager()
        item = self.get_item(manager, args.id)

        if not item:
            return ResponseFormatter.not_found(self.item_name, args.id)

        print(f"\n{self.item_name}: {args.id}")
        print("-" * 40)
        self.show_item_details(item)

        if hasattr(item, 'to_dict'):
            return {self.item_name.lower(): item.to_dict()}
        return {self.item_name.lower(): item}

    def get_item(self, manager, item_id: str) -> Optional[Any]:
        """Get a single item. Override for custom lookup.

        Args:
            manager: Manager instance
            item_id: Item identifier

        Returns:
            Item or None
        """
        getter = getattr(manager, f"get_{self.item_name.lower()}", None)
        if getter:
            return getter(item_id)
        return None

    def show_item_details(self, item: Any) -> None:
        """Show detailed item info. Override for custom display.

        Args:
            item: Item to display
        """
        if hasattr(item, 'to_dict'):
            for key, value in item.to_dict().items():
                if key != 'id':
                    print(f"  {key}: {value}")

    def cmd_delete(self, args) -> Dict:
        """Generic delete command.

        Args:
            args: Command arguments with 'id' and optional 'force' attributes

        Returns:
            Result dict
        """
        manager = self.get_manager()
        item = self.get_item(manager, args.id)

        if not item:
            return ResponseFormatter.not_found(self.item_name, args.id)

        if not args.force:
            item_desc = getattr(item, 'name', args.id)
            print(f"\nDeleting {self.item_name.lower()}: {item_desc}")
            confirm = input("  Confirm delete? (yes/no): ").lower()
            if confirm != "yes":
                print("\nCancelled")
                return {"cancelled": True}

        if self.delete_item(manager, args.id):
            print(f"\n{self.item_name} deleted: {args.id}")
            return {"deleted": True}
        else:
            print(f"\nFailed to delete")
            return {"deleted": False}

    def delete_item(self, manager, item_id: str) -> bool:
        """Delete an item. Override for custom deletion.

        Args:
            manager: Manager instance
            item_id: Item identifier

        Returns:
            True if deleted
        """
        deleter = getattr(manager, f"delete_{self.item_name.lower()}", None)
        if deleter:
            return deleter(item_id)
        return False

    # Subcommand registration helpers

    def register_list_command(self, subparsers) -> argparse.ArgumentParser:
        """Register list subcommand.

        Args:
            subparsers: Subparsers object

        Returns:
            Created parser
        """
        parser = subparsers.add_parser("list", help=f"List all {self.item_name_plural.lower()}")
        parser.set_defaults(func=self.cmd_list)
        return parser

    def register_show_command(self, subparsers) -> argparse.ArgumentParser:
        """Register show subcommand.

        Args:
            subparsers: Subparsers object

        Returns:
            Created parser
        """
        parser = subparsers.add_parser("show", help=f"Show {self.item_name.lower()} details")
        parser.add_argument("id", help=f"{self.item_name} ID")
        parser.set_defaults(func=self.cmd_show)
        return parser

    def register_delete_command(self, subparsers) -> argparse.ArgumentParser:
        """Register delete subcommand.

        Args:
            subparsers: Subparsers object

        Returns:
            Created parser
        """
        parser = subparsers.add_parser("delete", help=f"Delete a {self.item_name.lower()}")
        parser.add_argument("id", help=f"{self.item_name} ID")
        parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
        parser.set_defaults(func=self.cmd_delete)
        return parser


def create_cli_cli(
    feature_name: str,
    item_name: str,
    item_name_plural: str = None,
    manager_module: str = None,
) -> type:
    """Factory function to create a CLI class for a feature.

    Args:
        feature_name: Feature directory name
        item_name: Name of items (e.g., "Tag", "Theme")
        item_name_plural: Plural form (defaults to item_name + "s")
        manager_module: Module path to manager

    Returns:
        CLI class for the feature
    """
    if item_name_plural is None:
        item_name_plural = item_name + "s"

    class FeatureCLI(BaseCLI):
        def __init__(self):
            super().__init__(feature_name)

        def get_manager(self):
            if manager_module:
                parts = manager_module.rsplit('.', 1)
                if len(parts) == 2:
                    module = __import__(parts[0], fromlist=[parts[1]])
                else:
                    module = __import__(manager_module)
                manager_class = getattr(module, f"{item_name}Manager")
            else:
                from importlib import import_module
                module = import_module(f"src.extensions.features.{feature_name}.manager")
                manager_class = getattr(module, f"{item_name}Manager")
            return manager_class()

        @property
        def item_name(self) -> str:
            return item_name

        @property
        def item_name_plural(self) -> str:
            return item_name_plural

    return FeatureCLI
