"""Macros Extension

Text shortcuts/macros for quick responses.
Uses the macro_manager library for core functionality.

CLI Usage:
    uv run backend/manage.py macros list
    uv run backend/manage.py macros add <name> <trigger> <content>
    uv run backend/manage.py macros expand <trigger> [vars...]
    uv run backend/manage.py macros remove <trigger>
    uv run backend/manage.py macros stats
"""

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class MacrosExtension(LogicExtension):
    """Macros extension - CLI wrapper for macro_manager.

    This extension provides CLI access to the macro_manager library
    for managing text shortcuts and macros.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._manager = None
        self._storage_dir = None

    async def initialize(self, context):
        """Initialize the macros extension."""
        await super().initialize(context)

        # Import and initialize macro manager
        from src.extensions.libraries.core_chat_engine.sillytavern.macros import create_macro_manager
        from src.shared.utils.api_helpers import get_inputs_root_dir

        self._manager = create_macro_manager()

        # Set up storage directory
        try:
            self._storage_dir = get_inputs_root_dir() / "macros"
        except (ImportError, AttributeError):
            # Fallback: go up to project root (backend/ -> project root)
            self._storage_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "data" / "inputs" / "macros"

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load saved macros
        self._load_macros()

        logger.info("Macros extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        # Save macros before shutdown
        self._save_macros()

    def _load_macros(self):
        """Load macros from storage."""
        macro_file = self._storage_dir / "macros.json"
        if macro_file.exists():
            try:
                with open(macro_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._manager.import_from_dict(data)
                logger.info(f"Loaded {len(data)} macros from storage")
            except Exception as e:
                logger.error(f"Error loading macros: {e}")

    def _save_macros(self):
        """Save macros to storage."""
        macro_file = self._storage_dir / "macros.json"
        try:
            data = self._manager.export_to_dict()
            with open(macro_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} macros to storage")
        except Exception as e:
            logger.error(f"Error saving macros: {e}")

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all macros
            add <name> <trigger> <content> - Add a new macro
            expand <trigger>         - Expand a macro
            remove <trigger>         - Remove a macro
            enable <trigger>         - Enable a macro
            disable <trigger>        - Disable a macro
            stats                   - Show usage statistics
            import <file>           - Import macros from file
            export <file>           - Export macros to file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("macros", "macro", "shortcut") or args:
            subcommand = args[0] if args else "list"
            subargs = args[1:] if len(args) > 1 else []

            async for result in self._handle_cli_command(subcommand, subargs):
                yield result
        else:
            # API mode
            operation = input_data.get("operation")
            async for result in self._handle_api_operation(operation, input_data):
                yield result

    async def _handle_cli_command(
        self,
        subcommand: str,
        args: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle CLI subcommands."""
        if subcommand == "list":
            yield await self._cmd_list()
        elif subcommand == "add":
            if len(args) < 3:
                yield {"error": "Usage: add <name> <trigger> <content> [--description text]"}
                return
            yield await self._cmd_add(args[0], args[1], " ".join(args[2:]))
        elif subcommand == "expand":
            if not args:
                yield {"error": "Usage: expand <trigger> [vars...]"}
                return
            yield await self._cmd_expand(args[0], args[1:])
        elif subcommand == "remove":
            if not args:
                yield {"error": "Usage: remove <trigger>"}
                return
            yield await self._cmd_remove(args[0])
        elif subcommand == "enable":
            if not args:
                yield {"error": "Usage: enable <trigger>"}
                return
            yield await self._cmd_enable(args[0])
        elif subcommand == "disable":
            if not args:
                yield {"error": "Usage: disable <trigger>"}
                return
            yield await self._cmd_disable(args[0])
        elif subcommand == "stats":
            yield await self._cmd_stats()
        elif subcommand == "import":
            if not args:
                yield {"error": "Usage: import <file>"}
                return
            yield await self._cmd_import(args[0])
        elif subcommand == "export":
            if not args:
                yield {"error": "Usage: export <file>"}
                return
            yield await self._cmd_export(args[0])
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_macros":
            macros = self._manager.list_macros()
            yield {"macros": [m.__dict__ for m in macros]}
        elif operation == "add_macro":
            name = input_data.get("name")
            trigger = input_data.get("trigger")
            content = input_data.get("content")
            if not all([name, trigger, content]):
                yield {"error": "name, trigger, and content required"}
                return
            from src.extensions.libraries.core_chat_engine.sillytavern.macros import Macro
            macro = Macro(
                name=name,
                trigger=trigger,
                content=content,
                description=input_data.get("description", ""),
            )
            self._manager.add_macro(macro)
            self._save_macros()
            yield {"success": True, "trigger": trigger}
        elif operation == "expand_macro":
            trigger = input_data.get("trigger")
            variables = input_data.get("variables", {})
            if not trigger:
                yield {"error": "trigger required"}
                return
            expanded = self._manager.expand_macro(trigger, variables)
            if expanded:
                yield {"expanded": expanded, "trigger": trigger}
            else:
                yield {"error": "Macro not found"}
        elif operation == "remove_macro":
            trigger = input_data.get("trigger")
            if not trigger:
                yield {"error": "trigger required"}
                return
            success = self._manager.remove_macro(trigger)
            if success:
                self._save_macros()
                yield {"success": True}
            else:
                yield {"error": "Macro not found"}
        elif operation == "get_stats":
            stats = self._manager.get_usage_stats()
            yield {"stats": stats}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all macros."""
        try:
            macros = self._manager.list_macros()
            stats = self._get_stats()

            if not macros:
                return {
                    "output": "No macros found.",
                    "data": {"stats": stats}
                }

            lines = ["\nMacros:"]
            for macro in macros:
                status = "enabled" if macro.enabled else "disabled"
                lines.append(f"  [{status}] {macro.trigger} - {macro.name}")
                if macro.description:
                    lines.append(f"      {macro.description}")
                if macro.variables:
                    lines.append(f"      Variables: {', '.join(macro.variables)}")

            lines.append(f"\nTotal: {len(macros)} macros")

            return {
                "output": "\n".join(lines),
                "data": {"macros": [m.__dict__ for m in macros], "stats": stats}
            }
        except Exception as e:
            logger.error(f"Error listing macros: {e}")
            return {"error": str(e)}

    async def _cmd_add(self, name: str, trigger: str, content: str) -> Dict[str, Any]:
        """Add a new macro."""
        try:
            from src.extensions.libraries.core_chat_engine.sillytavern.macros import Macro

            # Parse out description if provided
            description = ""
            if "--description" in content:
                parts = content.split("--description")
                content = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""

            macro = Macro(
                name=name,
                trigger=trigger if trigger.startswith("/") else f"/{trigger}",
                content=content,
                description=description,
            )

            self._manager.add_macro(macro)
            self._save_macros()

            return {
                "output": f"Macro '{name}' added with trigger '{macro.trigger}'!",
                "data": {"name": name, "trigger": macro.trigger}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_expand(self, trigger: str, args: List[str]) -> Dict[str, Any]:
        """Expand a macro."""
        try:
            # Ensure trigger starts with /
            if not trigger.startswith("/"):
                trigger = f"/{trigger}"

            # Parse variables from args (format: key=value)
            variables = {}
            for arg in args:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    variables[key.strip()] = value.strip()

            expanded = self._manager.expand_macro(trigger, variables)
            if not expanded:
                return {"error": f"Macro '{trigger}' not found or disabled"}

            lines = [f"\nExpanded Macro: {trigger}"]
            lines.append("=" * 60)
            lines.append(expanded)
            lines.append("=" * 60)

            return {
                "output": "\n".join(lines),
                "data": {"trigger": trigger, "expanded": expanded, "variables": variables}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_remove(self, trigger: str) -> Dict[str, Any]:
        """Remove a macro."""
        try:
            # Ensure trigger starts with /
            if not trigger.startswith("/"):
                trigger = f"/{trigger}"

            success = self._manager.remove_macro(trigger)
            if success:
                self._save_macros()
                return {"output": f"Macro '{trigger}' removed!"}
            else:
                return {"error": f"Macro '{trigger}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_enable(self, trigger: str) -> Dict[str, Any]:
        """Enable a macro."""
        try:
            if not trigger.startswith("/"):
                trigger = f"/{trigger}"

            macro = self._manager.get_macro(trigger)
            if not macro:
                return {"error": f"Macro '{trigger}' not found"}

            macro.enabled = True
            self._save_macros()
            return {"output": f"Macro '{trigger}' enabled!"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_disable(self, trigger: str) -> Dict[str, Any]:
        """Disable a macro."""
        try:
            if not trigger.startswith("/"):
                trigger = f"/{trigger}"

            macro = self._manager.get_macro(trigger)
            if not macro:
                return {"error": f"Macro '{trigger}' not found"}

            macro.enabled = False
            self._save_macros()
            return {"output": f"Macro '{trigger}' disabled!"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_stats(self) -> Dict[str, Any]:
        """Show usage statistics."""
        try:
            stats = self._manager.get_usage_stats()
            macros = self._manager.list_macros()

            lines = ["\nMacro Usage Statistics:"]
            if stats:
                for trigger, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  {trigger}: {count} times")
            else:
                lines.append("  No usage recorded yet")

            lines.append(f"\nTotal macros: {len(macros)}")

            return {
                "output": "\n".join(lines),
                "data": {"stats": stats, "total_macros": len(macros)}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import macros from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self._storage_dir / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            count = self._manager.import_from_dict(data)
            self._save_macros()

            return {
                "output": f"Imported {count} macros from {file_path}!",
                "data": {"count": count, "file": str(path)}
            }
        except Exception as e:
            logger.error(f"Error importing macros: {e}")
            return {"error": str(e)}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export macros to file."""
        try:
            data = self._manager.export_to_dict()
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return {
                "output": f"Exported {len(data)} macros to {path}!",
                "data": {"count": len(data), "file": str(path)}
            }
        except Exception as e:
            logger.error(f"Error exporting macros: {e}")
            return {"error": str(e)}

    def _get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            macros = self._manager.list_macros()
            stats = self._manager.get_usage_stats()

            return {
                "storage_dir": str(self._storage_dir),
                "total_macros": len(macros),
                "enabled_macros": len([m for m in macros if m.enabled]),
                "usage_stats": stats,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Macros CLI

Commands:
    list                    List all macros
    add <name> <trigger> <content> Add a new macro
    expand <trigger>         Expand a macro
    remove <trigger>         Remove a macro
    enable <trigger>         Enable a macro
    disable <trigger>        Disable a macro
    stats                   Show usage statistics
    import <file>           Import macros from file
    export <file>           Export macros to file

Add Options:
    --description <text>    Description of the macro

Expand Options:
    Variables can be passed as key=value pairs

Examples:
    uv run backend/manage.py macros list
    uv run backend/manage.py macros add greeting /hello "Hi there!" --description "Standard greeting"
    uv run backend/manage.py macros expand /hello
    uv run backend/manage.py macros disable /hello
    uv run backend/manage.py macros export my_macros.json
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/macros", tags=["macros"])

        @router.get("/")
        async def list_macros():
            """List all macros."""
            macros = self._manager.list_macros()
            return {"macros": [m.__dict__ for m in macros]}

        @router.post("/")
        async def add_macro(request: dict):
            """Add a new macro."""
            from src.extensions.libraries.core_chat_engine.sillytavern.macros import Macro
            macro = Macro(**request)
            self._manager.add_macro(macro)
            self._save_macros()
            return {"success": True, "trigger": macro.trigger}

        @router.post("/expand")
        async def expand_macro(request: dict):
            """Expand a macro."""
            trigger = request.get("trigger")
            variables = request.get("variables", {})
            expanded = self._manager.expand_macro(trigger, variables)
            if expanded:
                return {"expanded": expanded}
            return {"error": "Macro not found"}, 404

        @router.delete("/{trigger}")
        async def delete_macro(trigger: str):
            """Delete a macro."""
            success = self._manager.remove_macro(trigger)
            if success:
                self._save_macros()
                return {"success": True}
            return {"error": "Macro not found"}, 404

        @router.get("/stats")
        async def get_stats():
            """Get usage statistics."""
            return self._manager.get_usage_stats()

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(MacrosExtension)
