"""Lorebooks Extension

Lorebook (World Info) management CLI wrapper.
Uses the lorebook_engine library for core functionality.
"""

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class LorebooksExtension(LogicExtension):
    """Lorebooks extension - CLI wrapper for lorebook_engine.

    This extension provides CLI access to the lorebook_engine library
    for managing world info / lorebooks with keyword and semantic activation.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None

    async def initialize(self, context):
        """Initialize the lorebooks extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "lorebooks"
        except (ImportError, AttributeError):
            # Fallback: go up to project root (backend/ -> project root)
            self._storage_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "data" / "inputs" / "lorebooks"

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Lorebooks extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all lorebooks
            get <name>              - Get lorebook details
            show <name>             - Alias for get
            create <name>           - Create a new lorebook
            add <name>              - Add entry to lorebook
            remove <name> <uid>     - Remove entry from lorebook
            match <name> <text>      - Test keyword matching
            delete <name>           - Delete a lorebook
            stats                   - Show storage statistics
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("lorebooks", "lore", "worldinfo", "wi") or args:
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
        elif subcommand in ("get", "show"):
            if not args:
                yield {"error": "Please provide a lorebook name"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand == "create":
            if not args:
                yield {"error": "Please provide a lorebook name"}
                return
            yield await self._cmd_create(args[0], args[1:])
        elif subcommand == "add":
            if not args:
                yield {"error": "Please provide a lorebook name"}
                return
            yield await self._cmd_add(args[0], args[1:])
        elif subcommand == "remove":
            if len(args) < 2:
                yield {"error": "Usage: remove <lorebook> <entry_uid>"}
                return
            yield await self._cmd_remove(args[0], args[1])
        elif subcommand == "match":
            if len(args) < 2:
                yield {"error": "Usage: match <lorebook> <text>"}
                return
            yield await self._cmd_match(args[0], " ".join(args[1:]))
        elif subcommand == "delete":
            if not args:
                yield {"error": "Please provide a lorebook name"}
                return
            yield await self._cmd_delete(args[0])
        elif subcommand == "stats":
            yield await self._cmd_stats()
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        from src.extensions.libraries.lorebook_engine import Lorebook, LorebookEntry

        if operation == "list_lorebooks":
            lorebooks = self._list_lorebooks()
            yield {"lorebooks": lorebooks}
        elif operation == "get_lorebook":
            name = input_data.get("name")
            if not name:
                yield {"error": "Lorebook name required"}
                return
            lorebook = self._load_lorebook(name)
            if lorebook:
                yield {"lorebook": lorebook.model_dump()}
            else:
                yield {"error": "Lorebook not found"}
        elif operation == "save_lorebook":
            data = input_data.get("lorebook", {})
            lorebook = Lorebook(**data)
            self._save_lorebook(lorebook)
            yield {"success": True, "name": lorebook.name}
        elif operation == "delete_lorebook":
            name = input_data.get("name")
            if not name:
                yield {"error": "Lorebook name required"}
                return
            success = self._delete_lorebook(name)
            yield {"success": success}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _list_lorebooks(self) -> List[str]:
        """List all lorebook names."""
        names = set()
        for filepath in self._storage_dir.iterdir():
            if filepath.is_file() and filepath.suffix in [".json", ".yaml", ".yml"]:
                names.add(filepath.stem)
        return sorted(names)

    def _load_lorebook(self, name: str):
        """Load a lorebook by name."""
        from src.extensions.libraries.lorebook_engine import Lorebook

        safe_name = self._sanitize_filename(name)

        # Try JSON first
        filepath = self._storage_dir / f"{safe_name}.json"
        if not filepath.exists():
            filepath = self._storage_dir / f"{safe_name}.yaml"

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Lorebook(**data)
        except Exception as e:
            logger.error(f"Error loading lorebook: {e}")
            return None

    def _save_lorebook(self, lorebook):
        """Save a lorebook."""
        from src.extensions.libraries.lorebook_engine import Lorebook

        safe_name = self._sanitize_filename(lorebook.name)
        filepath = self._storage_dir / f"{safe_name}.json"

        # Use mode='json' for JSON-serializable output (handles datetime)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(lorebook.model_dump(mode='json'), f, indent=2, ensure_ascii=False)

    def _delete_lorebook(self, name: str) -> bool:
        """Delete a lorebook."""
        safe_name = self._sanitize_filename(name)
        for ext in ["json", "yaml", "yml"]:
            filepath = self._storage_dir / f"{safe_name}.{ext}"
            if filepath.exists():
                filepath.unlink()
                return True
        return False

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a name for safe filename usage."""
        unsafe = '<>:"/\\|?*'
        for char in unsafe:
            name = name.replace(char, "_")
        return name.strip()

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all lorebooks."""
        try:
            lorebooks = self._list_lorebooks()
            stats = self._get_stats()

            if not lorebooks:
                return {
                    "output": "No lorebooks found.",
                    "data": {"stats": stats}
                }

            lines = ["\nLorebooks:"]
            for name in lorebooks:
                lines.append(f"  - {name}")

            lines.append(f"\nTotal: {len(lorebooks)} lorebooks")
            return {
                "output": "\n".join(lines),
                "data": {"lorebooks": lorebooks, "stats": stats}
            }
        except Exception as e:
            logger.error(f"Error listing lorebooks: {e}")
            return {"error": str(e)}

    async def _cmd_get(self, name: str) -> Dict[str, Any]:
        """Get lorebook details."""
        try:
            lorebook = self._load_lorebook(name)
            if not lorebook:
                return {"error": f"Lorebook '{name}' not found"}

            lines = [f"\nLorebook: {lorebook.name}"]
            if lorebook.description:
                lines.append(f"Description: {lorebook.description}")
            lines.append(f"Scan Depth: {lorebook.scan_depth}")
            lines.append(f"Token Budget: {lorebook.token_budget}")
            lines.append(f"Entries: {len(lorebook.entries)}")

            if lorebook.entries:
                lines.append("\nEntries:")
                for entry in lorebook.entries[:10]:  # Show first 10
                    keywords = ", ".join(entry.keys[:5])  # Show first 5 keywords
                    status = "enabled" if entry.enabled else "disabled"
                    lines.append(f"  [{entry.id}] {status} - {keywords}...")
                    if entry.comment:
                        lines.append(f"      {entry.comment}")
                if len(lorebook.entries) > 10:
                    lines.append(f"  ... and {len(lorebook.entries) - 10} more")

            return {
                "output": "\n".join(lines),
                "data": {"lorebook": lorebook.model_dump(mode='json')}
            }
        except Exception as e:
            logger.error(f"Error getting lorebook: {e}")
            return {"error": str(e)}

    async def _cmd_create(self, name: str, args: List[str]) -> Dict[str, Any]:
        """Create a new lorebook."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Create lorebook")
            parser.add_argument("--description", default="", help="Description")
            parser.add_argument("--scan-depth", type=int, default=4, help="Scan depth")
            parser.add_argument("--budget", type=int, default=100, help="Token budget")

            parsed = parser.parse_args(args)

            from src.extensions.libraries.lorebook_engine import Lorebook

            lorebook = Lorebook(
                name=name,
                description=parsed.description,
                scan_depth=parsed.scan_depth,
                token_budget=parsed.budget,
            )

            self._save_lorebook(lorebook)

            return {
                "output": f"Lorebook '{name}' created successfully!",
                "data": {"name": name}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_add(self, name: str, args: List[str]) -> Dict[str, Any]:
        """Add entry to lorebook."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Add lorebook entry")
            parser.add_argument("--keywords", required=True, help="Comma-separated keywords")
            parser.add_argument("--content", required=True, help="Entry content")
            parser.add_argument("--comment", default="", help="Comment")
            parser.add_argument("--depth", type=int, default=4, help="Scan depth")

            parsed = parser.parse_args(args)

            # Get next UID
            lorebook = self._load_lorebook(name)
            if not lorebook:
                return {"error": f"Lorebook '{name}' not found"}

            from src.extensions.libraries.lorebook_engine import LorebookEntry

            max_uid = max([e.id for e in lorebook.entries], default=-1)
            uid = max_uid + 1

            entry = LorebookEntry(
                id=uid,
                keys=[k.strip() for k in parsed.keywords.split(",")],
                content=parsed.content,
                comment=parsed.comment,
                extensions={"depth": parsed.depth, "probability": 100},
            )

            lorebook.entries.append(entry)
            self._save_lorebook(lorebook)

            return {
                "output": f"Entry {uid} added to '{name}'!",
                "data": {"uid": uid, "keywords": parsed.keywords}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_remove(self, name: str, uid_str: str) -> Dict[str, Any]:
        """Remove entry from lorebook."""
        try:
            uid = int(uid_str)
            lorebook = self._load_lorebook(name)
            if not lorebook:
                return {"error": f"Lorebook '{name}' not found"}

            lorebook.entries = [e for e in lorebook.entries if e.id != uid]
            self._save_lorebook(lorebook)

            return {"output": f"Entry {uid} removed from '{name}'!"}
        except ValueError:
            return {"error": "Entry UID must be a number"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_match(self, name: str, text: str) -> Dict[str, Any]:
        """Test keyword matching."""
        try:
            from src.extensions.libraries.lorebook_engine import LorebookTrigger

            lorebook = self._load_lorebook(name)
            if not lorebook:
                return {"error": f"Lorebook '{name}' not found"}

            trigger = LorebookTrigger()
            result = trigger.trigger(lorebook.entries, text)

            lines = [f"\nMatching against '{name}':"]
            lines.append(f"Input: {text[:100]}...")
            lines.append(f"\nActivated {result.count()} entries:")

            # Iterate through activated entries
            for entry_id, (entry, content) in result.activated_entries.items():
                keys = ", ".join(entry.keys[:5])
                matched = result.matched_keywords.get(entry_id, set())
                lines.append(f"  [{entry.id}] {keys}")
                if matched:
                    lines.append(f"      Matched: {', '.join(matched)}")

            return {
                "output": "\n".join(lines),
                "data": {
                    "match_count": result.count(),
                    "result": result.to_dict()
                }
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_delete(self, name: str) -> Dict[str, Any]:
        """Delete a lorebook."""
        try:
            success = self._delete_lorebook(name)
            if success:
                return {"output": f"Lorebook '{name}' deleted successfully!"}
            else:
                return {"error": f"Lorebook '{name}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_stats(self) -> Dict[str, Any]:
        """Show storage statistics."""
        try:
            stats = self._get_stats()
            lorebooks = self._list_lorebooks()

            lines = ["\nLorebook Storage Statistics:"]
            lines.append(f"  Directory: {stats.get('directory', 'N/A')}")
            lines.append(f"  Total lorebooks: {stats.get('count', 0)}")
            lines.append(f"  Total entries: {stats.get('entry_count', 0)}")

            return {
                "output": "\n".join(lines),
                "data": stats
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            lorebooks = self._list_lorebooks()
            total_entries = 0
            for name in lorebooks:
                lorebook = self._load_lorebook(name)
                if lorebook:
                    total_entries += len(lorebook.entries)

            return {
                "directory": str(self._storage_dir),
                "count": len(lorebooks),
                "entry_count": total_entries,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Lorebooks CLI

Commands:
    list                    List all lorebooks
    get <name>              Get lorebook details
    create <name>           Create a new lorebook
    add <name>              Add entry to lorebook
    remove <name> <uid>     Remove entry from lorebook
    match <name> <text>      Test keyword matching
    delete <name>           Delete a lorebook
    stats                   Show storage statistics

Create Options:
    --description <text>    Description
    --scan-depth <int>      Scan depth (default: 4)
    --budget <int>          Token budget (default: 100)

Add Options:
    --keywords <keys>       Comma-separated keywords (required)
    --content <text>        Entry content (required)
    --comment <text>        Comment
    --depth <int>           Scan depth (default: 4)

Examples:
    uv run backend/manage.py lorebooks list
    uv run backend/manage.py lorebooks create fantasy_world
    uv run backend/manage.py lorebooks add fantasy_world --keywords dragon,fire --content "..."
    uv run backend/manage.py lorebooks match fantasy_world "The dragon breathed fire"
    uv run backend/manage.py lorebooks delete fantasy_world
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from .routes import router
        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(LorebooksExtension)
