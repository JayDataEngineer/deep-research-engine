"""Presets Extension

Generation parameter presets management.
Uses the presets library for core functionality.

CLI Usage:
    uv run backend/manage.py presets list
    uv run backend/manage.py presets create <name> --temperature 0.7 --top_p 0.9
    uv run backend/manage.py presets get <name>
    uv run backend/manage.py presets delete <name>
"""

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class PresetsExtension(LogicExtension):
    """Presets extension - CLI wrapper for preset management.

    This extension provides CLI access to the presets library
    for managing generation parameters and format presets.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None

    async def initialize(self, context):
        """Initialize the presets extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "presets"
        except (ImportError, AttributeError):
            # Fallback: go up to project root (backend/ -> project root)
            self._storage_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "data" / "inputs" / "presets"

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Presets extension initialized")

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
            list                    - List all presets
            get <name>              - Get preset details
            create <name>           - Create a new preset
            delete <name>           - Delete a preset
            import <file>           - Import preset from file
            export <name> <file>    - Export preset to file
            categories             - List preset categories
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("presets", "preset", "settings") or args:
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
                yield {"error": "Please provide a preset name"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand == "create":
            if not args:
                yield {"error": "Please provide a preset name"}
                return
            yield await self._cmd_create(args[0], args[1:])
        elif subcommand == "delete":
            if not args:
                yield {"error": "Please provide a preset name"}
                return
            yield await self._cmd_delete(args[0])
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        elif subcommand == "export":
            if len(args) < 2:
                yield {"error": "Usage: export <name> <file>"}
                return
            yield await self._cmd_export(args[0], args[1])
        elif subcommand == "categories":
            yield await self._cmd_categories()
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_presets":
            presets = self._list_presets()
            yield {"presets": presets}
        elif operation == "get_preset":
            name = input_data.get("name")
            if not name:
                yield {"error": "Preset name required"}
                return
            preset = self._load_preset(name)
            if preset:
                yield {"preset": preset}
            else:
                yield {"error": "Preset not found"}
        elif operation == "save_preset":
            data = input_data.get("preset", {})
            name = data.get("name")
            if not name:
                yield {"error": "Preset name required"}
                return
            self._save_preset(name, data)
            yield {"success": True, "name": name}
        elif operation == "delete_preset":
            name = input_data.get("name")
            if not name:
                yield {"error": "Preset name required"}
                return
            success = self._delete_preset(name)
            yield {"success": success}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _list_presets(self) -> List[str]:
        """List all preset names."""
        names = set()
        for filepath in self._storage_dir.iterdir():
            if filepath.is_file() and filepath.suffix in [".json", ".yaml", ".yml"]:
                names.add(filepath.stem)
        return sorted(names)

    def _load_preset(self, name: str) -> Optional[Dict]:
        """Load a preset by name."""
        safe_name = self._sanitize_filename(name)

        # Try JSON first
        filepath = self._storage_dir / f"{safe_name}.json"
        if not filepath.exists():
            filepath = self._storage_dir / f"{safe_name}.yaml"

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f) if filepath.suffix == ".json" else __import__("yaml").safe_load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading preset: {e}")
            return None

    def _save_preset(self, name: str, data: Dict) -> None:
        """Save a preset."""
        safe_name = self._sanitize_filename(name)
        filepath = self._storage_dir / f"{safe_name}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _delete_preset(self, name: str) -> bool:
        """Delete a preset."""
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
        """List all presets."""
        try:
            presets = self._list_presets()

            if not presets:
                return {
                    "output": "No presets found.",
                    "data": {"presets": []}
                }

            lines = ["\nPresets:"]
            for name in presets:
                preset = self._load_preset(name)
                if preset:
                    category = preset.get("category", "unknown")
                    desc = preset.get("description", "")
                    lines.append(f"  - {name} ({category})")
                    if desc:
                        lines.append(f"      {desc}")

            lines.append(f"\nTotal: {len(presets)} presets")

            return {
                "output": "\n".join(lines),
                "data": {"presets": presets}
            }
        except Exception as e:
            logger.error(f"Error listing presets: {e}")
            return {"error": str(e)}

    async def _cmd_get(self, name: str) -> Dict[str, Any]:
        """Get preset details."""
        try:
            preset = self._load_preset(name)
            if not preset:
                return {"error": f"Preset '{name}' not found"}

            lines = [f"\nPreset: {name}"]
            lines.append(f"Category: {preset.get('category', 'unknown')}")

            if preset.get("description"):
                lines.append(f"Description: {preset['description']}")

            # Show generation parameters
            if "temperature" in preset:
                lines.append(f"\nGeneration Parameters:")
                lines.append(f"  Temperature: {preset['temperature']}")
                lines.append(f"  Top P: {preset.get('top_p', 'N/A')}")
                lines.append(f"  Top K: {preset.get('top_k', 'N/A')}")
                lines.append(f"  Max Tokens: {preset.get('max_tokens', 'N/A')}")

            return {
                "output": "\n".join(lines),
                "data": {"preset": preset}
            }
        except Exception as e:
            logger.error(f"Error getting preset: {e}")
            return {"error": str(e)}

    async def _cmd_create(self, name: str, args: List[str]) -> Dict[str, Any]:
        """Create a new preset."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Create preset")
            parser.add_argument("--category", choices=["generation", "format", "character", "author_note", "complete"],
                               default="generation", help="Preset category")
            parser.add_argument("--description", default="", help="Description")
            parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
            parser.add_argument("--top_p", type=float, default=0.9, help="Top P")
            parser.add_argument("--top_k", type=int, default=0, help="Top K")
            parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens")

            parsed = parser.parse_args(args)

            preset = {
                "name": name,
                "category": parsed.category,
                "description": parsed.description,
                "temperature": parsed.temperature,
                "top_p": parsed.top_p,
                "top_k": parsed.top_k,
                "max_tokens": parsed.max_tokens,
            }

            self._save_preset(name, preset)

            return {
                "output": f"Preset '{name}' created successfully!",
                "data": {"name": name}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_delete(self, name: str) -> Dict[str, Any]:
        """Delete a preset."""
        try:
            success = self._delete_preset(name)
            if success:
                return {"output": f"Preset '{name}' deleted successfully!"}
            else:
                return {"error": f"Preset '{name}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import preset from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self._storage_dir / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) if path.suffix == ".json" else __import__("yaml").safe_load(f)

            name = data.get("name", path.stem)
            self._save_preset(name, data)

            return {
                "output": f"Preset '{name}' imported successfully!",
                "data": {"name": name}
            }
        except Exception as e:
            logger.error(f"Error importing preset: {e}")
            return {"error": str(e)}

    async def _cmd_export(self, name: str, file_path: str) -> Dict[str, Any]:
        """Export preset to file."""
        try:
            preset = self._load_preset(name)
            if not preset:
                return {"error": f"Preset '{name}' not found"}

            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset, f, indent=2, ensure_ascii=False)

            return {
                "output": f"Preset '{name}' exported to {path}!",
                "data": {"filepath": str(path)}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_categories(self) -> Dict[str, Any]:
        """List preset categories."""
        categories = {
            "generation": "Model generation parameters (temperature, top_p, etc.)",
            "format": "Chat/context template settings",
            "character": "Character-specific settings",
            "author_note": "Author's note settings",
            "complete": "Full preset (all categories)",
        }

        lines = ["\nPreset Categories:"]
        for cat, desc in categories.items():
            lines.append(f"  - {cat}: {desc}")

        return {
            "output": "\n".join(lines),
            "data": {"categories": categories}
        }

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Presets CLI

Commands:
    list                    List all presets
    get <name>              Get preset details
    create <name>           Create a new preset
    delete <name>           Delete a preset
    import <file>           Import preset from file
    export <name> <file>    Export preset to file
    categories             List preset categories

Create Options:
    --category <category>   Preset category (generation/format/character/author_note/complete)
    --description <text>    Description
    --temperature <float>   Temperature (default: 0.7)
    --top_p <float>         Top P (default: 0.9)
    --top_k <int>           Top K (default: 0)
    --max_tokens <int>      Max tokens (default: 500)

Examples:
    uv run backend/manage.py presets list
    uv run backend/manage.py presets create creative --temperature 0.9 --description "High creativity"
    uv run backend/manage.py presets get creative
    uv run backend/manage.py presets export creative preset_backup.json
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/presets", tags=["presets"])

        @router.get("/")
        async def list_presets():
            """List all presets."""
            presets = self._list_presets()
            return {"presets": presets}

        @router.get("/{name}")
        async def get_preset(name: str):
            """Get a preset by name."""
            preset = self._load_preset(name)
            if preset:
                return preset
            return {"error": "Preset not found"}, 404

        @router.post("/")
        async def save_preset(request: dict):
            """Save a preset."""
            name = request.get("name")
            if not name:
                return {"error": "name required"}, 400
            self._save_preset(name, request)
            return {"success": True, "name": name}

        @router.delete("/{name}")
        async def delete_preset(name: str):
            """Delete a preset."""
            success = self._delete_preset(name)
            if success:
                return {"success": True}
            return {"error": "Preset not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(PresetsExtension)
