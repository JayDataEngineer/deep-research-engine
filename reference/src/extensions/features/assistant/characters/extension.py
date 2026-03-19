"""Characters Extension

Character card management CLI wrapper.
Uses the character_engine library for core functionality.

Supports TavernCard v2 format with PNG embedding, JSON, and YAML.

CLI Usage:
    uv run backend/manage.py characters list
    uv run backend/manage.py characters get <name>
    uv run backend/manage.py characters import <file_path>
    uv run backend/manage.py characters export <name> [output_path] [--format png|json|yaml]
    uv run backend/manage.py characters delete <name>
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class CharactersExtension(LogicExtension):
    """Characters extension - CLI wrapper for character_engine.

    This extension provides CLI access to the character_engine library
    for managing character cards (TavernCard v2 format).
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage = None

    async def initialize(self, context):
        """Initialize the characters extension."""
        await super().initialize(context)

        # Import character storage from engine
        from src.extensions.libraries.character_engine import get_character_storage
        self._storage = get_character_storage()

        logger.info("Characters extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        self._storage = None

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all characters
            get <name>              - Get character details
            show <name>             - Alias for get
            import <file_path>       - Import character from file
            export <name>           - Export character to file
            delete <name>           - Delete a character
            stats                   - Show storage statistics

        Supported import formats:
            - PNG (TavernCard v2 embedded)
            - JSON
            - YAML

        Supported export formats:
            - PNG (TavernCard v2 embedded) --format png
            - JSON (default)
            - YAML --format yaml
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("characters", "char", "chars") or args:
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
                yield {"error": "Please provide a character name"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        elif subcommand == "export":
            if not args:
                yield {"error": "Please provide a character name"}
                return
            yield await self._cmd_export(args[0], args[1:])
        elif subcommand == "delete":
            if not args:
                yield {"error": "Please provide a character name"}
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
        from src.extensions.libraries.character_engine import CharacterCard, CharacterBook

        if operation == "list_characters":
            characters = self._storage.list_cards()
            yield {"characters": characters}
        elif operation == "get_character":
            name = input_data.get("name")
            if not name:
                yield {"error": "Character name required"}
                return
            card = self._storage.load_card(name)
            if card:
                yield {"character": card.model_dump()}
            else:
                yield {"error": "Character not found"}
        elif operation == "save_character":
            data = input_data.get("character", {})
            card = CharacterCard(**data)
            self._storage.save_card(card)
            yield {"success": True, "name": card.name}
        elif operation == "delete_character":
            name = input_data.get("name")
            if not name:
                yield {"error": "Character name required"}
                return
            success = self._storage.delete_card(name)
            yield {"success": success}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all characters."""
        try:
            characters = self._storage.list_cards()
            stats = self._get_stats()

            if not characters:
                return {
                    "output": "No characters found.",
                    "data": {"stats": stats}
                }

            lines = ["\nCharacters:"]
            for char_name in characters:
                lines.append(f"  - {char_name}")

            lines.append(f"\nTotal: {len(characters)} characters")
            return {
                "output": "\n".join(lines),
                "data": {"characters": characters, "stats": stats}
            }
        except Exception as e:
            logger.error(f"Error listing characters: {e}")
            return {"error": str(e)}

    async def _cmd_get(self, name: str) -> Dict[str, Any]:
        """Get character details."""
        try:
            card = self._storage.load_card(name)
            if not card:
                return {"error": f"Character '{name}' not found"}

            lines = [f"\nCharacter: {card.name}"]
            if card.description:
                lines.append(f"\nDescription:\n{card.description}")
            if card.personality:
                lines.append(f"\nPersonality:\n{card.personality}")
            if card.scenario:
                lines.append(f"\nScenario:\n{card.scenario}")
            if card.first_mes:
                lines.append(f"\nFirst Message:\n{card.first_mes}")
            if card.tags:
                lines.append(f"\nTags: {', '.join(card.tags)}")

            # Character book info
            if card.character_book and card.character_book.entries:
                lines.append(f"\nCharacter Book: {len(card.character_book.entries)} entries")

            return {
                "output": "\n".join(lines),
                "data": {"character": card.model_dump()}
            }
        except Exception as e:
            logger.error(f"Error getting character: {e}")
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import character from file."""
        try:
            from src.extensions.libraries.character_engine import parse_character_card

            path = Path(file_path)
            if not path.exists():
                path = Path.cwd() / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            # Parse character card
            card = parse_character_card(str(path))
            if not card:
                return {"error": "Failed to parse character file"}

            # Save to storage
            self._storage.save_card(card)

            return {
                "output": f"Character '{card.name}' imported successfully!",
                "data": {"name": card.name, "format": path.suffix}
            }
        except Exception as e:
            logger.error(f"Error importing character: {e}")
            return {"error": str(e)}

    async def _cmd_export(
        self,
        name: str,
        args: List[str]
    ) -> Dict[str, Any]:
        """Export character to file."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Export character")
            parser.add_argument("--format", choices=["json", "yaml", "png"],
                               default="json", help="Export format")
            parser.add_argument("--output", help="Output file path")
            parser.add_argument("--image", help="Avatar image path (for PNG export)")

            try:
                parsed = parser.parse_args(args)
            except:
                # Simple case: just output path as arg
                if len(args) == 1 and not args[0].startswith("--"):
                    parsed = type('obj', (object,), {'format': 'json', 'output': args[0], 'image': None})()
                else:
                    raise

            card = self._storage.load_card(name)
            if not card:
                return {"error": f"Character '{name}' not found"}

            # Export based on format
            output_path = parsed.output

            if parsed.format == "png":
                # PNG export with embedded data
                if not output_path:
                    output_path = f"{name}.png"

                # Get image data
                if parsed.image:
                    image_path = Path(parsed.image)
                    if image_path.exists():
                        image_data = image_path.read_bytes()
                    else:
                        return {"error": f"Image not found: {parsed.image}"}
                else:
                    # Create default image
                    from PIL import Image
                    img = Image.new("RGB", (400, 600), color=(75, 85, 99))
                    from io import BytesIO
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    image_data = buf.getvalue()

                # Export as PNG (need to add this method to character_engine)
                return await self._export_png(card, image_data, output_path)
            else:
                # JSON or YAML export
                if not output_path:
                    output_path = f"{name}.{parsed.format}"

                output_path = Path(output_path)

                if parsed.format == "yaml":
                    import yaml
                    with open(output_path, "w", encoding="utf-8") as f:
                        yaml.dump(card.model_dump(), f, default_flow_style=False, allow_unicode=True)
                else:
                    import json
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(card.model_dump(), f, indent=2, ensure_ascii=False)

                return {
                    "output": f"Character exported to: {output_path}",
                    "data": {"filepath": str(output_path)}
                }

        except Exception as e:
            logger.error(f"Error exporting character: {e}")
            return {"error": str(e)}

    async def _export_png(self, card, image_data: bytes, output_path: str) -> Dict[str, Any]:
        """Export character as PNG with embedded data."""
        try:
            from PIL import Image, PngImagePlugin
            from io import BytesIO
            import base64
            import json

            # Open or create image
            img = Image.open(BytesIO(image_data))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Prepare character data
            char_data = json.dumps(card.model_dump(), ensure_ascii=False)
            char_bytes = char_data.encode("utf-8")
            b64_data = base64.b64encode(char_bytes).decode("ascii")

            # Create tEXt chunks with character data
            chunks = [b64_data[i:i+4000] for i in range(0, len(b64_data), 4000)]

            meta = PngImagePlugin.PngInfo()
            for i, chunk in enumerate(chunks):
                meta.add_text(f"chara{i}", chunk)
            meta.add_text("chara-version", "2")

            # Save
            output_path = Path(output_path)
            img.save(output_path, pnginfo=meta, optimize=True)

            return {
                "output": f"Character exported to: {output_path}",
                "data": {"filepath": str(output_path)}
            }
        except ImportError:
            return {"error": "PIL not installed, cannot export PNG"}
        except Exception as e:
            return {"error": f"PNG export failed: {e}"}

    async def _cmd_delete(self, name: str) -> Dict[str, Any]:
        """Delete a character."""
        try:
            success = self._storage.delete_card(name)
            if success:
                return {"output": f"Character '{name}' deleted successfully!"}
            else:
                return {"error": f"Character '{name}' not found"}
        except Exception as e:
            logger.error(f"Error deleting character: {e}")
            return {"error": str(e)}

    async def _cmd_stats(self) -> Dict[str, Any]:
        """Show storage statistics."""
        try:
            stats = self._get_stats()
            lines = ["\nCharacter Storage Statistics:"]
            lines.append(f"  Directory: {stats.get('directory', 'N/A')}")
            lines.append(f"  Total characters: {stats.get('count', 0)}")
            return {
                "output": "\n".join(lines),
                "data": stats
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            characters = self._storage.list_cards()
            return {
                "directory": str(self._storage.base_dir),
                "count": len(characters),
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Characters CLI

Commands:
    list                    List all characters
    get <name>              Get character details
    show <name>             Alias for get
    import <file_path>       Import character from file
    export <name>           Export character to file
    delete <name>           Delete a character
    stats                   Show storage statistics

Import Formats:
    PNG (TavernCard v2), JSON, YAML

Export Options:
    --format <json|yaml|png>  Export format (default: json)
    --output <path>          Output file path
    --image <path>           Avatar image path (for PNG)

Examples:
    uv run backend/manage.py characters list
    uv run backend/manage.py characters get Alice
    uv run backend/manage.py characters import character.png
    uv run backend/manage.py characters export Alice --format png --image avatar.png
    uv run backend/manage.py characters export Alice --format yaml
    uv run backend/manage.py characters delete Alice
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        # Use character_engine's router
        from src.extensions.libraries.character_engine import router as char_router
        return char_router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(CharactersExtension)
