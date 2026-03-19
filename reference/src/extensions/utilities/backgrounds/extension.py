"""Backgrounds Extension

Background image management for chats.
Provides CLI tools for managing chat background images.

CLI Usage:
    uv run backend/manage.py backgrounds list
    uv run backend/manage.py backgrounds set beach.jpg
    uv run backend/manage.py backgrounds get
    uv run backend/manage.py backgrounds random
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class Background:
    """A background image."""

    def __init__(
        self,
        filename: str,
        path: str,
        category: str = "general",
        enabled: bool = True,
    ):
        self.filename = filename
        self.path = path
        self.category = category
        self.enabled = enabled
        self.added_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "category": self.category,
            "enabled": self.enabled,
            "added_at": self.added_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Background":
        bg = cls(
            filename=data["filename"],
            path=data["path"],
            category=data.get("category", "general"),
            enabled=data.get("enabled", True),
        )
        if "added_at" in data:
            bg.added_at = datetime.fromisoformat(data["added_at"])
        return bg


class BackgroundsExtension(LogicExtension):
    """Backgrounds extension - CLI wrapper for background management.

    This extension provides CLI access to background image management
    for customizing chat appearance.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._backgrounds_dir = None
        self._backgrounds: List[Background] = []
        self._current_background: Optional[str] = None

    async def initialize(self, context):
        """Initialize the backgrounds extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "backgrounds"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "backgrounds"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Set up backgrounds directory (where images are stored)
        self._backgrounds_dir = self._storage_dir / "images"
        self._backgrounds_dir.mkdir(parents=True, exist_ok=True)

        # Load backgrounds
        self._load_backgrounds()

        logger.info(f"Backgrounds extension initialized with {len(self._backgrounds)} backgrounds")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_backgrounds()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all backgrounds
            set <filename>          - Set current background
            get                     - Get current background
            random                  - Set random background
            add <file>              - Add a new background
            remove <filename>       - Remove a background
            categories              - List background categories
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("backgrounds", "bg", "background", "wallpaper") or args:
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
        elif subcommand in ("set", "use", "select"):
            if not args:
                yield {"error": "Please provide a background filename"}
                return
            yield await self._cmd_set(args[0])
        elif subcommand == "get":
            yield await self._cmd_get()
        elif subcommand in ("random", "rand"):
            yield await self._cmd_random()
        elif subcommand in ("add", "import"):
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_add(args[0])
        elif subcommand in ("remove", "delete", "rm"):
            if not args:
                yield {"error": "Please provide a background filename"}
                return
            yield await self._cmd_remove(args[0])
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
        if operation == "list_backgrounds":
            yield {"backgrounds": [b.to_dict() for b in self._backgrounds]}
        elif operation == "set_background":
            filename = input_data.get("filename")
            if not filename:
                yield {"error": "filename required"}
                return
            success = self._set_background(filename)
            if success:
                yield {"success": True, "current": self._current_background}
            else:
                yield {"error": "Background not found"}
        elif operation == "get_background":
            yield {"current": self._current_background}
        elif operation == "set_random":
            result = self._set_random_background()
            yield {"current": result}
        elif operation == "add_background":
            file_path = input_data.get("file_path")
            category = input_data.get("category", "general")
            if not file_path:
                yield {"error": "file_path required"}
                return
            bg = await self._add_background(file_path, category)
            yield {"background": bg.to_dict()}
        elif operation == "remove_background":
            filename = input_data.get("filename")
            if not filename:
                yield {"error": "filename required"}
                return
            success = self._remove_background(filename)
            if success:
                self._save_backgrounds()
                yield {"success": True}
            else:
                yield {"error": "Background not found"}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_backgrounds(self):
        """Load backgrounds from storage."""
        # Load index file
        index_file = self._storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                    self._backgrounds = [Background.from_dict(b) for b in data.get("backgrounds", [])]
                    self._current_background = data.get("current")
            except Exception as e:
                logger.error(f"Error loading backgrounds: {e}")

        # Scan backgrounds directory for new images
        self._scan_backgrounds_dir()

    def _save_backgrounds(self):
        """Save backgrounds to storage."""
        index_file = self._storage_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump({
                "backgrounds": [b.to_dict() for b in self._backgrounds],
                "current": self._current_background,
            }, f, indent=2)

    def _scan_backgrounds_dir(self):
        """Scan backgrounds directory for image files."""
        if not self._backgrounds_dir.exists():
            return

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

        # Scan for images
        existing_filenames = {b.filename for b in self._backgrounds}
        for file_path in self._backgrounds_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                filename = file_path.name
                if filename not in existing_filenames:
                    # Add new background
                    bg = Background(
                        filename=filename,
                        path=str(file_path),
                        category="general",
                    )
                    self._backgrounds.append(bg)
                    logger.info(f"Found new background: {filename}")

    # ========================================================================
    # Background Management
    # ========================================================================

    def _set_background(self, filename: str) -> bool:
        """Set the current background."""
        # Try to find by filename or partial match
        for bg in self._backgrounds:
            if bg.filename == filename or filename.lower() in bg.filename.lower():
                self._current_background = bg.filename
                self._save_backgrounds()
                return True
        return False

    def _set_random_background(self) -> Optional[str]:
        """Set a random background."""
        enabled = [b for b in self._backgrounds if b.enabled]
        if not enabled:
            return None

        bg = random.choice(enabled)
        self._current_background = bg.filename
        self._save_backgrounds()
        return bg.filename

    async def _add_background(self, file_path: str, category: str = "general") -> Background:
        """Add a new background."""
        import shutil

        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Copy to backgrounds directory
        dest = self._backgrounds_dir / source.name
        shutil.copy2(source, dest)

        bg = Background(
            filename=source.name,
            path=str(dest),
            category=category,
        )
        self._backgrounds.append(bg)
        self._save_backgrounds()

        return bg

    def _remove_background(self, filename: str) -> bool:
        """Remove a background."""
        for i, bg in enumerate(self._backgrounds):
            if bg.filename == filename or filename.lower() in bg.filename.lower():
                # Delete file
                try:
                    Path(bg.path).unlink()
                except Exception as e:
                    logger.warning(f"Could not delete background file: {e}")

                self._backgrounds.pop(i)

                # Clear current if this was the active background
                if self._current_background == bg.filename:
                    self._current_background = None

                return True
        return False

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all backgrounds."""
        try:
            if not self._backgrounds:
                return {
                    "output": "No backgrounds found.",
                    "data": {"backgrounds": []}
                }

            # Group by category
            by_category: Dict[str, List[Background]] = {}
            for bg in self._backgrounds:
                if bg.category not in by_category:
                    by_category[bg.category] = []
                by_category[bg.category].append(bg)

            lines = [f"\nBackgrounds ({len(self._backgrounds)} total):"]

            for category, bgs in sorted(by_category.items()):
                current_marker = " [CURRENT]" if self._current_background in [b.filename for b in bgs] else ""
                lines.append(f"\n{category}{current_marker}:")
                for bg in bgs:
                    current = " * " if bg.filename == self._current_background else "   "
                    status = "✓" if bg.enabled else "✗"
                    lines.append(f"{current}[{status}] {bg.filename}")

            return {
                "output": "\n".join(lines),
                "data": {"backgrounds": [b.to_dict() for b in self._backgrounds]},
            }
        except Exception as e:
            logger.error(f"Error listing backgrounds: {e}")
            return {"error": str(e)}

    async def _cmd_set(self, filename: str) -> Dict[str, Any]:
        """Set current background."""
        success = self._set_background(filename)
        if success:
            return {
                "output": f"Background set to: {filename}",
                "data": {"current": self._current_background},
            }
        return {"error": f"Background not found: {filename}"}

    async def _cmd_get(self) -> Dict[str, Any]:
        """Get current background."""
        if self._current_background:
            bg = next((b for b in self._backgrounds if b.filename == self._current_background), None)
            if bg:
                return {
                    "output": f"Current background: {bg.filename}",
                    "data": {"background": bg.to_dict()},
                }
        return {
            "output": "No background set",
            "data": {"current": None},
        }

    async def _cmd_random(self) -> Dict[str, Any]:
        """Set random background."""
        result = self._set_random_background()
        if result:
            return {
                "output": f"Random background set to: {result}",
                "data": {"current": result},
            }
        return {"error": "No backgrounds available"}

    async def _cmd_add(self, file_path: str) -> Dict[str, Any]:
        """Add a new background."""
        try:
            bg = await self._add_background(file_path)
            return {
                "output": f"Background added: {bg.filename}",
                "data": {"background": bg.to_dict()},
            }
        except FileNotFoundError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_remove(self, filename: str) -> Dict[str, Any]:
        """Remove a background."""
        success = self._remove_background(filename)
        if success:
            self._save_backgrounds()
            return {"output": f"Background removed: {filename}"}
        return {"error": f"Background not found: {filename}"}

    async def _cmd_categories(self) -> Dict[str, Any]:
        """List background categories."""
        categories = sorted(set(b.category for b in self._backgrounds))

        lines = ["\nBackground Categories:"]
        for cat in categories:
            count = sum(1 for b in self._backgrounds if b.category == cat)
            lines.append(f"  - {cat}: {count}")

        return {
            "output": "\n".join(lines),
            "data": {"categories": categories},
        }

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Backgrounds CLI

Commands:
    list                    List all backgrounds
    set <filename>          Set current background
    get                     Get current background
    random                  Set random background
    add <file>              Add a new background
    remove <filename>       Remove a background
    categories              List background categories

Examples:
    uv run backend/manage.py backgrounds list
    uv run backend/manage.py backgrounds set beach.jpg
    uv run backend/manage.py backgrounds random
    uv run backend/manage.py backgrounds add /path/to/image.jpg

Supported formats: jpg, jpeg, png, gif, webp, bmp
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/backgrounds", tags=["backgrounds"])

        @router.get("/")
        async def list_backgrounds():
            """List all backgrounds."""
            return {"backgrounds": [b.to_dict() for b in self._backgrounds]}

        @router.get("/current")
        async def get_current():
            """Get current background."""
            return {"current": self._current_background}

        @router.put("/current")
        async def set_background(request: dict):
            """Set current background."""
            filename = request.get("filename")
            if not filename:
                return {"error": "filename required"}, 400
            success = self._set_background(filename)
            if success:
                return {"success": True, "current": self._current_background}
            return {"error": "Background not found"}, 404

        @router.put("/random")
        async def set_random():
            """Set random background."""
            result = self._set_random_background()
            if result:
                return {"current": result}
            return {"error": "No backgrounds available"}, 404

        @router.post("/add")
        async def add_background(request: dict):
            """Add a new background."""
            file_path = request.get("file_path")
            category = request.get("category", "general")
            if not file_path:
                return {"error": "file_path required"}, 400
            try:
                bg = await self._add_background(file_path, category)
                return {"background": bg.to_dict()}
            except FileNotFoundError:
                return {"error": "File not found"}, 404

        @router.delete("/{filename}")
        async def remove_background(filename: str):
            """Remove a background."""
            success = self._remove_background(filename)
            if success:
                self._save_backgrounds()
                return {"success": True}
            return {"error": "Background not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(BackgroundsExtension)
