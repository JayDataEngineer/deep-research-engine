"""Author Notes Extension

Author's notes for conversation guidance.
Uses the author_notes library for core functionality.

CLI Usage:
    uv run backend/manage.py author_notes list
    uv run backend/manage.py author_notes create <name> --content "Write in first person"
    uv run backend/manage.py author_notes get <name>
    uv run backend/manage.py author_notes set <name>
"""

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class AuthorNote:
    """An author's note entry."""

    def __init__(
        self,
        name: str,
        content: str,
        priority: int = 50,
        position: str = "before_chat",
        enabled: bool = True,
    ):
        self.id = f"{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.name = name
        self.content = content
        self.priority = priority  # 0-100, higher = more important
        self.position = position  # before_chat, after_chat, inline
        self.enabled = enabled
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "priority": self.priority,
            "position": self.position,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthorNote":
        note = cls(
            name=data["name"],
            content=data["content"],
            priority=data.get("priority", 50),
            position=data.get("position", "before_chat"),
            enabled=data.get("enabled", True),
        )
        note.id = data["id"]
        if "created_at" in data:
            note.created_at = datetime.fromisoformat(data["created_at"])
        return note


class AuthorNotesExtension(LogicExtension):
    """Author Notes extension - CLI wrapper for author note management.

    This extension provides CLI access to the author notes library
    for managing contextual notes that guide AI responses.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._notes: List[AuthorNote] = []
        self._active_note_id: Optional[str] = None

    async def initialize(self, context):
        """Initialize the author notes extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "author_notes"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "author_notes"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing notes
        self._load_notes()

        logger.info("Author Notes extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_notes()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all author notes
            create <name>           - Create a new author note
            get <name>              - Get author note details
            set <name>              - Set active author note
            update <name>           - Update an author note
            delete <name>           - Delete an author note
            enable <name>           - Enable an author note
            disable <name>          - Disable an author note
            export <file>           - Export notes to file
            import <file>           - Import notes from file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("author_notes", "notes", "author", "an") or args:
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
        elif subcommand in ("create", "add", "new"):
            if not args:
                yield {"error": "Please provide a note name"}
                return
            yield await self._cmd_create(args[0], args[1:])
        elif subcommand in ("get", "show"):
            if not args:
                yield {"error": "Please provide a note name or ID"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand in ("set", "activate", "use"):
            if not args:
                yield {"error": "Please provide a note name or ID"}
                return
            yield await self._cmd_set(args[0])
        elif subcommand in ("update", "edit"):
            if not args:
                yield {"error": "Please provide a note name or ID"}
                return
            yield await self._cmd_update(args[0], args[1:])
        elif subcommand in ("delete", "remove", "rm"):
            if not args:
                yield {"error": "Please provide a note name or ID"}
                return
            yield await self._cmd_delete(args[0])
        elif subcommand == "enable":
            if not args:
                yield {"error": "Please provide a note name or ID"}
                return
            yield await self._cmd_enable(args[0])
        elif subcommand == "disable":
            if not args:
                yield {"error": "Please provide a note name or ID"}
                return
            yield await self._cmd_disable(args[0])
        elif subcommand == "export":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_export(args[0])
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_notes":
            yield {"notes": [n.to_dict() for n in self._notes]}
        elif operation == "create_note":
            name = input_data.get("name")
            content = input_data.get("content")
            if not name or not content:
                yield {"error": "name and content required"}
                return
            note = self._add_note(
                name=name,
                content=content,
                priority=input_data.get("priority", 50),
                position=input_data.get("position", "before_chat"),
            )
            self._save_notes()
            yield {"note": note.to_dict()}
        elif operation == "get_note":
            note_id = input_data.get("id") or input_data.get("name")
            if not note_id:
                yield {"error": "id or name required"}
                return
            note = self._find_note(note_id)
            if note:
                yield {"note": note.to_dict()}
            else:
                yield {"error": "Note not found"}
        elif operation == "set_active":
            note_id = input_data.get("id") or input_data.get("name")
            if not note_id:
                yield {"error": "id or name required"}
                return
            success = self._set_active(note_id)
            if success:
                yield {"success": True, "active_id": self._active_note_id}
            else:
                yield {"error": "Note not found"}
        elif operation == "delete_note":
            note_id = input_data.get("id") or input_data.get("name")
            if not note_id:
                yield {"error": "id or name required"}
                return
            success = self._remove_note(note_id)
            if success:
                self._save_notes()
                yield {"success": True}
            else:
                yield {"error": "Note not found"}
        elif operation == "get_active":
            if self._active_note_id:
                note = self._find_note(self._active_note_id)
                if note:
                    yield {"note": note.to_dict()}
                else:
                    yield {"error": "Active note not found"}
            else:
                yield {"note": None}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_notes(self):
        """Load notes from storage."""
        notes_file = self._storage_dir / "notes.json"

        if not notes_file.exists():
            # Create default notes
            self._create_default_notes()
            return

        try:
            with open(notes_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._notes = [AuthorNote.from_dict(n) for n in data]

            # Load active note ID
            active_file = self._storage_dir / "active.json"
            if active_file.exists():
                with open(active_file, "r") as f:
                    active_data = json.load(f)
                    self._active_note_id = active_data.get("active_id")
        except Exception as e:
            logger.error(f"Error loading notes: {e}")
            self._notes = []

    def _save_notes(self):
        """Save notes to storage."""
        notes_file = self._storage_dir / "notes.json"
        with open(notes_file, "w", encoding="utf-8") as f:
            json.dump([n.to_dict() for n in self._notes], f, indent=2, ensure_ascii=False)

        # Save active note
        active_file = self._storage_dir / "active.json"
        with open(active_file, "w") as f:
            json.dump({"active_id": self._active_note_id}, f)

    def _create_default_notes(self):
        """Create default author notes."""
        defaults = [
            {
                "name": "Writing Style",
                "content": "Write in a descriptive, immersive style with vivid details. Show, don't tell.",
                "priority": 70,
            },
            {
                "name": "Character Voice",
                "content": "Maintain consistent character voice and personality throughout responses.",
                "priority": 80,
            },
            {
                "name": "Pacing",
                "content": "Keep the story moving forward. Avoid getting stuck in repetitive dialogue loops.",
                "priority": 60,
            },
        ]

        self._notes = [AuthorNote(**d) for d in defaults]
        self._save_notes()

    # ========================================================================
    # Note Management
    # ========================================================================

    def _add_note(
        self,
        name: str,
        content: str,
        priority: int = 50,
        position: str = "before_chat",
    ) -> AuthorNote:
        """Add a new author note."""
        note = AuthorNote(
            name=name,
            content=content,
            priority=priority,
            position=position,
        )
        self._notes.append(note)
        # Sort by priority (highest first)
        self._notes.sort(key=lambda n: n.priority, reverse=True)
        return note

    def _find_note(self, identifier: str) -> Optional[AuthorNote]:
        """Find a note by ID or name."""
        for note in self._notes:
            if note.id == identifier or note.name.lower() == identifier.lower():
                return note
        return None

    def _remove_note(self, identifier: str) -> bool:
        """Remove a note by ID or name."""
        note = self._find_note(identifier)
        if note:
            self._notes.remove(note)
            if self._active_note_id == note.id:
                self._active_note_id = None
            return True
        return False

    def _set_active(self, identifier: str) -> bool:
        """Set the active note by ID or name."""
        note = self._find_note(identifier)
        if note and note.enabled:
            self._active_note_id = note.id
            return True
        return False

    def _set_enabled(self, identifier: str, enabled: bool) -> bool:
        """Set enabled state of a note."""
        note = self._find_note(identifier)
        if note:
            note.enabled = enabled
            return True
        return False

    def get_active_note(self) -> Optional[AuthorNote]:
        """Get the currently active note."""
        if self._active_note_id:
            return self._find_note(self._active_note_id)
        return None

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all author notes."""
        try:
            if not self._notes:
                return {
                    "output": "No author notes found.",
                    "data": {"notes": []}
                }

            lines = ["\nAuthor Notes:"]
            for note in self._notes:
                status = "✓" if note.enabled else "✗"
                active = " [ACTIVE]" if note.id == self._active_note_id else ""
                lines.append(f"  {status} {note.name}{active}")
                lines.append(f"      ID: {note.id}")
                lines.append(f"      Priority: {note.priority} | Position: {note.position}")
                content_preview = note.content[:60] + "..." if len(note.content) > 60 else note.content
                lines.append(f"      Content: {content_preview}")
                lines.append("")

            lines.append(f"Total: {len(self._notes)} notes")

            return {
                "output": "\n".join(lines),
                "data": {"notes": [n.to_dict() for n in self._notes]}
            }
        except Exception as e:
            logger.error(f"Error listing notes: {e}")
            return {"error": str(e)}

    async def _cmd_create(self, name: str, args: List[str]) -> Dict[str, Any]:
        """Create a new author note."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Create author note")
            parser.add_argument("--content", required=True, help="Note content")
            parser.add_argument("--priority", type=int, default=50, help="Priority (0-100)")
            parser.add_argument("--position", choices=["before_chat", "after_chat", "inline"],
                               default="before_chat", help="Insertion position")

            parsed = parser.parse_args(args)

            note = self._add_note(
                name=name,
                content=parsed.content,
                priority=parsed.priority,
                position=parsed.position,
            )
            self._save_notes()

            return {
                "output": f"Author note '{name}' created (ID: {note.id})",
                "data": {"note": note.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_get(self, identifier: str) -> Dict[str, Any]:
        """Get author note details."""
        try:
            note = self._find_note(identifier)
            if not note:
                return {"error": f"Author note '{identifier}' not found"}

            active = " [ACTIVE]" if note.id == self._active_note_id else ""
            lines = [f"\nAuthor Note: {note.name}{active}"]
            lines.append(f"ID: {note.id}")
            lines.append(f"Priority: {note.priority}")
            lines.append(f"Position: {note.position}")
            lines.append(f"Enabled: {note.enabled}")
            lines.append(f"Created: {note.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"\nContent:")
            lines.append(note.content)

            return {
                "output": "\n".join(lines),
                "data": {"note": note.to_dict()}
            }
        except Exception as e:
            logger.error(f"Error getting note: {e}")
            return {"error": str(e)}

    async def _cmd_set(self, identifier: str) -> Dict[str, Any]:
        """Set active author note."""
        try:
            note = self._find_note(identifier)
            if not note:
                return {"error": f"Author note '{identifier}' not found"}

            if not note.enabled:
                return {"error": f"Author note '{note.name}' is disabled"}

            self._set_active(identifier)
            self._save_notes()

            return {
                "output": f"Active author note set to '{note.name}'",
                "data": {"active_id": self._active_note_id, "note": note.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_update(self, identifier: str, args: List[str]) -> Dict[str, Any]:
        """Update an author note."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Update author note")
            parser.add_argument("--content", help="New content")
            parser.add_argument("--priority", type=int, help="New priority (0-100)")
            parser.add_argument("--position", choices=["before_chat", "after_chat", "inline"],
                               help="New position")

            parsed = parser.parse_args(args)

            note = self._find_note(identifier)
            if not note:
                return {"error": f"Author note '{identifier}' not found"}

            if parsed.content:
                note.content = parsed.content
            if parsed.priority is not None:
                note.priority = parsed.priority
            if parsed.position:
                note.position = parsed.position

            # Re-sort by priority
            self._notes.sort(key=lambda n: n.priority, reverse=True)
            self._save_notes()

            return {
                "output": f"Author note '{note.name}' updated",
                "data": {"note": note.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_delete(self, identifier: str) -> Dict[str, Any]:
        """Delete an author note."""
        try:
            note = self._find_note(identifier)
            if not note:
                return {"error": f"Author note '{identifier}' not found"}

            name = note.name
            success = self._remove_note(identifier)
            if success:
                self._save_notes()
                return {"output": f"Author note '{name}' deleted"}
            return {"error": f"Failed to delete author note '{identifier}'"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_enable(self, identifier: str) -> Dict[str, Any]:
        """Enable an author note."""
        try:
            note = self._find_note(identifier)
            if not note:
                return {"error": f"Author note '{identifier}' not found"}

            self._set_enabled(identifier, True)
            self._save_notes()
            return {"output": f"Author note '{note.name}' enabled"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_disable(self, identifier: str) -> Dict[str, Any]:
        """Disable an author note."""
        try:
            note = self._find_note(identifier)
            if not note:
                return {"error": f"Author note '{identifier}' not found"}

            self._set_enabled(identifier, False)
            if self._active_note_id == note.id:
                self._active_note_id = None
            self._save_notes()
            return {"output": f"Author note '{note.name}' disabled"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export author notes to file."""
        try:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            data = {
                "notes": [n.to_dict() for n in self._notes],
                "active_id": self._active_note_id,
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return {
                "output": f"Author notes exported to {path}",
                "data": {"filepath": str(path)}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import author notes from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self._storage_dir / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            imported = 0
            if "notes" in data:
                for note_data in data["notes"]:
                    note = AuthorNote.from_dict(note_data)
                    self._notes.append(note)
                    imported += 1

                if "active_id" in data:
                    self._active_note_id = data["active_id"]

            # Sort by priority
            self._notes.sort(key=lambda n: n.priority, reverse=True)
            self._save_notes()

            return {
                "output": f"Imported {imported} author notes from {path}",
                "data": {"imported": imported}
            }
        except Exception as e:
            logger.error(f"Error importing notes: {e}")
            return {"error": str(e)}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Author Notes CLI

Commands:
    list                    List all author notes
    create <name>           Create a new author note
    get <name>              Get author note details
    set <name>              Set active author note
    update <name>           Update an author note
    delete <name>           Delete an author note
    enable <name>           Enable an author note
    disable <name>          Disable an author note
    export <file>           Export notes to file
    import <file>           Import notes from file

Create Options:
    --content <text>        Note content (required)
    --priority <int>        Priority (0-100, default: 50)
    --position <pos>        Position: before_chat, after_chat, inline

Update Options:
    --content <text>        New content
    --priority <int>        New priority (0-100)
    --position <pos>        New position: before_chat, after_chat, inline

Examples:
    uv run backend/manage.py author_notes list
    uv run backend/manage.py author_notes create "Writing Style" --content "Be descriptive" --priority 80
    uv run backend/manage.py author_notes set "Writing Style"
    uv run backend/manage.py author_notes export backup.json
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/author-notes", tags=["author-notes"])

        @router.get("/")
        async def list_notes():
            """List all author notes."""
            return {"notes": [n.to_dict() for n in self._notes]}

        @router.post("/")
        async def create_note(request: dict):
            """Create a new author note."""
            name = request.get("name")
            content = request.get("content")
            if not name or not content:
                return {"error": "name and content required"}, 400
            note = self._add_note(
                name,
                content,
                priority=request.get("priority", 50),
                position=request.get("position", "before_chat"),
            )
            self._save_notes()
            return {"note": note.to_dict()}

        @router.get("/active")
        async def get_active():
            """Get the active author note."""
            if self._active_note_id:
                note = self._find_note(self._active_note_id)
                if note:
                    return {"note": note.to_dict()}
            return {"note": None}

        @router.put("/active")
        async def set_active(request: dict):
            """Set the active author note."""
            note_id = request.get("id") or request.get("name")
            if not note_id:
                return {"error": "id or name required"}, 400
            success = self._set_active(note_id)
            if success:
                self._save_notes()
                return {"success": True, "active_id": self._active_note_id}
            return {"error": "Note not found"}, 404

        @router.delete("/{note_id}")
        async def delete_note(note_id: str):
            """Delete an author note."""
            success = self._remove_note(note_id)
            if success:
                self._save_notes()
                return {"success": True}
            return {"error": "Note not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(AuthorNotesExtension)
