"""Quick Replies Extension

Quick response buttons for common phrases.
Uses the quick_replies library for core functionality.

CLI Usage:
    uv run backend/manage.py quick_replies list
    uv run backend/manage.py quick_replies add "Hello!" --label Greeting
    uv run backend/manage.py quick_replies remove <id>
    uv run backend/manage.py quick_replies trigger <id>
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class QuickReply:
    """A single quick reply button."""

    def __init__(
        self,
        text: str,
        label: Optional[str] = None,
        icon: Optional[str] = None,
        color: Optional[str] = None,
        enabled: bool = True,
    ):
        self.id = str(uuid.uuid4())
        self.text = text
        self.label = label or text[:30]
        self.icon = icon or "zap"
        self.color = color or "default"
        self.enabled = enabled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "icon": self.icon,
            "color": self.color,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuickReply":
        reply = cls(
            text=data["text"],
            label=data.get("label"),
            icon=data.get("icon"),
            color=data.get("color"),
            enabled=data.get("enabled", True),
        )
        reply.id = data["id"]
        return reply


class QuickRepliesExtension(LogicExtension):
    """Quick Replies extension - CLI wrapper for quick reply management.

    This extension provides CLI access to the quick replies library
    for managing quick response buttons.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_file = None
        self._replies: List[QuickReply] = []

    async def initialize(self, context):
        """Initialize the quick replies extension."""
        await super().initialize(context)

        # Set up storage file
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            storage_dir = get_inputs_root_dir() / "quick_replies"
        except (ImportError, AttributeError):
            storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "quick_replies"
            )

        storage_dir.mkdir(parents=True, exist_ok=True)
        self._storage_file = storage_dir / "replies.json"

        # Load existing replies
        self._load_replies()

        logger.info("Quick Replies extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_replies()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all quick replies
            add <text>             - Add a new quick reply
            remove <id>            - Remove a quick reply
            trigger <id>           - Trigger (use) a quick reply
            enable <id>            - Enable a quick reply
            disable <id>           - Disable a quick reply
            export <file>          - Export quick replies to file
            import <file>          - Import quick replies from file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("quick_replies", "quick", "replies", "qr") or args:
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
        elif subcommand in ("add", "create", "new"):
            if not args:
                yield {"error": "Please provide the reply text"}
                return
            yield await self._cmd_add(args[0], args[1:])
        elif subcommand in ("remove", "delete", "rm"):
            if not args:
                yield {"error": "Please provide a reply ID"}
                return
            yield await self._cmd_remove(args[0])
        elif subcommand in ("trigger", "use", "send"):
            if not args:
                yield {"error": "Please provide a reply ID"}
                return
            yield await self._cmd_trigger(args[0])
        elif subcommand == "enable":
            if not args:
                yield {"error": "Please provide a reply ID"}
                return
            yield await self._cmd_enable(args[0])
        elif subcommand == "disable":
            if not args:
                yield {"error": "Please provide a reply ID"}
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
        if operation == "list_replies":
            yield {"replies": [r.to_dict() for r in self._replies]}
        elif operation == "add_reply":
            text = input_data.get("text")
            if not text:
                yield {"error": "text required"}
                return
            reply = self._add_reply(
                text,
                label=input_data.get("label"),
                icon=input_data.get("icon"),
                color=input_data.get("color"),
            )
            self._save_replies()
            yield {"reply": reply.to_dict()}
        elif operation == "remove_reply":
            reply_id = input_data.get("id")
            if not reply_id:
                yield {"error": "id required"}
                return
            success = self._remove_reply(reply_id)
            if success:
                self._save_replies()
                yield {"success": True}
            else:
                yield {"error": "Reply not found"}
        elif operation == "trigger_reply":
            reply_id = input_data.get("id")
            if not reply_id:
                yield {"error": "id required"}
                return
            result = self._trigger_reply(reply_id)
            if result:
                yield {"text": result}
            else:
                yield {"error": "Reply not found or disabled"}
        elif operation == "enable_reply":
            reply_id = input_data.get("id")
            if not reply_id:
                yield {"error": "id required"}
                return
            success = self._set_enabled(reply_id, True)
            if success:
                self._save_replies()
                yield {"success": True}
            else:
                yield {"error": "Reply not found"}
        elif operation == "disable_reply":
            reply_id = input_data.get("id")
            if not reply_id:
                yield {"error": "id required"}
                return
            success = self._set_enabled(reply_id, False)
            if success:
                self._save_replies()
                yield {"success": True}
            else:
                yield {"error": "Reply not found"}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_replies(self):
        """Load replies from storage."""
        if not self._storage_file.exists():
            # Create default replies
            self._create_default_replies()
            return

        try:
            with open(self._storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._replies = [QuickReply.from_dict(r) for r in data]
        except Exception as e:
            logger.error(f"Error loading replies: {e}")
            self._replies = []

    def _save_replies(self):
        """Save replies to storage."""
        if self._storage_file:
            with open(self._storage_file, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._replies], f, indent=2, ensure_ascii=False)

    def _create_default_replies(self):
        """Create default quick replies."""
        defaults = [
            {"text": "Continue, please.", "label": "Continue", "icon": "play"},
            {"text": "Can you elaborate on that?", "label": "Elaborate", "icon": "help"},
            {"text": "That's interesting, tell me more.", "label": "More", "icon": "plus"},
            {"text": "Let's change the topic.", "label": "Change Topic", "icon": "refresh"},
            {"text": "Can you summarize what we've discussed?", "label": "Summarize", "icon": "list"},
        ]

        self._replies = [QuickReply(**d) for d in defaults]
        self._save_replies()

    # ========================================================================
    # Quick Reply Management
    # ========================================================================

    def _add_reply(
        self,
        text: str,
        label: Optional[str] = None,
        icon: Optional[str] = None,
        color: Optional[str] = None,
    ) -> QuickReply:
        """Add a new quick reply."""
        reply = QuickReply(
            text=text,
            label=label,
            icon=icon,
            color=color,
        )
        self._replies.append(reply)
        return reply

    def _remove_reply(self, reply_id: str) -> bool:
        """Remove a quick reply by ID."""
        for i, reply in enumerate(self._replies):
            if reply.id == reply_id:
                self._replies.pop(i)
                return True
        return False

    def _trigger_reply(self, reply_id: str) -> Optional[str]:
        """Trigger a quick reply and return its text."""
        for reply in self._replies:
            if reply.id == reply_id and reply.enabled:
                return reply.text
        return None

    def _set_enabled(self, reply_id: str, enabled: bool) -> bool:
        """Set enabled state of a quick reply."""
        for reply in self._replies:
            if reply.id == reply_id:
                reply.enabled = enabled
                return True
        return False

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all quick replies."""
        try:
            if not self._replies:
                return {
                    "output": "No quick replies found.",
                    "data": {"replies": []}
                }

            lines = ["\nQuick Replies:"]
            for reply in self._replies:
                status = "✓" if reply.enabled else "✗"
                color_display = f" [{reply.color}]" if reply.color != "default" else ""
                lines.append(f"  {status} {reply.id[:8]} - {reply.label}{color_display}")
                lines.append(f"      Text: {reply.text[:60]}{'...' if len(reply.text) > 60 else ''}")
                lines.append(f"      Icon: {reply.icon}")

            lines.append(f"\nTotal: {len(self._replies)} replies")

            return {
                "output": "\n".join(lines),
                "data": {"replies": [r.to_dict() for r in self._replies]}
            }
        except Exception as e:
            logger.error(f"Error listing replies: {e}")
            return {"error": str(e)}

    async def _cmd_add(self, text: str, args: List[str]) -> Dict[str, Any]:
        """Add a new quick reply."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Add quick reply")
            parser.add_argument("--label", default=None, help="Button label")
            parser.add_argument("--icon", default="zap", help="Icon name")
            parser.add_argument("--color", choices=["default", "primary", "success", "warning", "danger"],
                               default="default", help="Button color")

            parsed = parser.parse_args(args)

            reply = self._add_reply(
                text=text,
                label=parsed.label,
                icon=parsed.icon,
                color=parsed.color,
            )
            self._save_replies()

            return {
                "output": f"Quick reply '{reply.label}' added (ID: {reply.id[:8]})",
                "data": {"reply": reply.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_remove(self, reply_id: str) -> Dict[str, Any]:
        """Remove a quick reply."""
        try:
            # Find by partial ID match
            for reply in self._replies:
                if reply.id.startswith(reply_id):
                    label = reply.label
                    success = self._remove_reply(reply.id)
                    if success:
                        self._save_replies()
                        return {"output": f"Quick reply '{label}' removed"}

            return {"error": f"Quick reply with ID '{reply_id}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_trigger(self, reply_id: str) -> Dict[str, Any]:
        """Trigger a quick reply."""
        try:
            for reply in self._replies:
                if reply.id.startswith(reply_id):
                    text = self._trigger_reply(reply.id)
                    if text:
                        return {
                            "output": f"Triggered: {text}",
                            "data": {"text": text, "reply": reply.to_dict()}
                        }
                    else:
                        return {"error": f"Quick reply '{reply.label}' is disabled"}

            return {"error": f"Quick reply with ID '{reply_id}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_enable(self, reply_id: str) -> Dict[str, Any]:
        """Enable a quick reply."""
        try:
            for reply in self._replies:
                if reply.id.startswith(reply_id):
                    self._set_enabled(reply.id, True)
                    self._save_replies()
                    return {"output": f"Quick reply '{reply.label}' enabled"}

            return {"error": f"Quick reply with ID '{reply_id}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_disable(self, reply_id: str) -> Dict[str, Any]:
        """Disable a quick reply."""
        try:
            for reply in self._replies:
                if reply.id.startswith(reply_id):
                    self._set_enabled(reply.id, False)
                    self._save_replies()
                    return {"output": f"Quick reply '{reply.label}' disabled"}

            return {"error": f"Quick reply with ID '{reply_id}' not found"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export quick replies to file."""
        try:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            with open(path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._replies], f, indent=2, ensure_ascii=False)

            return {
                "output": f"Quick replies exported to {path}",
                "data": {"filepath": str(path)}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import quick replies from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                # Fallback: go up to project root (backend/ -> project root)
                path = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "data" / "inputs" / "quick_replies" / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            imported = 0
            for reply_data in data:
                reply = QuickReply.from_dict(reply_data)
                self._replies.append(reply)
                imported += 1

            self._save_replies()

            return {
                "output": f"Imported {imported} quick replies from {path}",
                "data": {"imported": imported}
            }
        except Exception as e:
            logger.error(f"Error importing replies: {e}")
            return {"error": str(e)}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Quick Replies CLI

Commands:
    list                    List all quick replies
    add <text>             Add a new quick reply
    remove <id>            Remove a quick reply
    trigger <id>           Trigger (use) a quick reply
    enable <id>            Enable a quick reply
    disable <id>           Disable a quick reply
    export <file>          Export quick replies to file
    import <file>          Import quick replies from file

Add Options:
    --label <text>         Button label (default: first 30 chars of text)
    --icon <name>          Icon name (default: zap)
    --color <color>        Button color: default, primary, success, warning, danger

Examples:
    uv run backend/manage.py quick_replies list
    uv run backend/manage.py quick_replies add "Continue, please." --label Continue --icon play
    uv run backend/manage.py quick_replies trigger <id>
    uv run backend/manage.py quick_replies export backup.json
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/quick-replies", tags=["quick-replies"])

        @router.get("/")
        async def list_replies():
            """List all quick replies."""
            return {"replies": [r.to_dict() for r in self._replies]}

        @router.post("/")
        async def add_reply(request: dict):
            """Add a new quick reply."""
            text = request.get("text")
            if not text:
                return {"error": "text required"}, 400
            reply = self._add_reply(
                text,
                label=request.get("label"),
                icon=request.get("icon"),
                color=request.get("color"),
            )
            self._save_replies()
            return {"reply": reply.to_dict()}

        @router.delete("/{reply_id}")
        async def remove_reply(reply_id: str):
            """Remove a quick reply."""
            success = self._remove_reply(reply_id)
            if success:
                self._save_replies()
                return {"success": True}
            return {"error": "Reply not found"}, 404

        @router.post("/{reply_id}/trigger")
        async def trigger_reply(reply_id: str):
            """Trigger a quick reply."""
            text = self._trigger_reply(reply_id)
            if text:
                return {"text": text}
            return {"error": "Reply not found or disabled"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(QuickRepliesExtension)
