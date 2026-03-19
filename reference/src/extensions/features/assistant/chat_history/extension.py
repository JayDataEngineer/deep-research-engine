"""Chat History Extension

Chat history management with JSONL import/export.
Compatible with SillyTavern chat format.

CLI Usage:
    uv run backend/manage.py chat list
    uv run backend/manage.py chat show <chat_name>
    uv run backend/manage.py chat export <chat_name> [output_path]
    uv run backend/manage.py chat import <file_path>
    uv run backend/manage.py chat delete <chat_name>
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

from .storage import (
    ChatMessage,
    ChatMetadata,
    ChatHistoryStorage,
    get_chat_storage,
)

logger = logging.getLogger(__name__)


class ChatHistoryExtension(LogicExtension):
    """Chat history extension for SillyTavern-compatible chat management."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage: Optional[ChatHistoryStorage] = None
        self._router = None

    async def initialize(self, context):
        """Initialize the chat history extension."""
        await super().initialize(context)

        # Initialize storage
        self._storage = get_chat_storage()

        logger.info("Chat History extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        self._storage = None

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands or API requests.

        CLI Commands:
            list                    - List all chats
            show <chat_name>         - Show chat details
            export <chat_name>       - Export chat to file
            import <file_path>       - Import chat from file
            delete <chat_name>       - Delete a chat
            stats                   - Show storage statistics

        API Operations:
            list_chats              - Get all chats
            get_chat                - Get specific chat
            save_chat               - Save/update chat
            delete_chat             - Delete chat
            import_chat             - Import from file
            export_chat             - Export to file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command == "chat" or args:
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
        elif subcommand == "show" or subcommand == "get":
            if not args:
                yield {"error": "Please provide a chat name"}
                return
            yield await self._cmd_show(args[0])
        elif subcommand == "export":
            if not args:
                yield {"error": "Please provide a chat name"}
                return
            yield await self._cmd_export(args[0], args[1:])
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        elif subcommand == "delete":
            if not args:
                yield {"error": "Please provide a chat name"}
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
        if operation == "list_chats":
            chats = self._storage.list_chats()
            yield {"chats": chats}
        elif operation == "get_chat":
            name = input_data.get("name")
            if not name:
                yield {"error": "Chat name required"}
                return
            result = self._storage.load_chat(name)
            if result:
                messages, metadata = result
                yield {
                    "messages": [m.to_dict() for m in messages],
                    "metadata": metadata.to_dict()
                }
            else:
                yield {"error": "Chat not found"}
        elif operation == "save_chat":
            messages_data = input_data.get("messages", [])
            metadata_data = input_data.get("metadata", {})

            messages = [ChatMessage.from_dict(m) for m in messages_data]
            metadata = ChatMetadata(**metadata_data) if metadata_data else None

            filepath = self._storage.save_chat(messages, metadata)
            yield {"success": True, "filepath": str(filepath)}
        elif operation == "delete_chat":
            name = input_data.get("name")
            if not name:
                yield {"error": "Chat name required"}
                return
            success = self._storage.delete_chat(name)
            yield {"success": success}
        elif operation == "import_chat":
            file_path = input_data.get("file_path")
            if not file_path:
                yield {"error": "File path required"}
                return
            result = self._storage.import_chat(file_path)
            if result:
                messages, metadata = result
                yield {
                    "success": True,
                    "chat_name": metadata.chat_name,
                    "message_count": len(messages)
                }
            else:
                yield {"error": "Import failed"}
        elif operation == "export_chat":
            name = input_data.get("name")
            if not name:
                yield {"error": "Chat name required"}
                return
            output_path = input_data.get("output_path")
            result = self._storage.export_chat(name, output_path)
            if result:
                yield {"success": True, "filepath": str(result)}
            else:
                yield {"error": "Export failed"}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all chats."""
        chats = self._storage.list_chats()
        stats = self._storage.get_storage_stats()

        if not chats:
            return {
                "output": "No chats found.",
                "data": {"stats": stats}
            }

        lines = ["\nChats:"]
        for chat in chats[:20]:  # Show first 20
            from datetime import datetime
            create_date = datetime.fromtimestamp(chat.get("create_date", 0)).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  - {chat['chat_name']} ({chat['message_count']} messages, {create_date})")

        if len(chats) > 20:
            lines.append(f"  ... and {len(chats) - 20} more")

        lines.append(f"\nTotal: {len(chats)} chats")
        return {
            "output": "\n".join(lines),
            "data": {"chats": chats, "stats": stats}
        }

    async def _cmd_show(self, chat_name: str) -> Dict[str, Any]:
        """Show chat details."""
        result = self._storage.load_chat(chat_name)
        if not result:
            return {"error": f"Chat '{chat_name}' not found"}

        messages, metadata = result

        from datetime import datetime

        lines = [f"\nChat: {metadata.chat_name}"]
        if metadata.character_name:
            lines.append(f"Character: {metadata.character_name}")
        lines.append(f"Messages: {metadata.message_count}")
        lines.append(f"Created: {datetime.fromtimestamp(metadata.create_date).strftime('%Y-%m-%d %H:%M')}")

        if messages:
            lines.append(f"\nLast {min(5, len(messages))} messages:")
            for msg in messages[-5:]:
                role = "User" if msg.is_user else msg.name
                preview = msg.mes[:100] + "..." if len(msg.mes) > 100 else msg.mes
                lines.append(f"  [{role}]: {preview}")

        return {
            "output": "\n".join(lines),
            "data": {
                "metadata": metadata.to_dict(),
                "message_count": len(messages),
            }
        }

    async def _cmd_export(
        self,
        chat_name: str,
        args: List[str]
    ) -> Dict[str, Any]:
        """Export chat to file."""
        output_path = args[0] if args else None

        result = self._storage.export_chat(chat_name, output_path)
        if result:
            return {
                "output": f"Chat exported to: {result}",
                "data": {"filepath": str(result)}
            }
        else:
            return {"error": "Export failed"}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import chat from file."""
        result = self._storage.import_chat(file_path)
        if result:
            messages, metadata = result
            return {
                "output": f"Imported chat '{metadata.chat_name}' with {len(messages)} messages!",
                "data": {
                    "chat_name": metadata.chat_name,
                    "message_count": len(messages),
                }
            }
        else:
            return {"error": "Import failed"}

    async def _cmd_delete(self, chat_name: str) -> Dict[str, Any]:
        """Delete a chat."""
        success = self._storage.delete_chat(chat_name)
        if success:
            return {"output": f"Chat '{chat_name}' deleted successfully!"}
        else:
            return {"error": f"Chat '{chat_name}' not found"}

    async def _cmd_stats(self) -> Dict[str, Any]:
        """Show storage statistics."""
        stats = self._storage.get_storage_stats()
        chats = self._storage.list_chats()

        lines = ["\nChat Storage Statistics:"]
        lines.append(f"  Directory: {stats['directory']}")
        lines.append(f"  Total chats: {stats['chat_count']}")
        lines.append(f"  Total messages: {stats['total_messages']}")

        return {
            "output": "\n".join(lines),
            "data": stats
        }

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Chat History CLI

Commands:
    list                    List all chats
    show <chat_name>         Show chat details
    export <chat_name>       Export chat to file
    import <file_path>       Import chat from file
    delete <chat_name>       Delete a chat
    stats                   Show storage statistics

Export Options:
    [output_path]            Optional output file path

Examples:
    uv run backend/manage.py chat list
    uv run backend/manage.py chat show Alice
    uv run backend/manage.py chat export Alice alice_backup.jsonl
    uv run backend/manage.py chat import /path/to/chat.jsonl
    uv run backend/manage.py chat delete Alice
    uv run backend/manage.py chat stats
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def storage(self) -> ChatHistoryStorage:
        """Get the chat storage instance."""
        return self._storage

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        if self._router is None:
            from .routes import router
            self._router = router
        return self._router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(ChatHistoryExtension)
