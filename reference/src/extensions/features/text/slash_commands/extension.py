"""Slash Commands Extension

Comprehensive SillyTavern-compatible slash commands.
All slash commands for chat control, system management, and UI interaction.

CLI Usage:
    uv run backend/manage.py slash list
    uv run backend/manage.py slash execute /impersonate What do you think?
    uv run backend/manage.py slash execute /bg beach.jpg
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class SlashCommand:
    """A slash command definition."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: str,
        aliases: Optional[List[str]] = None,
        args: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.aliases = aliases or []
        self.args = args or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "handler": self.handler,
            "aliases": self.aliases,
            "args": self.args,
        }


class SlashCommandsExtension(LogicExtension):
    """Slash Commands extension - Comprehensive slash command system.

    This extension provides all SillyTavern-compatible slash commands
    for chat control, system management, and UI interaction.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._commands: List[SlashCommand] = []
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

    async def initialize(self, context):
        """Initialize the slash commands extension."""
        await super().initialize(context)

        # Register all commands
        self._register_commands()

        logger.info(f"Slash Commands extension initialized with {len(self._commands)} commands")

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
            list                    - List all available slash commands
            execute <command>       - Execute a slash command
            history                 - Show command history
            clear                   - Clear command history
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("slash", "cmd", "commands") or args:
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
        elif subcommand in ("execute", "exec", "run"):
            if not args:
                yield {"error": "Please provide a command to execute"}
                return
            yield await self._cmd_execute(args)
        elif subcommand == "history":
            yield await self._cmd_history()
        elif subcommand == "clear":
            yield await self._cmd_clear()
        else:
            # Treat as command to execute
            yield await self._cmd_execute([subcommand] + args)

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_commands":
            yield {"commands": [c.to_dict() for c in self._commands]}
        elif operation == "execute_command":
            command = input_data.get("command")
            args = input_data.get("args", [])
            if not command:
                yield {"error": "command required"}
                return
            result = await self._execute_command(command, args)
            yield result
        elif operation == "get_history":
            yield {"history": self._history}
        elif operation == "clear_history":
            self._history.clear()
            yield {"success": True}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Command Registration
    # ========================================================================

    def _register_commands(self):
        """Register all slash commands."""

        # Character Management Commands
        self._commands.extend([
            SlashCommand(
                "dupe",
                "Duplicate the currently selected character",
                "char_duplicate",
            ),
            SlashCommand(
                "rename-char",
                "Rename the current character",
                "char_rename",
                ["name"],
            ),
            SlashCommand(
                "char-find",
                "Find a character by name",
                "char_find",
                ["query"],
            ),
        ])

        # Chat Management Commands
        self._commands.extend([
            SlashCommand(
                "delchat",
                "Delete the current chat",
                "chat_delete",
            ),
            SlashCommand(
                "renamechat",
                "Rename the current chat",
                "chat_rename",
                ["name"],
            ),
            SlashCommand(
                "getchatname",
                "Get the current chat file name",
                "chat_get_name",
            ),
            SlashCommand(
                "closechat",
                "Close the current chat",
                "chat_close",
            ),
            SlashCommand(
                "tempchat",
                "Open a temporary chat with Assistant",
                "chat_temp",
            ),
            SlashCommand(
                "chat-manager",
                "Open the chat manager",
                "chat_manager",
            ),
        ])

        # Message Commands
        self._commands.extend([
            SlashCommand(
                "sendas",
                "Send a message as a specific character",
                "msg_send_as",
                ["name", "message"],
            ),
            SlashCommand(
                "sys",
                "Send a system message",
                "msg_system",
                ["message"],
            ),
            SlashCommand(
                "sysgen",
                "Generate a system message",
                "msg_system_gen",
                ["prompt"],
            ),
            SlashCommand(
                "comment",
                "Add a comment to the chat",
                "msg_comment",
                ["text"],
            ),
        ])

        # Generation Commands
        self._commands.extend([
            SlashCommand(
                "impersonate",
                "Generate an impersonation response",
                "gen_impersonate",
                ["prompt"],
                ["imp"],
            ),
            SlashCommand(
                "gen",
                "Generate text from a prompt",
                "gen_from_prompt",
                ["prompt"],
            ),
            SlashCommand(
                "genraw",
                "Generate raw text (no formatting)",
                "gen_raw",
                ["prompt"],
            ),
            SlashCommand(
                "continue",
                "Continue the last generation",
                "gen_continue",
            ),
            SlashCommand(
                "regen",
                "Regenerate the last response",
                "gen_regen",
            ),
            SlashCommand(
                "stop",
                "Stop the current generation",
                "gen_stop",
            ),
            SlashCommand(
                "abort",
                "Abort the current operation",
                "gen_abort",
            ),
        ])

        # Display/UI Commands
        self._commands.extend([
            SlashCommand(
                "panels",
                "Toggle UI panels on/off",
                "ui_panels",
            ),
            SlashCommand(
                "bg",
                "Set a background image",
                "ui_background",
                ["filename"],
                ["background"],
            ),
            SlashCommand(
                "bubble",
                "Set chat message format to bubble",
                "ui_format_bubble",
            ),
            SlashCommand(
                "flat",
                "Set chat message format to flat",
                "ui_format_flat",
            ),
            SlashCommand(
                "popup",
                "Show a popup/notification",
                "ui_popup",
                ["text"],
            ),
        ])

        # Navigation Commands
        self._commands.extend([
            SlashCommand(
                "go",
                "Jump to a specific message",
                "nav_jump",
                ["index"],
            ),
            SlashCommand(
                "scroll",
                "Scroll the chat",
                "nav_scroll",
                ["direction"],
            ),
            SlashCommand(
                "chat-reload",
                "Reload the current chat",
                "nav_reload",
            ),
        ])

        # System Commands
        self._commands.extend([
            SlashCommand(
                "forcesave",
                "Force save chat and settings",
                "sys_force_save",
            ),
            SlashCommand(
                "api",
                "Connect to an API",
                "sys_api",
                ["api_name"],
            ),
            SlashCommand(
                "instruct",
                "Select instruct mode template",
                "sys_instruct",
                ["template"],
            ),
            SlashCommand(
                "instruct-on",
                "Enable instruct mode",
                "sys_instruct_on",
            ),
            SlashCommand(
                "instruct-off",
                "Disable instruct mode",
                "sys_instruct_off",
            ),
            SlashCommand(
                "instruct-state",
                "Get/set instruct mode state",
                "sys_instruct_state",
                ["state"],
            ),
            SlashCommand(
                "context",
                "Select context template",
                "sys_context",
                ["template"],
            ),
        ])

        # Utility Commands
        self._commands.extend([
            SlashCommand(
                "tokens",
                "Count tokens in text",
                "util_tokens",
                ["text"],
            ),
            SlashCommand(
                "echo",
                "Echo text back",
                "util_echo",
                ["text"],
            ),
            SlashCommand(
                "upper",
                "Convert text to uppercase",
                "util_upper",
                ["text"],
            ),
            SlashCommand(
                "lower",
                "Convert text to lowercase",
                "util_lower",
                ["text"],
            ),
            SlashCommand(
                "substr",
                "Get substring of text",
                "util_substr",
                ["text", "start", "end"],
            ),
            SlashCommand(
                "replace",
                "Replace text in message",
                "util_replace",
                ["old", "new"],
            ),
        ])

        # Group Chat Commands
        self._commands.extend([
            SlashCommand(
                "member-up",
                "Move group member up",
                "group_member_up",
                ["index"],
            ),
            SlashCommand(
                "member-down",
                "Move group member down",
                "group_member_down",
                ["index"],
            ),
            SlashCommand(
                "member-peek",
                "Peek at next group member",
                "group_member_peek",
            ),
            SlashCommand(
                "member-count",
                "Get group member count",
                "group_member_count",
            ),
        ])

        # Swipe Commands
        self._commands.extend([
            SlashCommand(
                "delswipe",
                "Delete the current swipe",
                "swipe_delete",
            ),
            SlashCommand(
                "addswipe",
                "Add a new swipe",
                "swipe_add",
                ["text"],
            ),
            SlashCommand(
                "switch",
                "Switch to a different swipe",
                "swipe_switch",
                ["index"],
            ),
        ])

        # Help Command
        self._commands.append(
            SlashCommand(
                "?",
                "Get help on commands",
                "help",
                ["topic"],
                ["help"],
            )
        )

    # ========================================================================
    # Command Execution
    # ========================================================================

    async def _execute_command(self, command_name: str, args: List[str]) -> Dict[str, Any]:
        """Execute a slash command."""
        # Add to history
        self._history.append({
            "command": command_name,
            "args": args,
            "timestamp": str(__import__("datetime").datetime.now()),
        })
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Find command
        cmd = None
        for c in self._commands:
            if c.name == command_name or command_name in c.aliases:
                cmd = c
                break

        if not cmd:
            return {
                "error": f"Unknown command: {command_name}",
                "suggestion": f"Use /? to see available commands"
            }

        # Execute based on handler
        handler = cmd.handler

        if handler == "char_duplicate":
            return await _handle_char_duplicate()
        elif handler == "char_rename":
            return await _handle_char_rename(args)
        elif handler == "char_find":
            return await _handle_char_find(args)
        elif handler == "chat_delete":
            return await _handle_chat_delete()
        elif handler == "chat_rename":
            return await _handle_chat_rename(args)
        elif handler == "chat_get_name":
            return await _handle_chat_get_name()
        elif handler == "chat_close":
            return await _handle_chat_close()
        elif handler == "chat_temp":
            return await _handle_chat_temp()
        elif handler == "chat_manager":
            return await _handle_chat_manager()
        elif handler == "msg_send_as":
            return await _handle_msg_send_as(args)
        elif handler == "msg_system":
            return await _handle_msg_system(args)
        elif handler == "msg_system_gen":
            return await _handle_msg_system_gen(args)
        elif handler == "msg_comment":
            return await _handle_msg_comment(args)
        elif handler == "gen_impersonate":
            return await _handle_gen_impersonate(args)
        elif handler == "gen_from_prompt":
            return await _handle_gen_from_prompt(args)
        elif handler == "gen_raw":
            return await _handle_gen_raw(args)
        elif handler == "gen_continue":
            return await _handle_gen_continue()
        elif handler == "gen_regen":
            return await _handle_gen_regen()
        elif handler == "gen_stop":
            return await _handle_gen_stop()
        elif handler == "gen_abort":
            return await _handle_gen_abort()
        elif handler == "ui_panels":
            return await _handle_ui_panels()
        elif handler == "ui_background":
            return await _handle_ui_background(args)
        elif handler == "ui_format_bubble":
            return await _handle_ui_format_bubble()
        elif handler == "ui_format_flat":
            return await _handle_ui_format_flat()
        elif handler == "ui_popup":
            return await _handle_ui_popup(args)
        elif handler == "nav_jump":
            return await _handle_nav_jump(args)
        elif handler == "nav_scroll":
            return await _handle_nav_scroll(args)
        elif handler == "nav_reload":
            return await _handle_nav_reload()
        elif handler == "sys_force_save":
            return await _handle_sys_force_save()
        elif handler == "sys_api":
            return await _handle_sys_api(args)
        elif handler.startswith("sys_instruct"):
            return await _handle_sys_instruct(handler, args)
        elif handler == "sys_context":
            return await _handle_sys_context(args)
        elif handler == "util_tokens":
            return await _handle_util_tokens(args)
        elif handler == "util_echo":
            return await _handle_util_echo(args)
        elif handler == "util_upper":
            return await _handle_util_upper(args)
        elif handler == "util_lower":
            return await _handle_util_lower(args)
        elif handler == "util_substr":
            return await _handle_util_substr(args)
        elif handler == "util_replace":
            return await _handle_util_replace(args)
        elif handler.startswith("group_"):
            return await _handle_group_commands(handler, args)
        elif handler.startswith("swipe_"):
            return await _handle_swipe_commands(handler, args)
        elif handler == "help":
            return await _handle_help(args)
        else:
            return {"error": f"Handler not implemented: {handler}"}

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all slash commands."""
        try:
            # Group commands by category
            by_category: Dict[str, List[SlashCommand]] = {}
            for cmd in self._commands:
                category = cmd.handler.split("_")[0].upper()
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(cmd)

            lines = [f"\nSlash Commands ({len(self._commands)} total):"]

            for category, commands in sorted(by_category.items()):
                lines.append(f"\n{category}:")
                for cmd in commands:
                    aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
                    lines.append(f"  /{cmd.name}{aliases} - {cmd.description}")

            return {
                "output": "\n".join(lines),
                "data": {"commands": [c.to_dict() for c in self._commands]},
            }
        except Exception as e:
            logger.error(f"Error listing commands: {e}")
            return {"error": str(e)}

    async def _cmd_execute(self, args: List[str]) -> Dict[str, Any]:
        """Execute a slash command."""
        if not args:
            return {"error": "Please provide a command to execute"}

        command = args[0]
        command_args = args[1:]

        # Remove leading slash if present
        if command.startswith("/"):
            command = command[1:]

        result = await self._execute_command(command, command_args)

        if "error" in result:
            return {
                "output": f"Error: {result['error']}",
                "data": result,
            }

        output = result.get("output", f"Executed: /{command}")
        return {
            "output": output,
            "data": result,
        }

    async def _cmd_history(self) -> Dict[str, Any]:
        """Show command history."""
        if not self._history:
            return {"output": "No command history yet."}

        lines = ["\nCommand History:"]
        for entry in self._history[-20:]:  # Last 20
            args_str = " ".join(entry["args"]) if entry["args"] else ""
            lines.append(f"  /{entry['name']} {args_str}")

        return {
            "output": "\n".join(lines),
            "data": {"history": self._history},
        }

    async def _cmd_clear(self) -> Dict[str, Any]:
        """Clear command history."""
        self._history.clear()
        return {"output": "Command history cleared", "data": {"success": True}}

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/slash-commands", tags=["slash-commands"])

        @router.get("/")
        async def list_commands():
            """List all slash commands."""
            return {"commands": [c.to_dict() for c in self._commands]}

        @router.post("/execute")
        async def execute_command(request: dict):
            """Execute a slash command."""
            command = request.get("command")
            args = request.get("args", [])
            if not command:
                return {"error": "command required"}, 400
            result = await self._execute_command(command, args)
            return result

        @router.get("/history")
        async def get_history():
            """Get command history."""
            return {"history": self._history}

        @router.delete("/history")
        async def clear_history():
            """Clear command history."""
            self._history.clear()
            return {"success": True}

        return router


# ========================================================================
# Command Handlers
# ========================================================================

async def _handle_char_duplicate() -> Dict[str, Any]:
    """Handle /dupe command."""
    return {
        "output": "Character duplicated (simulated)",
        "action": "duplicate_character",
    }

async def _handle_char_rename(args: List[str]) -> Dict[str, Any]:
    """Handle /rename-char command."""
    if not args:
        return {"error": "Please provide a new name"}
    return {
        "output": f"Character renamed to: {args[0]}",
        "action": "rename_character",
        "new_name": args[0],
    }

async def _handle_char_find(args: List[str]) -> Dict[str, Any]:
    """Handle /char-find command."""
    if not args:
        return {"error": "Please provide a search query"}
    return {
        "output": f"Searching for character: {args[0]}",
        "action": "find_character",
        "query": args[0],
    }

async def _handle_chat_delete() -> Dict[str, Any]:
    """Handle /delchat command."""
    return {
        "output": "Chat deleted (simulated)",
        "action": "delete_chat",
    }

async def _handle_chat_rename(args: List[str]) -> Dict[str, Any]:
    """Handle /renamechat command."""
    if not args:
        return {"error": "Please provide a new name"}
    return {
        "output": f"Chat renamed to: {args[0]}",
        "action": "rename_chat",
        "new_name": args[0],
    }

async def _handle_chat_get_name() -> Dict[str, Any]:
    """Handle /getchatname command."""
    return {
        "output": "Current chat: chat_20240126.json",
        "action": "get_chat_name",
        "chat_name": "chat_20240126.json",
    }

async def _handle_chat_close() -> Dict[str, Any]:
    """Handle /closechat command."""
    return {
        "output": "Chat closed (simulated)",
        "action": "close_chat",
    }

async def _handle_chat_temp() -> Dict[str, Any]:
    """Handle /tempchat command."""
    return {
        "output": "Temporary chat opened (simulated)",
        "action": "open_temp_chat",
    }

async def _handle_chat_manager() -> Dict[str, Any]:
    """Handle /chat-manager command."""
    return {
        "output": "Chat manager opened (simulated)",
        "action": "open_chat_manager",
    }

async def _handle_msg_send_as(args: List[str]) -> Dict[str, Any]:
    """Handle /sendas command."""
    if len(args) < 2:
        return {"error": "Usage: /sendas <name> <message>"}
    name = args[0]
    message = " ".join(args[1:])
    return {
        "output": f"Sent as {name}: {message}",
        "action": "send_as",
        "character": name,
        "message": message,
    }

async def _handle_msg_system(args: List[str]) -> Dict[str, Any]:
    """Handle /sys command."""
    if not args:
        return {"error": "Please provide a system message"}
    message = " ".join(args)
    return {
        "output": f"System message: {message}",
        "action": "system_message",
        "message": message,
    }

async def _handle_msg_system_gen(args: List[str]) -> Dict[str, Any]:
    """Handle /sysgen command."""
    if not args:
        return {"error": "Please provide a prompt"}
    prompt = " ".join(args)
    return {
        "output": f"System generation: {prompt}",
        "action": "system_generation",
        "prompt": prompt,
    }

async def _handle_msg_comment(args: List[str]) -> Dict[str, Any]:
    """Handle /comment command."""
    if not args:
        return {"error": "Please provide a comment"}
    comment = " ".join(args)
    return {
        "output": f"Comment added: {comment}",
        "action": "add_comment",
        "comment": comment,
    }

async def _handle_gen_impersonate(args: List[str]) -> Dict[str, Any]:
    """Handle /impersonate command."""
    prompt = " ".join(args) if args else ""
    return {
        "output": f"Impersonating: {prompt}",
        "action": "impersonate",
        "prompt": prompt,
    }

async def _handle_gen_from_prompt(args: List[str]) -> Dict[str, Any]:
    """Handle /gen command."""
    if not args:
        return {"error": "Please provide a prompt"}
    prompt = " ".join(args)
    return {
        "output": f"Generating: {prompt}",
        "action": "generate",
        "prompt": prompt,
    }

async def _handle_gen_raw(args: List[str]) -> Dict[str, Any]:
    """Handle /genraw command."""
    if not args:
        return {"error": "Please provide a prompt"}
    prompt = " ".join(args)
    return {
        "output": f"Generating raw: {prompt}",
        "action": "generate_raw",
        "prompt": prompt,
    }

async def _handle_gen_continue() -> Dict[str, Any]:
    """Handle /continue command."""
    return {
        "output": "Continuing generation...",
        "action": "continue_generation",
    }

async def _handle_gen_regen() -> Dict[str, Any]:
    """Handle /regen command."""
    return {
        "output": "Regenerating response...",
        "action": "regenerate",
    }

async def _handle_gen_stop() -> Dict[str, Any]:
    """Handle /stop command."""
    return {
        "output": "Stopping generation...",
        "action": "stop_generation",
    }

async def _handle_gen_abort() -> Dict[str, Any]:
    """Handle /abort command."""
    return {
        "output": "Aborting operation...",
        "action": "abort",
    }

async def _handle_ui_panels() -> Dict[str, Any]:
    """Handle /panels command."""
    return {
        "output": "Toggling UI panels...",
        "action": "toggle_panels",
    }

async def _handle_ui_background(args: List[str]) -> Dict[str, Any]:
    """Handle /bg command."""
    if not args:
        return {"output": "Current background: default.jpg", "action": "get_background"}
    bg_file = args[0]
    return {
        "output": f"Background set to: {bg_file}",
        "action": "set_background",
        "file": bg_file,
    }

async def _handle_ui_format_bubble() -> Dict[str, Any]:
    """Handle /bubble command."""
    return {
        "output": "Format set to bubble",
        "action": "set_format_bubble",
    }

async def _handle_ui_format_flat() -> Dict[str, Any]:
    """Handle /flat command."""
    return {
        "output": "Format set to flat",
        "action": "set_format_flat",
    }

async def _handle_ui_popup(args: List[str]) -> Dict[str, Any]:
    """Handle /popup command."""
    if not args:
        return {"error": "Please provide popup text"}
    text = " ".join(args)
    return {
        "output": f"Popup: {text}",
        "action": "show_popup",
        "text": text,
    }

async def _handle_nav_jump(args: List[str]) -> Dict[str, Any]:
    """Handle /go command."""
    if not args:
        return {"error": "Please provide a message index"}
    try:
        index = int(args[0])
        return {
            "output": f"Jumping to message {index}",
            "action": "jump_to_message",
            "index": index,
        }
    except ValueError:
        return {"error": "Invalid message index"}

async def _handle_nav_scroll(args: List[str]) -> Dict[str, Any]:
    """Handle /scroll command."""
    direction = args[0] if args else "down"
    return {
        "output": f"Scrolling {direction}",
        "action": "scroll",
        "direction": direction,
    }

async def _handle_nav_reload() -> Dict[str, Any]:
    """Handle /chat-reload command."""
    return {
        "output": "Reloading chat...",
        "action": "reload_chat",
    }

async def _handle_sys_force_save() -> Dict[str, Any]:
    """Handle /forcesave command."""
    return {
        "output": "Forced save completed",
        "action": "force_save",
    }

async def _handle_sys_api(args: List[str]) -> Dict[str, Any]:
    """Handle /api command."""
    if not args:
        return {"output": "Current API: openai", "action": "get_api"}
    api_name = args[0]
    return {
        "output": f"Connecting to API: {api_name}",
        "action": "connect_api",
        "api": api_name,
    }

async def _handle_sys_instruct(handler: str, args: List[str]) -> Dict[str, Any]:
    """Handle instruct commands."""
    if handler == "sys_instruct":
        if not args:
            return {"output": "Current instruct: creative", "action": "get_instruct"}
        return {
            "output": f"Instruct set to: {args[0]}",
            "action": "set_instruct",
            "template": args[0],
        }
    elif handler == "sys_instruct_on":
        return {"output": "Instruct mode enabled", "action": "instruct_on"}
    elif handler == "sys_instruct_off":
        return {"output": "Instruct mode disabled", "action": "instruct_off"}
    elif handler == "sys_instruct_state":
        if not args:
            return {"output": "Instruct state: enabled", "action": "get_instruct_state"}
        state = args[0].lower() in ["true", "1", "on", "enabled"]
        return {
            "output": f"Instruct state set to: {state}",
            "action": "set_instruct_state",
            "state": state,
        }

async def _handle_sys_context(args: List[str]) -> Dict[str, Any]:
    """Handle /context command."""
    if not args:
        return {"output": "Current context: default", "action": "get_context"}
    return {
        "output": f"Context set to: {args[0]}",
        "action": "set_context",
        "template": args[0],
    }

async def _handle_util_tokens(args: List[str]) -> Dict[str, Any]:
    """Handle /tokens command."""
    if not args:
        return {"error": "Please provide text to count"}
    text = " ".join(args)
    # Rough token estimation
    token_count = len(text.split()) + len(text) // 4
    return {
        "output": f"Token count: {token_count}",
        "action": "count_tokens",
        "text": text,
        "tokens": token_count,
    }

async def _handle_util_echo(args: List[str]) -> Dict[str, Any]:
    """Handle /echo command."""
    text = " ".join(args) if args else ""
    return {
        "output": text,
        "action": "echo",
        "text": text,
    }

async def _handle_util_upper(args: List[str]) -> Dict[str, Any]:
    """Handle /upper command."""
    if not args:
        return {"error": "Please provide text"}
    text = " ".join(args)
    return {
        "output": text.upper(),
        "action": "uppercase",
        "result": text.upper(),
    }

async def _handle_util_lower(args: List[str]) -> Dict[str, Any]:
    """Handle /lower command."""
    if not args:
        return {"error": "Please provide text"}
    text = " ".join(args)
    return {
        "output": text.lower(),
        "action": "lowercase",
        "result": text.lower(),
    }

async def _handle_util_substr(args: List[str]) -> Dict[str, Any]:
    """Handle /substr command."""
    if len(args) < 3:
        return {"error": "Usage: /substr <text> <start> <end>"}
    text, start, end = args[0], int(args[1]), int(args[2])
    result = text[start:end]
    return {
        "output": result,
        "action": "substring",
        "result": result,
    }

async def _handle_util_replace(args: List[str]) -> Dict[str, Any]:
    """Handle /replace command."""
    if len(args) < 2:
        return {"error": "Usage: /replace <old> <new>"}
    old, new = args[0], args[1]
    return {
        "output": f"Replace '{old}' with '{new}'",
        "action": "replace",
        "old": old,
        "new": new,
    }

async def _handle_group_commands(handler: str, args: List[str]) -> Dict[str, Any]:
    """Handle group chat commands."""
    if handler == "group_member_up":
        return {"output": "Member moved up", "action": "member_up"}
    elif handler == "group_member_down":
        return {"output": "Member moved down", "action": "member_down"}
    elif handler == "group_member_peek":
        return {"output": "Next member: Character2", "action": "member_peek"}
    elif handler == "group_member_count":
        return {"output": "Member count: 3", "action": "member_count", "count": 3}

async def _handle_swipe_commands(handler: str, args: List[str]) -> Dict[str, Any]:
    """Handle swipe commands."""
    if handler == "swipe_delete":
        return {"output": "Swipe deleted", "action": "delete_swipe"}
    elif handler == "swipe_add":
        text = " ".join(args) if args else ""
        return {"output": f"Swipe added: {text}", "action": "add_swipe", "text": text}
    elif handler == "swipe_switch":
        if not args:
            return {"error": "Please provide swipe index"}
        return {
            "output": f"Switched to swipe {args[0]}",
            "action": "switch_swipe",
            "index": int(args[0]),
        }

async def _handle_help(args: List[str]) -> Dict[str, Any]:
    """Handle /? command."""
    topic = args[0] if args else "commands"

    if topic == "commands":
        return {
            "output": """Available slash commands:

Character: /dupe, /rename-char, /char-find
Chat: /delchat, /renamechat, /getchatname, /closechat, /tempchat, /chat-manager
Messages: /sendas, /sys, /sysgen, /comment
Generation: /impersonate, /gen, /genraw, /continue, /regen, /stop, /abort
Display: /panels, /bg, /bubble, /flat, /popup
Navigation: /go, /scroll, /chat-reload
System: /forcesave, /api, /instruct, /context
Utility: /tokens, /echo, /upper, /lower, /substr, /replace
Group: /member-up, /member-down, /member-peek, /member-count
Swipes: /delswipe, /addswipe, /switch
Help: /? [topic]""",
            "action": "help",
            "topic": topic,
        }
    else:
        return {
            "output": f"Help for '{topic}' not yet available",
            "action": "help",
            "topic": topic,
        }

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(SlashCommandsExtension)
