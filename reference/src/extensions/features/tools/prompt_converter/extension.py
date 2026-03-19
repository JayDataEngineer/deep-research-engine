"""Prompt Converter Extension

Convert prompts between different formats.
Supports SillyTavern, Claude, ChatGPT, and generic formats.

CLI Usage:
    uv run backend/manage.py convert list
    uv run backend/manage.py convert convert <input_format> <output_format> "<text>"
    uv run backend/manage.py convert macros expand "<text>"
    uv run backend/manage.py convert macros list
"""

import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class PromptConverterExtension(LogicExtension):
    """Prompt converter extension - Convert between different prompt formats.

    Supports conversion between:
    - SillyTavern format (with {{macros}})
    - Claude format (with <human>/<assistant>)
    - ChatGPT format (with ### instruction)
    - Generic plain text
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)

    async def initialize(self, context):
        """Initialize the prompt converter extension."""
        await super().initialize(context)
        logger.info("Prompt converter extension initialized")

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
            list                    - List supported formats
            convert <from> <to> <text> - Convert prompt format
            macros expand <text>     - Expand SillyTavern macros
            macros list             - List available macros
            templates list          - List template formats
            help                    - Show help
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("convert", "prompt", "converter") or args:
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
        elif subcommand == "convert":
            if len(args) < 3:
                yield {"error": "Usage: convert <from_format> <to_format> \"<text>\""}
                return
            yield await self._cmd_convert(args[0], args[1], " ".join(args[2:]))
        elif subcommand == "macros":
            if not args:
                yield {"error": "Usage: macros <expand|list> [text]"}
                return
            if args[0] == "expand":
                if len(args) < 2:
                    yield {"error": "Usage: macros expand \"<text>\""}
                    return
                yield await self._cmd_macros_expand(" ".join(args[1:]))
            elif args[0] == "list":
                yield await self._cmd_macros_list()
            else:
                yield {"error": f"Unknown macros subcommand: {args[0]}"}
        elif subcommand == "templates":
            if not args:
                yield {"error": "Usage: templates <list|convert>"}
                return
            if args[0] == "list":
                yield await self._cmd_templates_list()
            else:
                yield {"error": f"Unknown templates subcommand: {args[0]}"}
        elif subcommand == "help":
            yield self._help_text()
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_formats":
            yield {"formats": self._get_supported_formats()}
        elif operation == "convert":
            from_format = input_data.get("from_format")
            to_format = input_data.get("to_format")
            text = input_data.get("text", "")
            if not from_format or not to_format:
                yield {"error": "from_format and to_format required"}
                return
            result = self._convert_prompt(text, from_format, to_format)
            yield {"result": result, "from_format": from_format, "to_format": to_format}
        elif operation == "expand_macros":
            text = input_data.get("text", "")
            context = input_data.get("context", {})
            result = self._expand_macros(text, context)
            yield {"result": result}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List supported formats."""
        formats = self._get_supported_formats()

        lines = ["\nSupported Prompt Formats:"]
        for category, format_list in formats.items():
            lines.append(f"\n  {category.capitalize()}:")
            for fmt in format_list:
                lines.append(f"    - {fmt}")

        lines.append("\nConversion Examples:")
        lines.append("  convert sillytavern claude \"{{char}} is {{description}}\"")
        lines.append("  convert chatgpt generic \"### Instruction\\n\\nHello\"")

        return {
            "output": "\n".join(lines),
            "data": {"formats": formats}
        }

    async def _cmd_convert(self, from_format: str, to_format: str, text: str) -> Dict[str, Any]:
        """Convert prompt from one format to another."""
        try:
            result = self._convert_prompt(text, from_format, to_format)

            lines = [f"\nConverted from {from_format} to {to_format}:"]
            lines.append("=" * 60)
            lines.append(result)
            lines.append("=" * 60)

            return {
                "output": "\n".join(lines),
                "data": {
                    "from_format": from_format,
                    "to_format": to_format,
                    "original": text,
                    "converted": result
                }
            }
        except Exception as e:
            logger.error(f"Error converting prompt: {e}")
            return {"error": str(e)}

    async def _cmd_macros_expand(self, text: str) -> Dict[str, Any]:
        """Expand SillyTavern-style macros."""
        try:
            result = self._expand_macros(text, {})

            lines = [f"\nExpanded Macros:"]
            lines.append("=" * 60)
            lines.append(f"Original: {text}")
            lines.append(f"Expanded: {result}")
            lines.append("=" * 60)

            return {
                "output": "\n".join(lines),
                "data": {"original": text, "expanded": result}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_macros_list(self) -> Dict[str, Any]:
        """List available macros."""
        macros = self._get_macro_definitions()

        lines = ["\nSillyTavern Macros:"]
        for macro, info in macros.items():
            lines.append(f"  {{{{{macro}}}}} - {info['description']}")
            if "example" in info:
                lines.append(f"      Example: {info['example']}")

        return {
            "output": "\n".join(lines),
            "data": {"macros": macros}
        }

    async def _cmd_templates_list(self) -> Dict[str, Any]:
        """List template formats."""
        templates = [
            {"name": "chat", "description": "Chat completion format"},
            {"name": "instruct", "description": "Instruction following format"},
            {"name": "roleplay", "description": "Roleplay/character format"},
        ]

        lines = ["\nPrompt Templates:"]
        for template in templates:
            lines.append(f"  - {template['name']}: {template['description']}")

        return {
            "output": "\n".join(lines),
            "data": {"templates": templates}
        }

    # ========================================================================
    # Conversion Methods
    # ========================================================================

    def _get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported prompt formats."""
        return {
            "chat": ["sillytavern", "claude", "chatgpt", "generic", "alpaca"],
            "instruct": ["sillytavern", "claude", "chatgpt", "generic"],
            "roleplay": ["sillytavern", "claude", "generic"],
        }

    def _convert_prompt(self, text: str, from_format: str, to_format: str) -> str:
        """Convert prompt from one format to another."""
        # Normalize format names
        from_format = from_format.lower().replace("-", "_")
        to_format = to_format.lower().replace("-", "_")

        # If same format, just return
        if from_format == to_format:
            return text

        # Expand macros first if from SillyTavern
        if from_format in ("sillytavern", "silly_tavern"):
            text = self._expand_macros(text, {})

        # Convert based on target format
        if to_format in ("claude", "claude_v2"):
            return self._to_claude_format(text, from_format)
        elif to_format in ("chatgpt", "openai"):
            return self._to_chatgpt_format(text, from_format)
        elif to_format == "alpaca":
            return self._to_alpaca_format(text, from_format)
        elif to_format == "generic":
            return self._to_generic_format(text, from_format)
        else:
            return text

    def _to_claude_format(self, text: str, from_format: str) -> str:
        """Convert to Claude format."""
        # Check if already in Claude format
        if "<human>" in text.lower() or "<assistant>" in text.lower():
            return text

        # Convert from various formats to Claude
        if from_format in ("sillytavern", "silly_tavern"):
            # SillyTavern to Claude
            lines = []
            current_role = "human"

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check for role markers
                if line.lower().startswith(("you:", "assistant:", "ai:")):
                    if current_role != "assistant":
                        lines.append(f"\n<{current_role}>")
                        current_role = "assistant"
                    lines.append(line.split(":", 1)[1].strip() if ":" in line else line)
                elif line.lower().startswith(("user:", "human:", "player:")):
                    if current_role != "human":
                        lines.append(f"\n<{current_role}>")
                        current_role = "human"
                    lines.append(line.split(":", 1)[1].strip() if ":" in line else line)
                else:
                    lines.append(line)

            # Add final tag
            lines.append(f"\n<{current_role}>")

            return "\n".join(lines).strip()

        elif from_format in ("chatgpt", "openai"):
            # ChatGPT to Claude
            if "### Instruction" in text:
                parts = text.split("###")
                result = "<human>\n"
                for part in parts:
                    part = part.strip()
                    if part.lower().startswith("instruction"):
                        result += part.replace("Instruction", "", 1).strip() + "\n\n"
                    elif part.lower().startswith("response"):
                        result += f"\n<assistant>\n{part.replace('Response', '', 1).strip()}\n"
                        result += "\n<human>\n"
                return result.strip() + "\n"
            return f"<human>\n{text}\n"

        return f"<human>\n{text}\n"

    def _to_chatgpt_format(self, text: str, from_format: str) -> str:
        """Convert to ChatGPT format."""
        # Check if already in ChatGPT format
        if "###" in text:
            return text

        if from_format in ("sillytavern", "silly_tavern"):
            # Extract instruction/response pattern
            lines = []
            in_response = False

            for line in text.split("\n"):
                line_lower = line.lower().strip()
                if line_lower.startswith(("you:", "assistant:", "ai:")):
                    if not in_response:
                        lines.append("### Response:")
                        in_response = True
                    lines.append(line.split(":", 1)[1].strip() if ":" in line else line)
                elif line_lower.startswith(("user:", "human:", "player:")):
                    if in_response:
                        in_response = False
                    lines.append(line.split(":", 1)[1].strip() if ":" in line else line)
                else:
                    lines.append(line)

            return "\n".join(lines)

        elif from_format in ("claude", "claude_v2"):
            # Claude to ChatGPT
            result = text.replace("<human>", "### Instruction:")
            result = result.replace("</human>", "")
            result = result.replace("<assistant>", "### Response:")
            result = result.replace("</assistant>", "")
            return result.strip()

        return f"### Instruction:\n{text}\n\n### Response:"

    def _to_alpaca_format(self, text: str, from_format: str) -> str:
        """Convert to Alpaca format."""
        if from_format in ("sillytavern", "silly_tavern"):
            # Look for instruction/response pattern
            lines = []
            in_instruction = True

            for line in text.split("\n"):
                line_lower = line.lower().strip()
                if line_lower.startswith(("you:", "assistant:", "ai:")):
                    if in_instruction:
                        in_instruction = False
                    lines.append(line.split(":", 1)[1].strip() if ":" in line else line)
                elif line_lower.startswith(("user:", "human:", "player:")):
                    if not in_instruction:
                        in_instruction = True
                    lines.append(line.split(":", 1)[1].strip() if ":" in line else line)
                else:
                    lines.append(line)

            instruction = "\n".join(lines).strip()
            return f"### Instruction:\n{instruction}\n\n### Response:"

        return f"### Instruction:\n{text}\n\n### Response:"

    def _to_generic_format(self, text: str, from_format: str) -> str:
        """Convert to generic plain text."""
        # Remove format-specific markers
        text = re.sub(r"<(human|assistant|user|system|bot)>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</(human|assistant|user|system|bot)>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"#{2,}\s*(Instruction|Response|Input|Output)\s*:{0,1}", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\{\{.+?\}\}", "", text)  # Remove macros

        return text.strip()

    def _expand_macros(self, text: str, context: Dict[str, Any]) -> str:
        """Expand SillyTavern-style macros."""
        # Common SillyTavern macros
        macro_patterns = {
            r"\{\{char\}\}": "Character",
            r"\{\{user\}\}": "User",
            r"\{\{description\}\}": "character description",
            r"\{\{personality\}\}": "personality traits",
            r"\{\{scenario\}\}": "current scenario",
            r"\{\{first_mes\}\}": "first message",
            r"\{\{system\}\}": "system prompt",
        }

        result = text
        for pattern, replacement in macro_patterns.items():
            # Check if macro has a custom value in context
            macro_name = pattern.strip(r"\{\}")
            if macro_name in context:
                result = re.sub(pattern, str(context[macro_name]), result)
            else:
                result = re.sub(pattern, replacement, result)

        return result

    def _get_macro_definitions(self) -> Dict[str, Dict[str, str]]:
        """Get SillyTavern macro definitions."""
        return {
            "char": {
                "description": "Character name",
                "example": "{{char}} = Alice"
            },
            "user": {
                "description": "User/player name",
                "example": "{{user}} = Bob"
            },
            "description": {
                "description": "Character description",
                "example": "{{description}} = A brave warrior"
            },
            "personality": {
                "description": "Character personality",
                "example": "{{personality}} = Brave, kind, adventurous"
            },
            "scenario": {
                "description": "Current scenario/context",
                "example": "{{scenario}} = In a medieval tavern"
            },
            "first_mes": {
                "description": "First message from character",
                "example": "{{first_mes}} = Hello there, traveler!"
            },
            "system": {
                "description": "System prompt",
                "example": "{{system}} = You are a helpful assistant"
            },
        }

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Prompt Converter CLI

Commands:
    list                    List supported formats
    convert <from> <to> <text>  Convert prompt format
    macros expand <text>     Expand SillyTavern macros
    macros list             List available macros
    templates list          List template formats
    help                    Show this help

Supported Formats:
    sillytavern, tavern     - SillyTavern format (with {{macros}})
    claude                  - Claude format (<human>/<assistant>)
    chatgpt, openai         - ChatGPT format (### instruction)
    alpaca                  - Alpaca format
    generic                 - Plain text format

Examples:
    uv run backend/manage.py convert list
    uv run backend/manage.py convert convert sillytavern claude "{{char}} is {{description}}"
    uv run backend/manage.py convert macros expand "{{char}} meets {{user}}"
    uv run backend/manage.py convert macros list
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/convert", tags=["prompt_converter"])

        @router.get("/formats")
        async def list_formats():
            """List supported prompt formats."""
            return self._get_supported_formats()

        @router.post("/convert")
        async def convert_prompt(request: dict):
            """Convert prompt from one format to another."""
            from_format = request.get("from_format")
            to_format = request.get("to_format")
            text = request.get("text", "")
            if not from_format or not to_format:
                return {"error": "from_format and to_format required"}
            result = self._convert_prompt(text, from_format, to_format)
            return {"result": result, "from_format": from_format, "to_format": to_format}

        @router.post("/macros/expand")
        async def expand_macros(request: dict):
            """Expand SillyTavern macros."""
            text = request.get("text", "")
            context = request.get("context", {})
            result = self._expand_macros(text, context)
            return {"result": result, "macros_expanded": True}

        @router.get("/macros")
        async def list_macros():
            """List available macros."""
            return self._get_macro_definitions()

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(PromptConverterExtension)
