"""Regex Engine Extension

Regex pattern management and text processing.
Provides CLI tools for managing regex patterns and applying them to text.

CLI Usage:
    uv run backend/manage.py regex list
    uv run backend/manage.py regex add <name> <pattern>
    uv run backend/manage.py regex apply <name> <text>
    uv run backend/manage.py regex test <pattern> <text>
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


@dataclass
class RegexPattern:
    """A regex pattern definition."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    pattern: str = ""
    replacement: str = ""  # Replacement string (for find/replace)
    description: str = ""

    # Pattern flags
    flags: int = 0  # re.IGNORECASE, re.MULTILINE, etc.
    ignore_case: bool = False
    multiline: bool = False
    dotall: bool = False

    # Behavior
    enabled: bool = True
    find_only: bool = False  # If true, only find matches, don't replace

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        # Convert flags to list of names
        flag_names = []
        if self.flags & re.IGNORECASE:
            flag_names.append("IGNORECASE")
        if self.flags & re.MULTILINE:
            flag_names.append("MULTILINE")
        if self.flags & re.DOTALL:
            flag_names.append("DOTALL")
        d["flag_names"] = flag_names
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegexPattern":
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)

    def compile(self) -> re.Pattern:
        """Compile the regex pattern with flags."""
        flags = 0
        if self.ignore_case:
            flags |= re.IGNORECASE
        if self.multiline:
            flags |= re.MULTILINE
        if self.dotall:
            flags |= re.DOTALL
        self.flags = flags
        return re.compile(self.pattern, flags)

    def apply(self, text: str) -> Union[List[str], str]:
        """Apply the pattern to text."""
        regex = self.compile()
        if self.find_only:
            return regex.findall(text)
        else:
            return regex.sub(self.replacement, text)


class RegexEngineExtension(LogicExtension):
    """Regex Engine extension - CLI wrapper for regex pattern management.

    This extension provides CLI access to regex pattern management
    for text processing and transformations.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._patterns: Dict[str, RegexPattern] = {}

    async def initialize(self, context):
        """Initialize the regex engine extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "regex"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "regex"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load patterns
        self._load_patterns()
        self._create_default_patterns()

        logger.info(f"Regex Engine extension initialized with {len(self._patterns)} patterns")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_patterns()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all regex patterns
            get <name>              - Get pattern details
            add <name> <pattern>    - Add a new pattern
            delete <name>           - Delete a pattern
            apply <name> <text>     - Apply pattern to text
            test <pattern> <text>   - Test a pattern on text
            enable <name>           - Enable a pattern
            disable <name>          - Disable a pattern
            replace <name> <rep>    - Set replacement string
            import <file>           - Import patterns from file
            export <file>           - Export patterns to file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("regex", "regexengine", "re") or args:
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
        elif subcommand in ("get", "show", "info"):
            if not args:
                yield {"error": "Please provide a pattern name"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand in ("add", "create"):
            if len(args) < 2:
                yield {"error": "Please provide pattern name and regex pattern"}
                return
            yield await self._cmd_add(args[0], args[1])
        elif subcommand in ("delete", "remove", "rm", "del"):
            if not args:
                yield {"error": "Please provide a pattern name"}
                return
            yield await self._cmd_delete(args[0])
        elif subcommand == "apply":
            if len(args) < 2:
                yield {"error": "Please provide pattern name and text"}
                return
            yield await self._cmd_apply(args[0], " ".join(args[1:]))
        elif subcommand == "test":
            if len(args) < 2:
                yield {"error": "Please provide pattern and text"}
                return
            yield await self._cmd_test(args[0], " ".join(args[1:]))
        elif subcommand == "enable":
            if not args:
                yield {"error": "Please provide a pattern name"}
                return
            yield await self._cmd_enable(args[0])
        elif subcommand == "disable":
            if not args:
                yield {"error": "Please provide a pattern name"}
                return
            yield await self._cmd_disable(args[0])
        elif subcommand == "replace":
            if len(args) < 2:
                yield {"error": "Please provide pattern name and replacement"}
                return
            yield await self._cmd_replace(args[0], " ".join(args[1:]))
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        elif subcommand == "export":
            if not args:
                yield {"error": "Please provide a file path"}
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
        if operation == "list_patterns":
            yield {"patterns": {k: v.to_dict() for k, v in self._patterns.items()}}
        elif operation == "get_pattern":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            pattern = self._patterns.get(name)
            if pattern:
                yield {"pattern": pattern.to_dict()}
            else:
                yield {"error": "Pattern not found"}
        elif operation == "add_pattern":
            name = input_data.get("name")
            pattern_str = input_data.get("pattern")
            if not name or not pattern_str:
                yield {"error": "name and pattern required"}
                return
            pattern = self._add_pattern(name, pattern_str)
            self._save_patterns()
            yield {"pattern": pattern.to_dict()}
        elif operation == "delete_pattern":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            success = self._delete_pattern(name)
            if success:
                self._save_patterns()
                yield {"success": True}
            else:
                yield {"error": "Pattern not found"}
        elif operation == "apply_pattern":
            name = input_data.get("name")
            text = input_data.get("text", "")
            if not name:
                yield {"error": "name required"}
                return
            result = await self._cmd_apply(name, text)
            yield result
        elif operation == "test_pattern":
            pattern_str = input_data.get("pattern")
            text = input_data.get("text", "")
            if not pattern_str:
                yield {"error": "pattern required"}
                return
            result = await self._cmd_test(pattern_str, text)
            yield result
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_patterns(self):
        """Load patterns from storage."""
        index_file = self._storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                    for pattern_data in data.get("patterns", []):
                        pattern = RegexPattern.from_dict(pattern_data)
                        self._patterns[pattern.name] = pattern
            except Exception as e:
                logger.error(f"Error loading regex patterns: {e}")

    def _save_patterns(self):
        """Save patterns to storage."""
        index_file = self._storage_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump({
                "patterns": [p.to_dict() for p in self._patterns.values()],
            }, f, indent=2)

    def _create_default_patterns(self):
        """Create default regex patterns if none exist."""
        if self._patterns:
            return

        # Common patterns
        defaults = [
            RegexPattern(
                name="email",
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                description="Find email addresses",
                find_only=True,
            ),
            RegexPattern(
                name="url",
                pattern=r"https?://[^\s]+",
                description="Find URLs",
                find_only=True,
            ),
            RegexPattern(
                name="cleanup_quotes",
                pattern=r"\"",
                replacement="'",
                description="Replace double quotes with single quotes",
            ),
            RegexPattern(
                name="remove_extra_spaces",
                pattern=r" +",
                replacement=" ",
                description="Replace multiple spaces with single space",
            ),
            RegexPattern(
                name="fix_ellipsis",
                pattern=r"\.{3,}",
                replacement="...",
                description="Fix multiple dots to ellipsis",
            ),
        ]

        for pattern in defaults:
            self._patterns[pattern.name] = pattern

        self._save_patterns()
        logger.info("Created default regex patterns")

    # ========================================================================
    # Pattern Management
    # ========================================================================

    def _add_pattern(self, name: str, pattern_str: str, **kwargs) -> RegexPattern:
        """Add a new pattern."""
        pattern = RegexPattern(name=name, pattern=pattern_str, **kwargs)
        self._patterns[name] = pattern
        return pattern

    def _delete_pattern(self, name: str) -> bool:
        """Delete a pattern."""
        if name in self._patterns:
            del self._patterns[name]
            return True
        return False

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all patterns."""
        try:
            if not self._patterns:
                return {
                    "output": "No regex patterns found.",
                    "data": {"patterns": {}}
                }

            lines = [f"\nRegex Patterns ({len(self._patterns)} total):"]

            for name, pattern in sorted(self._patterns.items()):
                status = "✓" if pattern.enabled else "✗"
                mode = "FIND" if pattern.find_only else "REPLACE"
                lines.append(f"\n  [{status}] {name} ({mode})")
                lines.append(f"      Pattern: {pattern.pattern}")
                if pattern.replacement:
                    lines.append(f"      Replacement: {pattern.replacement}")
                if pattern.description:
                    lines.append(f"      Description: {pattern.description}")
                if pattern.ignore_case or pattern.multiline or pattern.dotall:
                    flags = []
                    if pattern.ignore_case:
                        flags.append("IGNORECASE")
                    if pattern.multiline:
                        flags.append("MULTILINE")
                    if pattern.dotall:
                        flags.append("DOTALL")
                    lines.append(f"      Flags: {', '.join(flags)}")

            return {
                "output": "\n".join(lines),
                "data": {"patterns": {k: v.to_dict() for k, v in self._patterns.items()}},
            }
        except Exception as e:
            logger.error(f"Error listing patterns: {e}")
            return {"error": str(e)}

    async def _cmd_get(self, name: str) -> Dict[str, Any]:
        """Get pattern details."""
        pattern = self._patterns.get(name)
        if pattern:
            lines = [f"\nPattern: {pattern.name}"]
            lines.append(f"Status: {'Enabled' if pattern.enabled else 'Disabled'}")
            lines.append(f"Mode: {'Find Only' if pattern.find_only else 'Find and Replace'}")
            lines.append(f"Pattern: {pattern.pattern}")
            lines.append(f"Replacement: {pattern.replacement or 'None'}")
            lines.append(f"Description: {pattern.description}")
            if pattern.ignore_case or pattern.multiline or pattern.dotall:
                flags = []
                if pattern.ignore_case:
                    flags.append("IGNORECASE")
                if pattern.multiline:
                    flags.append("MULTILINE")
                if pattern.dotall:
                    flags.append("DOTALL")
                lines.append(f"Flags: {', '.join(flags)}")

            return {
                "output": "\n".join(lines),
                "data": {"pattern": pattern.to_dict()},
            }
        return {"error": f"Pattern not found: {name}"}

    async def _cmd_add(self, name: str, pattern_str: str) -> Dict[str, Any]:
        """Add a new pattern."""
        if name in self._patterns:
            return {"error": f"Pattern already exists: {name}"}

        pattern = self._add_pattern(name, pattern_str)
        self._save_patterns()

        return {
            "output": f"Pattern added: {name}",
            "data": {"pattern": pattern.to_dict()},
        }

    async def _cmd_delete(self, name: str) -> Dict[str, Any]:
        """Delete a pattern."""
        success = self._delete_pattern(name)
        if success:
            self._save_patterns()
            return {"output": f"Pattern deleted: {name}"}
        return {"error": f"Pattern not found: {name}"}

    async def _cmd_apply(self, name: str, text: str) -> Dict[str, Any]:
        """Apply pattern to text."""
        pattern = self._patterns.get(name)
        if not pattern:
            return {"error": f"Pattern not found: {name}"}
        if not pattern.enabled:
            return {"error": f"Pattern is disabled: {name}"}

        try:
            result = pattern.apply(text)
            if pattern.find_only:
                return {
                    "output": f"Matches found: {result}",
                    "data": {"matches": result, "pattern": pattern.to_dict()},
                }
            else:
                return {
                    "output": f"Result: {result}",
                    "data": {"result": result, "pattern": pattern.to_dict()},
                }
        except re.error as e:
            return {"error": f"Regex error: {e}"}

    async def _cmd_test(self, pattern_str: str, text: str) -> Dict[str, Any]:
        """Test a pattern on text without saving."""
        try:
            regex = re.compile(pattern_str)
            matches = regex.findall(text)
            return {
                "output": f"Matches found: {matches}",
                "data": {"pattern": pattern_str, "matches": matches},
            }
        except re.error as e:
            return {"error": f"Regex error: {e}"}

    async def _cmd_enable(self, name: str) -> Dict[str, Any]:
        """Enable a pattern."""
        pattern = self._patterns.get(name)
        if pattern:
            pattern.enabled = True
            self._save_patterns()
            return {"output": f"Pattern enabled: {name}"}
        return {"error": f"Pattern not found: {name}"}

    async def _cmd_disable(self, name: str) -> Dict[str, Any]:
        """Disable a pattern."""
        pattern = self._patterns.get(name)
        if pattern:
            pattern.enabled = False
            self._save_patterns()
            return {"output": f"Pattern disabled: {name}"}
        return {"error": f"Pattern not found: {name}"}

    async def _cmd_replace(self, name: str, replacement: str) -> Dict[str, Any]:
        """Set replacement string for a pattern."""
        pattern = self._patterns.get(name)
        if pattern:
            pattern.replacement = replacement
            pattern.find_only = False
            self._save_patterns()
            return {"output": f"Replacement set for: {name}"}
        return {"error": f"Pattern not found: {name}"}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import patterns from file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                for pattern_data in data.get("patterns", []):
                    pattern = RegexPattern.from_dict(pattern_data)
                    self._patterns[pattern.name] = pattern
                self._save_patterns()
                return {"output": f"Patterns imported from: {file_path}"}
        except FileNotFoundError:
            return {"error": f"File not found: {file_path}"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export patterns to file."""
        try:
            with open(file_path, "w") as f:
                json.dump({
                    "patterns": [p.to_dict() for p in self._patterns.values()],
                }, f, indent=2)
            return {"output": f"Patterns exported to: {file_path}"}
        except Exception as e:
            return {"error": str(e)}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Regex Engine CLI

Commands:
    list                    List all regex patterns
    get <name>              Get pattern details
    add <name> <pattern>    - Add a new pattern
    delete <name>           Delete a pattern
    apply <name> <text>     Apply pattern to text
    test <pattern> <text>   Test a pattern on text (without saving)
    enable <name>           Enable a pattern
    disable <name>          Disable a pattern
    replace <name> <rep>    Set replacement string for a pattern
    import <file>           Import patterns from file
    export <file>           Export patterns to file

Pattern Flags:
    ignore_case             Make pattern case-insensitive
    multiline               Make $ match end of line, not just end of string
    dotall                   Make . match newlines as well

Default Patterns:
    email                   Find email addresses (FIND mode)
    url                     Find URLs (FIND mode)
    cleanup_quotes          Replace double quotes with single quotes
    remove_extra_spaces     Replace multiple spaces with single space
    fix_ellipsis            Fix multiple dots to ellipsis

Examples:
    uv run backend/manage.py regex list
    uv run backend/manage.py regex add phone "\\\\d{3}-\\\\d{3}-\\\\d{4}"
    uv run backend/manage.py regex apply email "Contact me at test@example.com"
    uv run backend/manage.py regex test "\\\\d+" "There are 123 apples"
    uv run backend/manage.py regex replace cleanup_quotes "'"
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/regex", tags=["regex"])

        @router.get("/")
        async def list_patterns():
            """List all patterns."""
            return {"patterns": {k: v.to_dict() for k, v in self._patterns.items()}}

        @router.get("/{name}")
        async def get_pattern(name: str):
            """Get pattern details."""
            pattern = self._patterns.get(name)
            if pattern:
                return {"pattern": pattern.to_dict()}
            return {"error": "Pattern not found"}, 404

        @router.post("/")
        async def add_pattern(request: dict):
            """Add a new pattern."""
            name = request.get("name")
            pattern_str = request.get("pattern")
            if not name or not pattern_str:
                return {"error": "name and pattern required"}, 400
            pattern = self._add_pattern(name, pattern_str)
            self._save_patterns()
            return {"pattern": pattern.to_dict()}

        @router.delete("/{name}")
        async def delete_pattern(name: str):
            """Delete a pattern."""
            success = self._delete_pattern(name)
            if success:
                self._save_patterns()
                return {"success": True}
            return {"error": "Pattern not found"}, 404

        @router.post("/apply")
        async def apply_pattern(request: dict):
            """Apply pattern to text."""
            name = request.get("name")
            text = request.get("text", "")
            pattern = self._patterns.get(name)
            if not pattern:
                return {"error": "Pattern not found"}, 404
            if not pattern.enabled:
                return {"error": "Pattern is disabled"}, 400

            result = pattern.apply(text)
            if pattern.find_only:
                return {"matches": result, "pattern": pattern.to_dict()}
            else:
                return {"result": result, "pattern": pattern.to_dict()}

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(RegexEngineExtension)
