"""Inline Scripts Extension

Inline code/JavaScript execution for advanced customization.
Provides CLI tools for managing and executing inline scripts.

CLI Usage:
    uv run backend/manage.py inlines list
    uv run backend/manage.py inlines add <name> --code "return text.toUpperCase();"
    uv run backend/manage.py inlines execute <name> --input "hello"
    uv run backend/manage.py inlines test --code "return 2 + 2;"
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime

try:
    import js2py
    JS_AVAILABLE = True
except ImportError:
    JS_AVAILABLE = False

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class InlineScript:
    """An inline script entry."""

    def __init__(
        self,
        name: str,
        code: str,
        trigger: str = "manual",  # manual, on_send, on_receive, on_generate
        enabled: bool = True,
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.code = code
        self.trigger = trigger
        self.enabled = enabled
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "trigger": self.trigger,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InlineScript":
        script = cls(
            name=data["name"],
            code=data["code"],
            trigger=data.get("trigger", "manual"),
            enabled=data.get("enabled", True),
        )
        script.id = data["id"]
        if "created_at" in data:
            script.created_at = datetime.fromisoformat(data["created_at"])
        return script


class InlineScriptsExtension(LogicExtension):
    """Inline Scripts extension - CLI wrapper for inline script management.

    This extension provides CLI access to inline script tools
    for managing and executing JavaScript code.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._scripts: List[InlineScript] = []

    async def initialize(self, context):
        """Initialize the inline scripts extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "inlines"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "inlines"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load existing scripts
        self._load_scripts()

        # Check JS availability
        if not JS_AVAILABLE:
            logger.warning("js2py not installed - JavaScript execution limited")

        logger.info("Inline Scripts extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_scripts()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all inline scripts
            add <name>              - Add a new inline script
            get <name>              - Get script details
            execute <name>          - Execute a script
            test                    - Test a code snippet
            remove <name>           - Remove a script
            enable <name>           - Enable a script
            disable <name>          - Disable a script
            export <file>           - Export scripts to file
            import <file>           - Import scripts from file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("inlines", "inline", "scripts", "js") or args:
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
                yield {"error": "Please provide a script name"}
                return
            yield await self._cmd_add(args[0], args[1:])
        elif subcommand in ("get", "show"):
            if not args:
                yield {"error": "Please provide a script name or ID"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand in ("execute", "run", "exec"):
            if not args:
                yield {"error": "Please provide a script name or ID"}
                return
            yield await self._cmd_execute(args[0], args[1:])
        elif subcommand == "test":
            yield await self._cmd_test(args)
        elif subcommand in ("remove", "delete", "rm"):
            if not args:
                yield {"error": "Please provide a script name or ID"}
                return
            yield await self._cmd_remove(args[0])
        elif subcommand == "enable":
            if not args:
                yield {"error": "Please provide a script name or ID"}
                return
            yield await self._cmd_enable(args[0])
        elif subcommand == "disable":
            if not args:
                yield {"error": "Please provide a script name or ID"}
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
        if operation == "list_scripts":
            yield {"scripts": [s.to_dict() for s in self._scripts]}
        elif operation == "add_script":
            name = input_data.get("name")
            code = input_data.get("code")
            if not name or not code:
                yield {"error": "name and code required"}
                return
            script = self._add_script(
                name=name,
                code=code,
                trigger=input_data.get("trigger", "manual"),
            )
            self._save_scripts()
            yield {"script": script.to_dict()}
        elif operation == "execute_script":
            script_id = input_data.get("id") or input_data.get("name")
            script_input = input_data.get("input", "")
            if not script_id:
                yield {"error": "id or name required"}
                return
            result = self._execute_script(script_id, script_input)
            if result:
                yield {"result": result}
            else:
                yield {"error": "Script not found or disabled"}
        elif operation == "test_code":
            code = input_data.get("code", "")
            script_input = input_data.get("input", "")
            result = self._execute_code(code, script_input)
            yield {"result": result}
        elif operation == "remove_script":
            script_id = input_data.get("id") or input_data.get("name")
            if not script_id:
                yield {"error": "id or name required"}
                return
            success = self._remove_script(script_id)
            if success:
                self._save_scripts()
                yield {"success": True}
            else:
                yield {"error": "Script not found"}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_scripts(self):
        """Load scripts from storage."""
        scripts_file = self._storage_dir / "scripts.json"

        if not scripts_file.exists():
            # Create default scripts
            self._create_default_scripts()
            return

        try:
            with open(scripts_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._scripts = [InlineScript.from_dict(s) for s in data]
        except Exception as e:
            logger.error(f"Error loading scripts: {e}")
            self._scripts = []

    def _save_scripts(self):
        """Save scripts to storage."""
        scripts_file = self._storage_dir / "scripts.json"
        with open(scripts_file, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self._scripts], f, indent=2, ensure_ascii=False)

    def _create_default_scripts(self):
        """Create default inline scripts."""
        defaults = [
            {
                "name": "Uppercase",
                "code": "return text.toUpperCase();",
                "trigger": "manual",
            },
            {
                "name": "Lowercase",
                "code": "return text.toLowerCase();",
                "trigger": "manual",
            },
            {
                "name": "Reverse",
                "code": "return text.split('').reverse().join('');",
                "trigger": "manual",
            },
            {
                "name": "Word Count",
                "code": "return text.split(/\\s+/).length.toString();",
                "trigger": "manual",
            },
        ]

        self._scripts = [InlineScript(**d) for d in defaults]
        self._save_scripts()

    # ========================================================================
    # Script Management
    # ========================================================================

    def _add_script(
        self,
        name: str,
        code: str,
        trigger: str = "manual",
    ) -> InlineScript:
        """Add a new inline script."""
        script = InlineScript(
            name=name,
            code=code,
            trigger=trigger,
        )
        self._scripts.append(script)
        return script

    def _find_script(self, identifier: str) -> Optional[InlineScript]:
        """Find a script by ID or name."""
        for script in self._scripts:
            if script.id == identifier or script.name.lower() == identifier.lower():
                return script
        return None

    def _remove_script(self, identifier: str) -> bool:
        """Remove a script by ID or name."""
        script = self._find_script(identifier)
        if script:
            self._scripts.remove(script)
            return True
        return False

    def _set_enabled(self, identifier: str, enabled: bool) -> bool:
        """Set enabled state of a script."""
        script = self._find_script(identifier)
        if script:
            script.enabled = enabled
            return True
        return False

    # ========================================================================
    # Script Execution
    # ========================================================================

    def _execute_code(self, code: str, script_input: str = "") -> Dict[str, Any]:
        """Execute JavaScript code."""
        if not JS_AVAILABLE:
            return {
                "error": "js2py not installed - cannot execute JavaScript",
                "code": code,
                "input": script_input,
            }

        try:
            # Wrap the code in a function
            wrapped_code = f"""
            function execute(text) {{
                {code}
            }}
            execute("{script_input.replace(chr(34), '\\"')}");
            """

            result = js2py.eval_js(wrapped_code)

            return {
                "result": str(result),
                "code": code,
                "input": script_input,
            }
        except Exception as e:
            return {
                "error": str(e),
                "code": code,
                "input": script_input,
            }

    def _execute_script(self, identifier: str, script_input: str = "") -> Optional[Dict[str, Any]]:
        """Execute a script by ID or name."""
        script = self._find_script(identifier)
        if script and script.enabled:
            return self._execute_code(script.code, script_input)
        return None

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all inline scripts."""
        try:
            if not self._scripts:
                return {
                    "output": "No inline scripts found.",
                    "data": {"scripts": []}
                }

            lines = ["\nInline Scripts:"]
            for script in self._scripts:
                status = "✓" if script.enabled else "✗"
                code_preview = script.code[:40] + "..." if len(script.code) > 40 else script.code
                lines.append(f"  {status} [{script.id}] {script.name} ({script.trigger})")
                lines.append(f"      {code_preview}")

            return {
                "output": "\n".join(lines),
                "data": {"scripts": [s.to_dict() for s in self._scripts]}
            }
        except Exception as e:
            logger.error(f"Error listing scripts: {e}")
            return {"error": str(e)}

    async def _cmd_add(self, name: str, args: List[str]) -> Dict[str, Any]:
        """Add a new inline script."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Add inline script")
            parser.add_argument("--code", required=True, help="JavaScript code")
            parser.add_argument("--trigger",
                               choices=["manual", "on_send", "on_receive", "on_generate"],
                               default="manual", help="Trigger type")

            parsed = parser.parse_args(args)

            script = self._add_script(
                name=name,
                code=parsed.code,
                trigger=parsed.trigger,
            )
            self._save_scripts()

            return {
                "output": f"Inline script '{name}' added (ID: {script.id})",
                "data": {"script": script.to_dict()}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_get(self, identifier: str) -> Dict[str, Any]:
        """Get script details."""
        try:
            script = self._find_script(identifier)
            if not script:
                return {"error": f"Inline script '{identifier}' not found"}

            lines = [f"\nInline Script: {script.name}"]
            lines.append(f"ID: {script.id}")
            lines.append(f"Trigger: {script.trigger}")
            lines.append(f"Enabled: {script.enabled}")
            lines.append(f"Created: {script.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"\nCode:")
            lines.append(script.code)

            return {
                "output": "\n".join(lines),
                "data": {"script": script.to_dict()}
            }
        except Exception as e:
            logger.error(f"Error getting script: {e}")
            return {"error": str(e)}

    async def _cmd_execute(self, identifier: str, args: List[str]) -> Dict[str, Any]:
        """Execute a script."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Execute script")
            parser.add_argument("--input", default="", help="Input text")

            parsed = parser.parse_args(args)

            script = self._find_script(identifier)
            if not script:
                return {"error": f"Inline script '{identifier}' not found"}

            if not script.enabled:
                return {"error": f"Inline script '{script.name}' is disabled"}

            result = self._execute_code(script.code, parsed.input)

            lines = [f"\nExecuting: {script.name}"]
            lines.append(f"Input: {parsed.input}")
            if "error" in result:
                lines.append(f"Error: {result['error']}")
            else:
                lines.append(f"Result: {result['result']}")

            return {
                "output": "\n".join(lines),
                "data": {"result": result}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_test(self, args: List[str]) -> Dict[str, Any]:
        """Test a code snippet."""
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Test code")
            parser.add_argument("--code", required=True, help="JavaScript code to test")
            parser.add_argument("--input", default="", help="Input text")

            parsed = parser.parse_args(args)

            result = self._execute_code(parsed.code, parsed.input)

            lines = [f"\nTesting code:"]
            lines.append(f"Input: {parsed.input}")
            if "error" in result:
                lines.append(f"Error: {result['error']}")
            else:
                lines.append(f"Result: {result['result']}")

            return {
                "output": "\n".join(lines),
                "data": {"result": result}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_remove(self, identifier: str) -> Dict[str, Any]:
        """Remove an inline script."""
        try:
            script = self._find_script(identifier)
            if not script:
                return {"error": f"Inline script '{identifier}' not found"}

            name = script.name
            success = self._remove_script(identifier)
            if success:
                self._save_scripts()
                return {"output": f"Inline script '{name}' removed"}
            return {"error": f"Failed to remove inline script '{identifier}'"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_enable(self, identifier: str) -> Dict[str, Any]:
        """Enable an inline script."""
        try:
            script = self._find_script(identifier)
            if not script:
                return {"error": f"Inline script '{identifier}' not found"}

            self._set_enabled(identifier, True)
            self._save_scripts()
            return {"output": f"Inline script '{script.name}' enabled"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_disable(self, identifier: str) -> Dict[str, Any]:
        """Disable an inline script."""
        try:
            script = self._find_script(identifier)
            if not script:
                return {"error": f"Inline script '{identifier}' not found"}

            self._set_enabled(identifier, False)
            self._save_scripts()
            return {"output": f"Inline script '{script.name}' disabled"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_export(self, file_path: str) -> Dict[str, Any]:
        """Export inline scripts to file."""
        try:
            path = Path(file_path)
            if not path.suffix:
                path = path.with_suffix(".json")

            with open(path, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in self._scripts], f, indent=2, ensure_ascii=False)

            return {
                "output": f"Inline scripts exported to {path}",
                "data": {"filepath": str(path)}
            }
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import inline scripts from file."""
        try:
            path = Path(file_path)
            if not path.exists():
                path = self._storage_dir / file_path
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            imported = 0
            for script_data in data:
                script = InlineScript.from_dict(script_data)
                self._scripts.append(script)
                imported += 1

            self._save_scripts()

            return {
                "output": f"Imported {imported} inline scripts from {path}",
                "data": {"imported": imported}
            }
        except Exception as e:
            logger.error(f"Error importing scripts: {e}")
            return {"error": str(e)}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Inline Scripts CLI

Commands:
    list                    List all inline scripts
    add <name>              Add a new inline script
    get <name>              Get script details
    execute <name>          Execute a script
    test                    Test a code snippet
    remove <name>           Remove a script
    enable <name>           Enable a script
    disable <name>          Disable a script
    export <file>           Export scripts to file
    import <file>           Import scripts from file

Add Options:
    --code <js>             JavaScript code (required)
    --trigger <type>        Trigger: manual, on_send, on_receive, on_generate

Execute Options:
    --input <text>          Input text (available as 'text' variable)

Test Options:
    --code <js>             JavaScript code to test
    --input <text>          Input text

Variables available in scripts:
    text                    The input text

Examples:
    uv run backend/manage.py inlines list
    uv run backend/manage.py inlines add Uppercase --code "return text.toUpperCase();"
    uv run backend/manage.py inlines execute Uppercase --input "hello"
    uv run backend/manage.py inlines test --code "return text.split('').reverse().join('');" --input "hello"
    uv run backend/manage.py inlines export backup.json

Note: Requires js2py package for JavaScript execution.
      Install with: pip install js2py
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/inlines", tags=["inlines"])

        @router.get("/")
        async def list_scripts():
            """List all inline scripts."""
            return {"scripts": [s.to_dict() for s in self._scripts]}

        @router.post("/")
        async def add_script(request: dict):
            """Add a new inline script."""
            name = request.get("name")
            code = request.get("code")
            if not name or not code:
                return {"error": "name and code required"}, 400
            script = self._add_script(
                name,
                code,
                trigger=request.get("trigger", "manual"),
            )
            self._save_scripts()
            return {"script": script.to_dict()}

        @router.post("/execute")
        async def execute_script(request: dict):
            """Execute an inline script."""
            script_id = request.get("id") or request.get("name")
            script_input = request.get("input", "")
            if not script_id:
                return {"error": "id or name required"}, 400
            result = self._execute_script(script_id, script_input)
            if result:
                return {"result": result}
            return {"error": "Script not found or disabled"}, 404

        @router.post("/test")
        async def test_code(request: dict):
            """Test JavaScript code."""
            code = request.get("code", "")
            script_input = request.get("input", "")
            result = self._execute_code(code, script_input)
            return {"result": result}

        @router.delete("/{script_id}")
        async def remove_script(script_id: str):
            """Remove an inline script."""
            success = self._remove_script(script_id)
            if success:
                self._save_scripts()
                return {"success": True}
            return {"error": "Script not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(InlineScriptsExtension)
