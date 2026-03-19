"""Variables Extension

Variable management system.
Provides CLI tools for managing template variables and dynamic replacements.

CLI Usage:
    uv run backend/manage.py vars list
    uv run backend/manage.py vars set <name> <value>
    uv run backend/manage.py vars get <name>
    uv run backend/manage.py vars delete <name>
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

from src.extensions.core.base import CLIExtension, BaseDataModel

logger = logging.getLogger(__name__)


@dataclass
class Variable(BaseDataModel):
    """A variable definition.

    Extends BaseDataModel for automatic serialization/deserialization.
    """

    name: str = ""
    value: Any = ""
    type: str = "string"  # string, number, boolean, json, array
    description: str = ""

    # Scope control
    global_var: bool = True  # If true, available across all chats
    chat_id: Optional[str] = None  # If not global, tied to specific chat

    # Persistence
    persistent: bool = True  # If true, saved to disk

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_value(self, value: Any, var_type: Optional[str] = None):
        """Set the variable value with type conversion."""
        var_type = var_type or self.type

        if var_type == "number":
            self.value = float(value)
        elif var_type == "boolean":
            if isinstance(value, str):
                self.value = value.lower() in ("true", "1", "yes", "on")
            else:
                self.value = bool(value)
        elif var_type == "json":
            if isinstance(value, str):
                self.value = json.loads(value)
            else:
                self.value = value
        elif var_type == "array":
            if isinstance(value, str):
                self.value = json.loads(value)
            else:
                self.value = list(value) if not isinstance(value, list) else value
        else:  # string
            self.value = str(value)

        self.type = var_type
        self.touch()  # Update updated_at timestamp


class VariablesExtension(CLIExtension):
    """Variables extension - CLI wrapper for variable management.

    This extension provides CLI access to the variable management system
    for defining and using template variables throughout the application.

    Uses CLIExtension for storage management.
    Uses BaseDataModel for Variable serialization.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._variables: Dict[str, Variable] = {}
        self._variable_pattern = re.compile(r"\{\{(\w+)\}\}")
        self._storage_subdir = "variables"  # Set storage subdir for CLIExtension

    async def initialize(self, context):
        """Initialize the variables extension."""
        await super().initialize(context)
        self._create_default_variables()
        logger.info(f"Variables extension initialized with {len(self._variables)} variables")

    def _load_data(self) -> None:
        """Load variables from storage."""
        if not self._storage_dir:
            return

        index_file = self._storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                    for var_data in data.get("variables", []):
                        var = Variable.from_dict(var_data)
                        self._variables[var.name] = var
            except Exception as e:
                logger.error(f"Error loading variables: {e}")

    def _save_data(self) -> None:
        """Save variables to storage."""
        if not self._storage_dir:
            return

        # Only save persistent variables
        persistent_vars = {k: v for k, v in self._variables.items() if v.persistent}
        index_file = self._storage_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump({
                "variables": [v.to_dict() for v in persistent_vars.values()],
            }, f, indent=2)

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all variables
            get <name>              - Get variable value
            set <name> <value>      - Set variable value
            delete <name>           - Delete a variable
            replace <text>          - Replace variables in text
            types                   - Show variable types
            export                  - Export variables to JSON
            import <file>           - Import variables from JSON
            clear                   - Clear all non-persistent variables
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("vars", "variables", "var") or args:
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
                yield {"error": "Please provide a variable name"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand in ("set", "add", "create"):
            if len(args) < 2:
                yield {"error": "Please provide variable name and value"}
                return
            yield await self._cmd_set(args[0], " ".join(args[1:]))
        elif subcommand in ("delete", "remove", "rm", "del"):
            if not args:
                yield {"error": "Please provide a variable name"}
                return
            yield await self._cmd_delete(args[0])
        elif subcommand == "replace":
            if not args:
                yield {"error": "Please provide text to replace"}
                return
            yield await self._cmd_replace(" ".join(args))
        elif subcommand == "types":
            yield await self._cmd_types()
        elif subcommand == "export":
            yield await self._cmd_export()
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        elif subcommand == "clear":
            yield await self._cmd_clear()
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_variables":
            yield {"variables": {k: v.to_dict() for k, v in self._variables.items()}}
        elif operation == "get_variable":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            var = self._variables.get(name)
            if var:
                yield {"variable": var.to_dict()}
            else:
                yield {"error": "Variable not found"}
        elif operation == "set_variable":
            name = input_data.get("name")
            value = input_data.get("value")
            var_type = input_data.get("type", "string")
            if not name or value is None:
                yield {"error": "name and value required"}
                return
            var = self._set_variable(name, value, var_type)
            self._save_data()
            yield {"variable": var.to_dict()}
        elif operation == "delete_variable":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            success = self._delete_variable(name)
            if success:
                self._save_data()
                yield {"success": True}
            else:
                yield {"error": "Variable not found"}
        elif operation == "replace_variables":
            text = input_data.get("text", "")
            result = self.replace_variables(text)
            yield {"text": result}
        elif operation == "clear_variables":
            self._clear_variables()
            self._save_data()
            yield {"success": True}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Variable Management
    # ========================================================================

    def _create_default_variables(self):
        """Create default variables if none exist."""
        if self._variables:
            return

        # Common default variables
        defaults = [
            Variable(name="name", value="User", type="string", description="User's name"),
            Variable(name="time", value="", type="string", description="Current time (updated dynamically)"),
            Variable(name="date", value="", type="string", description="Current date (updated dynamically)"),
        ]

        for var in defaults:
            self._variables[var.name] = var

        self._save_data()
        logger.info("Created default variables")

    # ========================================================================
    # Variable Management
    # ========================================================================

    def _set_variable(self, name: str, value: Any, var_type: str = "string") -> Variable:
        """Set a variable value."""
        if name in self._variables:
            self._variables[name].set_value(value, var_type)
            return self._variables[name]
        else:
            var = Variable(name=name)
            var.set_value(value, var_type)
            self._variables[name] = var
            return var

    def _delete_variable(self, name: str) -> bool:
        """Delete a variable."""
        if name in self._variables:
            del self._variables[name]
            return True
        return False

    def _clear_variables(self):
        """Clear all non-persistent variables."""
        self._variables = {k: v for k, v in self._variables.items() if v.persistent}

    def replace_variables(self, text: str) -> str:
        """Replace variables in text."""
        def replace_match(match):
            var_name = match.group(1)
            if var_name in self._variables:
                value = self._variables[var_name].value
                return str(value)
            return match.group(0)

        # Handle dynamic variables
        result = text
        if "time" in self._variables:
            import datetime as dt
            self._variables["time"].value = dt.datetime.now().strftime("%H:%M:%S")
        if "date" in self._variables:
            import datetime as dt
            self._variables["date"].value = dt.datetime.now().strftime("%Y-%m-%d")

        result = self._variable_pattern.sub(replace_match, result)
        return result

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all variables."""
        try:
            if not self._variables:
                return {
                    "output": "No variables found.",
                    "data": {"variables": {}}
                }

            lines = [f"\nVariables ({len(self._variables)} total):"]

            for name, var in sorted(self._variables.items()):
                persist_marker = " [P]" if var.persistent else ""
                global_marker = " [G]" if var.global_var else ""
                value_str = str(var.value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                lines.append(f"\n  {var.type:10} {name}{persist_marker}{global_marker}")
                lines.append(f"      Value: {value_str}")
                if var.description:
                    lines.append(f"      Description: {var.description}")

            return {
                "output": "\n".join(lines),
                "data": {"variables": {k: v.to_dict() for k, v in self._variables.items()}},
            }
        except Exception as e:
            logger.error(f"Error listing variables: {e}")
            return {"error": str(e)}

    async def _cmd_get(self, name: str) -> Dict[str, Any]:
        """Get variable value."""
        var = self._variables.get(name)
        if var:
            return {
                "output": f"{name} = {var.value}",
                "data": {"variable": var.to_dict()},
            }
        return {"error": f"Variable not found: {name}"}

    async def _cmd_set(self, name: str, value: str) -> Dict[str, Any]:
        """Set variable value."""
        # Auto-detect type
        var_type = "string"
        if value.lower() in ("true", "false"):
            var_type = "boolean"
        elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            var_type = "number"
        elif value.startswith("{") or value.startswith("["):
            var_type = "json"

        var = self._set_variable(name, value, var_type)
        self._save_data()

        return {
            "output": f"Variable set: {name} = {var.value}",
            "data": {"variable": var.to_dict()},
        }

    async def _cmd_delete(self, name: str) -> Dict[str, Any]:
        """Delete a variable."""
        success = self._delete_variable(name)
        if success:
            self._save_data()
            return {"output": f"Variable deleted: {name}"}
        return {"error": f"Variable not found: {name}"}

    async def _cmd_replace(self, text: str) -> Dict[str, Any]:
        """Replace variables in text."""
        result = self.replace_variables(text)
        return {
            "output": f"Original: {text}\nResult: {result}",
            "data": {"original": text, "result": result},
        }

    async def _cmd_types(self) -> Dict[str, Any]:
        """Show variable types."""
        lines = ["\nVariable Types:"]
        lines.append("  string  - Text values")
        lines.append("  number  - Numeric values (integers and floats)")
        lines.append("  boolean - True/False values")
        lines.append("  json    - JSON objects")
        lines.append("  array   - JSON arrays")

        return {
            "output": "\n".join(lines),
            "data": {"types": ["string", "number", "boolean", "json", "array"]},
        }

    async def _cmd_export(self) -> Dict[str, Any]:
        """Export variables to JSON."""
        output = {}
        for name, var in self._variables.items():
            output[name] = {
                "value": var.value,
                "type": var.type,
                "description": var.description,
            }

        return {
            "output": json.dumps(output, indent=2),
            "data": {"variables": output},
        }

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import variables from JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                for name, var_data in data.items():
                    value = var_data.get("value", "")
                    var_type = var_data.get("type", "string")
                    self._set_variable(name, value, var_type)
                self._save_data()
                return {"output": f"Variables imported from: {file_path}"}
        except FileNotFoundError:
            return {"error": f"File not found: {file_path}"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_clear(self) -> Dict[str, Any]:
        """Clear all non-persistent variables."""
        before = len(self._variables)
        self._clear_variables()
        after = len(self._variables)
        self._save_data()

        return {
            "output": f"Cleared {before - after} non-persistent variables. {after} variables remaining.",
        }

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Variables CLI

Commands:
    list                    List all variables
    get <name>              Get variable value
    set <name> <value>      Set variable value (auto-detects type)
    delete <name>           Delete a variable
    replace <text>          Replace variables in text (use {{name}} syntax)
    types                   Show variable types
    export                  Export variables to JSON
    import <file>           Import variables from JSON file
    clear                   Clear all non-persistent variables

Variable Types:
    string  - Text values (default)
    number  - Numeric values
    boolean - True/False values
    json    - JSON objects
    array   - JSON arrays

Usage in Templates:
    Use {{variable_name}} syntax in templates to reference variables.
    Example: "Hello {{name}}, today is {{date}}."

Examples:
    uv run backend/manage.py vars list
    uv run backend/manage.py vars set name "John Doe"
    uv run backend/manage.py vars set age 25
    uv run backend/manage.py vars get name
    uv run backend/manage.py vars replace "Hello {{name}}"
    uv run backend/manage.py vars delete age
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/variables", tags=["variables"])

        @router.get("/")
        async def list_variables():
            """List all variables."""
            return {"variables": {k: v.to_dict() for k, v in self._variables.items()}}

        @router.get("/{name}")
        async def get_variable(name: str):
            """Get variable value."""
            var = self._variables.get(name)
            if var:
                return {"variable": var.to_dict()}
            return {"error": "Variable not found"}, 404

        @router.post("/")
        async def set_variable(request: dict):
            """Set variable value."""
            name = request.get("name")
            value = request.get("value")
            var_type = request.get("type", "string")
            if not name or value is None:
                return {"error": "name and value required"}, 400
            var = self._set_variable(name, value, var_type)
            self._save_data()
            return {"variable": var.to_dict()}

        @router.delete("/{name}")
        async def delete_variable(name: str):
            """Delete a variable."""
            success = self._delete_variable(name)
            if success:
                self._save_data()
                return {"success": True}
            return {"error": "Variable not found"}, 404

        @router.post("/replace")
        async def replace_variables(request: dict):
            """Replace variables in text."""
            text = request.get("text", "")
            result = self.replace_variables(text)
            return {"original": text, "result": result}

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(VariablesExtension)
