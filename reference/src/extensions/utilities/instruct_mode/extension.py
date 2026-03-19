"""Instruct Mode Extension

Instruct mode template management.
Provides CLI tools for managing instruct mode presets/templates.

CLI Usage:
    uv run backend/manage.py instruct list
    uv run backend/manage.py instruct get <name>
    uv run backend/manage.py instruct create <name>
    uv run backend/manage.py instruct set <name>
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


@dataclass
class InstructTemplate:
    """An instruct mode template."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    enabled: bool = True

    # Template strings
    system_prompt: str = ""  # System message
    user_template: str = "{{user}}"  # How to format user messages
    assistant_template: str = "{{char}}"  # How to format assistant messages
    stop_sequence: str = ""  # Stop sequence for generation

    # Instruct mode settings
    wrap: bool = True  # Whether to wrap messages in templates
    marker_enabled: bool = True  # Whether to use markers

    # Position control
    system_prompt_position: str = "top"  # "top" or "bottom"

    # Advanced settings
    include_names: bool = True  # Include names in wrapped messages
    context_template: str = ""  # Optional context template
    examples_template: str = ""  # Optional examples template

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructTemplate":
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class InstructModeExtension(LogicExtension):
    """Instruct Mode extension - CLI wrapper for instruct mode template management.

    This extension provides CLI access to instruct mode template management
    for defining how the model behaves in instruct mode.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._storage_dir = None
        self._templates: List[InstructTemplate] = []
        self._current_template: Optional[str] = None

    async def initialize(self, context):
        """Initialize the instruct mode extension."""
        await super().initialize(context)

        # Set up storage directory
        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / "instruct_mode"
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / "instruct_mode"
            )

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Load templates
        self._load_templates()
        self._create_default_templates()

        logger.info(f"Instruct Mode extension initialized with {len(self._templates)} templates")

    async def shutdown(self):
        """Shutdown the extension."""
        self._save_templates()

    async def process(
        self,
        input_data: dict,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process CLI commands.

        CLI Commands:
            list                    - List all instruct mode templates
            get <name>              - Get template details
            create <name>           - Create a new template
            delete <name>           - Delete a template
            set <name>              - Set current template
            get-current             - Get current template
            enable <name>           - Enable a template
            disable <name>          - Disable a template
            update <name>           - Update template settings
            import <file>           - Import template from file
            export <name> <file>    - Export template to file
        """
        args = input_data.get("args", [])
        command = input_data.get("command", "")

        # Handle CLI command routing
        if command in ("instruct", "instructmode", "im") or args:
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
                yield {"error": "Please provide a template name"}
                return
            yield await self._cmd_get(args[0])
        elif subcommand == "create":
            if not args:
                yield {"error": "Please provide a template name"}
                return
            yield await self._cmd_create(args[0])
        elif subcommand in ("delete", "remove", "rm", "del"):
            if not args:
                yield {"error": "Please provide a template name"}
                return
            yield await self._cmd_delete(args[0])
        elif subcommand in ("set", "use", "select"):
            if not args:
                yield {"error": "Please provide a template name"}
                return
            yield await self._cmd_set(args[0])
        elif subcommand in ("get-current", "current"):
            yield await self._cmd_get_current()
        elif subcommand == "enable":
            if not args:
                yield {"error": "Please provide a template name"}
                return
            yield await self._cmd_enable(args[0])
        elif subcommand == "disable":
            if not args:
                yield {"error": "Please provide a template name"}
                return
            yield await self._cmd_disable(args[0])
        elif subcommand == "update":
            if not args:
                yield {"error": "Please provide a template name"}
                return
            yield await self._cmd_update(args[0])
        elif subcommand == "import":
            if not args:
                yield {"error": "Please provide a file path"}
                return
            yield await self._cmd_import(args[0])
        elif subcommand == "export":
            if len(args) < 2:
                yield {"error": "Please provide template name and output file"}
                return
            yield await self._cmd_export(args[0], args[1])
        else:
            yield {"error": f"Unknown subcommand: {subcommand}"}
            yield self._help_text()

    async def _handle_api_operation(
        self,
        operation: Optional[str],
        input_data: dict
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle API operations."""
        if operation == "list_templates":
            yield {"templates": [t.to_dict() for t in self._templates]}
        elif operation == "get_template":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            template = self._get_template(name)
            if template:
                yield {"template": template.to_dict()}
            else:
                yield {"error": "Template not found"}
        elif operation == "create_template":
            data = input_data.get("template", {})
            template = InstructTemplate.from_dict(data)
            self._templates.append(template)
            self._save_templates()
            yield {"template": template.to_dict()}
        elif operation == "delete_template":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            success = self._delete_template(name)
            if success:
                self._save_templates()
                yield {"success": True}
            else:
                yield {"error": "Template not found"}
        elif operation == "set_template":
            name = input_data.get("name")
            if not name:
                yield {"error": "name required"}
                return
            success = self._set_template(name)
            if success:
                self._save_templates()
                yield {"current": self._current_template}
            else:
                yield {"error": "Template not found"}
        elif operation == "get_current":
            yield {"current": self._current_template}
        elif operation == "update_template":
            name = input_data.get("name")
            data = input_data.get("template", {})
            if not name:
                yield {"error": "name required"}
                return
            template = self._update_template(name, data)
            if template:
                self._save_templates()
                yield {"template": template.to_dict()}
            else:
                yield {"error": "Template not found"}
        else:
            yield {"error": f"Unknown operation: {operation}"}

    # ========================================================================
    # Storage Methods
    # ========================================================================

    def _load_templates(self):
        """Load templates from storage."""
        index_file = self._storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                    self._templates = [InstructTemplate.from_dict(t) for t in data.get("templates", [])]
                    self._current_template = data.get("current")
            except Exception as e:
                logger.error(f"Error loading instruct templates: {e}")

    def _save_templates(self):
        """Save templates to storage."""
        index_file = self._storage_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump({
                "templates": [t.to_dict() for t in self._templates],
                "current": self._current_template,
            }, f, indent=2)

    def _create_default_templates(self):
        """Create default instruct mode templates if none exist."""
        if self._templates:
            return

        # Default: Alpaca style
        alpaca = InstructTemplate(
            name="Alpaca",
            description="Alpaca-style instruction format",
            system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            user_template="### Instruction:\n{{user}}\n\n### Response:\n",
            assistant_template="{{char}}",
            stop_sequence="###",
            wrap=True,
            marker_enabled=True,
            include_names=False,
        )

        # Default: ChatML style
        chatml = InstructTemplate(
            name="ChatML",
            description="ChatML-style instruction format",
            system_prompt="You are a helpful assistant.",
            user_template="<|im_start|>user\n{{user}}<|im_end|>\n",
            assistant_template="<|im_start|>assistant\n{{char}}<|im_end|>\n",
            stop_sequence="<|im_end|>",
            wrap=True,
            marker_enabled=True,
            include_names=False,
        )

        # Default: Vicuna style
        vicuna = InstructTemplate(
            name="Vicuna",
            description="Vicuna-style instruction format",
            system_prompt="A chat between a curious user and an artificial intelligence assistant.",
            user_template="USER: {{user}}\n",
            assistant_template="ASSISTANT: {{char}}\n",
            stop_sequence="",
            wrap=True,
            marker_enabled=False,
            include_names=True,
        )

        self._templates.extend([alpaca, chatml, vicuna])
        self._current_template = "Alpaca"
        self._save_templates()

        logger.info("Created default instruct mode templates")

    # ========================================================================
    # Template Management
    # ========================================================================

    def _get_template(self, name: str) -> Optional[InstructTemplate]:
        """Get a template by name."""
        for template in self._templates:
            if template.name.lower() == name.lower():
                return template
        return None

    def _set_template(self, name: str) -> bool:
        """Set the current template."""
        template = self._get_template(name)
        if template:
            self._current_template = template.name
            self._save_templates()
            return True
        return False

    def _delete_template(self, name: str) -> bool:
        """Delete a template."""
        for i, template in enumerate(self._templates):
            if template.name.lower() == name.lower():
                # Don't allow deleting the current template
                if template.name == self._current_template:
                    logger.warning("Cannot delete current template")
                    return False

                self._templates.pop(i)
                return True
        return False

    def _update_template(self, name: str, data: Dict[str, Any]) -> Optional[InstructTemplate]:
        """Update a template."""
        template = self._get_template(name)
        if template:
            for key, value in data.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            template.updated_at = datetime.now()
            return template
        return None

    # ========================================================================
    # CLI Commands
    # ========================================================================

    async def _cmd_list(self) -> Dict[str, Any]:
        """List all templates."""
        try:
            if not self._templates:
                return {
                    "output": "No instruct mode templates found.",
                    "data": {"templates": []}
                }

            lines = [f"\nInstruct Mode Templates ({len(self._templates)} total):"]

            for template in self._templates:
                current_marker = " [CURRENT]" if template.name == self._current_template else ""
                status = "✓" if template.enabled else "✗"
                lines.append(f"\n  [{status}] {template.name}{current_marker}")
                lines.append(f"      Description: {template.description}")
                lines.append(f"      System Prompt: {template.system_prompt[:50]}..." if len(template.system_prompt) > 50 else f"      System Prompt: {template.system_prompt}")

            return {
                "output": "\n".join(lines),
                "data": {"templates": [t.to_dict() for t in self._templates]},
            }
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return {"error": str(e)}

    async def _cmd_get(self, name: str) -> Dict[str, Any]:
        """Get template details."""
        template = self._get_template(name)
        if template:
            lines = [f"\nTemplate: {template.name}"]
            lines.append(f"Description: {template.description}")
            lines.append(f"Status: {'Enabled' if template.enabled else 'Disabled'}")
            lines.append(f"System Prompt: {template.system_prompt}")
            lines.append(f"User Template: {template.user_template}")
            lines.append(f"Assistant Template: {template.assistant_template}")
            lines.append(f"Stop Sequence: {template.stop_sequence or 'None'}")
            lines.append(f"Wrap: {template.wrap}")
            lines.append(f"Marker Enabled: {template.marker_enabled}")
            lines.append(f"Include Names: {template.include_names}")
            lines.append(f"Position: {template.system_prompt_position}")

            return {
                "output": "\n".join(lines),
                "data": {"template": template.to_dict()},
            }
        return {"error": f"Template not found: {name}"}

    async def _cmd_create(self, name: str) -> Dict[str, Any]:
        """Create a new template."""
        if self._get_template(name):
            return {"error": f"Template already exists: {name}"}

        template = InstructTemplate(name=name)
        self._templates.append(template)
        self._save_templates()

        return {
            "output": f"Template created: {name}",
            "data": {"template": template.to_dict()},
        }

    async def _cmd_delete(self, name: str) -> Dict[str, Any]:
        """Delete a template."""
        success = self._delete_template(name)
        if success:
            self._save_templates()
            return {"output": f"Template deleted: {name}"}
        return {"error": f"Template not found or cannot be deleted: {name}"}

    async def _cmd_set(self, name: str) -> Dict[str, Any]:
        """Set current template."""
        success = self._set_template(name)
        if success:
            return {
                "output": f"Current template set to: {name}",
                "data": {"current": self._current_template},
            }
        return {"error": f"Template not found: {name}"}

    async def _cmd_get_current(self) -> Dict[str, Any]:
        """Get current template."""
        if self._current_template:
            template = self._get_template(self._current_template)
            if template:
                return {
                    "output": f"Current template: {template.name}",
                    "data": {"template": template.to_dict()},
                }
        return {
            "output": "No template set",
            "data": {"current": None},
        }

    async def _cmd_enable(self, name: str) -> Dict[str, Any]:
        """Enable a template."""
        template = self._get_template(name)
        if template:
            template.enabled = True
            self._save_templates()
            return {"output": f"Template enabled: {name}"}
        return {"error": f"Template not found: {name}"}

    async def _cmd_disable(self, name: str) -> Dict[str, Any]:
        """Disable a template."""
        template = self._get_template(name)
        if template:
            template.enabled = False
            self._save_templates()
            return {"output": f"Template disabled: {name}"}
        return {"error": f"Template not found: {name}"}

    async def _cmd_update(self, name: str) -> Dict[str, Any]:
        """Update a template (simple mode - just update description)."""
        template = self._get_template(name)
        if template:
            # For now, just update the timestamp
            template.updated_at = datetime.now()
            self._save_templates()
            return {"output": f"Template updated: {name}"}
        return {"error": f"Template not found: {name}"}

    async def _cmd_import(self, file_path: str) -> Dict[str, Any]:
        """Import a template from file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                template = InstructTemplate.from_dict(data)
                self._templates.append(template)
                self._save_templates()
                return {
                    "output": f"Template imported: {template.name}",
                    "data": {"template": template.to_dict()},
                }
        except FileNotFoundError:
            return {"error": f"File not found: {file_path}"}
        except Exception as e:
            return {"error": str(e)}

    async def _cmd_export(self, name: str, output_path: str) -> Dict[str, Any]:
        """Export a template to file."""
        template = self._get_template(name)
        if template:
            try:
                with open(output_path, "w") as f:
                    json.dump(template.to_dict(), f, indent=2)
                return {"output": f"Template exported to: {output_path}"}
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Template not found: {name}"}

    def _help_text(self) -> Dict[str, str]:
        """Return help text."""
        return {
            "output": """
Instruct Mode CLI

Commands:
    list                    List all instruct mode templates
    get <name>              Get template details
    create <name>           Create a new template
    delete <name>           Delete a template
    set <name>              Set current template
    get-current             Get current template
    enable <name>           Enable a template
    disable <name>          Disable a template
    update <name>           Update template settings
    import <file>           Import template from file
    export <name> <file>    Export template to file

Examples:
    uv run backend/manage.py instruct list
    uv run backend/manage.py instruct set Alpaca
    uv run backend/manage.py instruct create MyTemplate
    uv run backend/manage.py instruct export Alpaca alpaca.json

Default Templates:
    Alpaca - Alpaca-style instruction format
    ChatML - ChatML-style instruction format
    Vicuna - Vicuna-style instruction format
"""
        }

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def router(self):
        """Get the FastAPI router for this extension."""
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/instruct", tags=["instruct"])

        @router.get("/")
        async def list_templates():
            """List all templates."""
            return {"templates": [t.to_dict() for t in self._templates]}

        @router.get("/current")
        async def get_current():
            """Get current template."""
            return {"current": self._current_template}

        @router.put("/current")
        async def set_template(request: dict):
            """Set current template."""
            name = request.get("name")
            if not name:
                return {"error": "name required"}, 400
            success = self._set_template(name)
            if success:
                return {"current": self._current_template}
            return {"error": "Template not found"}, 404

        @router.get("/{name}")
        async def get_template(name: str):
            """Get template details."""
            template = self._get_template(name)
            if template:
                return {"template": template.to_dict()}
            return {"error": "Template not found"}, 404

        @router.post("/")
        async def create_template(request: dict):
            """Create a new template."""
            data = request.get("template", {})
            template = InstructTemplate.from_dict(data)
            self._templates.append(template)
            self._save_templates()
            return {"template": template.to_dict()}

        @router.delete("/{name}")
        async def delete_template(name: str):
            """Delete a template."""
            success = self._delete_template(name)
            if success:
                self._save_templates()
                return {"success": True}
            return {"error": "Template not found"}, 404

        return router

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(InstructModeExtension)
