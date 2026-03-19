"""Chat Templates Manager

Manages pre-built chat structures and starter conversations.
Similar to SillyTavern's chat templates feature.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import uuid

from src.extensions.features.assistant.chat_history.storage import ChatMessage, ChatMetadata

logger = logging.getLogger(__name__)


@dataclass
class ChatTemplate:
    """A pre-built chat template."""
    id: str
    name: str
    description: str
    category: str  # starter, scenario, roleplay, etc.
    character_name: str
    messages: List[Dict[str, Any]]
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v or v == []}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatTemplate":
        return cls(**data)


# Built-in templates
BUILTIN_TEMPLATES = [
    ChatTemplate(
        id="starter_greeting",
        name="First Greeting",
        description="Character introduces themselves",
        category="starter",
        character_name="",
        messages=[
            {"name": "char", "is_user": False, "is_name": False, "mes": "Hello! My name is {character_name}. It's nice to meet you."},
            {"name": "user", "is_user": True, "is_name": False, "mes": "Hi! Nice to meet you too."},
        ],
        tags=["greeting", "introduction"],
        author="System",
    ),
    ChatTemplate(
        id="scenario_cafe",
        name="Cafe Meeting",
        description="Meet at a cozy cafe",
        category="scenario",
        character_name="",
        messages=[
            {"name": "narrator", "is_user": False, "is_name": True, "mes": "*You sit down at a cozy corner table in the cafe. The aroma of fresh coffee fills the air.*"},
            {"name": "char", "is_user": False, "is_name": False, "mes": "*walks over to your table* Hi there! I'm {character_name}. What can I get for you today?"},
            {"name": "user", "is_user": True, "is_name": False, "mes": "Oh, hello! Just a coffee would be nice, please."},
        ],
        tags=["scenario", "cafe", "casual"],
        author="System",
    ),
    ChatTemplate(
        id="roleplay_fantasy",
        name="Fantasy Adventure",
        description="Start a fantasy adventure roleplay",
        category="roleplay",
        character_name="",
        messages=[
            {"name": "narrator", "is_user": False, "is_name": True, "mes": "*You find yourself at the entrance of an ancient dungeon. Torches flicker on the stone walls.*"},
            {"name": "char", "is_user": False, "is_name": False, "mes": "*draws their sword* Stay close, {user}. This dungeon is rumored to be filled with treasures... and dangers."},
            {"name": "user", "is_user": True, "is_name": False, "mes": "I'm ready. Lead the way!"},
        ],
        tags=["fantasy", "adventure", "action"],
        author="System",
    ),
]


class ChatTemplateManager:
    """Manages chat templates."""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            storage_path = get_inputs_root_dir() / "chat_templates"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.templates_file = self.storage_path / "templates.json"

        self._templates: Dict[str, ChatTemplate] = {}
        self._load_templates()
        self._register_builtins()

        logger.info(f"Chat Template Manager initialized at: {self.storage_path}")

    def _load_templates(self) -> None:
        """Load custom templates from disk."""
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r') as f:
                    data = json.load(f)
                    for template_data in data:
                        template = ChatTemplate.from_dict(template_data)
                        self._templates[template.id] = template
                logger.debug(f"Loaded {len(self._templates)} custom templates")
            except Exception as e:
                logger.error(f"Failed to load templates: {e}")
                self._templates = {}

    def _save_templates(self) -> None:
        """Save custom templates to disk."""
        try:
            # Only save custom templates (not built-in)
            custom_templates = [
                t.to_dict() for t in self._templates.values()
                if t.author != "System"
            ]
            with open(self.templates_file, 'w') as f:
                json.dump(custom_templates, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")

    def _register_builtins(self) -> None:
        """Register built-in templates."""
        for template in BUILTIN_TEMPLATES:
            if template.id not in self._templates:
                self._templates[template.id] = template

    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ChatTemplate]:
        """List templates.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)

        Returns:
            List of ChatTemplates
        """
        templates = list(self._templates.values())

        if category:
            templates = [t for t in templates if t.category == category]

        if tags:
            templates = [
                t for t in templates
                if any(tag in t.tags for tag in tags)
            ]

        return templates

    def get_template(self, template_id: str) -> Optional[ChatTemplate]:
        """Get a template by ID.

        Args:
            template_id: The template ID

        Returns:
            The ChatTemplate or None
        """
        return self._templates.get(template_id)

    def create_template(
        self,
        name: str,
        description: str,
        category: str,
        messages: List[Dict[str, Any]],
        character_name: str = "",
        tags: Optional[List[str]] = None,
        author: str = "User"
    ) -> ChatTemplate:
        """Create a new template.

        Args:
            name: Template name
            description: Description
            category: Category
            messages: List of message dicts
            character_name: Optional character name
            tags: Optional tags
            author: Creator name

        Returns:
            The created ChatTemplate
        """
        template = ChatTemplate(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            category=category,
            character_name=character_name,
            messages=messages,
            tags=tags or [],
            author=author
        )

        self._templates[template.id] = template
        self._save_templates()
        logger.info(f"Created template: {template.name}")
        return template

    def apply_template(
        self,
        template_id: str,
        character_name: str = "",
        user_name: str = "You"
    ) -> List[ChatMessage]:
        """Apply a template to get chat messages.

        Args:
            template_id: The template ID
            character_name: Name to substitute for {character_name}
            user_name: Name to substitute for {user}

        Returns:
            List of ChatMessages
        """
        template = self.get_template(template_id)
        if not template:
            return []

        messages = []
        for msg_dict in template.messages:
            # Substitute placeholders
            mes = msg_dict.get("mes", "")
            mes = mes.replace("{character_name}", character_name or template.character_name)
            mes = mes.replace("{user}", user_name)

            message = ChatMessage(
                name=msg_dict.get("name", "char"),
                is_user=msg_dict.get("is_user", False),
                is_name=msg_dict.get("is_name", False),
                mes=mes,
            )
            messages.append(message)

        return messages

    def delete_template(self, template_id: str) -> bool:
        """Delete a template.

        Args:
            template_id: The template ID

        Returns:
            True if deleted, False if not found or built-in
        """
        template = self.get_template(template_id)
        if not template or template.author == "System":
            return False  # Can't delete built-in templates

        del self._templates[template_id]
        self._save_templates()
        logger.info(f"Deleted template: {template_id}")
        return True

    def export_template(self, template_id: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Export a template to a file.

        Args:
            template_id: The template ID
            output_path: Optional output path

        Returns:
            Path to exported file or None
        """
        template = self.get_template(template_id)
        if not template:
            return None

        if output_path is None:
            safe_name = template.name.replace(" ", "_").replace("/", "_")
            output_path = Path.cwd() / f"template_{safe_name}.json"

        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)

        logger.info(f"Exported template to: {output_path}")
        return output_path

    def import_template(self, source: Path) -> Optional[ChatTemplate]:
        """Import a template from a file.

        Args:
            source: Path to template file

        Returns:
            The imported ChatTemplate or None
        """
        source_path = Path(source)
        if not source_path.exists():
            return None

        try:
            with open(source_path, 'r') as f:
                data = json.load(f)

            # Generate new ID to avoid conflicts
            data["id"] = str(uuid.uuid4())[:8]

            template = ChatTemplate.from_dict(data)
            self._templates[template.id] = template
            self._save_templates()
            logger.info(f"Imported template: {template.name}")
            return template

        except Exception as e:
            logger.error(f"Failed to import template: {e}")
            return None

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return sorted(set(t.category for t in self._templates.values()))

    def get_stats(self) -> Dict[str, Any]:
        """Get template statistics."""
        templates = list(self._templates.values())
        custom_count = sum(1 for t in templates if t.author != "System")

        return {
            "total_templates": len(templates),
            "builtin_templates": len(templates) - custom_count,
            "custom_templates": custom_count,
            "categories": self.get_categories(),
            "storage_path": str(self.storage_path),
        }
