"""Shared Prompt Template Engine

Provides Jinja2-based prompt templating for all chat modes.
Supports character cards, world info, and advanced prompt construction.

Compatible with SillyTavern's prompt template system.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from jinja2 import (
        Environment,
        BaseLoader,
        TemplateError,
        TemplateSyntaxError,
        select_autoescape,
        StrictUndefined,
    )
    _HAS_JINJA = True
except ImportError:
    _HAS_JINJA = False

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Filters and Functions
# =============================================================================

def format_datetime(value: Union[str, datetime], format_str: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime value."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except:
            return value
    if isinstance(value, datetime):
        return value.strftime(format_str)
    return str(value)


def truncate_words(value: str, num_words: int = 50, suffix: str = "...") -> str:
    """Truncate text to a specific number of words."""
    words = value.split()
    if len(words) <= num_words:
        return value
    return " ".join(words[:num_words]) + suffix


def truncate_chars(value: str, num_chars: int = 500, suffix: str = "...") -> str:
    """Truncate text to a specific number of characters."""
    if len(value) <= num_chars:
        return value
    return value[:num_chars] + suffix


def repeat(value: Any, times: int) -> List:
    """Repeat a value multiple times."""
    return [value] * times


def get_attr(obj: Dict[str, Any], key: str, default: Any = "") -> Any:
    """Get an attribute from an object/dict safely."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def count_words(value: str) -> int:
    """Count words in text."""
    return len(value.split())


def count_chars(value: str) -> int:
    """Count characters in text."""
    return len(value)


def to_json(value: Any) -> str:
    """Convert value to JSON string."""
    return json.dumps(value, ensure_ascii=False)


def from_json(value: str) -> Any:
    """Parse JSON string."""
    return json.loads(value)


# =============================================================================
# Template Context Builders
# =============================================================================

class PromptContext:
    """Builds template context for various scenarios."""

    @staticmethod
    def for_character(character: Any, **extra_vars) -> Dict[str, Any]:
        """Build context for character-based prompting."""
        # Handle both dataclass and dict-like objects
        if hasattr(character, '__dict__'):
            data = {k: v for k, v in vars(character).items() if not k.startswith('_')}
        elif isinstance(character, dict):
            data = character
        else:
            data = {"name": str(character)}

        context = {
            "character": data,
            "char": data,  # Alias
            "now": datetime.now(),
            "today": datetime.today(),
        }
        context.update(extra_vars)
        return context

    @staticmethod
    def for_conversation(
        messages: List[Dict[str, str]],
        character: Optional[Any] = None,
        world_info: Optional[str] = None,
        **extra_vars
    ) -> Dict[str, Any]:
        """Build context for conversation prompting."""
        context = {
            "messages": messages,
            "conversation": messages,  # Alias
            "message_count": len(messages),
        }

        if character:
            if hasattr(character, '__dict__'):
                char_data = {k: v for k, v in vars(character).items() if not k.startswith('_')}
            else:
                char_data = {"name": str(character)}
            context["character"] = char_data
            context["char"] = char_data  # Alias

        if world_info:
            context["world_info"] = world_info
            context["lorebook"] = world_info  # Alias

        context.update(extra_vars)
        return context

    @staticmethod
    def for_rpg_system(system: str, **extra_vars) -> Dict[str, Any]:
        """Build context for RPG system prompting."""
        context = {
            "system": system,
            "now": datetime.now(),
        }
        context.update(extra_vars)
        return context


# =============================================================================
# Template Engine
# =============================================================================

class PromptTemplateEngine:
    """Jinja2-based prompt template engine.

    Features:
    - Variable substitution with {{ variable }}
    - Conditionals with {% if %} ... {% endif %}
    - Loops with {% for %} ... {% endfor %}
    - Custom filters and functions
    - Template inheritance and includes
    - Strict mode for catching undefined variables
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        strict: bool = False,
        auto_escape: bool = False,
    ):
        """Initialize the template engine.

        Args:
            template_dir: Directory for template files (optional)
            strict: Enable strict undefined variable checking
            auto_escape: Enable auto-escaping (not needed for prompts)
        """
        if not _HAS_JINJA:
            raise RuntimeError(
                "Jinja2 is not installed. "
                "Install it with: pip install jinja2"
            )

        self.template_dir = Path(template_dir) if template_dir else None
        self.strict = strict

        # Create Jinja environment
        env_kwargs = {
            "loader": BaseLoader(),
            "trim_blocks": True,
            "lstrip_blocks": True,
        }

        if strict:
            env_kwargs["undefined"] = StrictUndefined

        if auto_escape:
            env_kwargs["autoescape"] = select_autoescape()

        self.env = Environment(**env_kwargs)

        # Register custom filters
        self.env.filters["datetime"] = format_datetime
        self.env.filters["truncate_words"] = truncate_words
        self.env.filters["truncate_chars"] = truncate_chars
        self.env.filters["repeat"] = repeat
        self.env.filters["get"] = get_attr
        self.env.filters["count_words"] = count_words
        self.env.filters["count_chars"] = count_chars
        self.env.filters["to_json"] = to_json
        self.env.filters["from_json"] = from_json

        # Add global helper functions
        self.env.globals["now"] = datetime.now
        self.env.globals["today"] = datetime.today
        self.env.globals["range"] = range
        self.env.globals["len"] = len

        logger.info("PromptTemplateEngine initialized with custom filters")

    def render(
        self,
        template: str,
        variables: Dict[str, Any],
        strict: Optional[bool] = None,
    ) -> str:
        """Render a template string with variables.

        Args:
            template: Template string with Jinja syntax
            variables: Variables for substitution
            strict: Override default strict mode

        Returns:
            Rendered string

        Raises:
            TemplateError: If template syntax is invalid
            TemplateSyntaxError: If template has syntax errors
        """
        try:
            jinja_template = self.env.from_string(
                template,
                globals=variables,
            )

            # Use template-level strict setting if provided
            old_undefined = None
            if strict is not None:
                if strict:
                    self.env.undefined = StrictUndefined
                else:
                    old_undefined = self.env.undefined
                    self.env.undefined = lambda **kwargs: ""

            result = jinja_template.render(**variables)

            # Restore old undefined if we changed it
            if old_undefined is not None:
                self.env.undefined = old_undefined

            logger.debug(f"Template rendered successfully, length: {len(result)}")
            return result

        except (TemplateError, TemplateSyntaxError) as e:
            logger.error(f"Template rendering error: {e}")
            if strict or self.strict:
                raise
            # Return original template on error if not strict
            return template

    def render_template_file(
        self,
        template_path: Union[str, Path],
        variables: Dict[str, Any],
        strict: Optional[bool] = None,
    ) -> str:
        """Render a template from file.

        Args:
            template_path: Path to template file
            variables: Variables for substitution
            strict: Override default strict mode

        Returns:
            Rendered string
        """
        template_path = Path(template_path)

        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            if strict or self.strict:
                raise FileNotFoundError(f"Template not found: {template_path}")
            return ""

        # Read template file
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        return self.render(template_content, variables, strict=strict)

    def render_character_prompt(
        self,
        character: Any,
        template: Optional[str] = None,
        additional_vars: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render a character-specific prompt.

        Args:
            character: Character object or dict
            template: Template string (uses default if None)
            additional_vars: Additional variables

        Returns:
            Rendered character prompt
        """
        if template is None:
            # Default character template
            template = """You are {{ character.name }}.

{% if character.description %}
{{ character.description }}
{% endif %}

{% if character.personality %}
Personality: {{ character.personality }}
{% endif %}

{% if character.scenario %}
Current Scenario: {{ character.scenario }}
{% endif %}

{% if character.mes_example %}
Examples:
{{ character.mes_example }}
{% endif %}

Stay in character at all times. Respond as {{ character.name }} would."""

        context = PromptContext.for_character(character, **(additional_vars or {}))
        return self.render(template, context)

    def render_with_messages(
        self,
        template: str,
        messages: List[Dict[str, str]],
        character: Optional[Any] = None,
        world_info: Optional[str] = None,
        **extra_vars
    ) -> str:
        """Render a template with conversation context.

        Args:
            template: Template string
            messages: Conversation messages
            character: Optional character data
            world_info: Optional world info content
            **extra_vars: Additional variables

        Returns:
            Rendered prompt
        """
        context = PromptContext.for_conversation(
            messages, character, world_info, **extra_vars
        )
        return self.render(template, context)

    def validate_template(self, template: str) -> tuple[bool, Optional[str]]:
        """Validate a template string.

        Args:
            template: Template string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.env.from_string(template)
            return True, None
        except (TemplateError, TemplateSyntaxError) as e:
            return False, str(e)


# =============================================================================
# Template Registry
# =============================================================================

class TemplateRegistry:
    """Registry for named prompt templates."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the template registry.

        Args:
            base_dir: Base directory for template storage
        """
        if base_dir is None:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            base_dir = get_inputs_root_dir()

        self.base_dir = Path(base_dir)
        self.templates_dir = self.base_dir / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        self._cached_templates: Dict[str, str] = {}

        logger.info(f"TemplateRegistry initialized at: {self.templates_dir}")

    def register(self, name: str, template: str) -> None:
        """Register a named template.

        Args:
            name: Template name
            template: Template content
        """
        self._cached_templates[name] = template

        # Optionally save to file
        filepath = self.templates_dir / f"{name}.jinja2"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(template)

        logger.info(f"Registered template: {name}")

    def get(self, name: str) -> Optional[str]:
        """Get a registered template.

        Args:
            name: Template name

        Returns:
            Template content or None if not found
        """
        # Check cache first
        if name in self._cached_templates:
            return self._cached_templates[name]

        # Try loading from file
        filepath = self.templates_dir / f"{name}.jinja2"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                template = f.read()
            self._cached_templates[name] = template
            return template

        return None

    def list_templates(self) -> List[str]:
        """List all available template names."""
        templates = set(self._cached_templates.keys())

        # Add templates from directory
        for filepath in self.templates_dir.glob("*.jinja2"):
            templates.add(filepath.stem)

        return sorted(templates)

    def delete(self, name: str) -> bool:
        """Delete a registered template.

        Args:
            name: Template name

        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        if name in self._cached_templates:
            del self._cached_templates[name]

        # Remove file
        filepath = self.templates_dir / f"{name}.jinja2"
        if filepath.exists():
            filepath.unlink()
            return True

        return False


# =============================================================================
# Singleton Instances
# =============================================================================

_template_engine: Optional[PromptTemplateEngine] = None
_template_registry: Optional[TemplateRegistry] = None


def get_template_engine(
    template_dir: Optional[Path] = None,
    strict: bool = False,
    force_new: bool = False,
) -> PromptTemplateEngine:
    """Get the singleton template engine instance.

    Args:
        template_dir: Directory for template files
        strict: Enable strict mode
        force_new: Force creation of new instance

    Returns:
        PromptTemplateEngine instance
    """
    global _template_engine

    if force_new or _template_engine is None:
        _template_engine = PromptTemplateEngine(template_dir=template_dir, strict=strict)

    return _template_engine


def get_template_registry(base_dir: Optional[Path] = None) -> TemplateRegistry:
    """Get the singleton template registry instance.

    Args:
        base_dir: Optional custom base directory

    Returns:
        TemplateRegistry instance
    """
    global _template_registry

    if _template_registry is None:
        _template_registry = TemplateRegistry(base_dir=base_dir)

    return _template_registry


# =============================================================================
# Convenience Functions
# =============================================================================

def render_prompt(
    template: str,
    variables: Dict[str, Any],
    strict: bool = False,
) -> str:
    """Quick render function using the singleton engine.

    Args:
        template: Template string
        variables: Template variables
        strict: Raise on errors

    Returns:
        Rendered string
    """
    engine = get_template_engine()
    return engine.render(template, variables, strict=strict)


def render_character(character: Any, template: Optional[str] = None) -> str:
    """Render a character prompt using the singleton engine.

    Args:
        character: Character data
        template: Optional custom template

    Returns:
        Rendered prompt
    """
    engine = get_template_engine()
    return engine.render_character_prompt(character, template)
