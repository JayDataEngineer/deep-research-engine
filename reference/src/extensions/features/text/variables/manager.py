"""Variables Manager

Manages variable functionality with Jinja2 templating.
Refactored to use BaseManager and StorageManager.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

from jinja2 import Environment, BaseLoader, TemplateSyntaxError
from jinja2.filters import FILTERS

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class Variable(BaseItem):
    """A variable definition."""

    id: str
    name: str
    value: str
    description: str = ""
    scope: str = "global"  # global, chat, character

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Variable":
        return cls(**data)


class VariableManager(BaseManager):
    """Manages variable functionality with Jinja2 templating.

    Supports SillyTavern-style variable templating:
    - Variables: {{variable_name}}
    - Conditionals: {% if variable %}...{% endif %}
    - Loops: {% for item in items %}...{% endfor %}
    - Filters: {{variable|upper}}, {{value|default("default")}}
    """

    # Default variables (actual values, not templates)
    # These can be overridden at runtime via context
    DEFAULT_VARIABLES = {
        "char": Variable(
            id="char",
            name="char",
            value="Character",  # Resolved value, not a template
            description="Current character name",
        ),
        "user": Variable(
            id="user",
            name="user",
            value="You",  # Resolved value, not a template
            description="Current user name",
        ),
        "date": Variable(
            id="date",
            name="date",
            value="{{current_date}}",  # Dynamic - keeps template
            description="Current date",
        ),
        "time": Variable(
            id="time",
            name="time",
            value="{{current_time}}",  # Dynamic - keeps template
            description="Current time",
        ),
        "now": Variable(
            id="now",
            name="now",
            value="{{now}}",  # Dynamic - keeps template
            description="Current datetime",
        ),
    }

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the variable manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="variables", storage_file="variables.json")

        # Load defaults if empty
        if not self._items:
            self._load_defaults()

        # Initialize Jinja2 environment
        self._jinja_env = Environment(loader=BaseLoader())
        self._setup_custom_filters()

    def _load_defaults(self):
        """Load default variables."""
        for var in self.DEFAULT_VARIABLES.values():
            if var.id not in self._items:
                self._items[var.id] = var
        self._save()

    def _setup_custom_filters(self):
        """Setup custom Jinja2 filters."""
        # Add common SillyTavern-style filters

        # String manipulation filters
        def uppercase(s):
            return str(s).upper() if s else s

        def lowercase(s):
            return str(s).lower() if s else s

        def capitalize(s):
            return str(s).capitalize() if s else s

        def title_case(s):
            return str(s).title() if s else s

        def trim(s):
            return str(s).strip() if s else s

        def reverse(s):
            return str(s)[::-1] if s else s

        # Numeric filters
        def abs_filter(value):
            return abs(value) if value is not None else 0

        def round_filter(value, precision=0):
            try:
                return round(float(value), precision)
            except (TypeError, ValueError):
                return value

        # List filters
        def length_filter(value):
            try:
                return len(value)
            except TypeError:
                return 0

        def first_filter(value):
            try:
                return value[0] if value else None
            except (TypeError, IndexError):
                return None

        def last_filter(value):
            try:
                return value[-1] if value else None
            except (TypeError, IndexError):
                return None

        def join_filter(value, separator=""):
            try:
                return separator.join(str(v) for v in value)
            except TypeError:
                return str(value)

        # Random filter
        def random_filter(value):
            import random
            try:
                return random.choice(list(value))
            except (TypeError, IndexError):
                return None

        # Date/time filters
        def date_filter(value, format_str="%Y-%m-%d"):
            if value:
                try:
                    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
                    return dt.strftime(format_str)
                except Exception:
                    return str(value)
            return datetime.now().strftime(format_str)

        # Default filter
        def default_filter(value, default_value=""):
            return value if value not in (None, "") else default_value

        # Register filters
        self._jinja_env.filters['upper'] = uppercase
        self._jinja_env.filters['lower'] = lowercase
        self._jinja_env.filters['capitalize'] = capitalize
        self._jinja_env.filters['title'] = title_case
        self._jinja_env.filters['trim'] = trim
        self._jinja_env.filters['reverse'] = reverse
        self._jinja_env.filters['abs'] = abs_filter
        self._jinja_env.filters['round'] = round_filter
        self._jinja_env.filters['length'] = length_filter
        self._jinja_env.filters['len'] = length_filter
        self._jinja_env.filters['first'] = first_filter
        self._jinja_env.filters['last'] = last_filter
        self._jinja_env.filters['join'] = join_filter
        self._jinja_env.filters['random'] = random_filter
        self._jinja_env.filters['date'] = date_filter
        self._jinja_env.filters['default'] = default_filter

    def _get_item_class(self):
        """Get the Variable item class."""
        return Variable

    # ========================================================================
    # Variable-specific methods
    # ========================================================================

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable by name.

        Args:
            name: Variable name

        Returns:
            Variable or None
        """
        return self._items.get(name)

    def get_variable_value(self, name: str) -> Optional[str]:
        """Get a variable's value.

        Args:
            name: Variable name

        Returns:
            Variable value or None
        """
        var = self._items.get(name)
        return var.value if var else None

    def set_variable(self, name: str, value: str, description: str = "", scope: str = "global") -> bool:
        """Set a variable.

        Args:
            name: Variable name
            value: Variable value
            description: Optional description
            scope: Variable scope

        Returns:
            True if set
        """
        var = Variable(
            id=StorageManager.generate_id_from_name(name),
            name=name,
            value=value,
            description=description,
            scope=scope,
        )

        # Update if exists, add if not
        if var.id in self._items:
            return self.update_item(var)
        return self.add_item(var)

    def delete_variable(self, name: str) -> bool:
        """Delete a variable (won't delete built-ins).

        Args:
            name: Variable name

        Returns:
            True if deleted
        """
        if name in self.DEFAULT_VARIABLES:
            return False

        return self.delete_item(name)

    def replace_variables(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Replace variables in text using Jinja2 templating.

        Args:
            text: Text to replace variables in
            context: Optional context dict for additional variables

        Returns:
            Text with variables replaced
        """
        # Build the template context
        template_context = {}

        # Add runtime values first (lowest precedence)
        runtime_values = {
            'now': datetime.now().isoformat(),
            'current_date': datetime.now().strftime("%Y-%m-%d"),
            'current_time': datetime.now().strftime("%H:%M:%S"),
            'character_name': 'Character',
            'user_name': 'You',
            'TIME': datetime.now().strftime("%H:%M:%S"),
            'DATE': datetime.now().strftime("%Y-%m-%d"),
            'NOW': datetime.now().isoformat(),
        }
        template_context.update(runtime_values)

        # Add stored variable values (these override runtime defaults)
        # If a variable's value is itself a template, resolve it first
        for var in self._items.values():
            value = var.value
            # Check if the value looks like a template
            if value and '{{' in value and value.startswith('{{'):
                # Try to resolve it as a template
                try:
                    resolved = self._jinja_env.from_string(value).render(**runtime_values)
                    template_context[var.name] = resolved
                except Exception:
                    # If resolution fails, use the literal value
                    template_context[var.name] = value
            else:
                template_context[var.name] = value

        # Add runtime context (highest precedence - overrides everything)
        if context:
            template_context.update(context)

        try:
            # Use Jinja2 for templating
            template = self._jinja_env.from_string(text)
            result = template.render(**template_context)
            return result
        except TemplateSyntaxError as e:
            logger.warning(f"Jinja2 template syntax error: {e}")
            # Fall back to simple replacement
            return self._simple_replace(text, template_context)
        except Exception as e:
            logger.warning(f"Template rendering error: {e}")
            # Fall back to simple replacement
            return self._simple_replace(text, template_context)

    def _simple_replace(self, text: str, context: Dict[str, Any]) -> str:
        """Simple fallback replacement without Jinja2 features.

        Args:
            text: Text to process
            context: Variables

        Returns:
            Text with simple {{var}} replacements
        """
        pattern = r'\{\{(\w+)\}\}'

        def replacer(match):
            var_name = match.group(1)
            value = context.get(var_name, match.group(0))
            # If value is a dict, try to get a 'value' key
            if isinstance(value, dict):
                value = value.get('value', match.group(0))
            return str(value)

        return re.sub(pattern, replacer, text)

    def get_variables_by_scope(self, scope: str) -> List[Variable]:
        """Get variables by scope.

        Args:
            scope: Scope to filter by

        Returns:
            List of variables
        """
        return [v for v in self._items.values() if v.scope == scope]

    def search_variables(self, query: str) -> List[Variable]:
        """Search variables by name or description.

        Args:
            query: Search query

        Returns:
            Matching variables
        """
        query = query.lower()
        return [
            v for v in self._items.values()
            if query in v.name.lower() or query in v.description.lower()
        ]