"""
Macros Feature

User-defined text shortcuts for common phrases/actions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
from datetime import datetime


@dataclass
class Macro:
    """A text macro."""

    name: str
    trigger: str  # The shortcut (e.g., "/intro")
    content: str  # What it expands to
    description: str = ""
    enabled: bool = True
    variables: List[str] = None  # Variables in the macro (e.g., {name})

    def __post_init__(self):
        if self.variables is None:
            # Extract variables from content
            self.variables = re.findall(r"\{(\w+)\}", self.content)


class MacroManager:
    """
    Manage text macros for quick shortcuts.

    Examples:
        /intro → Expands to a long introduction
        /setting → Expands to scene description
        /action → Expands to a common action pattern
    """

    def __init__(self):
        """Initialize macro manager."""
        self.macros: Dict[str, Macro] = {}
        self.usage_history: List[tuple[str, datetime]] = []

        # Add default macros
        self._add_default_macros()

    def _add_default_macros(self) -> None:
        """Add default macros."""
        default_macros = [
            Macro(
                name="greeting",
                trigger="/hello",
                content="Hello! It's great to see you. How can I help you today?",
                description="Standard greeting",
            ),
            Macro(
                name="thinking",
                trigger="/hmm",
                content="*pauses thoughtfully* Hmm, let me think about that...",
                description="Thinking action",
            ),
            Macro(
                name="affirmation",
                trigger="/yes",
                content="I understand and agree with that.",
                description="Affirmation",
            ),
        ]

        for macro in default_macros:
            self.add_macro(macro)

    def add_macro(self, macro: Macro) -> None:
        """Add a macro."""
        self.macros[macro.trigger] = macro

    def remove_macro(self, trigger: str) -> bool:
        """Remove a macro by trigger."""
        if trigger in self.macros:
            del self.macros[trigger]
            return True
        return False

    def get_macro(self, trigger: str) -> Optional[Macro]:
        """Get a macro by trigger."""
        return self.macros.get(trigger)

    def expand_macro(
        self,
        trigger: str,
        variables: Dict[str, str] = None,
    ) -> Optional[str]:
        """
        Expand a macro trigger into its content.

        Args:
            trigger: The macro trigger (e.g., "/intro")
            variables: Variables to substitute in the macro

        Returns:
            Expanded content, or None if macro not found
        """
        macro = self.get_macro(trigger)
        if not macro or not macro.enabled:
            return None

        # Record usage
        self.usage_history.append((trigger, datetime.now()))

        # Substitute variables
        content = macro.content

        if variables:
            for var_name, var_value in variables.items():
                content = content.replace(f"{{{var_name}}}", var_value)

        return content

    def detect_and_expand(
        self,
        text: str,
        variables: Dict[str, str] = None,
    ) -> tuple[str, bool]:
        """
        Detect macro in text and expand it.

        Args:
            text: Input text
            variables: Variables for substitution

        Returns:
            Tuple of (expanded_text, was_expanded)
        """
        # Check if text starts with a macro trigger
        for trigger, macro in self.macros.items():
            if text.startswith(trigger) and macro.enabled:
                expanded = self.expand_macro(trigger, variables)
                if expanded:
                    # Keep any text after the trigger
                    remaining = text[len(trigger):].strip()
                    if remaining:
                        expanded = expanded + " " + remaining
                    return expanded, True

        return text, False

    def list_macros(
        self,
        include_disabled: bool = False,
    ) -> List[Macro]:
        """List all macros."""
        macros = list(self.macros.values())
        if not include_disabled:
            macros = [m for m in macros if m.enabled]
        return macros

    def get_usage_stats(self) -> Dict[str, int]:
        """Get macro usage statistics."""
        stats = {}
        for trigger, _ in self.usage_history:
            stats[trigger] = stats.get(trigger, 0) + 1
        return stats

    def import_from_dict(self, data: List[Dict[str, Any]]) -> int:
        """Import macros from list of dictionaries."""
        count = 0
        for macro_data in data:
            macro = Macro(
                name=macro_data.get("name", ""),
                trigger=macro_data.get("trigger", ""),
                content=macro_data.get("content", ""),
                description=macro_data.get("description", ""),
                enabled=macro_data.get("enabled", True),
            )
            self.add_macro(macro)
            count += 1
        return count

    def export_to_dict(self) -> List[Dict[str, Any]]:
        """Export macros to list of dictionaries."""
        return [
            {
                "name": m.name,
                "trigger": m.trigger,
                "content": m.content,
                "description": m.description,
                "enabled": m.enabled,
                "variables": m.variables,
            }
            for m in self.macros.values()
        ]


def create_macro_manager() -> MacroManager:
    """Factory function to create MacroManager."""
    return MacroManager()
