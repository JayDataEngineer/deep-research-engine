"""
Context Templates (W++, STDP, Boostyle, etc.)

Pre-built prompt structures popular in the RP community.
These are common persona formatting styles.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path


class ContextTemplate(Enum):
    """Supported context template types."""

    WPLUS = "wplus"  # W++ format
    STDP = "stdp"  # STDP format
    BOOSTYLE = "boostyle"  # Boostyle format
    SUCCUBST = "succubst"  # Succubus format (romantic RP)
    SAYORI = "sayori"  # Sayori format
    PLAIN = "plain"  # Plain text


def _load_context_templates() -> Dict[str, Any]:
    """Load context templates from YAML.

    Returns:
        Dict with templates and metadata
    """
    try:
        import yaml
        from src.constants import BACKEND_ROOT
        backend_dir = BACKEND_ROOT
        prompts_dir = backend_dir / "prompts"
        templates_file = prompts_dir / "context_templates.yaml"

        if templates_file.exists():
            with open(templates_file) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass

    # Fallback to empty - will use hardcoded templates below
    return {}


class ContextTemplateManager:
    """
    Manage persona context templates.

    These are popular formatting styles for character personas
    in the RP community (Character.AI, SillyTavern, etc.).
    """

    # Template definitions (fallback if YAML fails to load)
    FALLBACK_TEMPLATES = {
        ContextTemplate.WPLUS: """You are {{char}}.

Details:
{{persona}}

Scenario:
{{scenario}}

Additional information:
{{description}}""",

        ContextTemplate.STDP: """[Character Persona for {{char}}]
[Name: {{char}}]
[Description: {{persona}}]
[Scenario: {{scenario}}]
[Personality: {{description}}]""",

        ContextTemplate.BOOSTYLE: """
════════════════════════════════════════
{{char}}
════════════════════════════════════════

✦ Basic Information ✦
→ Name: {{char}}
→ Description: {{description}}

✦ Personality ✦
{{persona}}

✦ Current Scenario ✦
{{scenario}}

✦ Guidelines ✦
- Stay in character at all times
- Respond naturally, not as an AI assistant
- Keep responses engaging and authentic
""",

        ContextTemplate.SUCCUBST: """【Character Profile】
Name: {{char}}

【Personality】
{{persona}}

【Appearance】
{{description}}

【Scenario】
{{scenario}}

【Notes】
- Romantic RP personality
- Affectionate and attentive
- Stay in character
""",

        ContextTemplate.SAYORI: """
◇ ────────────────────────────────────────── ◇
          {{char}}
◇ ────────────────────────────────────────── ◇

✦ Who I Am ✦
{{persona}}

✦ Where I Am ✦
{{scenario}}

✦ What I'm Like ✦
{{description}}

◇ ────────────────────────────────────────── ◇
""",

        ContextTemplate.PLAIN: """You are {{char}}.

{{persona}}

Current situation: {{scenario}}

Additional details: {{description}}""",
    }

    # Aliases
    ALIASES = {
        "w++": ContextTemplate.WPLUS,
        "wplus": ContextTemplate.WPLUS,
        "wpp": ContextTemplate.WPLUS,
        "stdp": ContextTemplate.STDP,
        "boostyle": ContextTemplate.BOOSTYLE,
        "succubst": ContextTemplate.SUCCUBST,
        "sayori": ContextTemplate.SAYORI,
        "plain": ContextTemplate.PLAIN,
    }

    def __init__(self):
        """Initialize context template manager."""
        self._yaml_data = _load_context_templates()

    def _get_templates(self) -> Dict[ContextTemplate, str]:
        """Get templates from YAML or fallback."""
        if self._yaml_data and "templates" in self._yaml_data:
            # Convert YAML format to enum dict
            templates = {}
            yaml_templates = self._yaml_data["templates"]
            for key, template_data in yaml_templates.items():
                enum_key = key.upper()
                if hasattr(ContextTemplate, enum_key):
                    templates[ContextTemplate[enum_key]] = template_data.get("content", "")
            return templates if templates else self.FALLBACK_TEMPLATES
        return self.FALLBACK_TEMPLATES

    def apply_template(
        self,
        template: ContextTemplate,
        variables: Dict[str, str],
    ) -> str:
        """
        Apply a template to format persona content.

        Args:
            template: Template to use
            variables: Dict with keys: char, persona, scenario, description

        Returns:
            Formatted persona string
        """
        # Support aliases
        if isinstance(template, str):
            template = self.ALIASES.get(template.lower(), ContextTemplate.PLAIN)

        templates = self._get_templates()
        template_str = templates.get(template, templates.get(ContextTemplate.PLAIN, ""))

        # Substitute variables
        result = template_str
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", value)

        return result

    def get_template(self, template: ContextTemplate) -> str:
        """Get the raw template string."""
        if isinstance(template, str):
            template = self.ALIASES.get(template.lower(), ContextTemplate.PLAIN)

        templates = self._get_templates()
        return templates.get(template, templates.get(ContextTemplate.PLAIN, ""))

    def list_templates(self) -> List[str]:
        """List all available templates."""
        return [t.value for t in ContextTemplate]

    def get_template_description(self, template: ContextTemplate) -> str:
        """Get description of a template."""
        # Check YAML first
        if self._yaml_data and "templates" in self._yaml_data:
            yaml_templates = self._yaml_data["templates"]
            if isinstance(template, str):
                template_key = template.lower()
            else:
                template_key = template.value.lower()
            if template_key in yaml_templates:
                return yaml_templates[template_key].get("description", "Unknown format")

        # Fallback descriptions
        descriptions = {
            ContextTemplate.WPLUS: "W++ format - Detailed structured persona format",
            ContextTemplate.STDP: "STDP format - Section-based persona format",
            ContextTemplate.BOOSTYLE: "Boostyle format - Decorative bordered format",
            ContextTemplate.SUCCUBST: "Succubus format - Romantic RP format",
            ContextTemplate.SAYORI: "Sayori format - Clean minimalist format",
            ContextTemplate.PLAIN: "Plain format - Simple unformatted",
        }

        if isinstance(template, str):
            template = self.ALIASES.get(template.lower(), ContextTemplate.PLAIN)

        return descriptions.get(template, "Unknown format")


def create_context_template_manager() -> ContextTemplateManager:
    """Factory function to create ContextTemplateManager."""
    return ContextTemplateManager()
