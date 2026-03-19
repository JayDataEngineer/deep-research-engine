"""Prompt Manager for Auto Studio.

Assembles prompts from modular config sections using Jinja2.
"""

import logging
from dataclasses import dataclass
from typing import Any

from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass
class AssembledPrompt:
    """An assembled prompt from config sections."""
    system: str
    user: str


class ConfigPromptManager:
    """
    Manages loading and rendering of prompts from extension config.

    Instead of separate YAML files, prompts are defined in sections
    within the extension config.yaml and assembled using Jinja2.
    """

    def __init__(self, config: dict):
        """
        Initialize the prompt manager with extension config.

        Args:
            config: The full extension config dict (from config.yaml)
        """
        self.config = config
        self.deep_agent_config = config.get("deep_agent", {})
        self.fast_mode_config = config.get("fast_mode", {})

    def assemble_prompt(self, variables: dict[str, Any], mode: str = "deep_agent") -> AssembledPrompt:
        """
        Assemble the full prompt from config sections.

        Args:
            variables: Dictionary of variable values for Jinja2 rendering
            mode: Prompt mode - "deep_agent" or "fast"

        Returns:
            AssembledPrompt with system and user prompts
        """
        if mode == "fast":
            return self._assemble_fast_prompt(variables)
        else:
            return self._assemble_deep_agent_prompt(variables)

    def _assemble_deep_agent_prompt(self, variables: dict[str, Any]) -> AssembledPrompt:
        """Assemble Deep Agent prompt from config sections."""
        # Get prompt template sections from config
        prompts = self.deep_agent_config.get("prompts", {})
        system_template = prompts.get("system", "")
        user_template = prompts.get("user", "")

        # Get all the prompt sections (these are the Jinja variables)
        prompt_sections = {
            "general_direction": self.config.get("general_direction", ""),
            "clip_breaks": self.config.get("clip_breaks", ""),
            "emotion_vectors": self.config.get("emotion_vectors", ""),
            "qwen_edit": self.config.get("qwen_edit", ""),
            "ltx2_video": self.config.get("ltx2_video", ""),
            "camera_angles": self.config.get("camera_angles", ""),
            "character_consistency": self.config.get("character_consistency", ""),
            "duration_calculation": self.config.get("duration_calculation", ""),
        }

        # Merge with user-provided variables
        all_variables = {**prompt_sections, **variables}

        # Render system prompt
        system_rendered = self._render_template(system_template, all_variables)

        # Render user prompt
        user_rendered = self._render_template(user_template, all_variables)

        return AssembledPrompt(
            system=system_rendered,
            user=user_rendered,
        )

    def _assemble_fast_prompt(self, variables: dict[str, Any]) -> AssembledPrompt:
        """Assemble Fast Mode prompt from config sections."""
        # Get prompt template sections from config
        prompts = self.fast_mode_config.get("prompts", {})
        system_template = prompts.get("system", "")
        user_template = prompts.get("user", "")

        # Get style_guidelines section for fast mode
        prompt_sections = {
            "style_guidelines": self.config.get("style_guidelines", ""),
        }

        # Merge with user-provided variables
        all_variables = {**prompt_sections, **variables}

        # Render system prompt
        system_rendered = self._render_template(system_template, all_variables)

        # Render user prompt
        user_rendered = self._render_template(user_template, all_variables)

        return AssembledPrompt(
            system=system_rendered,
            user=user_rendered,
        )

    def _render_template(self, template: str, variables: dict[str, Any]) -> str:
        """Render a Jinja2 template with variables."""
        try:
            jinja_template = Template(template)
            return jinja_template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            # Return template as-is on error
            return template
