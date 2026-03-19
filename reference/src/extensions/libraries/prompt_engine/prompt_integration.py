"""
SillyTavern Prompt Manager Integration with Graph System

This module provides integration between the SillyTavern-compatible
PromptManager and the existing graph/companion brain system.

It bridges two systems:
- PromptManager: SillyTavern-compatible prompt ordering/injection
- PersonalityLayer/PromptStack: Existing prompt system

The integration allows the graph to use PromptManager prompts while
maintaining backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .prompt_manager import get_prompt_manager, PromptManager
from .prompt_models import Prompt, PromptOrderEntry, InjectionPosition

logger = logging.getLogger(__name__)


class PromptManagerIntegration:
    """
    Integration layer for SillyTavern Prompt Manager.

    This class provides methods to:
    1. Load prompts from PromptManager into PersonalityLayer
    2. Export PersonalityLayer prompts to PromptManager format
    3. Build system prompts with proper ordering/injection
    """

    def __init__(
        self,
        prompt_manager: Optional[PromptManager] = None,
        enabled: bool = True,
    ):
        """
        Initialize the integration.

        Args:
            prompt_manager: PromptManager instance (uses global if None)
            enabled: Whether integration is enabled
        """
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.enabled = enabled

    def inject_into_personality_layer(
        self,
        personality: PersonalityLayer,
        character_id: str,
        character_data: Optional[Dict[str, Any]] = None,
        generation_type: str = "normal",
    ) -> None:
        """
        Inject PromptManager prompts into PersonalityLayer.

        This modifies the PersonalityLayer's prompt_stack to include
        prompts from PromptManager while preserving existing prompts.

        Args:
            personality: Target PersonalityLayer to inject into
            character_id: Character ID for prompt lookup
            character_data: Optional character data for marker resolution
            generation_type: Generation type (normal, continue, quiet)
        """
        if not self.enabled:
            return

        try:
            # Build injection context from PromptManager
            injection_context = self.prompt_manager.build_injection_context(
                character_id=character_id,
                character_data=character_data,
                persona_data=None,  # Could extract from personality
                generation_type=generation_type,
            )

            logger.info(
                f"[PromptIntegration] Injecting {len(injection_context)} prompts "
                f"into personality layer for character {character_id}"
            )

            # Inject each prompt into the prompt stack
            for prompt_item in injection_context:
                identifier = prompt_item["identifier"]
                role = prompt_item["role"]
                content = prompt_item["content"]
                depth = prompt_item["depth"]

                # Convert depth: -1 (relative) -> "before", other -> depth number
                if depth == -1:
                    # Relative injection - add to beginning
                    injection_position = "before"
                    injection_depth = 0
                else:
                    # Absolute injection - at specific depth
                    injection_position = "after"
                    injection_depth = depth

                # Create PromptLayer and add to stack
                prompt_layer = PromptLayer(
                    content=content,
                    role=role,
                    injection_depth=injection_depth,
                    injection_position=injection_position,
                    identifier=identifier,
                )

                personality.prompt_stack.add_layer(prompt_layer)

            logger.debug(
                f"[PromptIntegration] Successfully injected prompts. "
                f"New stack size: {len(personality.prompt_stack.layers)}"
            )

        except Exception as e:
            logger.error(f"[PromptIntegration] Failed to inject prompts: {e}")
            # Don't raise - allow graph to continue with default prompts

    def export_personality_to_manager(
        self,
        personality: PersonalityLayer,
        character_id: str,
        character_name: str,
    ) -> bool:
        """
        Export PersonalityLayer prompts to PromptManager.

        This converts PersonalityLayer prompts to SillyTavern format
        and stores them in the PromptManager.

        Args:
            personality: Source PersonalityLayer
            character_id: Target character ID
            character_name: Character display name

        Returns:
            True if export succeeded
        """
        if not self.enabled:
            return False

        try:
            # Convert prompt stack to PromptManager format
            prompts = []
            order_entries = []

            for layer in personality.prompt_stack.layers:
                # Create Prompt from PromptLayer
                prompt = Prompt(
                    identifier=layer.identifier or f"custom_{id(layer)}",
                    name=layer.identifier or "Custom Prompt",
                    role=layer.role,  # type: ignore
                    content=layer.content,
                    system_prompt=False,  # Custom prompts are not system prompts
                    marker=False,
                    injection_position=(
                        InjectionPosition.RELATIVE
                        if layer.injection_position == "before"
                        else InjectionPosition.ABSOLUTE
                    ),
                    injection_depth=layer.injection_depth,
                )
                prompts.append(prompt)

                # Create order entry
                order_entries.append(
                    PromptOrderEntry(identifier=prompt.identifier, enabled=True)
                )

            # Add to PromptManager
            for prompt in prompts:
                self.prompt_manager.add_prompt(prompt)

            # Set order for character
            self.prompt_manager.set_prompt_order(character_id, order_entries)

            logger.info(
                f"[PromptIntegration] Exported {len(prompts)} prompts "
                f"for character {character_name}"
            )

            return True

        except Exception as e:
            logger.error(f"[PromptIntegration] Failed to export prompts: {e}")
            return False

    def build_system_prompt_from_manager(
        self,
        character_id: str,
        character_data: Optional[Dict[str, Any]] = None,
        generation_type: str = "normal",
    ) -> str:
        """
        Build a system prompt string from PromptManager.

        This combines all enabled relative (pre-chat) prompts into
        a single system prompt string for compatibility with
        systems that expect a single system prompt.

        Args:
            character_id: Character ID for prompt lookup
            character_data: Optional character data
            generation_type: Generation type

        Returns:
            Combined system prompt string
        """
        if not self.enabled:
            return ""

        try:
            injection_context = self.prompt_manager.build_injection_context(
                character_id=character_id,
                character_data=character_data,
                persona_data=None,
                generation_type=generation_type,
            )

            # Filter for relative prompts (depth == -1) and system role
            system_parts = []
            for item in injection_context:
                if item["depth"] == -1 and item["role"] == "system":
                    content = item.get("content", "").strip()
                    if content:
                        system_parts.append(content)

            combined_prompt = "\n\n".join(system_parts)

            logger.debug(
                f"[PromptIntegration] Built system prompt: {len(system_parts)} parts, "
                f"{len(combined_prompt)} characters"
            )

            return combined_prompt

        except Exception as e:
            logger.error(f"[PromptIntegration] Failed to build system prompt: {e}")
            return ""

    def get_enabled_prompt_count(
        self,
        character_id: str,
    ) -> int:
        """Get count of enabled prompts for a character."""
        if not self.enabled:
            return 0

        try:
            collection = self.prompt_manager.get_prompt_collection(
                character_id,
                only_enabled=True,
            )
            return len(collection.prompts)
        except Exception:
            return 0

    def list_character_prompts(
        self,
        character_id: str,
    ) -> List[Dict[str, Any]]:
        """List all prompts for a character with their status."""
        if not self.enabled:
            return []

        try:
            manager = self.prompt_manager
            order = manager.get_prompt_order(character_id)

            result = []
            for entry in order:
                prompt = manager.get_prompt(entry.identifier)
                if prompt:
                    result.append({
                        "identifier": entry.identifier,
                        "name": prompt.name,
                        "enabled": entry.enabled,
                        "role": prompt.role,
                        "marker": prompt.marker,
                        "system_prompt": prompt.system_prompt,
                        "injection_position": prompt.injection_position,
                        "injection_depth": prompt.injection_depth,
                    })

            return result

        except Exception as e:
            logger.error(f"[PromptIntegration] Failed to list prompts: {e}")
            return []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_prompt_integration() -> PromptManagerIntegration:
    """Get the global PromptManager integration instance."""
    return PromptManagerIntegration(
        prompt_manager=get_prompt_manager(),
        enabled=True,
    )


def inject_prompts_for_character(
    personality: PersonalityLayer,
    character_id: str,
    character_data: Optional[Dict[str, Any]] = None,
    generation_type: str = "normal",
) -> None:
    """
    Helper function to inject prompts for a character.

    This is a convenience wrapper around PromptManagerIntegration.
    """
    integration = get_prompt_integration()
    integration.inject_into_personality_layer(
        personality=personality,
        character_id=character_id,
        character_data=character_data,
        generation_type=generation_type,
    )


def build_system_prompt(
    character_id: str,
    character_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Helper function to build system prompt from PromptManager.

    Returns a combined system prompt string for the character.
    """
    integration = get_prompt_integration()
    return integration.build_system_prompt_from_manager(
        character_id=character_id,
        character_data=character_data,
    )
