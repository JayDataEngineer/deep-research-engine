"""Prompt Builder - Connects prompt_engine library to chat modes.

This module provides the "driveshaft" connection between the prompt_engine library
and the actual chat modes. It replaces the old prompt_loader system.

Usage:
    from src.extensions.libraries.core_chat_engine.prompts.builder import build_system_messages

    # Get system messages for a mode
    system_messages = build_system_messages(
        mode="roleplay",
        character=character,
        user_profile=user_profile,
        memory_context=memory_context,
    )
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.messages import SystemMessage, BaseMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Builder
# =============================================================================

class PromptBuilder:
    """Builds system prompts using the prompt_engine library.

    This replaces the old prompt_loader system with the enhanced prompt_engine
    that supports Jinja2 templates and LangChain message conversion.
    """

    def __init__(self):
        """Initialize the prompt builder."""
        self._stack_manager = None

    def _get_stack_manager(self):
        """Lazy-load the stack manager."""
        if self._stack_manager is None:
            from src.extensions.libraries.prompt_engine import get_stack_manager, init_stack_manager
            try:
                self._stack_manager = get_stack_manager()
            except RuntimeError:
                # Initialize if not already done
                import asyncio
                try:
                    # Check if we're in an async context
                    loop = asyncio.get_running_loop()
                    # We're in an async context - can't use asyncio.run()
                    # Create a task and run it synchronously
                    import concurrent.futures
                    import threading

                    result = [None]
                    exception = [None]

                    def run_in_new_loop():
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            result[0] = new_loop.run_until_complete(init_stack_manager())
                            new_loop.close()
                        except Exception as e:
                            exception[0] = e

                    thread = threading.Thread(target=run_in_new_loop)
                    thread.start()
                    thread.join(timeout=5)

                    if exception[0]:
                        raise exception[0]
                    if result[0]:
                        self._stack_manager = result[0]
                    else:
                        # Fallback to simple message
                        logger.warning("[PROMPT BUILDER] Could not initialize stack manager in async context")
                        raise RuntimeError("Could not initialize stack manager")
                except RuntimeError:
                    # No running loop, use asyncio.run()
                    self._stack_manager = asyncio.run(init_stack_manager())
        return self._stack_manager

    def build_system_messages(
        self,
        mode: str,
        character: Optional[Any] = None,
        user_profile: Optional[Any] = None,
        memory_context: Optional[str] = None,
        personality_config: Optional[Any] = None,
        additional_vars: Optional[Dict[str, Any]] = None,
    ) -> List[BaseMessage]:
        """Build system messages using prompt_engine.

        Args:
            mode: Chat mode (standard, companion, roleplay)
            character: Character object (for roleplay mode)
            user_profile: User profile object
            memory_context: Memory context string from memory system
            personality_config: Personality configuration
            additional_vars: Additional template variables

        Returns:
            List of LangChain BaseMessage objects (typically SystemMessages)
        """
        # Determine which stack to use
        stack_name = self._get_stack_name(mode, personality_config)

        # Load the stack
        manager = self._get_stack_manager()
        stack = manager.registry.get(stack_name)

        if not stack:
            logger.warning(f"[PROMPT BUILDER] Stack '{stack_name}' not found, using default")
            # Fallback to default stack
            stack = manager.registry.get("system_default")
            if not stack:
                # Ultimate fallback - hardcoded default
                return [SystemMessage(content="You are a helpful AI assistant.")]

        # Build template variables
        variables = self._build_variables(
            mode=mode,
            character=character,
            user_profile=user_profile,
            memory_context=memory_context,
            personality_config=personality_config,
        )

        # Add additional variables
        if additional_vars:
            variables.update(additional_vars)

        # Render stack to LangChain messages
        try:
            messages = stack.to_langchain_messages(**variables)
            logger.debug(f"[PROMPT BUILDER] Built {len(messages)} messages from stack '{stack_name}'")
            return messages
        except Exception as e:
            logger.error(f"[PROMPT BUILDER] Failed to render stack '{stack_name}': {e}")
            # Fallback to hardcoded prompt
            return [SystemMessage(content=self._get_fallback_prompt(mode))]

    def _get_stack_name(self, mode: str, personality_config: Optional[Any]) -> str:
        """Determine which stack to use for a mode.

        Args:
            mode: Chat mode
            personality_config: Personality configuration (may have stack override)

        Returns:
            Stack name/ID
        """
        # Check if personality config specifies a stack
        if personality_config and hasattr(personality_config, 'prompt_stack'):
            custom_stack = personality_config.prompt_stack
            if custom_stack:
                return custom_stack

        # Mode-based defaults
        stack_map = {
            "standard": "system_default",
            "companion": "personality_friendly",
            "roleplay": "roleplay_default",
            "rp": "roleplay_default",  # alias
            "deep_research": "deep_research",  # deep research workers with true prefills
        }

        return stack_map.get(mode, "system_default")

    def _build_variables(
        self,
        mode: str,
        character: Optional[Any],
        user_profile: Optional[Any],
        memory_context: Optional[str],
        personality_config: Optional[Any],
    ) -> Dict[str, Any]:
        """Build template variables from context.

        Args:
            mode: Chat mode
            character: Character object
            user_profile: User profile object
            memory_context: Memory context string
            personality_config: Personality configuration

        Returns:
            Dictionary of template variables
        """
        variables = {}

        # Mode
        variables["mode"] = mode

        # Character info (roleplay mode)
        if character:
            variables["char_name"] = getattr(character, "name", "Character")
            variables["char_description"] = getattr(character, "description", "")
            variables["char_backstory"] = getattr(character, "backstory", "")
            variables["char_personality"] = getattr(character, "personality", "")
            variables["scenario"] = getattr(character, "scenario", "the current situation")
        else:
            # Fallback for non-RP modes
            variables["char_name"] = "Assistant"
            variables["char_description"] = "A helpful AI assistant"
            variables["scenario"] = "general conversation"

        # User info
        if user_profile:
            variables["user_name"] = getattr(user_profile, "name", "User")
            variables["user_preferences"] = getattr(user_profile, "preferences", "")
        else:
            variables["user_name"] = "User"

        # Memory context
        if memory_context:
            variables["memory_context"] = memory_context
        else:
            variables["memory_context"] = ""

        # Personality config
        if personality_config:
            variables["personality_name"] = getattr(personality_config, "name", "")
            variables["personality_role"] = getattr(personality_config, "role", "")
            variables["personality_description"] = getattr(personality_config, "description", "").strip()
        else:
            variables["personality_name"] = ""
            variables["personality_role"] = ""
            variables["personality_description"] = ""

        return variables

    def _get_fallback_prompt(self, mode: str) -> str:
        """Get fallback hardcoded prompt if stack loading fails.

        Args:
            mode: Chat mode

        Returns:
            Fallback prompt string
        """
        if mode in ("roleplay", "rp"):
            return "You are roleplaying as a character. Stay in character and respond naturally."
        elif mode == "companion":
            return "You are a friendly and helpful companion. Be warm, supportive, and engaging."
        else:  # standard
            return "You are a helpful AI assistant."


# =============================================================================
# Singleton instance
# =============================================================================

_prompt_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get the global PromptBuilder instance."""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder


# =============================================================================
# Convenience functions
# =============================================================================

def build_system_messages(
    mode: str,
    character: Optional[Any] = None,
    user_profile: Optional[Any] = None,
    memory_context: Optional[str] = None,
    personality_config: Optional[Any] = None,
    additional_vars: Optional[Dict[str, Any]] = None,
) -> List[BaseMessage]:
    """Convenience function to build system messages.

    This is the main entry point for prompt engine integration.

    Args:
        mode: Chat mode (standard, companion, roleplay)
        character: Character object (for roleplay mode)
        user_profile: User profile object
        memory_context: Memory context string from memory system
        personality_config: Personality configuration
        additional_vars: Additional template variables

    Returns:
        List of LangChain BaseMessage objects (typically SystemMessages)

    Example:
        messages = build_system_messages(
            mode="roleplay",
            character=my_character,
            user_profile=user,
            memory_context=memory_string,
        )
    """
    builder = get_prompt_builder()
    return builder.build_system_messages(
        mode=mode,
        character=character,
        user_profile=user_profile,
        memory_context=memory_context,
        personality_config=personality_config,
        additional_vars=additional_vars,
    )


def get_system_prompt_string(
    mode: str,
    character: Optional[Any] = None,
    personality_config: Optional[Any] = None,
) -> str:
    """Get system prompt as a single string (for backward compatibility).

    This renders the stack and concatenates the content into a single string.

    Args:
        mode: Chat mode
        character: Character object (for roleplay mode)
        personality_config: Personality configuration

    Returns:
        Concatenated system prompt string
    """
    messages = build_system_messages(
        mode=mode,
        character=character,
        personality_config=personality_config,
    )

    # Concatenate all message content
    return "\n\n".join(msg.content for msg in messages if msg.content.strip())
