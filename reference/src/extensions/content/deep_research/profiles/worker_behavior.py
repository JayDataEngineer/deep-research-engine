"""Generic Text Injection System - Content-agnostic prompt stack builder.

This module provides the MECHANISM only, not the CONTENT.
It does NOT know about "speech_patterns", "emotion_bias", etc. - those are
extension-specific concepts.

KEY PRINCIPLE: The system is content-agnostic. It:
1. Takes a dict of personality data (ANY fields)
2. Extracts ALL top-level keys
3. Renders through Jinja2 template
4. Injects into prompt stack at the right location

Extensions define WHAT fields exist.
This system defines HOW they're injected.

Example:
    Podcast defines: speech_patterns, emotion_bias, ideological_framework
    Twitter defines: hashtags, tweet_style, engagement_tactics
    Generic system: "I see a dict with keys, I'll inject them all"
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger
from langchain_core.messages import SystemMessage, AIMessage

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


def extract_all_personality_fields(
    personality_data: Dict[str, Any],
    exclude_fields: Optional[set] = None
) -> Dict[str, Any]:
    """Extract ALL personality fields, excluding metadata.

    This is content-agnostic - it doesn't know WHAT fields exist,
    it just extracts everything except excluded metadata.

    Args:
        personality_data: Raw personality dict from YAML
        exclude_fields: Optional fields to exclude (e.g., _metadata, name)

    Returns:
        Dict of ALL personality fields (content-agnostic)
    """
    if exclude_fields is None:
        exclude_fields = {"_metadata", "_version", "_description"}

    return {
        key: value
        for key, value in personality_data.items()
        if key not in exclude_fields
    }


def build_system_messages_from_dict(
    base_prompt: str,
    personality_data: Dict[str, Any],
    template: Optional[str] = None,
    prefill: Optional[str] = None
) -> List[Dict[str, str]]:
    """Build system messages from personality dict (content-agnostic).

    The system does NOT know what fields exist in personality_data.
    It just renders them into the prompt.

    Args:
        base_prompt: The base worker prompt
        personality_data: Raw personality dict (ANY structure)
        template: Optional Jinja2 template string for formatting
        prefill: Optional "true prefill" for character voice

    Returns:
        List of message dicts with role and content

    Example:
        personality_data = {
            "name": "Emma",
            "role": "podcast host",
            "speech_patterns": ["bubbly", "energetic"],  # Podcast-specific
            "ideological_framework": "progressive"
        }

        System doesn't know what "speech_patterns" is, just that it's a field.
    """
    messages = []

    # Build personality block if data provided
    if personality_data:
        persona_block = _render_personality_block(personality_data, template)
    else:
        persona_block = ""

    # Combine base prompt with persona
    full_prompt = base_prompt
    if persona_block:
        full_prompt = f"{base_prompt}\n\n{persona_block}"

    # Add as system message
    messages.append({"role": "system", "content": full_prompt})

    # Add "true prefill" as assistant message if provided
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    return messages


def _render_personality_block(
    personality_data: Dict[str, Any],
    template: Optional[str] = None
) -> str:
    """Render personality block from data (content-agnostic).

    Args:
        personality_data: Raw personality dict
        template: Optional Jinja2 template string

    Returns:
        Rendered personality block string
    """
    if template:
        # Use Jinja2 template if provided
        from jinja2 import Template
        jinja_template = Template(template)
        return jinja_template.render(**personality_data)
    else:
        # Default rendering: all fields as key-value pairs
        lines = ["**CHARACTER DATA:**"]
        for key, value in personality_data.items():
            # Skip metadata fields
            if key.startswith("_"):
                continue

            if isinstance(value, list):
                lines.append(f"\n**{key.upper()}:**")
                for item in value:
                    lines.append(f"- {item}")
            elif isinstance(value, dict):
                lines.append(f"\n**{key.upper()}:**")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"**{key.upper()}:** {value}")

        return "\n".join(lines)


def get_worker_instructions(
    personality_data: Dict[str, Any],
    config_directive_key: str = "research_directive"
) -> tuple[str, Optional[str]]:
    """Extract worker instructions from personality (content-agnostic).

    Args:
        personality_data: Raw personality dict
        config_directive_key: Key to look for directive (default: research_directive)

    Returns:
        (directive, prefill) tuple

    Example:
        # Podcast provides these keys:
        personality_data = {
            "research_directive": "Research from far-left...",
            "prefill_prompt": "Understood. I will portray...",
        }

        # Twitter could provide different keys:
        personality_data = {
            "tweet_directive": "Create viral tweets...",
            "tweet_prefill": "Let's tweet...",
        }

        get_worker_instructions(personality_data, "tweet_directive")
        # Returns: ("Create viral tweets...", "Let's tweet...")
    """
    directive = personality_data.get(config_directive, "")
    prefill = personality_data.get("prefill_prompt", "")

    # Try common prefill key variants
    if not prefill:
        for key in ["prefill", "prefill_prompt", "true_prefill", "assistant_prefill"]:
            if key in personality_data:
                prefill = personality_data[key]
                break

    return directive, prefill


def resolve_character_mapping(
    worker_type: str,
    character_card_mappings: Dict[str, str],
) -> Optional[str]:
    """Resolve worker type to personality ID (content-agnostic).

    The system doesn't know WHAT workers exist, just how to map them.

    Args:
        worker_type: Worker type (ANY string)
        character_card_mappings: Direct worker->personality mappings

    Returns:
        Personality ID or None

    NOTE: Legacy pattern support removed - use explicit mappings only
    """
    # Direct mapping only
    if worker_type in character_card_mappings:
        return character_card_mappings[worker_type]

    return None


def load_personality_for_worker(
    worker_type: str,
    character_card_mappings: Dict[str, str],
    profile_loader,
) -> Optional[Dict[str, Any]]:
    """Load personality data for a worker (content-agnostic).

    Args:
        worker_type: Worker type (ANY string)
        character_card_mappings: Character mappings from config
        profile_loader: ProfileLoader instance

    Returns:
        Raw personality dict (ALL fields, content-agnostic) or None

    NOTE: Legacy pattern support removed - use explicit mappings only
    """
    # Resolve to personality ID
    personality_id = resolve_character_mapping(
        worker_type,
        character_card_mappings,
    )

    if not personality_id:
        return None

    # Load personality (user → system)
    try:
        return profile_loader.load_personality(personality_id)
    except FileNotFoundError:
        logger.warning(f"[TEXT_INJECTION] Personality '{personality_id}' not found for worker '{worker_type}'")
        return None
