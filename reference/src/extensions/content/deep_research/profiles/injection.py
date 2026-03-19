"""Generic Text Injection System - DRY worker behavior and character prompts.

This module provides the MECHANISM for text injection, not the CONTENT.
It generalizes the pattern used by Podcast for ANY extension.

EXPORTS (Content-Agnostic):
- extract_all_personality_fields() - Extract ALL fields from personality dict
- build_system_messages_from_dict() - Build messages from any personality dict
- get_worker_instructions() - Extract directive and prefill from personality
- resolve_character_mapping() - Generic character mapping resolution
- load_personality_for_worker() - Unified personality loading

The system is CONTENT-AGNOSTIC - it doesn't know about specific fields like
"speech_patterns" or "emotion_bias". It just takes ANY dict structure and
renders it into the prompt stack at the correct location.

Usage:
    from src.extensions.content.deep_research.profiles.injection import (
        extract_all_personality_fields,
        build_system_messages_from_dict,
        load_personality_for_worker,
    )

    # Load personality for worker
    personality = load_personality_for_worker(
        worker_type="left_character",
        character_card_mappings={"left_character": "EMMA"},
        profile_loader=loader
    )

    # Extract all fields content-agnostically
    personality_fields = extract_all_personality_fields(personality)

    # Build system messages with character injection
    messages = build_system_messages_from_dict(
        base_prompt=worker_prompt,
        personality_data=personality_fields,
        prefill=personality.get("prefill_prompt")
    )
"""

from .worker_behavior import (
    # Core functions (content-agnostic)
    extract_all_personality_fields,
    build_system_messages_from_dict,
    get_worker_instructions,
    _render_personality_block,  # Exported for advanced use cases

    # Character mapping
    resolve_character_mapping,
    load_personality_for_worker,
)

__all__ = [
    # Core functions (content-agnostic)
    "extract_all_personality_fields",
    "build_system_messages_from_dict",
    "get_worker_instructions",
    "_render_personality_block",  # Exported for advanced use cases

    # Character mapping
    "resolve_character_mapping",
    "load_personality_for_worker",
]
