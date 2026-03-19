"""Podcast format extension prompts.

This module provides podcast-specific prompts by loading character data
from the CENTRAL character repository (text_injections.yaml).

The podcast format owns the logic for loading and formatting character data.
The generic factory in utility.py knows nothing about podcasts.
"""

import yaml
from pathlib import Path
from typing import List, Any, Dict

# Add backend to path for absolute imports
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(backend_dir) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(backend_dir))

from src.constants import CHARACTER_TEXT_INJECTIONS_FILE


def get_prompts(character: Any = None, **kwargs: Any) -> List[str]:
    """Get podcast-specific prompts (character profiles).

    This function is called by the generic factory in utility.py.
    The podcast format owns the logic for what it returns.

    Args:
        character: Character name(s) - can be:
            - str: Single character name (e.g., "OLIVIA")
            - list: Multiple character names (e.g., ["EMMA", "OLIVIA"])
        **kwargs: Other arguments that may be passed (ignored for now)

    Returns:
        List of prompt strings to append to the base podcast formatting prompt.
        For multiple characters, returns all character profiles with clear separation.
    """
    prompts = []

    if not character:
        return prompts

    # Handle both single character (str) and multiple characters (list)
    if isinstance(character, str):
        characters = [character]
    elif isinstance(character, list):
        characters = character
    else:
        return prompts

    # Load each character from central repository
    for char_name in characters:
        char_data = _load_character_from_central_repo(char_name)
        if char_data:
            prompts.append(f"CHARACTER PROFILE ({char_name}):\n{char_data}\n")

    return prompts


def _load_character_from_central_repo(character_name: str) -> str:
    """Load character from CENTRAL source of truth.

    Uses CHARACTER_TEXT_INJECTIONS_FILE from constants.py so the path
    can be hot-swapped by changing one constant.

    Args:
        character_name: Name of the character to load (e.g., "OLIVIA")

    Returns:
        Formatted character data as string, or empty string if not found
    """
    if not CHARACTER_TEXT_INJECTIONS_FILE.exists():
        return ""

    with open(CHARACTER_TEXT_INJECTIONS_FILE, 'r') as f:
        data = yaml.safe_load(f)

    if not data or "personalities" not in data:
        return ""

    personalities = data["personalities"]
    if character_name not in personalities:
        return ""

    char = personalities[character_name]

    # Format character data as a structured prompt
    lines = [
        f"Name: {char.get('name', character_name)}",
        f"Role: {char.get('role', 'Podcast Host')}",
        "",
        "Description:",
        char.get("description", "No description available."),
        "",
    ]

    # Add speech patterns if available
    if "speech_patterns" in char:
        lines.extend([
            "Speech Patterns:",
            *[f"  - {pattern}" for pattern in char["speech_patterns"]],
            ""
        ])

    # Add ideological framework if available
    if "ideological_framework" in char:
        lines.extend([
            "Ideological Framework:",
            f"  {char['ideological_framework']}",
            ""
        ])

    # Add emotion bias if available (useful for TTS context)
    if "emotion_bias" in char:
        bias = char["emotion_bias"]
        lines.extend([
            "Emotional Baseline (for TTS):",
            f"  [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]",
            f"  {bias}",
            ""
        ])

    return "\n".join(lines)


# Optional: List available characters for discovery
def list_available_characters() -> List[str]:
    """List all available character names from central repository.

    Returns:
        List of character names
    """
    if not CHARACTER_TEXT_INJECTIONS_FILE.exists():
        return []

    with open(CHARACTER_TEXT_INJECTIONS_FILE, 'r') as f:
        data = yaml.safe_load(f)

    if not data or "personalities" not in data:
        return []

    return list(data["personalities"].keys())
