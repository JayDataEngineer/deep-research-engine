"""
SillyTavern-compatible Prompt Manager Models

This module provides Pydantic models compatible with SillyTavern's
prompt manager export format for prompt ordering and injection.

SillyTavern Prompt Format Reference:
https://github.com/SillyTavern/SillyTavern/blob/develop/public/scripts/PromptManager.js
"""

from typing import Literal, Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# =============================================================================
# CONSTANTS
# =============================================================================

class InjectionPosition:
    """Prompt injection position types."""
    RELATIVE = 0  # Before chat history (relative positioning)
    ABSOLUTE = 1  # At specific depth in chat history (absolute positioning)


# Prompt role types (as Literal for Pydantic compatibility)
PromptRole = Literal["system", "user", "assistant"]


# Default values from SillyTavern
DEFAULT_DEPTH = 4
DEFAULT_ORDER = 100

# System prompt identifiers (built-in prompts that cannot be deleted)
SYSTEM_PROMPT_IDS = [
    "main",
    "nsfw",
    "prefill",
    "enhanceDefinitions",
    "dialogueExamples",
    "chatHistory",
    "worldInfoBefore",
    "worldInfoAfter",
    "charDescription",
    "charPersonality",
    "scenario",
    "personaDescription",
]

# Marker prompt identifiers (placeholders for dynamic content)
MARKER_PROMPT_IDS = [
    "dialogueExamples",
    "chatHistory",
    "worldInfoBefore",
    "worldInfoAfter",
    "charDescription",
    "charPersonality",
    "scenario",
    "personaDescription",
]


# =============================================================================
# PROMPT MODELS
# =============================================================================

class Prompt(BaseModel):
    """
    A single prompt configuration compatible with SillyTavern.

    This matches the SillyTavern Prompt class from PromptManager.js
    """

    identifier: str = Field(
        ...,
        description="Unique identifier for the prompt"
    )
    name: str = Field(
        ...,
        description="Display name of the prompt"
    )
    role: PromptRole = Field(
        default="system",
        description="Role of the prompt (system, user, assistant)"
    )
    content: str = Field(
        default="",
        description="The prompt text content"
    )
    system_prompt: bool = Field(
        default=False,
        description="Whether this is a built-in system prompt"
    )
    marker: bool = Field(
        default=False,
        description="Whether this is a marker (placeholder for dynamic content)"
    )
    position: Optional[Union[str, int]] = Field(
        default=None,
        description="Position in the prompt list (for UI display)"
    )
    injection_position: int = Field(
        default=InjectionPosition.RELATIVE,
        description="0=relative (before chat), 1=absolute (at depth)"
    )
    injection_depth: int = Field(
        default=DEFAULT_DEPTH,
        description="Depth in chat history for absolute injection",
        ge=0
    )
    injection_order: int = Field(
        default=DEFAULT_ORDER,
        description="Order for prompts at same depth (lower=first)",
        ge=0
    )
    injection_trigger: List[str] = Field(
        default_factory=list,
        description="Generation type triggers (e.g., ['normal', 'continue'])"
    )
    forbid_overrides: bool = Field(
        default=False,
        description="Whether this prompt can be overridden by character cards"
    )
    extension: bool = Field(
        default=False,
        description="Whether this prompt was added by an extension"
    )

    @field_validator('identifier')
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        """Validate identifier format."""
        if not v or not v.strip():
            raise ValueError("identifier cannot be empty")
        # Sanitize to match ST format
        return v.strip().replace(" ", "_")

    @field_validator('injection_position')
    @classmethod
    def validate_injection_position(cls, v: int) -> int:
        """Validate injection position."""
        if v not in [InjectionPosition.RELATIVE, InjectionPosition.ABSOLUTE]:
            raise ValueError("injection_position must be 0 (relative) or 1 (absolute)")
        return v


class PromptOrderEntry(BaseModel):
    """
    An entry in the prompt order list for a character.

    This represents one entry in the prompt_order array from SillyTavern.
    """

    identifier: str = Field(
        ...,
        description="Prompt identifier to reference"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this prompt is currently enabled"
    )


class PromptCollection(BaseModel):
    """
    A collection of prompts ready for injection into the context.

    This represents the result of getPromptCollection() with filtering
    and substitution applied.
    """

    prompts: List[Prompt] = Field(
        default_factory=list,
        description="List of prompts (filtered and enabled)"
    )
    overridden: List[str] = Field(
        default_factory=list,
        description="Identifiers of prompts that were overridden"
    )


# =============================================================================
# EXPORT/IMPORT MODELS
# =============================================================================

class PromptExportData(BaseModel):
    """
    The data section of a SillyTavern prompt export.

    Matches the 'data' field in ST export format:
    {
        "version": 1,
        "type": "full" | "character",
        "data": {
            "prompts": [...],
            "prompt_order": [...]
        }
    }
    """

    prompts: List[Prompt] = Field(
        default_factory=list,
        description="All prompt definitions"
    )
    prompt_order: List[PromptOrderEntry] = Field(
        default_factory=list,
        description="Prompt order for a character (optional, for character exports)"
    )


class PromptExport(BaseModel):
    """
    Complete SillyTavern prompt export format.

    This is the root object in ST export files.
    """

    version: int = Field(
        default=1,
        description="Export format version"
    )
    type: Literal["full", "character"] = Field(
        ...,
        description="'full' = all prompts, 'character' = character-specific"
    )
    data: PromptExportData = Field(
        ...,
        description="Prompts and order data"
    )


class CharacterPromptConfig(BaseModel):
    """
    Character-specific prompt configuration.

    Stores prompt order per character, compatible with ST's
    character-specific prompt ordering.
    """

    character_id: str = Field(
        ...,
        description="Character ID (or name for identification)"
    )
    character_name: str = Field(
        ...,
        description="Character display name"
    )
    prompt_order: List[PromptOrderEntry] = Field(
        default_factory=list,
        description="Prompt order for this character"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this config was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this config was last updated"
    )


# =============================================================================
# DEFAULT PROMPTS (SillyTavern-compatible)
# =============================================================================

def get_default_prompts() -> List[Prompt]:
    """
    Get the default system prompts compatible with SillyTavern.

    These match the chatCompletionDefaultPrompts from PromptManager.js
    """
    return [
        Prompt(
            identifier="main",
            name="Main Prompt",
            role="system",
            content="Write {{char}}'s next reply in a fictional chat between {{charIfNotGroup}} and {{user}}.",
            system_prompt=True,
            marker=False,
            forbid_overrides=False,
        ),
        Prompt(
            identifier="nsfw",
            name="Auxiliary Prompt",
            role="system",
            content="",
            system_prompt=True,
            marker=False,
        ),
        Prompt(
            identifier="dialogueExamples",
            name="Chat Examples",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
        Prompt(
            identifier="prefill",
            name="Prefill",
            role="assistant",
            content="",
            system_prompt=True,
            marker=False,
        ),
        Prompt(
            identifier="chatHistory",
            name="Chat History",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
        Prompt(
            identifier="worldInfoAfter",
            name="World Info (after)",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
        Prompt(
            identifier="worldInfoBefore",
            name="World Info (before)",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
        Prompt(
            identifier="enhanceDefinitions",
            name="Enhance Definitions",
            role="system",
            content="If you have more knowledge of {{char}}, add to the character's lore and personality to enhance them but keep the Character Sheet's definitions absolute.",
            system_prompt=True,
            marker=False,
        ),
        Prompt(
            identifier="charDescription",
            name="Character Description",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
        Prompt(
            identifier="charPersonality",
            name="Character Personality",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
        Prompt(
            identifier="scenario",
            name="Scenario",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
        Prompt(
            identifier="personaDescription",
            name="Persona Description",
            role="system",
            content="",
            system_prompt=True,
            marker=True,
        ),
    ]


def get_default_prompt_order() -> List[PromptOrderEntry]:
    """
    Get the default prompt order from SillyTavern.

    This matches promptManagerDefaultPromptOrder from PromptManager.js
    """
    return [
        PromptOrderEntry(identifier="main", enabled=True),
        PromptOrderEntry(identifier="worldInfoBefore", enabled=True),
        PromptOrderEntry(identifier="personaDescription", enabled=True),
        PromptOrderEntry(identifier="charDescription", enabled=True),
        PromptOrderEntry(identifier="charPersonality", enabled=True),
        PromptOrderEntry(identifier="scenario", enabled=True),
        PromptOrderEntry(identifier="enhanceDefinitions", enabled=False),
        PromptOrderEntry(identifier="nsfw", enabled=True),
        PromptOrderEntry(identifier="worldInfoAfter", enabled=True),
        PromptOrderEntry(identifier="dialogueExamples", enabled=True),
        PromptOrderEntry(identifier="chatHistory", enabled=True),
        PromptOrderEntry(identifier="prefill", enabled=True),
    ]
