"""Character Card Models.

Pydantic models for character cards and related data.
Compatible with SillyTavern and other RP card formats.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import re
import json


class CharacterCard(BaseModel):
    """Character card model.

    Compatible with SillyTavern character card format.
    """

    # Identification
    name: str = Field(..., min_length=1, max_length=200, description="Character name")
    description: str = Field("", description="Character description/appearance")

    # Personality
    personality: str = Field("", description="Personality traits and behavior")
    scenario: str = Field("", description="Scenario/setting for the character")

    # Dialogue
    mes_example: str = Field("", description="Example dialogue (message examples)")
    first_mes: str = Field("", description="First message/greeting")

    # Metadata
    creator: str = Field("", description="Character creator")
    character_version: str = Field("1.0", description="Character version")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    # Extensions - for custom fields
    extensions: Dict[str, Any] = Field(default_factory=dict, description="Custom extensions")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is valid."""
        if not v or not v.strip():
            raise ValueError("Character name cannot be empty")
        return v.strip()


class CharacterBookEntry(BaseModel):
    """Character Book / Lorebook entry.

    Keyword-activated entry that can be embedded in character cards.
    Compatible with SillyTavern CharacterBook V2 format.
    """

    # Identification
    id: int = Field(..., ge=0, description="Unique entry ID")
    name: Optional[str] = Field(None, description="Entry name")
    comment: Optional[str] = Field(None, description="Notes about this entry")

    # Content
    content: str = Field(..., min_length=1, description="Content to inject when activated")

    # Keywords (SillyTavern field names)
    keys: List[str] = Field(default_factory=list, description="Primary trigger keywords")
    secondary_keys: List[str] = Field(default_factory=list, description="Secondary keywords (AND logic)")

    # Activation control
    enabled: bool = Field(True, description="Whether entry is active")
    constant: bool = Field(False, description="Always active (keyword-independent)")

    # Positioning (SillyTavern positions)
    position: str = Field("after_char", description="Where to inject: before_char, after_char, at_depth")

    # Priority
    priority: int = Field(0, ge=0, description="Priority for sorting (higher = earlier)")

    # Filtering
    case_sensitive: Optional[bool] = Field(False, description="Keyword case sensitivity")
    selective: bool = Field(False, description="Require ALL keywords (AND) vs ANY (OR)")
    use_regex: bool = Field(False, description="Treat keys as regex patterns")

    # Extensions (SillyTavern stores depth/probability here)
    extensions: Dict[str, Any] = Field(
        default_factory=lambda: {"depth": 4, "probability": 100},
        description="Extensions (depth, probability, etc.)"
    )

    @field_validator("keys", "secondary_keys")
    @classmethod
    def validate_keys(cls, v: List[str]) -> List[str]:
        """Clean and normalize keys."""
        return [k.strip() for k in v if k.strip()]

    @property
    def depth(self) -> int:
        """Get scan depth from extensions."""
        return self.extensions.get("depth", 4)

    @property
    def probability(self) -> int:
        """Get probability from extensions (0-100 scale)."""
        return self.extensions.get("probability", 100)


class CharacterBook(BaseModel):
    """Character Book - collection of lorebook entries.

    Can be embedded in a character card or standalone.
    """

    # Identification
    name: str = Field(..., min_length=1, description="Character book name")
    description: Optional[str] = Field(None, description="Description")

    # Scanning configuration
    scan_depth: int = Field(10, ge=0, description="Messages to scan back")
    token_budget: int = Field(2000, ge=0, description="Max tokens for activated entries")

    # Entries
    entries: List[CharacterBookEntry] = Field(default_factory=list, description="Book entries")

    # Extensions
    extensions: Dict[str, Any] = Field(default_factory=dict, description="Custom extensions")

    def get_enabled_entries(self) -> List[CharacterBookEntry]:
        """Get only enabled entries sorted by priority."""
        enabled = [e for e in self.entries if e.enabled]
        return sorted(enabled, key=lambda e: (-e.priority, e.id))

    def add_entry(self, entry: CharacterBookEntry) -> None:
        """Add an entry to the book."""
        existing_ids = {e.id for e in self.entries}
        if entry.id in existing_ids:
            raise ValueError(f"Entry ID {entry.id} already exists")
        self.entries.append(entry)

    def remove_entry(self, entry_id: int) -> bool:
        """Remove an entry by ID."""
        for i, e in enumerate(self.entries):
            if e.id == entry_id:
                self.entries.pop(i)
                return True
        return False

    def get_entry(self, entry_id: int) -> Optional[CharacterBookEntry]:
        """Get an entry by ID."""
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None
