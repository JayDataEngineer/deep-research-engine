"""Lorebook Schema - Pydantic models for SillyTavern-compatible lorebooks.

Lorebooks provide keyword-activated context entries that inject relevant
information into the conversation based on triggers.
"""

from typing import Optional, List, Dict, Any, Literal, Union
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re


class LorebookExtensions(BaseModel):
    """Custom extensions for lorebook entries."""

    # Custom mood/tone for this entry
    mood: Optional[str] = Field(None, description="Mood/tone override")

    # Related characters, factions, locations
    characters: Optional[List[str]] = Field(default_factory=list, description="Related character IDs")
    factions: Optional[List[str]] = Field(default_factory=list, description="Related factions")
    locations: Optional[List[str]] = Field(default_factory=list, description="Related locations IDs")

    # Related entries for cross-referencing
    related_entries: Optional[List[str]] = Field(default_factory=list, description="Related entry IDs")

    # Probability override
    probability_override: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Token budget override
    token_budget_override: Optional[int] = Field(None, ge=0)

    # Additional custom fields
    custom: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata")


class LorebookEntry(BaseModel):
    """A single lorebook entry with keyword activation.

    Compatible with SillyTavern CharacterBook V2 format.
    """

    # Identification - use numeric id like SillyTavern
    id: int = Field(..., description="Unique entry identifier (numeric, auto-incremented)")
    name: Optional[str] = Field(None, description="Human-readable entry name")
    comment: Optional[str] = Field(None, description="Notes about this entry")

    # Content
    content: str = Field(..., min_length=1, description="Content to inject when activated")

    # Keywords for activation - SillyTavern field names
    keys: List[str] = Field(default_factory=list, description="Primary trigger keywords")
    secondary_keys: List[str] = Field(default_factory=list, description="Secondary keywords (for AND logic)")

    # Activation control
    enabled: bool = Field(True, description="Whether this entry is active")
    constant: bool = Field(False, description="Always active (keyword-independent)")

    # Positioning - SillyTavern uses before_char/after_char
    position: Literal["before_char", "after_char", "at_depth", "EMTop", "EMBottom", "ANTop", "ANBottom"] = Field(
        "after_char", description="Where to inject content"
    )

    # Priority for sorting (higher = earlier) - this is insertion_order in ST
    insertion_order: int = Field(0, ge=0, description="Order when priority ties (ST: priority field)")

    # Filtering
    case_sensitive: Optional[bool] = Field(False, description="Keyword case sensitivity")
    selective: bool = Field(False, description="Require ALL keywords (AND) vs ANY (OR)")
    use_regex: bool = Field(False, description="Treat keys as regex patterns")

    # Extensions - SillyTavern stores depth and probability here
    extensions: Dict[str, Any] = Field(
        default_factory=lambda: {"depth": 4, "probability": 100},
        description="SillyTavern extensions (depth, probability, etc.)"
    )

    @field_validator("keys", "secondary_keys")
    @classmethod
    def validate_keys(cls, v: List[str]) -> List[str]:
        """Clean and normalize keys."""
        return [k.strip() for k in v if k.strip()]  # Don't lower() - let user control case

    @property
    def depth(self) -> int:
        """Get scan depth from extensions (SillyTavern compatibility)."""
        return self.extensions.get("depth", 4)

    @property
    def probability(self) -> int:
        """Get probability from extensions (SillyTavern: 0-100 scale)."""
        return self.extensions.get("probability", 100)

    @property
    def priority(self) -> int:
        """Get priority (alias for insertion_order for compatibility)."""
        return self.insertion_order

    @property
    def keywords(self) -> List[str]:
        """Get primary keywords (alias for keys for compatibility)."""
        return self.keys

    @property
    def secondary_keywords(self) -> List[str]:
        """Get secondary keywords (alias for secondary_keys for compatibility)."""
        return self.secondary_keys

    @property
    def keysecondary(self) -> List[str]:
        """Get secondary keywords (SillyTavern field name compatibility)."""
        return self.secondary_keys

    def get_token_budget(self) -> int:
        """Get token budget for this entry."""
        # Check extensions first
        ext = LorebookExtensions(**self.extensions)
        if ext.token_budget_override:
            return ext.token_budget_override
        return None


class Lorebook(BaseModel):
    """A collection of lorebook entries.

    Can be stored as a standalone YAML file or embedded in a character card.
    """

    # Identification
    name: str = Field(..., min_length=1, max_length=100, description="Lorebook name")
    description: Optional[str] = Field(None, description="What this lorebook contains")
    category: Literal["context", "character", "world", "custom"] = Field("custom", description="Lorebook category")
    version: str = Field("1.0", description="Lorebook version")
    author: Optional[str] = Field(None, description="Creator name")

    # Scanning configuration
    scan_depth: int = Field(10, ge=0, description="How many messages back to scan for keywords")
    token_budget: int = Field(2000, ge=0, description="Max tokens for all activated entries")
    recursive_scanning: bool = Field(True, description="Scan entire conversation history")

    # Entries
    entries: List[LorebookEntry] = Field(default_factory=list, description="Lorebook entries")

    # Extensions
    extensions: Dict[str, Any] = Field(default_factory=dict, description="Global extensions")

    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation time")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Last update")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is filename-safe."""
        clean = re.sub(r'[<>:"/\\|?*]', '', v)
        clean = re.sub(r'\s+', '_', clean)
        if not clean:
            raise ValueError("Name must contain valid characters")
        return clean

    def get_enabled_entries(self) -> List[LorebookEntry]:
        """Get only enabled entries sorted by insertion_order."""
        enabled = [e for e in self.entries if e.enabled]
        # Sort by insertion_order (lower = first), then by id
        return sorted(enabled, key=lambda e: (e.insertion_order, e.id))

    def add_entry(self, entry: LorebookEntry) -> None:
        """Add an entry to the lorebook."""
        # Check for duplicate ID
        existing_ids = {e.id for e in self.entries}
        if entry.id in existing_ids:
            raise ValueError(f"Entry ID '{entry.id}' already exists")
        self.entries.append(entry)
        self.updated_at = datetime.now()

    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing entry."""
        for entry in self.entries:
            if str(entry.id) == entry_id:
                for key, value in updates.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)
                self.updated_at = datetime.now()
                return True
        return False

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry from the lorebook."""
        for i, entry in enumerate(self.entries):
            if str(entry.id) == entry_id:
                self.entries.pop(i)
                self.updated_at = datetime.now()
                return True
        return False

    def get_entry(self, entry_id: str) -> Optional[LorebookEntry]:
        """Get an entry by ID."""
        for entry in self.entries:
            if str(entry.id) == entry_id:
                return entry
        return None


class LorebookRegistry(BaseModel):
    """Registry of all available lorebooks."""

    lorebooks: Dict[str, Lorebook] = Field(default_factory=dict, description="Registered lorebooks by ID")
    last_sync: Optional[datetime] = Field(None, description="Last sync with filesystem")

    def register(self, lorebook: Lorebook) -> None:
        """Register a lorebook."""
        lorebook_id = lorebook.name.lower().replace(" ", "_")
        self.lorebooks[lorebook_id] = lorebook

    def get(self, lorebook_id: str) -> Optional[Lorebook]:
        """Get a lorebook by ID."""
        return self.lorebooks.get(lorebook_id)

    def list_all(self) -> List[Lorebook]:
        """List all registered lorebooks."""
        return list(self.lorebooks.values())

    def search(self, query: str) -> List[Lorebook]:
        """Search lorebooks by name, description, or entry keywords."""
        query_lower = query.lower()
        results = []

        for lorebook in self.lorebooks.values():
            # Search name/description
            if query_lower in lorebook.name.lower():
                results.append(lorebook)
                continue

            if lorebook.description and query_lower in lorebook.description.lower():
                results.append(lorebook)
                continue

            # Search entry keys (SillyTavern field name)
            for entry in lorebook.entries:
                if any(query_lower in kw.lower() for kw in entry.keys):
                    results.append(lorebook)
                    break

        return results


# Default lorebook templates
DEFAULT_LOREBOOKS = {
    "fantasy_world": Lorebook(
        name="fantasy_world",
        description="Generic fantasy world setting",
        category="context",
        scan_depth=10,
        token_budget=2000,
        entries=[
            LorebookEntry(
                id=0,
                name="The Capital City",
                comment="Main setting info",
                keys=["capital", "royal city", "throne room", "castle", "palace"],
                content="""The Kingdom of Aethoria's capital is Stormhold, a magnificent city built around the peaks of the Celestial Mountains. The Royal Palace sits at the highest point, its towers touching the clouds.

Key locations:
- The Grand Plaza: Central square with markets and festivities
- The Palace District: Noble residences and government buildings
- The Merchant Quarter: Trade hub with exotic goods
- The Old City: Ancient ruins from the founding era""",
                constant=False,
                position="after_char",
                insertion_order=0,
                enabled=True,
                extensions={"depth": 4, "probability": 100},
            ),
            LorebookEntry(
                id=1,
                name="Magic System",
                comment="Always include this context",
                keys=[],
                content="""Magic in this world flows from Ley Lines that crisscross the land. Mages can draw power from these lines, but must be careful not to deplete them.

Key rules:
- Magic requires training and focus
- Overuse can exhaust the mage
- Some locations are stronger in magic than others
- Ley Lines can be sensed by those with the gift""",
                constant=True,
                position="before_char",
                insertion_order=0,
                enabled=True,
                extensions={"depth": 4, "probability": 100},
            ),
        ],
    ),

    "character_backgrounds": Lorebook(
        name="character_backgrounds",
        description="Character backstories and relationships",
        category="custom",
        scan_depth=15,
        token_budget=1500,
        entries=[
            LorebookEntry(
                id=0,
                name="Asuka Langley Soryu",
                comment="Asuka's personality and background",
                keys=["asuka", "pilot", "evangelion", "redhead"],
                secondary_keys=["langley", "soryu"],
                content="""Asuka Langley Soryu is the pilot of Evangelion Unit-02. She is proud, competitive, and seeks validation through her piloting skills.

Personality traits:
- Fiercely independent and confident
- Hides vulnerability behind a tough exterior
- Competitive with others, especially Shinji
- Deeply affected by past trauma
- Seeks approval and recognition

Speech patterns:
- Assertive and direct
- Uses insults or teasing as defense mechanism
- Gets frustrated when things don't go her way
- Shows moments of genuine kindness rarely""",
                constant=False,
                position="before_char",
                insertion_order=0,
                enabled=True,
                extensions={"depth": 4, "probability": 100, "mood": "defensive", "characters": ["shinji", "rei", "misato"]},
            ),
        ],
    ),
}
