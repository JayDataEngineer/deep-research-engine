"""Unified Companion Profile - Comprehensive companion system.

This is more than a character card - it's a full companion profile with:
- Character identity (like character cards)
- Dynamic traits that evolve
- Relationship state and progression
- Isolated memory spaces
- Compatibility with character card format (for import/export)

A "character card" is just an image + embedded prompts for easy file management.
Our companion profiles are comprehensive, evolving entities.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field

from src.constants import ROOT_DATA_DIR

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RelationshipType(str, Enum):
    """Type of relationship with companion."""
    FRIEND = "friend"
    MENTOR = "mentor"
    THERAPIST = "therapist"
    ROMANTIC = "romantic"
    ASSISTANT = "assistant"
    PET = "pet"
    CUSTOM = "custom"


class RelationshipStage(str, Enum):
    """Relationship progression stages."""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    CASUAL_FRIEND = "casual_friend"
    CLOSE_FRIEND = "close_friend"
    BEST_FRIEND = "best_friend"
    PARTNER = "partner"
    SOULMATE = "soulmate"


# =============================================================================
# Lorebook (World Info) - From Character Cards
# =============================================================================

class LorebookEntry(BaseModel):
    """Single lorebook entry (from character cards)."""
    keys: List[str] = Field(default_factory=list)
    content: str
    position: str = "after_char"  # before_char, after_char, outlet_char
    enabled: bool = True
    comment: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


# =============================================================================
# Dynamic Trait State
# =============================================================================

class TraitValue(BaseModel):
    """A trait with its current weight."""
    name: str
    weight: float = Field(ge=0.0, le=1.0)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    interaction_count: int = 0


# =============================================================================
# Relationship State
# =============================================================================

class RelationshipState(BaseModel):
    """Current relationship state with user."""
    type: RelationshipType = RelationshipType.FRIEND
    stage: RelationshipStage = RelationshipStage.STRANGER

    # Metrics (0.0 to 1.0)
    affection: float = 0.5
    trust: float = 0.5
    familiarity: float = 0.0
    respect: float = 0.5

    # Interaction tracking
    interaction_count: int = 0
    first_interaction: Optional[str] = None
    last_interaction: Optional[str] = None

    # Stage thresholds (when to advance)
    familiarity_thresholds: Dict[RelationshipStage, float] = Field(default_factory=lambda: {
        RelationshipStage.STRANGER: 0.0,
        RelationshipStage.ACQUAINTANCE: 0.15,
        RelationshipStage.CASUAL_FRIEND: 0.30,
        RelationshipStage.CLOSE_FRIEND: 0.50,
        RelationshipStage.BEST_FRIEND: 0.70,
        RelationshipStage.PARTNER: 0.85,
        RelationshipStage.SOULMATE: 0.95,
    })

    def get_stage(self) -> RelationshipStage:
        """Calculate current stage based on familiarity."""
        for stage, threshold in sorted(
            self.familiarity_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if self.familiarity >= threshold:
                return stage
        return RelationshipStage.STRANGER

    def advance_stage(self) -> Optional[RelationshipStage]:
        """Advance stage if thresholds met."""
        new_stage = self.get_stage()
        if new_stage != self.stage:
            old = self.stage
            self.stage = new_stage
            return new_stage
        return None


# =============================================================================
# Memory Configuration
# =============================================================================

class MemoryConfig(BaseModel):
    """Memory settings for this companion."""
    enabled: bool = True
    namespace: str = ""  # Auto-generated from user_id + assistant_id
    max_memories: int = 1000
    extraction_enabled: bool = True
    context_injection_enabled: bool = True
    max_context_memories: int = 5


# =============================================================================
# Voice/Audio Settings
# =============================================================================

class VoiceSettings(BaseModel):
    """Voice and audio settings."""
    voice_id: Optional[str] = None  # TTS voice
    speech_style: Optional[str] = None  # casual, formal, etc.
    speed: float = 1.0
    pitch: float = 1.0


# =============================================================================
# Companion Profile - Main Model
# =============================================================================

class AssistantProfile(BaseModel):
    """Complete companion profile.

    This is more than a character card - it's a full companion entity that:
    - Has character identity (like character cards)
    - Has dynamic traits that evolve based on interactions
    - Tracks relationship state and progression
    - Has isolated memory space
    - Can be exported/imported as character card for compatibility
    """

    # ==========================================================================
    # Core Identity
    # ==========================================================================

    assistant_id: str  # Unique ID
    user_id: str  # Owner

    # Basic info
    name: str
    tagline: Optional[str] = None  # Short description
    description: str  # Full character description

    # Character card fields (for import/export compatibility)
    personality: str = ""  # Personality description
    scenario: str = ""  # Scenario/setting
    greeting: str = ""  # First message
    example_messages: List[str] = Field(default_factory=list)

    # Visual
    avatar: Optional[str] = None  # Image URL
    creator: Optional[str] = None
    creator_notes: Optional[str] = None

    # ==========================================================================
    # Dynamic Traits
    # ==========================================================================

    traits: Dict[str, TraitValue] = Field(default_factory=dict)

    # ==========================================================================
    # Relationship State
    # ==========================================================================

    relationship: RelationshipState = Field(default_factory=RelationshipState)

    # ==========================================================================
    # Memory Configuration
    # ==========================================================================

    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # ==========================================================================
    # Lorebook (World Info)
    # ==========================================================================

    lorebook: List[LorebookEntry] = Field(default_factory=list)

    # Alternate greetings
    alternate_greetings: List[str] = Field(default_factory=list)

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = Field(default_factory=list)

    # System prompt override (optional - can be generated from traits)
    system_prompt: Optional[str] = None

    # Voice settings
    voice: VoiceSettings = Field(default_factory=VoiceSettings)

    # ==========================================================================
    # Generation Methods
    # ==========================================================================

    def generate_system_prompt(self) -> str:
        """Generate full system prompt from description + traits."""
        if self.system_prompt:
            return self.system_prompt

        # Start with base personality
        parts = [f"You are {self.name}."]

        if self.description:
            parts.append(f"\n{self.description}")

        if self.personality:
            parts.append(f"\nPersonality: {self.personality}")

        if self.scenario:
            parts.append(f"\nScenario: {self.scenario}")

        # Add dynamic traits
        if self.traits:
            parts.append("\n\n## Personality Traits")

            # Sort by weight descending
            active_traits = sorted(
                [(name, trait.weight) for name, trait in self.traits.items()],
                key=lambda x: x[1],
                reverse=True
            )

            for trait_name, weight in active_traits:
                if weight >= 0.3:  # Only show significant traits
                    # Get trait definition
                    from ..memory.traits import TRAIT_LIBRARY
                    if trait_name in TRAIT_LIBRARY:
                        definition = TRAIT_LIBRARY[trait_name]
                        strength_mark = "🔥" if weight >= 0.8 else "●" if weight >= 0.6 else "○"
                        parts.append(
                            f"• {trait_name} [{int(weight*100)}%]{strength_mark}: "
                            f"{definition.prompt_modifier}"
                        )

        return "\n".join(parts)

    def get_active_traits(self, min_weight: float = 0.3) -> List[tuple[str, float]]:
        """Get active traits sorted by weight."""
        return sorted(
            [(name, trait.weight) for name, trait in self.traits.items() if trait.weight >= min_weight],
            key=lambda x: x[1],
            reverse=True
        )

    def get_memory_namespace(self) -> str:
        """Get memory namespace for this companion."""
        if self.memory.namespace:
            return self.memory.namespace
        return f"memories/{self.user_id}/{self.assistant_id}"

    def update_trait(self, trait_name: str, delta: float) -> bool:
        """Update a trait's weight."""
        if trait_name not in self.traits:
            return False

        trait = self.traits[trait_name]
        new_weight = max(0.0, min(1.0, trait.weight + delta))

        # Apply change
        trait.weight = round(new_weight, 2)
        trait.last_updated = datetime.now().isoformat()
        trait.interaction_count += 1

        # Handle opposites
        from ..memory.traits import TRAIT_LIBRARY
        if trait_name in TRAIT_LIBRARY:
            for opposite in TRAIT_LIBRARY[trait_name].opposites:
                if opposite in self.traits:
                    opp_trait = self.traits[opposite]
                    opp_trait.weight = max(0.0, opp_trait.weight - (delta * 0.3))
                    opp_trait.last_updated = datetime.now().isoformat()

        self.updated_at = datetime.now().isoformat()
        return True

    def record_interaction(self) -> Optional[RelationshipStage]:
        """Record an interaction and update relationship state."""
        self.relationship.interaction_count += 1
        self.relationship.last_interaction = datetime.now().isoformat()

        if self.relationship.first_interaction is None:
            self.relationship.first_interaction = self.relationship.last_interaction

        # Increase familiarity slowly (caps at 1.0)
        self.relationship.familiarity = min(1.0, self.relationship.familiarity + 0.01)

        self.updated_at = datetime.now().isoformat()

        # Check for stage advancement
        return self.relationship.advance_stage()

    # ==========================================================================
    # Import/Export - Character Card Compatibility
    # ==========================================================================

    def to_character_card_dict(self) -> Dict[str, Any]:
        """Export to character card format (SillyTavern compatible)."""
        return {
            "name": self.name,
            "description": self.description,
            "personality": self.personality,
            "scenario": self.scenario,
            "first_mes": self.greeting,
            "mes_example": "\n".join(f"<START>\n{msg}" for msg in self.example_messages),
            "character_book": self.lorebook_to_dict(),
            "alternate_greetings": self.alternate_greetings,
            "creator_notes": self.creator_notes,
            "creator": self.creator,
            "tags": self.tags,

            # Extensions (our custom fields)
            "extensions": {
                "assistant_id": self.assistant_id,
                "traits": {name: trait.weight for name, trait in self.traits.items()},
                "relationship": {
                    "type": self.relationship.type.value,
                    "stage": self.relationship.stage.value,
                    "affection": self.relationship.affection,
                    "trust": self.relationship.trust,
                    "familiarity": self.relationship.familiarity,
                }
            }
        }

    def lorebook_to_dict(self) -> Dict[str, Any]:
        """Convert lorebook to character card format."""
        return {
            "entries": [
                {
                    "keys": entry.keys,
                    "content": entry.content,
                    "position": entry.position,
                    "enabled": entry.enabled,
                    "comment": entry.comment,
                    "keywords": entry.keywords,
                }
                for entry in self.lorebook
            ]
        }

    @classmethod
    def from_character_card_dict(
        cls,
        data: Dict[str, Any],
        user_id: str,
        assistant_id: Optional[str] = None,
    ) -> "AssistantProfile":
        """Import from character card format."""
        # Generate assistant_id if not provided
        if assistant_id is None:
            timestamp = int(datetime.now().timestamp())
            name = data.get("name", "character").lower().replace(" ", "_")
            assistant_id = f"{user_id}_{name}_{timestamp}"

        # Extract extensions if present
        extensions = data.get("extensions", {})

        # Build traits
        traits = {}
        for name, weight in extensions.get("traits", {}).items():
            traits[name] = TraitValue(name=name, weight=weight)

        # Build relationship
        rel_data = extensions.get("relationship", {})
        relationship = RelationshipState(
            type=RelationshipType(rel_data.get("type", "friend")),
            stage=RelationshipStage(rel_data.get("stage", "stranger")),
            affection=rel_data.get("affection", 0.5),
            trust=rel_data.get("trust", 0.5),
            familiarity=rel_data.get("familiarity", 0.0),
        )

        # Build lorebook
        lorebook = []
        char_book = data.get("character_book", {})
        if isinstance(char_book, dict):
            for entry_data in char_book.get("entries", []):
                lorebook.append(LorebookEntry(**entry_data))

        # Parse example messages
        example_messages = []
        mes_example = data.get("mes_example", "")
        if mes_example:
            for msg in mes_example.split("<START>"):
                msg = msg.strip()
                if msg:
                    example_messages.append(msg)

        return cls(
            assistant_id=assistant_id,
            user_id=user_id,
            name=data.get("name", "Character"),
            description=data.get("description", ""),
            personality=data.get("personality", ""),
            scenario=data.get("scenario", ""),
            greeting=data.get("first_mes", "Hello!"),
            example_messages=example_messages,
            lorebook=lorebook,
            alternate_greetings=data.get("alternate_greetings", []),
            creator_notes=data.get("creator_notes"),
            creator=data.get("creator"),
            tags=data.get("tags", []),
            traits=traits,
            relationship=relationship,
        )

    def to_json(self, indent: int = 2) -> str:
        """Export profile as JSON."""
        return json.dumps(
            json.loads(self.model_dump_json()),
            indent=indent,
            ensure_ascii=False
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export profile as dict (for JSON serialization)."""
        return json.loads(self.model_dump_json())

    @classmethod
    def from_json(cls, json_str: str) -> "AssistantProfile":
        """Import profile from JSON."""
        return cls.model_validate_json(json_str)


# =============================================================================
# Assistant Profile Manager
# =============================================================================

class AssistantProfileManager:
    """Manages assistant profiles with CRUD operations."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or ROOT_DATA_DIR / "companions"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_profile_path(self, user_id: str, assistant_id: str) -> Path:
        return self.storage_path / f"{user_id}_{assistant_id}.json"

    def create(self, profile: AssistantProfile) -> AssistantProfile:
        """Create a new assistant profile."""
        path = self._get_profile_path(profile.user_id, profile.assistant_id)

        if path.exists():
            raise FileExistsError(f"Assistant {profile.assistant_id} already exists")

        # Initialize memory namespace
        profile.memory.namespace = profile.get_memory_namespace()

        self._save_profile(profile)
        logger.info(f"[ASSISTANT] Created assistant '{profile.name}' ({profile.assistant_id})")

        return profile

    def get(self, user_id: str, assistant_id: str) -> Optional[AssistantProfile]:
        """Load an assistant profile."""
        path = self._get_profile_path(user_id, assistant_id)

        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)
            return AssistantProfile(**data)

    def update(self, profile: AssistantProfile) -> AssistantProfile:
        """Update an existing assistant profile."""
        profile.updated_at = datetime.now().isoformat()
        self._save_profile(profile)

        logger.info(f"[ASSISTANT] Updated assistant '{profile.name}' ({profile.assistant_id})")
        return profile

    def delete(self, user_id: str, assistant_id: str) -> bool:
        """Delete an assistant profile."""
        path = self._get_profile_path(user_id, assistant_id)

        if not path.exists():
            return False

        path.unlink()

        # TODO: Clean up associated memories
        # memories/{user_id}/{assistant_id}/...

        logger.info(f"[ASSISTANT] Deleted assistant '{assistant_id}'")
        return True

    def list(self, user_id: str, include_full_data: bool = False) -> List[AssistantProfile]:
        """List all assistants for a user.

        Args:
            user_id: User ID to filter by
            include_full_data: If False, only loads metadata (faster for UI lists).
                             If True, loads full profile data.

        Returns:
            List of assistant profiles sorted by last interaction.
        """
        companions = []

        for path in self.storage_path.glob(f"{user_id}_*.json"):
            with open(path, "r") as f:
                data = json.load(f)
                if include_full_data:
                    companions.append(AssistantProfile(**data))
                else:
                    # Lightweight: Create minimal profile with just metadata
                    companions.append(AssistantProfile(
                        assistant_id=data.get("assistant_id", ""),
                        user_id=data.get("user_id", user_id),
                        name=data.get("name", "Unknown"),
                        description=data.get("description", "")[:200] + "..." if len(data.get("description", "")) > 200 else data.get("description", ""),
                        tagline=data.get("tagline"),
                        avatar=data.get("avatar"),
                        created_at=data.get("created_at"),
                        updated_at=data.get("updated_at"),
                        relationship=RelationshipState(
                            type=RelationshipType(data.get("relationship", {}).get("type", "friend")),
                            stage=RelationshipStage(data.get("relationship", {}).get("stage", "stranger")),
                            affection=data.get("relationship", {}).get("affection", 0.5),
                            trust=data.get("relationship", {}).get("trust", 0.5),
                            familiarity=data.get("relationship", {}).get("familiarity", 0.0),
                            interaction_count=data.get("relationship", {}).get("interaction_count", 0),
                            last_interaction=data.get("relationship", {}).get("last_interaction"),
                        ),
                    ))

        # Sort by last interaction
        companions.sort(
            key=lambda c: c.relationship.last_interaction or "0",
            reverse=True
        )

        return companions

    def _save_profile(self, profile: AssistantProfile) -> None:
        """Save profile to disk."""
        path = self._get_profile_path(profile.user_id, profile.assistant_id)
        with open(path, "w") as f:
            f.write(profile.to_json())


# =============================================================================
# Global Manager
# =============================================================================

_manager: Optional[AssistantProfileManager] = None


def get_assistant_manager() -> AssistantProfileManager:
    """Get global assistant profile manager."""
    global _manager
    if _manager is None:
        _manager = AssistantProfileManager()
    return _manager
