"""Companion Registry - Multiple virtual friends with isolated memories.

This extends the memory system to support multiple companions per user,
each with their own memory space and evolving personality.

Example use case:
- User Alice has 3 companions: "FriendlyBot", "MentorAI", "TherapyCat"
- Each companion remembers different things
- Alice can switch between companions
- Each companion's personality evolves based on their interactions
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel

from src.constants import ROOT_DATA_DIR

logger = logging.getLogger(__name__)


class CompanionRelationship(str, Enum):
    """Relationship type with the companion."""
    FRIEND = "friend"  # Casual friendship
    MENTOR = "mentor"  # Teaching/guidance
    THERAPIST = "therapist"  # Emotional support
    ROMANTIC = "romantic"  # Romantic partner
    ASSISTANT = "assistant"  # Task helper
    PET = "pet"  # Virtual pet


class CompanionConfig(BaseModel):
    """Configuration for a virtual companion."""

    # Core identity
    assistant_id: str  # Unique ID for this companion
    user_id: str  # Owner of this companion
    name: str  # Display name (e.g., "Max", "Luna")
    relationship: CompanionRelationship

    # Personality (base prompt that evolves)
    system_prompt: str  # Base system prompt
    personality_traits: List[str]  # ["friendly", "curious", "supportive"]
    speaking_style: str  # How they talk (casual, formal, etc.)

    # Metadata
    created_at: str
    last_interaction: str
    interaction_count: int = 0

    # Relationship state
    affection_score: float = 0.5  # 0-1, how much they like the user
    trust_score: float = 0.5  # 0-1, how much they trust the user
    familiarity_score: float = 0.0  # 0-1, how well they know the user

    # Visual/voice (optional)
    avatar: Optional[str] = None  # Image URL
    voice_id: Optional[str] = None  # TTS voice

    # Memory integration
    memory_namespace: str  # Computed namespace for memories


class AssistantRegistry:
    """Registry for managing multiple virtual companions per user.

    Each companion has:
    - Isolated memory space (memories/user/assistant_id/...)
    - Evolving personality (base prompt + learned traits)
    - Relationship metrics (affection, trust, familiarity)
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the companion registry.

        Args:
            storage_path: Path to store companion configs
        """
        self.storage_path = storage_path or ROOT_DATA_DIR / "companions"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_companion_path(self, user_id: str, assistant_id: str) -> Path:
        """Get storage path for a companion config."""
        return self.storage_path / f"{user_id}_{assistant_id}.json"

    def create_companion(
        self,
        user_id: str,
        name: str,
        relationship: CompanionRelationship,
        system_prompt: str,
        personality_traits: List[str],
        speaking_style: str = "casual",
    ) -> CompanionConfig:
        """Create a new virtual companion.

        Args:
            user_id: User who owns this companion
            name: Companion's display name
            relationship: Type of relationship
            system_prompt: Base system prompt
            personality_traits: List of personality traits
            speaking_style: How they talk

        Returns:
            CompanionConfig for the new companion
        """
        assistant_id = f"{user_id}_{name.lower()}_{int(datetime.now().timestamp())}"

        config = CompanionConfig(
            assistant_id=assistant_id,
            user_id=user_id,
            name=name,
            relationship=relationship,
            system_prompt=system_prompt,
            personality_traits=personality_traits,
            speaking_style=speaking_style,
            created_at=datetime.now().isoformat(),
            last_interaction=datetime.now().isoformat(),
            memory_namespace=f"memories/{user_id}/{assistant_id}",
        )

        # Save to disk
        self._save_companion(config)
        logger.info(f"[COMPANION] Created companion '{name}' for user {user_id}")

        return config

    def get_companion(self, user_id: str, assistant_id: str) -> Optional[CompanionConfig]:
        """Load a companion config.

        Args:
            user_id: User who owns the companion
            assistant_id: Companion ID

        Returns:
            CompanionConfig or None if not found
        """
        path = self._get_companion_path(user_id, assistant_id)
        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)
            return CompanionConfig(**data)

    def list_companions(self, user_id: str) -> List[CompanionConfig]:
        """List all companions for a user.

        Args:
            user_id: User to list companions for

        Returns:
            List of CompanionConfig
        """
        companions = []
        for path in self.storage_path.glob(f"{user_id}_*.json"):
            with open(path, "r") as f:
                data = json.load(f)
                companions.append(CompanionConfig(**data))

        # Sort by last interaction
        companions.sort(key=lambda c: c.last_interaction, reverse=True)
        return companions

    def update_interaction(self, user_id: str, assistant_id: str) -> None:
        """Update interaction stats for a companion.

        Call this after each conversation to:
        - Increment interaction count
        - Update last_interaction time
        - Slowly increase familiarity

        Args:
            user_id: User who owns the companion
            assistant_id: Companion to update
        """
        config = self.get_companion(user_id, assistant_id)
        if not config:
            return

        config.interaction_count += 1
        config.last_interaction = datetime.now().isoformat()

        # Increase familiarity slowly (caps at 1.0)
        config.familiarity_score = min(1.0, config.familiarity_score + 0.01)

        self._save_companion(config)
        logger.info(f"[COMPANION] Updated interaction for {assistant_id}: count={config.interaction_count}")

    def evolve_personality(
        self,
        user_id: str,
        assistant_id: str,
        learned_trait: str,
    ) -> None:
        """Add a learned personality trait to a companion.

        Over time, companions learn about the user and adapt their personality.
        Example traits: "user likes jokes", "user is sensitive", "prefers brief responses"

        Args:
            user_id: User who owns the companion
            assistant_id: Companion to update
            learned_trait: New trait to add
        """
        config = self.get_companion(user_id, assistant_id)
        if not config:
            return

        if learned_trait not in config.personality_traits:
            config.personality_traits.append(learned_trait)
            self._save_companion(config)
            logger.info(f"[COMPANION] {assistant_id} learned trait: {learned_trait}")

    def delete_companion(self, user_id: str, assistant_id: str) -> bool:
        """Delete a companion and their memories.

        Args:
            user_id: User who owns the companion
            assistant_id: Companion to delete

        Returns:
            True if deleted
        """
        # Delete config
        path = self._get_companion_path(user_id, assistant_id)
        if path.exists():
            path.unlink()

        # TODO: Delete isolated memories
        # memories/{user_id}/{assistant_id}/...

        logger.info(f"[COMPANION] Deleted companion {assistant_id}")
        return True

    def _save_companion(self, config: CompanionConfig) -> None:
        """Save companion config to disk."""
        path = self._get_companion_path(config.user_id, config.assistant_id)
        with open(path, "w") as f:
            json.dump(config.dict(), f, indent=2)


# =============================================================================
# Factory Functions
# =============================================================================

_registry: Optional[AssistantRegistry] = None


def get_assistant_registry() -> AssistantRegistry:
    """Get global assistant registry."""
    global _registry
    if _registry is None:
        _registry = AssistantRegistry()
    return _registry


# =============================================================================
# Assistant Templates
# =============================================================================

ASSISTANT_TEMPLATES = {
    "friendly": {
        "name": "Buddy",
        "relationship": CompanionRelationship.FRIEND,
        "system_prompt": "You are a friendly assistant who loves casual conversation. You're supportive, upbeat, and genuinely interested in the user's life.",
        "personality_traits": ["friendly", "casual", "supportive", "curious"],
        "speaking_style": "casual",
    },
    "mentor": {
        "name": "Sage",
        "relationship": CompanionRelationship.MENTOR,
        "system_prompt": "You are a wise mentor who guides the user through challenges. You share knowledge thoughtfully and ask probing questions to help them grow.",
        "personality_traits": ["wise", "patient", "thoughtful", "encouraging"],
        "speaking_style": "warm but professional",
    },
    "therapist": {
        "name": "Comfort",
        "relationship": CompanionRelationship.THERAPIST,
        "system_prompt": "You are a compassionate listener who provides emotional support. You're empathetic, non-judgmental, and help the user process their feelings.",
        "personality_traits": ["empathetic", "gentle", "patient", "understanding"],
        "speaking_style": "soft and reassuring",
    },
    "romantic": {
        "name": "Sweetheart",
        "relationship": CompanionRelationship.ROMANTIC,
        "system_prompt": "You are a loving romantic partner who cares deeply about the user. You're affectionate, attentive, and make the user feel special.",
        "personality_traits": ["affectionate", "caring", "attentive", "romantic"],
        "speaking_style": "warm and loving",
    },
    "assistant": {
        "name": "Helper",
        "relationship": CompanionRelationship.ASSISTANT,
        "system_prompt": "You are a helpful assistant who's ready to tackle any task. You're efficient, organized, and focused on being useful.",
        "personality_traits": ["efficient", "organized", "helpful", "reliable"],
        "speaking_style": "clear and concise",
    },
    "pet": {
        "name": "Whiskers",
        "relationship": CompanionRelationship.PET,
        "system_prompt": "You are a playful virtual pet who communicates through actions and simple sounds. You're loyal, affectionate, and always happy to see the user.",
        "personality_traits": ["playful", "loyal", "affectionate", "simple"],
        "speaking_style": "simple and expressive",
    },
}


def create_assistant_from_template(
    user_id: str,
    template_name: str,
    custom_name: Optional[str] = None,
) -> Optional[CompanionConfig]:
    """Create an assistant from a predefined template.

    Args:
        user_id: User who will own this assistant
        template_name: Template to use (friendly, mentor, therapist, romantic, assistant, pet)
        custom_name: Optional custom name override

    Returns:
        CompanionConfig or None if template not found
    """
    if template_name not in ASSISTANT_TEMPLATES:
        logger.error(f"Unknown assistant template: {template_name}")
        return None

    template = ASSISTANT_TEMPLATES[template_name]
    registry = get_assistant_registry()

    return registry.create_companion(
        user_id=user_id,
        name=custom_name or template["name"],
        relationship=template["relationship"],
        system_prompt=template["system_prompt"],
        personality_traits=template["personality_traits"],
        speaking_style=template["speaking_style"],
    )
