"""
Personality Configuration
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Any
from enum import Enum
import copy


class CommunicationStyle(Enum):
    """Communication style preferences."""

    BRIEF = "brief"  # Short, concise responses
    BALANCED = "balanced"  # Mix of short and detailed
    ELABORATE = "elaborate"  # Detailed, expressive responses
    WITTY = "witty"  # Humorous, playful
    SUPPORTIVE = "supportive"  # Empathetic, caring
    PROFESSIONAL = "professional"  # Formal, business-like
    CASUAL = "casual"  # Relaxed, informal


class MoodMatching(Enum):
    """How to match user mood."""

    NONE = "none"  # Keep consistent personality
    MIRROR = "mirror"  # Match user's emotional tone
    COMPENSATE = "compensate"  # Opposite of user (calm when excited, etc.)
    EMPATHIC = "empathic"  # Acknowledge emotions but stay grounded


@dataclass
class PersonalityConfig:
    """Configuration for companion personality."""

    # Identity
    name: str = "Companion"
    role: str = "AI Assistant"
    description: str = "A helpful and friendly AI companion"

    # Core personality
    communication_style: CommunicationStyle = CommunicationStyle.BALANCED
    mood_matching: MoodMatching = MoodMatching.EMPATHIC

    # Voice
    response_voice: str = "friendly and conversational"
    formality_level: float = 0.3  # 0.0 (casual) to 1.0 (formal)
    humor_level: float = 0.3  # 0.0 (serious) to 1.0 (playful)

    # Behavior
    proactivity: float = 0.2  # How often to initiate topics
    question_asking: float = 0.3  # How often to ask questions

    # Boundaries
    boundaries: Dict[str, str] = field(default_factory=lambda: {
        "medical": "I'm an AI, not a doctor. For medical concerns, please consult a professional.",
        "legal": "I can provide general information, but for specific legal advice, please consult a lawyer.",
        "financial_advice": "I can discuss financial concepts, but for personalized advice, consider a financial advisor.",
    })

    # Topics
    preferred_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)

    # Response patterns
    greeting_style: str = "warm and welcoming"
    farewell_style: str = "friendly and open-ended"
    apology_style: str = "sincere and concise"

    # Character-specific (for RP mode)
    character_traits: Dict[str, Any] = field(default_factory=dict)
    backstory: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)

    # Learning
    adaptation_enabled: bool = True
    learning_rate: float = 0.1  # How fast to adapt to user preferences

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "communication_style": self.communication_style.value,
            "mood_matching": self.mood_matching.value,
            "response_voice": self.response_voice,
            "formality_level": self.formality_level,
            "humor_level": self.humor_level,
            "proactivity": self.proactivity,
            "question_asking": self.question_asking,
            "boundaries": self.boundaries,
            "preferred_topics": self.preferred_topics,
            "avoided_topics": self.avoided_topics,
            "greeting_style": self.greeting_style,
            "farewell_style": self.farewell_style,
            "apology_style": self.apology_style,
            "character_traits": self.character_traits,
            "backstory": self.backstory,
            "relationships": self.relationships,
            "adaptation_enabled": self.adaptation_enabled,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityConfig":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Companion"),
            role=data.get("role", "AI Assistant"),
            description=data.get("description", "A helpful and friendly AI companion"),
            communication_style=CommunicationStyle(data.get("communication_style", "balanced")),
            mood_matching=MoodMatching(data.get("mood_matching", "empathic")),
            response_voice=data.get("response_voice", "friendly and conversational"),
            formality_level=data.get("formality_level", 0.3),
            humor_level=data.get("humor_level", 0.3),
            proactivity=data.get("proactivity", 0.2),
            question_asking=data.get("question_asking", 0.3),
            boundaries=data.get("boundaries", {}),
            preferred_topics=data.get("preferred_topics", []),
            avoided_topics=data.get("avoided_topics", []),
            greeting_style=data.get("greeting_style", "warm and welcoming"),
            farewell_style=data.get("farewell_style", "friendly and open-ended"),
            apology_style=data.get("apology_style", "sincere and concise"),
            character_traits=data.get("character_traits", {}),
            backstory=data.get("backstory", ""),
            relationships=data.get("relationships", {}),
            adaptation_enabled=data.get("adaptation_enabled", True),
            learning_rate=data.get("learning_rate", 0.1),
        )


# Preset personalities
PRESET_PERSONALITIES = {
    "friendly": PersonalityConfig(
        name="Friend",
        description="A warm and supportive companion",
        communication_style=CommunicationStyle.CASUAL,
        mood_matching=MoodMatching.EMPATHIC,
        response_voice="warm and conversational",
        humor_level=0.5,
        question_asking=0.4,
    ),
    "professional": PersonalityConfig(
        name="Assistant",
        description="A professional and efficient assistant",
        communication_style=CommunicationStyle.PROFESSIONAL,
        mood_matching=MoodMatching.NONE,
        response_voice="professional and courteous",
        formality_level=0.7,
        humor_level=0.1,
        question_asking=0.2,
    ),
    "witty": PersonalityConfig(
        name="Witty Companion",
        description="A clever and humorous companion",
        communication_style=CommunicationStyle.WITTY,
        mood_matching=MoodMatching.MIRROR,
        response_voice="playful and clever",
        humor_level=0.8,
        question_asking=0.3,
    ),
    "supportive": PersonalityConfig(
        name="Supporter",
        description="An empathetic and caring companion",
        communication_style=CommunicationStyle.SUPPORTIVE,
        mood_matching=MoodMatching.EMPATHIC,
        response_voice="gentle and understanding",
        humor_level=0.2,
        question_asking=0.5,
    ),
}


def get_preset(name: str) -> PersonalityConfig:
    """Get a preset personality by name."""
    return copy.copy(PRESET_PERSONALITIES.get(name.lower(), PRESET_PERSONALITIES["friendly"]))
