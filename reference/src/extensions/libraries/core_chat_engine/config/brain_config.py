"""
Brain Configuration
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List

from .personality_config import PersonalityConfig, CommunicationStyle


@dataclass
class BrainConfig:
    """Configuration for the companion brain."""

    # Core settings
    mode: str = "companion"  # "companion" or "rp"
    model: Any = None  # LLM model

    # Personality
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)

    # Components
    enable_tool_routing: bool = True
    enable_conversation_summary: bool = True
    enable_mood_tracking: bool = True

    # Thresholds
    intent_confidence_threshold: float = 0.6
    routing_cost_threshold: float = 0.5

    # Conversation settings
    max_context_turns: int = 20
    summarize_interval: int = 10
    context_window_size: int = 10

    # Generation settings
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0

    # Special features
    enable_ooc: bool = True
    enable_story_mode: bool = True
    enable_slash_commands: bool = True

    # Registered tools
    available_tools: List[str] = field(default_factory=lambda: [
        "research",
        "investing",
        "coding",
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode,
            "personality": self.personality.to_dict(),
            "enable_tool_routing": self.enable_tool_routing,
            "enable_conversation_summary": self.enable_conversation_summary,
            "enable_mood_tracking": self.enable_mood_tracking,
            "intent_confidence_threshold": self.intent_confidence_threshold,
            "routing_cost_threshold": self.routing_cost_threshold,
            "max_context_turns": self.max_context_turns,
            "summarize_interval": self.summarize_interval,
            "context_window_size": self.context_window_size,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_ooc": self.enable_ooc,
            "enable_story_mode": self.enable_story_mode,
            "enable_slash_commands": self.enable_slash_commands,
            "available_tools": self.available_tools,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrainConfig":
        """Create from dictionary."""
        personality_data = data.get("personality", {})
        personality = PersonalityConfig.from_dict(personality_data)

        return cls(
            mode=data.get("mode", "companion"),
            model=data.get("model"),
            personality=personality,
            enable_tool_routing=data.get("enable_tool_routing", True),
            enable_conversation_summary=data.get("enable_conversation_summary", True),
            enable_mood_tracking=data.get("enable_mood_tracking", True),
            intent_confidence_threshold=data.get("intent_confidence_threshold", 0.6),
            routing_cost_threshold=data.get("routing_cost_threshold", 0.5),
            max_context_turns=data.get("max_context_turns", 20),
            summarize_interval=data.get("summarize_interval", 10),
            context_window_size=data.get("context_window_size", 10),
            max_tokens=data.get("max_tokens", 500),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k", 0),
            enable_ooc=data.get("enable_ooc", True),
            enable_story_mode=data.get("enable_story_mode", True),
            enable_slash_commands=data.get("enable_slash_commands", True),
            available_tools=data.get("available_tools", ["research", "investing", "coding"]),
        )


# Preset configurations
PRESET_CONFIGS = {
    "standard": BrainConfig(
        mode="companion",
        personality=PersonalityConfig(
            name="Standard",
            role="AI Assistant",
            description="A helpful AI assistant with tool access",
        ),
        enable_tool_routing=True,
        enable_conversation_summary=False,  # No memory for standard
        enable_mood_tracking=False,  # No mood tracking for standard
        max_context_turns=5,  # Smaller context for standard
    ),
    "companion": BrainConfig(
        mode="companion",
        personality=PersonalityConfig(
            name="Companion",
            role="AI Assistant",
            description="A helpful and friendly AI companion",
        ),
    ),
    "rp": BrainConfig(
        mode="rp",
        personality=PersonalityConfig(
            name="Character",
            role="Roleplay Character",
            description="A roleplay character",
            communication_style=CommunicationStyle.BALANCED,
        ),
    ),
}


def get_preset_config(name: str) -> BrainConfig:
    """Get a preset configuration by name."""
    return PRESET_CONFIGS.get(name.lower(), PRESET_CONFIGS["companion"])
