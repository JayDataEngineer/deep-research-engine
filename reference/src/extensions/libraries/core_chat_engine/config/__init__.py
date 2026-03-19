"""
Configuration for Companion Brain
"""

from .personality_config import (
    PersonalityConfig,
    CommunicationStyle,
    MoodMatching,
    PRESET_PERSONALITIES,
    get_preset,
)
from .brain_config import (
    BrainConfig,
    PRESET_CONFIGS,
    get_preset_config,
)

__all__ = [
    "PersonalityConfig",
    "CommunicationStyle",
    "MoodMatching",
    "PRESET_PERSONALITIES",
    "get_preset",
    "BrainConfig",
    "PRESET_CONFIGS",
    "get_preset_config",
]
