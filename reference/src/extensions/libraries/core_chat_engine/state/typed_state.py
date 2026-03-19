"""
Typed state models for Companion Brain v2.

Provides properly typed state models for LangGraph using dataclasses.
Compatible with the existing semantic-rp-system codebase.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from datetime import datetime
from enum import Enum

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


class UserMood(str, Enum):
    """Detected user mood states."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    BORED = "bored"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"


class IntentType(str, Enum):
    """Types of detected user intents."""
    CASUAL_CHAT = "casual_chat"
    CREATIVE_REQUEST = "creative_request"
    QUESTION = "question"
    ROLEPLAY_INITIATION = "roleplay_initiation"
    COMMAND = "command"
    EMOTIONAL_SUPPORT = "emotional_support"
    OOC_COMMENT = "ooc_comment"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    primary: IntentType
    confidence: float = 1.0
    secondary: Optional[IntentType] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class ProcessedResponse:
    """Final processed response ready for output."""
    text: str
    intent: IntentResult
    source: str = "direct"
    processing_time: float = 0.0
    was_routed: bool = False
    confidence: float = 1.0
    tts_params: Dict[str, Any] = field(default_factory=dict)
    avatar_params: Dict[str, Any] = field(default_factory=dict)
    ui_elements: List[Dict[str, Any]] = field(default_factory=list)
