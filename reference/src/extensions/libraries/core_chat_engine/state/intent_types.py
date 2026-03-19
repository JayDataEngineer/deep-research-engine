"""
Intent Types and Results
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class IntentType(Enum):
    """Types of user intents."""

    # Handle directly (with personality)
    CASUAL_CHAT = "casual_chat"
    EMOTIONAL = "emotional"
    EXCITED = "excited"
    GREETING = "greeting"
    FAREWELL = "farewell"

    # Route to specialists
    RESEARCH_QUERY = "research_query"
    INVESTMENT_TASK = "investment_task"
    CODING_TASK = "coding_task"
    CREATIVE_REQUEST = "creative_request"
    INFORMATION_QUERY = "information"
    ENTERTAINMENT = "entertainment"
    NEWS_QUERY = "news_query"
    WEATHER_QUERY = "weather_query"

    # System
    UNKNOWN = "unknown"
    COMMAND = "command"  # Slash command


@dataclass
class IntentResult:
    """Result of intent classification."""

    primary: IntentType
    confidence: float  # 0.0 to 1.0
    entities: Dict[str, Any] = field(default_factory=dict)
    should_route: bool = False
    route_target: Optional[str] = None  # Tool name if routing
    reasoning: str = ""  # For debugging


# Intent to tool mapping
INTENT_TOOL_MAP = {
    IntentType.RESEARCH_QUERY: "research",
    IntentType.INVESTMENT_TASK: "investing",
    IntentType.CODING_TASK: "coding",
    IntentType.CREATIVE_REQUEST: "creative",
    IntentType.NEWS_QUERY: "news",
    IntentType.WEATHER_QUERY: "weather",
}
