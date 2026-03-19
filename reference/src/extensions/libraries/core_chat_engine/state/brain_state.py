"""
Brain State Management
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime
from enum import Enum

from .intent_types import IntentType, IntentResult


class UserMood(Enum):
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


@dataclass
class ConversationTurn:
    """A single conversation turn."""

    turn_number: int
    user_input: str
    assistant_response: str
    timestamp: datetime
    intent: IntentType
    was_routed: bool = False
    tool_used: Optional[str] = None
    mood_detected: Optional[UserMood] = None


@dataclass
class ConversationState:
    """State of the ongoing conversation."""

    turn_count: int = 0
    turns: List[ConversationTurn] = field(default_factory=list)

    # Topic tracking
    current_topic: Optional[str] = None
    topic_history: List[str] = field(default_factory=list)

    # Summary for long conversations
    summary: str = ""
    summary_turn: int = 0  # Which turn we last summarized at

    # Context window (recent messages)
    context_window: List[Dict[str, str]] = field(default_factory=list)
    max_context_turns: int = 20

    # User state
    user_mood: Optional[UserMood] = None
    user_mood_history: List[tuple[datetime, UserMood]] = field(default_factory=list)

    # Session metadata
    session_start: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def add_turn(
        self,
        user_input: str,
        response: str,
        intent: IntentType,
        was_routed: bool = False,
        tool_used: Optional[str] = None,
    ) -> ConversationTurn:
        """Add a turn to conversation history."""
        self.turn_count += 1

        turn = ConversationTurn(
            turn_number=self.turn_count,
            user_input=user_input,
            assistant_response=response,
            timestamp=datetime.now(),
            intent=intent,
            was_routed=was_routed,
            tool_used=tool_used,
            mood_detected=self.user_mood,
        )

        self.turns.append(turn)
        self.last_activity = datetime.now()

        # Update context window
        self.context_window.append({"role": "user", "content": user_input})
        self.context_window.append({"role": "assistant", "content": response})

        # Trim context window if needed
        if len(self.context_window) > self.max_context_turns * 2:
            self.context_window = self.context_window[-self.max_context_turns * 2 :]

        return turn

    def get_recent_context(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation context."""
        recent_turns = self.context_window[-max_turns * 2 :]
        return recent_turns

    def should_summarize(self, interval: int = 10) -> bool:
        """Check if conversation needs summarization."""
        return (self.turn_count - self.summary_turn) >= interval

    def update_summary(self, summary: str) -> None:
        """Update conversation summary."""
        self.summary = summary
        self.summary_turn = self.turn_count

    def update_mood(self, mood: UserMood) -> None:
        """Update user mood."""
        self.user_mood = mood
        self.user_mood_history.append((datetime.now(), mood))

    def get_conversation_duration(self) -> float:
        """Get conversation duration in minutes."""
        return (datetime.now() - self.session_start).total_seconds() / 60


@dataclass
class UserProfile:
    """Learned user preferences and profile."""

    user_id: str = "default"

    # Communication preferences
    preferred_style: str = "balanced"  # brief, balanced, elaborate
    formality: str = "casual"  # casual, formal, mixed
    humor_tolerance: float = 0.5  # 0.0 to 1.0

    # Topic interests
    interests: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)

    # Relationship
    relationship_level: str = "new"  # new, familiar, close
    shared_jokes: List[str] = field(default_factory=list)
    shared_memory: List[str] = field(default_factory=list)

    # Behavior patterns
    typical_turn_count: float = 5.0  # Rolling average
    session_count: int = 0
    total_turns: int = 0

    # Emotional patterns
    mood_history: List[tuple[datetime, UserMood]] = field(default_factory=list)
    support_triggers: List[str] = field(default_factory=list)  # Topics needing care

    def record_session(self, turn_count: int, duration: float) -> None:
        """Record a completed session."""
        self.session_count += 1
        self.total_turns += turn_count

        # Update rolling average
        alpha = 0.2  # Smoothing factor
        self.typical_turn_count = alpha * turn_count + (1 - alpha) * self.typical_turn_count

    def add_interest(self, topic: str) -> None:
        """Add a topic interest."""
        if topic not in self.interests:
            self.interests.append(topic)

    def avoid_topic(self, topic: str) -> None:
        """Add a topic to avoid."""
        if topic not in self.avoided_topics:
            self.avoided_topics.append(topic)

    def add_shared_memory(self, memory: str) -> None:
        """Add a shared memory."""
        self.shared_memory.append(memory)


@dataclass
class ProcessedInput:
    """Processed user input."""

    raw: str
    clean: str  # Stripped of markers
    commands: List[str] = field(default_factory=list)  # Detected commands
    ooc_content: Optional[str] = None  # OOC content if detected
    story_mode: bool = False
    multimodal: Dict[str, Any] = field(default_factory=dict)  # Images, files, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a specialist tool."""

    source: str  # Tool name or "direct"
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class ProcessedResponse:
    """Final processed response ready for output."""

    text: str
    intent: IntentResult
    source: str  # "direct" or tool name

    # Multimodal parameters
    tts_params: Dict[str, Any] = field(default_factory=dict)
    avatar_params: Dict[str, Any] = field(default_factory=dict)
    ui_elements: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    processing_time: float = 0.0
    was_routed: bool = False
    confidence: float = 1.0


@dataclass
class BrainState:
    """Complete state of the companion brain."""

    # Core components state
    conversation: ConversationState = field(default_factory=ConversationState)
    user_profile: UserProfile = field(default_factory=UserProfile)

    # Current processing
    current_input: Optional[ProcessedInput] = None
    current_intent: Optional[IntentResult] = None
    current_response: Optional[str] = None

    # Routing history
    last_tool_used: Optional[str] = None
    tool_history: List[str] = field(default_factory=list)

    # Mode
    mode: str = "companion"  # "companion" or "rp"

    def reset(self) -> None:
        """Reset state (start new conversation)."""
        self.conversation = ConversationState()
        self.current_input = None
        self.current_intent = None
        self.current_response = None
        self.last_tool_used = None
        self.tool_history = []
