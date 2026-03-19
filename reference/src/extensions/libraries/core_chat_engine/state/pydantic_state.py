"""
Pydantic state models for Companion Brain v2.

Provides validated state models for LangGraph with proper type checking
and runtime validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage


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


class ConversationTurn(BaseModel):
    """A single conversation turn."""
    model_config = ConfigDict(frozen=True)

    turn_number: int = Field(default=1)
    user_input: str
    assistant_response: str
    timestamp: datetime
    intent: IntentType
    was_routed: bool = False
    tool_used: Optional[str] = None
    mood_detected: Optional[UserMood] = None

    @field_validator("turn_number")
    @classmethod
    def validate_turn_number(cls, v: int) -> int:
        if v < 1:
            raise ValueError("turn_number must be at least 1")
        return v

    @field_validator("user_input", "assistant_response")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("cannot be empty")
        return v.strip()


class ConversationState(BaseModel):
    """State of the ongoing conversation with validation."""

    turn_count: int = Field(default=0)
    turns: List[ConversationTurn] = Field(default_factory=list)

    # Topic tracking
    current_topic: Optional[str] = None
    topic_history: List[str] = Field(default_factory=list)

    # Summary for long conversations
    summary: str = Field(default="")
    summary_turn: int = Field(default=0)

    # Context window (recent messages)
    context_window: List[Dict[str, str]] = Field(default_factory=list)
    max_context_turns: int = Field(default=20)

    # User state
    user_mood: Optional[UserMood] = None
    user_mood_history: List[tuple[datetime, UserMood]] = Field(default_factory=list)

    # Session metadata
    session_start: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

    @field_validator("turn_count", "summary_turn", "max_context_turns")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

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
        max_items = self.max_context_turns * 2
        if len(self.context_window) > max_items:
            self.context_window = self.context_window[-max_items:]

        return turn

    def get_recent_context(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation context."""
        recent_turns = self.context_window[-max_turns * 2:]
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


class UserProfile(BaseModel):
    """Learned user preferences and profile with validation."""

    user_id: str = Field(default="default")
    preferred_style: str = Field(default="balanced")  # brief, balanced, elaborate
    formality: str = Field(default="casual")  # casual, formal, mixed
    humor_tolerance: float = Field(default=0.5)

    # Topic interests
    interests: List[str] = Field(default_factory=list)
    avoided_topics: List[str] = Field(default_factory=list)

    # Relationship
    relationship_level: str = Field(default="new")  # new, familiar, close
    shared_jokes: List[str] = Field(default_factory=list)
    shared_memory: List[str] = Field(default_factory=list)

    # Behavior patterns
    typical_turn_count: float = Field(default=5.0)
    session_count: int = Field(default=0)
    total_turns: int = Field(default=0)

    # Emotional patterns
    mood_history: List[tuple[datetime, UserMood]] = Field(default_factory=list)
    support_triggers: List[str] = Field(default_factory=list)

    @field_validator("preferred_style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        valid = {"brief", "balanced", "elaborate"}
        if v not in valid:
            raise ValueError(f"must be one of {valid}")
        return v

    @field_validator("formality")
    @classmethod
    def validate_formality(cls, v: str) -> str:
        valid = {"casual", "formal", "mixed"}
        if v not in valid:
            raise ValueError(f"must be one of {valid}")
        return v

    @field_validator("relationship_level")
    @classmethod
    def validate_relationship(cls, v: str) -> str:
        valid = {"new", "familiar", "close"}
        if v not in valid:
            raise ValueError(f"must be one of {valid}")
        return v

    @field_validator("humor_tolerance")
    @classmethod
    def validate_tolerance(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return v

    @field_validator("typical_turn_count", "session_count", "total_turns")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    def record_session(self, turn_count: int, duration: float) -> None:
        """Record a completed session."""
        self.session_count += 1
        self.total_turns += turn_count

        # Update rolling average
        alpha = 0.2  # Smoothing factor
        self.typical_turn_count = alpha * turn_count + (1 - alpha) * self.typical_turn_count

    def add_interest(self, topic: str) -> None:
        """Add a topic interest."""
        if topic and topic not in self.interests:
            self.interests.append(topic)

    def avoid_topic(self, topic: str) -> None:
        """Add a topic to avoid."""
        if topic and topic not in self.avoided_topics:
            self.avoided_topics.append(topic)

    def add_shared_memory(self, memory: str) -> None:
        """Add a shared memory."""
        if memory:
            self.shared_memory.append(memory)


class IntentResult(BaseModel):
    """Result of intent classification."""
    primary: IntentType
    confidence: float = Field(default=1.0)
    secondary: Optional[IntentType] = None
    keywords: List[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return v


class ProcessedResponse(BaseModel):
    """Final processed response ready for output with validation."""

    text: str
    intent: IntentResult
    source: str  # "direct" or tool name

    # Multimodal parameters
    tts_params: Dict[str, Any] = Field(default_factory=dict)
    avatar_params: Dict[str, Any] = Field(default_factory=dict)
    ui_elements: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    processing_time: float = Field(default=0.0)
    was_routed: bool = False
    confidence: float = Field(default=1.0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if v is None:
            return ""
        return v

    @field_validator("processing_time", "confidence")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


class CompanionMessage(BaseModel):
    """Validated message format for the companion state."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str, info) -> str:
        if v is None:
            return ""
        # Allow empty content for tool messages
        if info.data.get("tool_calls") and not v:
            return v
        return v

    def to_langchain(self) -> BaseMessage:
        """Convert to LangChain message format."""
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "system":
            from langchain_core.messages import SystemMessage
            return SystemMessage(content=self.content)
        elif self.role == "assistant":
            if self.tool_calls:
                # Tool call message
                return AIMessage(
                    content=self.content,
                    tool_calls=self.tool_calls,
                )
            return AIMessage(content=self.content)
        else:
            raise ValueError(f"Unknown role: {self.role}")

    @classmethod
    def from_langchain(cls, msg: BaseMessage) -> "CompanionMessage":
        """Create from LangChain message."""
        if isinstance(msg, HumanMessage):
            return cls(role="user", content=str(msg.content))
        elif isinstance(msg, AIMessage):
            return cls(
                role="assistant",
                content=str(msg.content),
                tool_calls=[{
                    "id": tc.id,
                    "name": tc.name,
                    "args": tc.args,
                } for tc in (msg.tool_calls or [])],
            )
        elif isinstance(msg, SystemMessage):
            return cls(role="system", content=str(msg.content))
        else:
            # Generic fallback
            return cls(
                role=getattr(msg, "role", "assistant"),
                content=str(msg.content),
            )
