"""
Conversation Management

Tracks conversation state across turns.
Handles context, summarization, mood tracking.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from ..state.brain_state import (
    ConversationState,
    ConversationTurn,
    UserMood,
)


class ConversationManager:
    """Track conversation state across turns."""

    def __init__(
        self,
        max_context_turns: int = 20,
        summarize_interval: int = 10,
        context_window_size: int = 10,
    ):
        """
        Initialize conversation manager.

        Args:
            max_context_turns: Maximum turns to keep in full context
            summarize_interval: Summarize every N turns
            context_window_size: Size of sliding context window
        """
        self.max_context_turns = max_context_turns
        self.summarize_interval = summarize_interval
        self.context_window_size = context_window_size

    def create_state(self) -> ConversationState:
        """Create a new conversation state."""
        return ConversationState(
            max_context_turns=self.max_context_turns,
        )

    def add_turn(
        self,
        state: ConversationState,
        user_input: str,
        response: str,
        intent: Any,
        was_routed: bool = False,
        tool_used: Optional[str] = None,
    ) -> ConversationTurn:
        """Add a turn to conversation history."""
        return state.add_turn(
            user_input=user_input,
            response=response,
            intent=intent,
            was_routed=was_routed,
            tool_used=tool_used,
        )

    def get_context(self, state: ConversationState, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation context for LLM."""
        if max_turns is None:
            max_turns = self.context_window_size

        return state.get_recent_context(max_turns)

    def should_summarize(self, state: ConversationState) -> bool:
        """Check if conversation needs summarization."""
        return state.should_summarize(self.summarize_interval)

    def update_summary(
        self,
        state: ConversationState,
        summary: str,
    ) -> None:
        """Update conversation summary."""
        state.update_summary(summary)

    def detect_mood(self, text: str, previous_mood: Optional[UserMood] = None) -> UserMood:
        """Detect user mood from text (simple heuristic)."""
        text_lower = text.lower()

        # Check for emotion indicators
        mood_keywords = {
            UserMood.HAPPY: ["happy", "great", "awesome", "excited", "love", "yay", ":)", ":d"],
            UserMood.SAD: ["sad", "down", "depressed", "unhappy", "crying", ":("],
            UserMood.ANGRY: ["angry", "mad", "furious", "hate", "annoying"],
            UserMood.ANXIOUS: ["anxious", "worried", "nervous", "scared", "afraid"],
            UserMood.EXCITED: ["excited", "can't wait", "looking forward", "!"],
            UserMood.BORED: ["bored", "boring", "nothing to do"],
            UserMood.FRUSTRATED: ["frustrated", "annoying", "doesn't work"],
            UserMood.CONFUSED: ["confused", "don't understand", "what do you mean", "huh"],
        }

        # Count matches for each mood
        mood_scores = {}
        for mood, keywords in mood_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                mood_scores[mood] = score

        # Return mood with highest score, or previous mood, or neutral
        if mood_scores:
            return max(mood_scores, key=mood_scores.get)
        elif previous_mood:
            return previous_mood
        else:
            return UserMood.NEUTRAL

    def update_mood(
        self,
        state: ConversationState,
        mood: UserMood,
    ) -> None:
        """Update user mood."""
        state.update_mood(mood)

    def track_topic(self, state: ConversationState, topic: Optional[str]) -> None:
        """Track current topic."""
        if topic and topic != state.current_topic:
            # Add previous topic to history
            if state.current_topic:
                state.topic_history.append(state.current_topic)

            state.current_topic = topic

    def get_conversation_stats(self, state: ConversationState) -> Dict[str, Any]:
        """Get conversation statistics."""
        duration = state.get_conversation_duration()

        # Count intents
        intent_counts = {}
        for turn in state.turns:
            intent = turn.intent.value if hasattr(turn.intent, 'value') else str(turn.intent)
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Count tool usage
        tool_counts = {}
        for turn in state.turns:
            if turn.tool_used:
                tool_counts[turn.tool_used] = tool_counts.get(turn.tool_used, 0) + 1

        return {
            "turn_count": state.turn_count,
            "duration_minutes": duration,
            "current_topic": state.current_topic,
            "topics_discussed": len(state.topic_history),
            "intent_distribution": intent_counts,
            "tool_usage": tool_counts,
            "current_mood": state.user_mood.value if state.user_mood else None,
            "mood_changes": len(state.user_mood_history),
        }

    def format_conversation_for_display(
        self,
        state: ConversationState,
        max_turns: int = 5,
    ) -> str:
        """Format recent conversation for display."""
        recent_turns = state.turns[-max_turns:]

        lines = []
        for turn in recent_turns:
            lines.append(f"Turn {turn.turn_number}:")
            lines.append(f"  User: {turn.user_input[:100]}...")
            lines.append(f"  Assistant: {turn.assistant_response[:100]}...")
            if turn.was_routed:
                lines.append(f"  [Used tool: {turn.tool_used}]")
            lines.append("")

        return "\n".join(lines)


def create_conversation_manager(
    max_context_turns: int = 20,
    summarize_interval: int = 10,
    context_window_size: int = 10,
) -> ConversationManager:
    """Factory function to create ConversationManager."""
    return ConversationManager(
        max_context_turns=max_context_turns,
        summarize_interval=summarize_interval,
        context_window_size=context_window_size,
    )
