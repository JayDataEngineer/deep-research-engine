"""
Summarization Middleware

Compresses conversation context to save tokens while preserving important information.
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class ConversationSummarizer:
    """
    Summarizes conversation turns to compress context.

    When conversations get long, old turns can be summarized into
    a condensed version that retains key information while using fewer tokens.
    """

    def __init__(
        self,
        model,  # LLM for generating summaries
        max_turns_before_summary: int = 10,
        summary_depth: int = 5,  # Number of turns to include in summary
    ):
        """
        Initialize conversation summarizer.

        Args:
            model: LLM instance for generating summaries
            max_turns_before_summary: When to trigger summarization
            summary_depth: How many recent turns to keep uncompressed
        """
        self.model = model
        self.max_turns_before_summary = max_turns_before_summary
        self.summary_depth = summary_depth

    def should_summarize(self, turn_count: int) -> bool:
        """Check if summarization should occur."""
        return turn_count >= self.max_turns_before_summary

    def create_summary(
        self,
        conversation_history: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a summary of conversation history.

        Args:
            conversation_history: List of conversation turns
            metadata: Optional metadata to include in summary

        Returns:
            Summary text
        """
        if not conversation_history:
            return ""

        # Build prompt for summarization
        turns_text = []
        for i, turn in enumerate(conversation_history):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            turns_text.append(f"{role.capitalize()}: {content}")

        conversation_text = "\n".join(turns_text)

        summary_prompt = f"""Summarize the following conversation, capturing the most important points, decisions, and personal information shared.

Conversation:
{conversation_text}

Provide a concise summary that preserves:
1. Key topics discussed
2. Important facts about the user
3. Any decisions or conclusions reached
4. Current context or ongoing topics

Summary:"""

        try:
            # Generate summary using the model
            response = self.model.invoke(summary_prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            return summary.strip()
        except Exception as e:
            # Fallback to simple extraction
            return self._simple_summary(conversation_history)

    def _simple_summary(
        self,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """
        Create a simple summary without LLM (fallback).

        Extracts key information using patterns.
        """
        topics = set()
        user_facts = []

        for turn in conversation_history:
            content = turn.get("content", "").lower()
            role = turn.get("role", "")

            # Extract potential topics (capitalized words)
            import re
            words = re.findall(r'\b[A-Z][a-z]+\b', turn.get("content", ""))
            topics.update(words[:5])  # Take first 5 capitalized words

            # Extract first-person statements from user
            if role == "user":
                if " i am " in content or " i'm " in content:
                    user_facts.append(turn.get("content", ""))

        summary_parts = []
        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(list(topics)[:10])}")
        if user_facts:
            summary_parts.append(f"User shared: {'; '.join(user_facts[:3])}")

        return " | ".join(summary_parts) if summary_parts else "Conversation continues..."

    def compress_context(
        self,
        conversation_history: List[Dict[str, str]],
        keep_recent: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Compress conversation by replacing old turns with a summary.

        Args:
            conversation_history: Full conversation history
            keep_recent: Number of recent turns to keep uncompressed

        Returns:
            Compressed context with summary and recent turns
        """
        if len(conversation_history) <= keep_recent:
            return [{"type": "full", "data": conversation_history}]

        # Split into old (to be summarized) and recent (keep as-is)
        old_turns = conversation_history[:-keep_recent]
        recent_turns = conversation_history[-keep_recent:]

        # Create summary
        summary = self.create_summary(old_turns)

        return [
            {
                "type": "summary",
                "content": summary,
                "turn_count": len(old_turns),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "type": "recent",
                "data": recent_turns,
            },
        ]

    def get_summarized_context(
        self,
        conversation_history: List[Dict[str, str]],
        compressed: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Reconstruct conversation context from compressed format.

        Args:
            conversation_history: Original full history (for reference)
            compressed: Compressed context format

        Returns:
            List of conversation turns for prompt
        """
        result = []

        for item in compressed:
            if item["type"] == "summary":
                result.append({
                    "role": "system",
                    "content": f"[Previous conversation summary: {item['content']}]",
                })
            elif item["type"] == "recent":
                result.extend(item["data"])
            elif item["type"] == "full":
                result.extend(item["data"])

        return result


class SummaryManager:
    """
    Manages conversation summaries and compression.

    Tracks when to summarize and maintains summary history.
    """

    def __init__(
        self,
        summarizer: ConversationSummarizer,
        max_summaries: int = 3,
    ):
        """
        Initialize summary manager.

        Args:
            summarizer: ConversationSummarizer instance
            max_summaries: Maximum number of summaries to keep
        """
        self.summarizer = summarizer
        self.max_summaries = max_summaries
        self.summaries: List[Dict[str, Any]] = []
        self.current_turn_count = 0

    def add_turn(self, turn: Dict[str, str]) -> Optional[List[Dict[str, Any]]]:
        """
        Add a conversation turn and potentially compress.

        Args:
            turn: Conversation turn to add

        Returns:
            Compressed context if compression occurred, None otherwise
        """
        self.current_turn_count += 1

        # Check if we should compress
        if self.summarizer.should_summarize(self.current_turn_count):
            # Get current conversation (summary + recent turns)
            current_context = self.get_full_context()

            # Compress
            compressed = self.summarizer.compress_context(
                current_context,
                keep_recent=self.summarizer.summary_depth,
            )

            # Store new summary
            for item in compressed:
                if item["type"] == "summary":
                    self.summaries.append(item)
                    # Keep only recent summaries
                    if len(self.summaries) > self.max_summaries:
                        self.summaries = self.summaries[-self.max_summaries:]

            # Reset counter (keep recent turns, so we count from those)
            self.current_turn_count = self.summarizer.summary_depth

            return compressed

        return None

    def get_full_context(self) -> List[Dict[str, str]]:
        """
        Get full conversation context (summaries + recent turns).

        Returns:
            List of all conversation turns
        """
        context = []

        # Add all summaries
        for summary in self.summaries:
            context.append({
                "role": "system",
                "content": f"[Summary: {summary['content']}]",
            })

        # Add recent turns that aren't in summaries yet
        # (This is a simplified version - in practice, you'd track recent turns separately)

        return context

    def reset(self):
        """Reset summary state."""
        self.summaries = []
        self.current_turn_count = 0


def create_summarizer(
    model,
    max_turns_before_summary: int = 10,
    summary_depth: int = 5,
) -> ConversationSummarizer:
    """Create a conversation summarizer."""
    return ConversationSummarizer(
        model=model,
        max_turns_before_summary=max_turns_before_summary,
        summary_depth=summary_depth,
    )


def create_summary_manager(
    summarizer: ConversationSummarizer,
    max_summaries: int = 3,
) -> SummaryManager:
    """Create a summary manager."""
    return SummaryManager(summarizer=summarizer, max_summaries=max_summaries)
