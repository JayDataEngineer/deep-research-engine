"""
Title Generator Service

Fast, non-blocking title generation using the LLM factory.
Runs in parallel with chat response - triggered immediately on user message.

Optimized for speed:
- No CoT/thinking mode (enable_thinking=False)
- Short prompt, max 20 tokens
- Hot-swappable backends via string configuration
"""

import asyncio
import logging
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

# Generation parameters
MAX_TITLE_LENGTH = 50
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.8

# Default model for title generation (can be overridden via settings)
# Use a faster/cheaper model for simple title generation
DEFAULT_TITLE_MODEL = "glm-4.7"  # Or configure via env


class TitleGenerator:
    """
    Singleton service for generating chat titles using the LLM factory.

    Uses the centralized get_llm() factory for hot-swappable backend support.
    """

    _instance: Optional["TitleGenerator"] = None
    _lock: Optional[object] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

    def is_available(self) -> bool:
        """Check if the title generator is available (factory always returns something)."""
        return True

    async def generate_title_async(
        self,
        user_message: str,
        ai_response: str = "",
        fallback_title: str = "",
    ) -> str:
        """
        Generate a title from the first 12 letters of the user's message.

        Args:
            user_message: The first user message
            ai_response: Optional snippet of the AI response (unused)
            fallback_title: Optional fallback title from frontend

        Returns:
            Title (first 12 letters with ellipsis if needed)
        """
        # Use frontend-provided fallback if available, otherwise generate
        if fallback_title:
            logger.info(f"[TITLE GEN] Using frontend-provided title: {fallback_title}")
            return fallback_title
        return self._fallback_title(user_message)

    def generate_title(
        self,
        user_message: str,
        ai_response: str = "",
        fallback_title: str = "",
    ) -> str:
        """
        Synchronous title generation from the first 12 letters.

        Args:
            user_message: The first user message
            ai_response: Optional snippet of the AI response (unused)
            fallback_title: Optional fallback title from frontend

        Returns:
            Title (first 12 letters with ellipsis if needed)
        """
        # Use frontend-provided fallback if available, otherwise generate
        if fallback_title:
            logger.info(f"[TITLE GEN] Using frontend-provided title: {fallback_title}")
            return fallback_title
        return self._fallback_title(user_message)

    def _build_prompt(self, user_message: str, ai_response: str) -> str:
        """Build the prompt for title generation."""

        # Truncate messages to fit context
        user_truncated = user_message[:200]

        # Simple, direct instruction for title generation
        prompt = f"""Generate a short, concise title (max {MAX_TITLE_LENGTH} characters) for this conversation.

User's first message: {user_truncated}

Requirements:
- Short and punchy (2-6 words)
- No quotation marks
- No "Title:" prefix
- Just the title text

Title:"""

        return prompt

    def _clean_title(self, generated: str) -> str:
        """Clean and validate the generated title."""
        if not generated:
            logger.debug("[TITLE GEN] Rejected: empty title")
            return ""

        # Remove common prefixes
        title = generated.strip()
        prefixes_to_remove = [
            "Title:",
            "title:",
            "The title is",
            "The title is:",
            '"',
            "'",
        ]

        for prefix in prefixes_to_remove:
            if title.lower().startswith(prefix.lower()):
                title = title[len(prefix):].strip()

        # Remove trailing quotes, newlines, and common suffixes
        suffixes_to_remove = ['"', "'", "\n"]
        for suffix in suffixes_to_remove:
            title = title.rstrip(suffix)

        title = title.strip()

        # If title is just underscores or special chars, reject it
        if not title or all(c in '_-=+*#@' for c in title):
            logger.debug(f"[TITLE GEN] Rejected: only special chars: '{title}'")
            return ""

        # Limit length
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0]

        # Filter out bad patterns - be less aggressive
        bad_patterns = [
            "conversation about",  # Meta descriptions
            "discussion of",       # Meta descriptions
            "new chat",            # Default placeholder
            "untitled",            # Default placeholder
        ]

        title_lower = title.lower()
        for pattern in bad_patterns:
            if pattern in title_lower:
                logger.debug(f"[TITLE GEN] Rejected: matches bad pattern '{pattern}': '{title}'")
                return ""

        return title

    def _fallback_title(self, user_message: str) -> str:
        """Generate a title from the first 12 letters of the user message."""
        # Remove leading whitespace and get first 12 characters
        trimmed = user_message.strip()
        title = trimmed[:12]

        # Add ellipsis if text is longer than 12 chars
        result = title + "..." if len(trimmed) > 12 else title

        # Ultimate fallback if message is empty or just whitespace
        if not result:
            from datetime import datetime
            now = datetime.now()
            return now.strftime("%b %d, %-I:%M %p").lstrip("0").replace(" 0", " ")

        return result


# Singleton instance
_title_generator: Optional[TitleGenerator] = None
_generator_lock = Lock()


def get_title_generator() -> TitleGenerator:
    """Get the singleton title generator instance."""
    global _title_generator

    if _title_generator is None:
        with _generator_lock:
            if _title_generator is None:
                _title_generator = TitleGenerator()

    return _title_generator
