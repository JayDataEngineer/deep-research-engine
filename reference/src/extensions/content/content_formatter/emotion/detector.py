"""Emotion Detector - Deep Agent-based emotion detection for dialogue.

Uses deepagents library with thinking mode for intelligent emotion tagging.
Uses built-in structured output with response_format for automatic validation.

Usage:
    from src.extensions.content.content_formatter.emotion import detect_emotions, DialogueLine

    dialogue_lines = [DialogueLine("EMMA", "Hello there!"), ...]
    detected_lines = await detect_emotions(dialogue_lines)
"""

import logging
from pathlib import Path
from typing import List, Any, Optional, Dict

from .schema import EmotionVector, DialogueLine, EmotionSegment, EmotionResponse

logger = logging.getLogger(__name__)

# Configuration for retry logic
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


# Default prompt fallback (used if config is not available)
DEFAULT_SYSTEM_PROMPT = """You are an expert audio director specializing in emotional delivery for TTS generation.

Your goal is NATURALISM. You despise "over-acting."

Your task is to analyze dialogue text and assign emotion vectors for TTS generation.

**Emotion Dimensions (8 values, each 0.0-1.0):**
1. happy - joy, cheerfulness
2. angry - frustration, aggression
3. sad - sorrow, grief
4. afraid - fear, anxiety
5. disgusted - revulsion, disapproval
6. melancholic - pensive, somber
7. surprised - shock, amazement
8. calm - peaceful, neutral (baseline)

**CRITICAL - How Emotions Work:**
- For neutral/calm speech: calm=0.8-1.0, ALL other emotions must be 0.0-0.1
- For emotional speech: Only 1-2 emotions should be elevated, calm should be lower (0.3-0.6)
- Most natural conversation has calm dominant (0.7-1.0) with tiny amounts of other emotions (0.0-0.15)
- Avoid "mixed high emotions" - e.g., don't set angry=0.5 AND disgusted=0.4 AND calm=0.7

**Your Freedom:**
- Split text into chunks however works best for the conversation flow (not tied to line breaks)
- Assign emotions based on context, dialogue flow, and speaker state
- Ensure smooth emotional transitions between segments

**Directorial Rules:**
- "calm" should dominate for neutral speech (0.7-1.0)
- Before assigning emotions, analyze the subtext and emotional arc
- Match speaker personality (e.g., narrator is usually calm)
"""


class EmotionDetector:
    """Emotion detection using deep_agent with built-in structured output.

    Uses deepagents' response_format for automatic Pydantic validation and retry.
    The agent has full freedom to split text and assign emotions however it wants.

    Attributes:
        DIMENSIONS: List of emotion dimension names
    """

    # Emotion dimensions: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    DIMENSIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the emotion detector.

        Args:
            config: Optional configuration dict with keys:
                - system_prompt: Custom system prompt (overrides default)
                - model: Model configuration dict with 'name' and 'enable_thinking'
        """
        self.config = config or {}
        self._llm = None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for deep agent.

        Loads from config if available, falls back to DEFAULT_SYSTEM_PROMPT.
        """
        prompt = self.config.get("system_prompt")
        if prompt:
            return prompt
        return DEFAULT_SYSTEM_PROMPT

    async def _get_llm(self):
        """Get LLM instance (lazy initialization)."""
        if self._llm is None:
            from src.shared.factory import get_llm

            model_config = self.config.get("model", {})
            enable_thinking = model_config.get("enable_thinking", True)

            self._llm = await get_llm(enable_thinking=enable_thinking)

        return self._llm

    async def detect(self, dialogue_lines: List[DialogueLine]) -> tuple[List[DialogueLine], dict]:
        """Detect emotions using deep_agent with built-in structured output.

        Includes retry logic with validation for robustness.

        Args:
            dialogue_lines: List of DialogueLine objects with speaker, text

        Returns:
            Tuple of (List of DialogueLine objects with emotion vectors assigned, llm_info dict)

        Note:
            Returns a tuple for backwards compatibility with audio_producer.
            The llm_info dict contains metadata about the detection process.
        """
        if not dialogue_lines:
            return [], {}

        # Build dialogue text for the agent
        dialogue_text = self._format_dialogue(dialogue_lines)

        # Get LLM from factory
        llm = await self._get_llm()

        # Import and create deep agent with structured output
        from deepagents import create_deep_agent
        import asyncio

        agent = create_deep_agent(
            model=llm,
            response_format=EmotionResponse,  # Built-in structured output with validation!
            system_prompt=self._get_system_prompt(),
        )

        prompt = f"""Analyze this dialogue and assign emotion vectors:

{dialogue_text}

Return the emotion vectors for each segment.
"""

        logger.info(f"[EmotionDetector] Invoking deep_agent with {len(dialogue_lines)} dialogue lines...")

        # Retry loop with validation
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # Invoke agent - structured output is automatic!
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )

                # Extract structured response from result
                structured = result.get("structured_response")
                if structured is None:
                    raise ValueError("Deep agent did not return structured_response")

                # Validate: check we got segments
                if not structured.segments:
                    raise ValueError("No segments returned from deep agent")

                # Validate: check segment count matches input (approximately)
                # Allow some flexibility as the agent may split/merge lines
                if len(structured.segments) < len(dialogue_lines) * 0.5:
                    raise ValueError(f"Too few segments: got {len(structured.segments)}, expected ~{len(dialogue_lines)}")

                # Convert segments to DialogueLines with emotions
                logger.info(f"[EmotionDetector] Got {len(structured.segments)} segments from structured output")

                result_lines = []
                for i, segment in enumerate(structured.segments):
                    # Validate emotion vector
                    if len(segment.emotion) != 8:
                        raise ValueError(f"Invalid emotion dimension count: {len(segment.emotion)}")

                    # Try to match with original line if available
                    original_line = dialogue_lines[i] if i < len(dialogue_lines) else None

                    # Create new DialogueLine with detected emotion
                    line = DialogueLine(
                        speaker=segment.speaker,
                        text=segment.text,
                        direction=original_line.direction if original_line else "",
                        section_id=original_line.section_id if original_line else 0
                    )
                    line.emotion = EmotionVector(segment.emotion)
                    result_lines.append(line)

                # Success! Return tuple for compatibility: (lines, llm_info)
                llm_info = {
                    "reasoning": "deep_agent",
                    "attempts": attempt + 1,
                    "segments": len(result_lines),
                }
                return result_lines, llm_info

            except Exception as e:
                last_error = e
                logger.warning(f"[EmotionDetector] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff

        # All retries failed - return with neutral emotions as fallback
        logger.error(f"[EmotionDetector] All {MAX_RETRIES} attempts failed, using neutral emotions")
        result_lines = []
        for line in dialogue_lines:
            new_line = DialogueLine(line.speaker, line.text, line.direction, line.section_id)
            new_line.emotion = EmotionVector.neutral()
            result_lines.append(new_line)

        llm_info = {
            "reasoning": "fallback_neutral",
            "error": str(last_error),
            "attempts": MAX_RETRIES,
        }
        return result_lines, llm_info

    def _format_dialogue(self, dialogue_lines: List[DialogueLine]) -> str:
        """Format dialogue lines for the agent."""
        lines = []
        for dl in dialogue_lines:
            speaker = getattr(dl, "speaker", "UNKNOWN")
            text = getattr(dl, "text", "")
            direction = getattr(dl, "direction", "")
            if direction:
                lines.append(f"[{speaker}] ({direction}) {text}")
            else:
                lines.append(f"[{speaker}] {text}")
        return "\n".join(lines)


# ============================================================================
# Module-level convenience functions
# ============================================================================

_default_detector: Optional[EmotionDetector] = None


async def detect_emotions(
    dialogue_lines: List[DialogueLine],
    config: Optional[Dict[str, Any]] = None
) -> tuple[List[DialogueLine], dict]:
    """Detect emotions in dialogue lines using deep_agent.

    This is the recommended simple function-based API.

    Args:
        dialogue_lines: List of DialogueLine objects with speaker, text
        config: Optional configuration dict (uses defaults if not provided)

    Returns:
        Tuple of (List of DialogueLine objects with .emotion set, llm_info dict)

    Example:
        from src.extensions.content.content_formatter.emotion import detect_emotions, DialogueLine

        lines = [
            DialogueLine("EMMA", "Hello there!"),
            DialogueLine("OLIVIA", "Hi! How are you?"),
        ]
        detected, llm_info = await detect_emotions(lines)
        for line in detected:
            print(f"{line.speaker}: {line.text} -> {line.emotion.to_list()}")
    """
    global _default_detector

    if config is not None:
        detector = EmotionDetector(config)
    else:
        if _default_detector is None:
            _default_detector = EmotionDetector()
        detector = _default_detector

    return await detector.detect(dialogue_lines)


def get_emotion_detector(config: Optional[Dict[str, Any]] = None) -> EmotionDetector:
    """Get an EmotionDetector instance.

    Factory function for creating configured EmotionDetector instances.

    Args:
        config: Optional configuration dict

    Returns:
        EmotionDetector instance
    """
    return EmotionDetector(config)
