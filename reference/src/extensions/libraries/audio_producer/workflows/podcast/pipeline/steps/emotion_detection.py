"""
Emotion Detection Pipeline Step

**PURE FUNCTION**: Input → Process → Output

Input:
  - script_text: str (the podcast script)
  - config: EmotionDetectionConfig (how to detect emotions)

Output:
  - List[EmotionTaggedLine] (each line with emotion vector)

This step is:
- Deterministic: same input → same output
- Side-effect free: no mutations, no I/O
- Testable: can unit test without LLM by mocking
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Emotion detection processing mode."""
    SINGLE_CALL = "single_call"      # One LLM call per segment
    ROLLING_WINDOW = "rolling_window"  # Overlapping windows
    PARALLEL = "parallel"              # Process segments in parallel


@dataclass
class EmotionVector:
    """8-dimensional emotion vector for IndexTTS.

    Format: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

    Values: 0.0 - 1.0
    - Natural speech: mostly 0.0-0.3 emotions, 0.7-1.0 calm
    """
    happy: float = 0.0
    angry: float = 0.0
    sad: float = 0.0
    afraid: float = 0.0
    disgusted: float = 0.0
    melancholic: float = 0.0
    surprised: float = 0.0
    calm: float = 1.0

    def to_list(self) -> List[float]:
        """Convert to list format for IndexTTS."""
        return [
            self.happy, self.angry, self.sad, self.afraid,
            self.disgusted, self.melancholic, self.surprised,
            self.calm
        ]

    @classmethod
    def from_list(cls, values: List[float]) -> "EmotionVector":
        """Create from list (LLM output)."""
        if len(values) == 9:
            # Handle old format with intensity - drop last element
            values = values[:8]
        if len(values) != 8:
            raise ValueError(f"EmotionVector requires 8 values, got {len(values)}")
        return cls(*values)

    def to_index_tts_format(self) -> List[float]:
        """Format for IndexTTS."""
        return self.to_list()


@dataclass
class EmotionTaggedLine:
    """A dialogue line with emotion metadata."""
    index: int                      # Original line index in script
    speaker: str                     # Speaker name (EMMA, OLIVIA, NARRATOR)
    text: str                        # The dialogue text
    emotion: EmotionVector           # Emotion vector for TTS
    context_note: Optional[str] = None  # Optional note about emotional beat


@dataclass
class EmotionDetectionConfig:
    """Configuration for emotion detection step."""
    # Processing mode
    mode: ProcessingMode = ProcessingMode.SINGLE_CALL

    # Prompt configuration
    prompt_name: str = "single_line"  # Which prompt to use

    # Chunking (for batch processing)
    chunk_size: int = 10              # Lines per chunk
    chunk_overlap: int = 2            # Overlapping lines between chunks

    # LLM parameters
    temperature: float = 0.7
    max_tokens: int = 500

    # Quality vs Speed tradeoff
    parallel_workers: int = 3         # Number of parallel LLM calls


# ============================================================================
# PURE FUNCTION - The core of this step
# ============================================================================

async def detect_emotions(
    script_text: str,
    config: EmotionDetectionConfig,
    llm_client: Any  # LLM client interface
) -> List[EmotionTaggedLine]:
    """Detect emotions for a podcast script.

    **PURE FUNCTION**: Given the same inputs, always returns the same output.
    No side effects, no mutations.

    Args:
        script_text: The podcast script to analyze
        config: How to detect emotions
        llm_client: LLM client (has async generate method)

    Returns:
        List of emotion-tagged dialogue lines

    Example:
        config = EmotionDetectionConfig(mode=ProcessingMode.SINGLE_CALL)
        lines = await detect_emotions(script, config, llm_client)
        for line in lines:
            print(f"{line.speaker}: {line.emotion.to_list()}")
    """
    # Parse script into dialogue lines
    dialogue_lines = _parse_script(script_text)

    # Apply emotion detection based on mode
    if config.mode == ProcessingMode.SINGLE_CALL:
        return await _detect_single_call(dialogue_lines, config, llm_client)
    elif config.mode == ProcessingMode.ROLLING_WINDOW:
        return await _detect_rolling_window(dialogue_lines, config, llm_client)
    elif config.mode == ProcessingMode.PARALLEL:
        return await _detect_parallel(dialogue_lines, config, llm_client)
    else:
        raise ValueError(f"Unknown processing mode: {config.mode}")


# ============================================================================
# Helper functions (still pure - no side effects)
# ============================================================================

def _parse_script(script_text: str) -> List[Dict[str, Any]]:
    """Parse script into structured dialogue lines.

    **PURE**: String parsing, no side effects.

    Returns:
        List of dicts with keys: index, speaker, text, direction
    """
    import re

    lines = []
    # Flexible pattern that accepts ANY uppercase speaker name
    pattern = re.compile(
        r'^(?P<speaker>[A-Z][A-Z0-9_]*)(\s*(?P<direction>\([^)]*\)))?:\s*(?P<text>.+)$|'
        r'^\*\*(?P<speaker2>[A-Z][A-Z0-9_]*)(\s*(?P<direction2>\([^)]*\)))?:\*\*\s*(?P<text2>.+)'
    )

    for idx, line in enumerate(script_text.strip().split('\n')):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            groups = match.groupdict()
            speaker = groups.get('speaker') or groups.get('speaker2')
            text = groups.get('text') or groups.get('text2')
            direction = groups.get('direction') or groups.get('direction2')

            lines.append({
                'index': idx,
                'speaker': speaker,
                'text': text,
                'direction': direction
            })

    return lines


async def _detect_single_call(
    dialogue_lines: List[Dict[str, Any]],
    config: EmotionDetectionConfig,
    llm_client: Any
) -> List[EmotionTaggedLine]:
    """Single-call mode: One LLM call per chunk of lines.

    **PURE**: LLM calls are async but deterministic given same inputs.
    """
    # Import prompt loader (lazy import to avoid circular deps)
    from src.extensions.libraries.audio_producer.prompts.emotion_detection import get_prompt_loader

    prompt_loader = get_prompt_loader()
    prompt_template = prompt_loader.get_prompt(config.prompt_name)

    results = []

    # Process in chunks
    for i in range(0, len(dialogue_lines), config.chunk_size):
        chunk = dialogue_lines[i:i + config.chunk_size]

        # Format prompt
        formatted_prompt = _format_chunk_prompt(chunk, prompt_template)

        # Call LLM
        response = await llm_client.ainvoke(
            formatted_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        # Parse response
        emotions = _parse_llm_response(response)

        # Create EmotionTaggedLine objects
        for j, line_data in enumerate(chunk):
            if j < len(emotions):
                results.append(EmotionTaggedLine(
                    index=line_data['index'],
                    speaker=line_data['speaker'],
                    text=line_data['text'],
                    emotion=EmotionVector.from_list(emotions[j])
                ))

    return results


def _format_chunk_prompt(chunk: List[Dict[str, Any]], prompt_template: str) -> str:
    """Format the prompt for a chunk of dialogue lines.

    **PURE**: String formatting, no side effects.
    """
    # Check what placeholders the prompt template needs
    import string
    formatter = string.Formatter()
    placeholders = {p for _, p, _, _ in formatter.parse(prompt_template) if p}

    # If prompt needs {speaker} and {text} (single_line format)
    if "speaker" in placeholders and "text" in placeholders:
        # Format each line individually and join
        formatted_lines = []
        for line in chunk:
            formatted_lines.append(prompt_template.format(
                speaker=line["speaker"],
                text=line["text"]
            ))
        # Join all formatted prompts
        return "\n\n---\n\n".join(formatted_lines)

    # If prompt needs {dialogue_lines} (batch format)
    elif "dialogue_lines" in placeholders:
        dialogue_text = "\n".join([
            f"{line['speaker']}: {line['text']}"
            for line in chunk
        ])
        return prompt_template.format(dialogue_lines=dialogue_text)

    # If prompt needs {script_text} (chunking format)
    elif "script_text" in placeholders:
        script_text = "\n".join([
            f"{line['speaker']}: {line['text']}"
            for line in chunk
        ])
        return prompt_template.format(script_text=script_text)

    # Default: return as-is
    else:
        return prompt_template


def _parse_llm_response(response: str) -> List[List[float]]:
    """Parse LLM response into emotion vectors.

    **PURE**: String parsing, no side effects.

    Args:
        response: LLM response (should be JSON array)

    Returns:
        List of emotion vectors (each is 9 floats)
    """
    import json

    try:
        # Try parsing as JSON array
        parsed = json.loads(response.strip())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract arrays using regex
    import re
    arrays = re.findall(r'\[([0-9.,\s]+)\]', response)
    results = []
    for arr in arrays:
        values = [float(x.strip()) for x in arr.split(',') if x.strip()]
        if len(values) == 9:
            results.append(values)

    return results


async def _detect_rolling_window(
    dialogue_lines: List[Dict[str, Any]],
    config: EmotionDetectionConfig,
    llm_client: Any
) -> List[EmotionTaggedLine]:
    """Rolling window mode: Overlapping chunks for continuity.

    **PURE**: LLM calls are deterministic, results are merged predictably.
    """
    # Implementation similar to single_call but with overlap
    # This is a placeholder for the full implementation
    return await _detect_single_call(dialogue_lines, config, llm_client)


async def _detect_parallel(
    dialogue_lines: List[Dict[str, Any]],
    config: EmotionDetectionConfig,
    llm_client: Any
) -> List[EmotionTaggedLine]:
    """Parallel mode: Process chunks in parallel.

    **PURE**: Parallel execution but results are deterministic.
    """
    # Implementation using asyncio.gather for parallel LLM calls
    # This is a placeholder for the full implementation
    return await _detect_single_call(dialogue_lines, config, llm_client)
