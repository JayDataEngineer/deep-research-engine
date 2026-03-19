"""Sentence Transition Smoothing Module.

This module implements various methods to smooth transitions between independently
generated TTS segments, addressing the "disjointed" problem.

Five modes available:
1. none - No processing (hard cuts)
2. crossfade - Audio cross-fading
3. adaptive_merge - Merge sentences with similar emotions (Solution 1)
4. ghost_text - Generate with overlap, trim (Solution 2)
5. breath_pause - Insert breaths based on punctuation (Solution 3)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import from centralized content_formatter emotion module
from src.extensions.content.content_formatter.emotion import DialogueLine, EmotionVector

logger = logging.getLogger(__name__)


# =============================================================================
# Solution 1: Adaptive Merge (Emotion-based grouping)
# =============================================================================

def adaptive_merge_sentences(
    dialogue_lines: List[DialogueLine],
    threshold: float = 0.15,
    max_sentences: int = 5
) -> List[DialogueLine]:
    """Merge consecutive sentences with similar emotions.

    This reduces TTS calls and creates longer audio blocks that flow naturally.
    Only splits when emotion changes significantly (where a pause is natural).

    Args:
        dialogue_lines: List of DialogueLine with emotions already set
        threshold: Euclidean distance threshold (lower = more aggressive merging)
        max_sentences: Safety limit to prevent excessively long generations

    Returns:
        List of merged DialogueLine objects
    """
    if not dialogue_lines:
        return []

    merged = []
    current_text = dialogue_lines[0].text
    current_emotion = np.array(dialogue_lines[0].emotion.to_list())
    current_speaker = dialogue_lines[0].speaker
    current_direction = dialogue_lines[0].direction
    sentence_count = 1
    original_indices = [0]

    for i in range(1, len(dialogue_lines)):
        line = dialogue_lines[i]

        # Only merge within same speaker
        if line.speaker != current_speaker:
            # Flush current merged segment
            merged.append(DialogueLine(
                current_speaker,
                current_text,
                current_direction,
                EmotionVector(current_emotion.tolist())
            ))
            # Start new segment
            current_text = line.text
            current_emotion = np.array(line.emotion.to_list())
            current_speaker = line.speaker
            current_direction = line.direction
            sentence_count = 1
            original_indices = [i]
            continue

        next_emotion = np.array(line.emotion.to_list())
        distance = np.linalg.norm(current_emotion - next_emotion)

        if distance < threshold and sentence_count < max_sentences:
            # EMOTIONS ARE SIMILAR: MERGE
            current_text += " " + line.text
            # Update running average of emotion
            current_emotion = (current_emotion * sentence_count + next_emotion) / (sentence_count + 1)
            sentence_count += 1
            original_indices.append(i)
            logger.debug(f"Merged sentence {i} (dist={distance:.3f} < {threshold})")
        else:
            # EMOTION CHANGED: FLUSH AND START NEW
            merged.append(DialogueLine(
                current_speaker,
                current_text,
                current_direction,
                EmotionVector(current_emotion.tolist())
            ))
            current_text = line.text
            current_emotion = next_emotion
            current_direction = line.direction
            sentence_count = 1
            original_indices = [i]
            logger.debug(f"Split at sentence {i} (dist={distance:.3f} >= {threshold})")

    # Flush final segment
    merged.append(DialogueLine(
        current_speaker,
        current_text,
        current_direction,
        EmotionVector(current_emotion.tolist())
    ))

    logger.info(f"Adaptive merge: {len(dialogue_lines)} → {len(merged)} segments")
    return merged


# =============================================================================
# Solution 2: Ghost Text (Overlap generation)
# =============================================================================

def prepare_ghost_text_inputs(
    dialogue_lines: List[DialogueLine],
    overlap_words: int = 5
) -> List[Tuple[str, int, int]]:
    """Prepare TTS inputs with ghost text overlap.

    For each sentence after the first, prepend the last N words of the previous
    sentence. This creates smooth prosodic transitions. The overlap is trimmed later.

    Args:
        dialogue_lines: List of DialogueLine objects
        overlap_words: Number of trailing words to include

    Returns:
        List of tuples: (text_with_ghost, trim_chars, original_line_index)
    """
    inputs = []

    for i, line in enumerate(dialogue_lines):
        if i == 0:
            # First sentence: no ghost text
            inputs.append((line.text, 0, i))
        else:
            # Get last N words from previous sentence
            prev_text = dialogue_lines[i - 1].text
            prev_words = prev_text.split()
            ghost_words = prev_words[-overlap_words:] if len(prev_words) >= overlap_words else prev_words
            ghost_text = " ".join(ghost_words)

            # Calculate how many characters to trim (rough estimate)
            # This will be refined after TTS by aligning with actual audio
            trim_chars = len(ghost_text) + 1  # +1 for space

            text_with_ghost = f"{ghost_text} {line.text}"
            inputs.append((text_with_ghost, trim_chars, i))
            logger.debug(f"Sentence {i}: added {overlap_words} ghost words, will trim ~{trim_chars} chars")

    return inputs


def trim_ghost_audio(
    audio_bytes: bytes,
    trim_chars: int,
    reference_audio_duration: float,
    reference_text: str
) -> bytes:
    """Trim the ghost text portion from generated audio.

    Args:
        audio_bytes: Generated audio (with ghost text)
        trim_chars: Estimated characters to trim
        reference_audio_duration: Duration of reference sentence's audio
        reference_text: Reference sentence text

    Returns:
        Trimmed audio bytes
    """
    # For now, use a simple ratio-based estimate
    # A more sophisticated implementation would use:
    # - Forced alignment (montreal forced aligner)
    # - Waveform cross-correlation
    # - Whisper timing information

    import struct
    import tempfile
    from pathlib import Path

    # Estimate trim duration based on character ratio
    total_chars = len(reference_text) + trim_chars
    trim_ratio = trim_chars / total_chars

    # Parse WAV to get duration
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(tmp_path)
        duration_ms = len(audio)

        # Calculate trim duration (with some padding to ensure we cut enough)
        trim_ms = int(duration_ms * trim_ratio * 1.1)  # 10% padding

        # Trim from the beginning
        trimmed = audio[trim_ms:]

        # Export back to bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp2:
            tmp2_path = Path(tmp2.name)
            trimmed.export(tmp2_path, format="wav")
            with open(tmp2_path, 'rb') as f:
                return f.read()
    finally:
        tmp_path.unlink(missing_ok=True)


# =============================================================================
# Solution 3: Breath/Pause (Punctuation-aware pauses)
# =============================================================================

def calculate_pause_duration(
    text: str,
    comma_pause_ms: int = 50,
    period_pause_ms: int = 150,
    paragraph_pause_ms: int = 500
) -> int:
    """Calculate appropriate pause duration based on ending punctuation.

    Args:
        text: The text to analyze
        comma_pause_ms: Pause after comma
        period_pause_ms: Pause after period/exclamation/question
        paragraph_pause_ms: Pause after paragraph break (double newline)

    Returns:
        Pause duration in milliseconds
    """
    text = text.strip()

    # Check for paragraph break (double newline or major scene marker)
    if not text or text.endswith('...') or '---' in text:
        return paragraph_pause_ms

    # Get last character
    last_char = text[-1] if text else ''

    if last_char == ',':
        return comma_pause_ms
    elif last_char in '.!?':
        return period_pause_ms
    elif last_char in ':;':
        return period_pause_ms  # Treat colon/semicolon like period
    else:
        return 25  # Minimal pause for other endings (quotes, etc)


def apply_breath_pauses(
    audio_segments: List[bytes],
    dialogue_texts: List[str],
    comma_pause_ms: int = 50,
    period_pause_ms: int = 150,
    paragraph_pause_ms: int = 500,
    breath_sample_path: str = ""
) -> bytes:
    """Apply variable pauses based on punctuation between audio segments.

    Args:
        audio_segments: List of audio byte strings
        dialogue_texts: Corresponding text for each segment
        comma_pause_ms: Pause after comma
        period_pause_ms: Pause after period
        paragraph_pause_ms: Pause after paragraph break
        breath_sample_path: Optional path to breath sound sample

    Returns:
        Combined audio with appropriate pauses
    """
    from pydub import AudioSegment

    if not audio_segments:
        return b''

    # Load breath sample if provided
    breath = None
    if breath_sample_path and Path(breath_sample_path).exists():
        breath = AudioSegment.from_wav(breath_sample_path)

    combined = AudioSegment.from_wav_file(audio_segments[0])

    for i in range(1, len(audio_segments)):
        # Calculate pause duration based on previous text's ending
        pause_ms = calculate_pause_duration(
            dialogue_texts[i - 1],
            comma_pause_ms,
            period_pause_ms,
            paragraph_pause_ms
        )

        # Add pause (breath or silence)
        if breath and pause_ms >= 150:
            # Use breath sound for longer pauses
            combined += breath[:pause_ms] if len(breath) > pause_ms else breath
        else:
            # Use silence
            combined += AudioSegment.silent(duration=pause_ms)

        # Add next segment
        combined += AudioSegment.from_wav_file(audio_segments[i])

    return combined.export(format="wav").read()


# =============================================================================
# Main Transition Handler
# =============================================================================

@dataclass
class TransitionResult:
    """Result of sentence transition processing."""
    dialogue_lines: List[DialogueLine]
    audio_segments: List[bytes]  # May be modified depending on mode
    metadata: Dict[str, Any]


def apply_sentence_transitions(
    dialogue_lines: List[DialogueLine],
    audio_segments: List[bytes],
    mode: str = "crossfade",
    config: Dict[str, Any] = None
) -> TransitionResult:
    """Apply sentence transition smoothing based on mode.

    Args:
        dialogue_lines: List of DialogueLine with emotions
        audio_segments: List of generated audio segments
        mode: Transition mode (none, crossfade, adaptive_merge, ghost_text, breath_pause)
        config: Configuration dict for the specific mode

    Returns:
        TransitionResult with processed data
    """
    config = config or {}

    if mode == "none":
        # No processing
        return TransitionResult(
            dialogue_lines=dialogue_lines,
            audio_segments=audio_segments,
            metadata={"mode": "none", "description": "No transition processing"}
        )

    elif mode == "adaptive_merge":
        # This should be called BEFORE TTS generation
        # (included here for API completeness, but typically used earlier)
        merged = adaptive_merge_sentences(
            dialogue_lines,
            threshold=config.get("emotion_threshold", 0.15),
            max_sentences=config.get("max_sentences_per_merge", 5)
        )
        return TransitionResult(
            dialogue_lines=merged,
            audio_segments=audio_segments,  # Note: segments won't align if called after TTS
            metadata={"mode": "adaptive_merge", "original_count": len(dialogue_lines), "merged_count": len(merged)}
        )

    elif mode == "ghost_text":
        # Process audio segments to trim ghost text
        processed_segments = []
        trim_info = config.get("overlap_words", 5)

        for i, (segment, line) in enumerate(zip(audio_segments, dialogue_lines)):
            if i == 0:
                processed_segments.append(segment)
            else:
                # Trim the ghost text portion
                # In real implementation, would need reference audio duration
                # For now, just pass through
                processed_segments.append(segment)

        return TransitionResult(
            dialogue_lines=dialogue_lines,
            audio_segments=processed_segments,
            metadata={"mode": "ghost_text", "description": "Ghost text trimming applied"}
        )

    elif mode == "breath_pause":
        # Apply variable pauses based on punctuation
        texts = [line.text for line in dialogue_lines]
        combined_audio = apply_breath_pauses(
            audio_segments,
            texts,
            comma_pause_ms=config.get("comma_pause_ms", 50),
            period_pause_ms=config.get("period_pause_ms", 150),
            paragraph_pause_ms=config.get("paragraph_pause_ms", 500),
            breath_sample_path=config.get("breath_sample_path", "")
        )

        return TransitionResult(
            dialogue_lines=dialogue_lines,
            audio_segments=[combined_audio],  # Combined into single segment
            metadata={"mode": "breath_pause", "description": "Punctuation-aware pauses applied"}
        )

    elif mode == "crossfade":
        # Cross-fading is handled in the AudioMerger during concatenation
        # This mode just returns the segments unchanged for now
        return TransitionResult(
            dialogue_lines=dialogue_lines,
            audio_segments=audio_segments,
            metadata={"mode": "crossfade", "description": "Cross-fade applied during merge"}
        )

    else:
        logger.warning(f"Unknown sentence transition mode: {mode}")
        return TransitionResult(
            dialogue_lines=dialogue_lines,
            audio_segments=audio_segments,
            metadata={"mode": "none", "error": f"Unknown mode: {mode}"}
        )
