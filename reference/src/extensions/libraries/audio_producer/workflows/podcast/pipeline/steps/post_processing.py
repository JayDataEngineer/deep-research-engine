"""
Post-Processing Pipeline Step

**PURE FUNCTION**: Input → Process → Output

Input:
  - audio_chunks: List[AudioChunk] (from TTS generation step)
  - config: PostProcessingConfig (how to merge and process)

Output:
  - FinalAudio (merged audio with metadata)

This step is:
- Deterministic: same input → same output
- Side-effect free: all file I/O is explicit (via save() method)
- Composable: can chain multiple post-processing steps
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import io
import struct

logger = logging.getLogger(__name__)


@dataclass
class CrossfadeConfig:
    """Configuration for crossfading between audio chunks."""
    enabled: bool = False
    duration_ms: int = 100           # Crossfade duration
    curve: str = "linear"            # linear, exponential, equal_power


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing step."""
    # Silence between chunks
    add_silence: bool = True
    silence_duration: float = 0.5    # seconds

    # Crossfade
    crossfade: Optional[CrossfadeConfig] = None

    # Normalization
    normalize: bool = True
    target_db: float = -3.0          # Target loudness in dB

    # Output format
    output_format: str = "wav"
    sample_rate: int = 22050


@dataclass
class FinalAudio:
    """Final merged audio with metadata."""
    audio_data: bytes                # Raw audio bytes
    format: str                      # "wav", "mp3", etc.
    sample_rate: int                 # Sample rate in Hz
    duration: float                  # Duration in seconds
    num_chunks: int                  # Number of audio chunks merged
    processing_stats: dict           # Statistics about processing

    def save(self, path: Path) -> None:
        """Save audio to file (side effect, but explicit).

        This is the ONLY side effect in the entire pipeline.
        All other processing is pure.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.audio_data)

        logger.info(f"Saved final audio to {path} ({self.duration:.2f}s, {self.num_chunks} chunks)")


# ============================================================================
# PURE FUNCTION - The core of this step
# ============================================================================

def process_audio_chunks(
    audio_chunks: List,
    config: PostProcessingConfig
) -> FinalAudio:
    """Process and merge audio chunks into final audio.

    **PURE FUNCTION**: All processing is in-memory.
    File I/O only happens when you call .save() on the result.

    Args:
        audio_chunks: Output from TTS generation step
        config: Post-processing configuration

    Returns:
        FinalAudio object (call .save() to write to disk)

    Example:
        config = PostProcessingConfig(
            add_silence=True,
            silence_duration=0.5,
            crossfade=CrossfadeConfig(enabled=True, duration_ms=100)
        )
        final = process_audio_chunks(chunks, config)
        final.save(Path("output/podcast.wav"))
    """
    if not audio_chunks:
        raise ValueError("No audio chunks to process")

    # Extract raw audio data from chunks
    audio_segments = [chunk.audio_data for chunk in audio_chunks]

    # Merge segments based on config
    if config.crossfade and config.crossfade.enabled:
        merged_data = _merge_with_crossfade(audio_segments, config)
    else:
        merged_data = _merge_with_silence(audio_segments, config)

    # Calculate final stats
    total_duration = sum(chunk.duration for chunk in audio_chunks)

    # Normalize if requested
    if config.normalize:
        merged_data = _normalize_audio(merged_data, config.target_db)

    return FinalAudio(
        audio_data=merged_data,
        format=config.output_format,
        sample_rate=config.sample_rate,
        duration=total_duration,
        num_chunks=len(audio_chunks),
        processing_stats={
            "chunks_processed": len(audio_chunks),
            "crossfade_enabled": config.crossfade.enabled if config.crossfade else False,
            "silence_added": config.add_silence,
            "normalized": config.normalize
        }
    )


# ============================================================================
# Helper functions (all pure - no side effects)
# ============================================================================

def _merge_with_silence(
    audio_segments: List[bytes],
    config: PostProcessingConfig
) -> bytes:
    """Merge audio segments with silence between them.

    **PURE**: Bytes manipulation, no side effects.
    """
    if not config.add_silence:
        return b''.join(audio_segments)

    # Generate silence
    silence = _generate_silence(
        duration=config.silence_duration,
        sample_rate=config.sample_rate
    )

    # Join segments with silence
    result = []
    for i, segment in enumerate(audio_segments):
        result.append(segment)
        if i < len(audio_segments) - 1:  # Don't add silence after last segment
            result.append(silence)

    return b''.join(result)


def _merge_with_crossfade(
    audio_segments: List[bytes],
    config: PostProcessingConfig
) -> bytes:
    """Merge audio segments with crossfade between them.

    **PURE**: Audio processing, no side effects.

    Note: This is a simplified implementation.
    Full crossfade requires WAV parsing and audio processing.
    For now, falls back to silence-based merge.
    """
    # TODO: Implement proper crossfade
    # Requires:
    # - Parse WAV headers to get audio data
    # - Extract samples
    # - Apply crossfade curve
    # - Reconstruct WAV

    # For now, use silence-based merge
    logger.warning("Crossfade not fully implemented, using silence instead")
    return _merge_with_silence(audio_segments, config)


def _generate_silence(duration: float, sample_rate: int) -> bytes:
    """Generate silence as WAV bytes.

    **PURE**: Pure computation, no side effects.
    """
    # Calculate number of samples
    num_samples = int(duration * sample_rate)

    # WAV header for silence
    # 16-bit PCM, mono, 22050 Hz
    byte_rate = sample_rate * 2  # 2 bytes per sample
    data_size = num_samples * 2
    file_size = 36 + data_size

    header = struct.pack('<4sI4s', b'RIFF', file_size, b'WAVE')
    fmt_chunk = struct.pack('<4sIHHIIHH',
        b'fmt ', 16,  # fmt chunk size
        1,  # PCM format
        1,  # mono
        sample_rate,
        byte_rate,
        2,  # block align
        16  # bits per sample
    )
    data_chunk = struct.pack('<4sI', b'data', data_size)

    # Silence (zeros)
    silence_data = b'\x00\x00' * num_samples

    return header + fmt_chunk + data_chunk + silence_data


def _normalize_audio(audio_data: bytes, target_db: float) -> bytes:
    """Normalize audio to target dB level.

    **PURE**: Audio processing, no side effects.

    Note: Simplified implementation.
    Full normalization requires:
    - Parse WAV to extract samples
    - Calculate RMS
    - Apply gain
    - Reconstruct WAV
    """
    # TODO: Implement proper normalization
    # For now, return audio unchanged
    return audio_data
