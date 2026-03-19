"""
Text Chunking Service for TTS models with limited context.

Some TTS models have difficulty generating long-form audio in a single pass.
This service splits text into appropriate chunks, generates audio for each chunk,
and concatenates the results.
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text with metadata."""
    text: str
    index: int
    char_count: int
    estimated_duration: float  # in seconds


@dataclass
class ChunkerConfig:
    """Configuration for text chunking."""
    max_chars: int = 500
    min_chars: int = 50
    prefer_sentence_boundary: bool = True
    overlap_chars: int = 0  # Overlap between chunks for smoother concatenation
    chars_per_second: float = 15.0  # For duration estimation


class ChunkingStrategy:
    """Base class for chunking strategies."""

    def chunk(self, text: str, config: ChunkerConfig) -> list[TextChunk]:
        """Split text into chunks."""
        raise NotImplementedError


class SentenceBoundaryStrategy(ChunkingStrategy):
    """Chunk text at sentence boundaries when possible."""

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    # Clause boundaries (commas, semicolons)
    CLAUSE_BOUNDARIES = re.compile(r'(?<=[,;])\s+')

    def chunk(self, text: str, config: ChunkerConfig) -> list[TextChunk]:
        """Split text at sentence or clause boundaries."""
        chunks = []
        text = text.strip()

        while len(text) > config.max_chars:
            # Find the best break point
            break_point = self._find_break_point(text, config.max_chars)

            # Extract chunk
            chunk_text = text[:break_point].strip()
            if len(chunk_text) >= config.min_chars:
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    char_count=len(chunk_text),
                    estimated_duration=len(chunk_text) / config.chars_per_second
                ))

            # Move to next chunk (with optional overlap)
            if config.overlap_chars > 0:
                text = text[break_point - config.overlap_chars:].strip()
            else:
                text = text[break_point:].strip()

        # Add remaining text
        if text:
            chunks.append(TextChunk(
                text=text,
                index=len(chunks),
                char_count=len(text),
                estimated_duration=len(text) / config.chars_per_second
            ))

        return chunks

    def _find_break_point(self, text: str, max_chars: int) -> int:
        """Find the best sentence or clause boundary within max_chars."""
        # Search for sentence boundaries first
        for match in reversed(list(self.SENTENCE_ENDINGS.finditer(text[:max_chars + 50]))):
            if match.end() <= max_chars:
                return match.end()

        # If no sentence boundary, try clause boundaries
        for match in reversed(list(self.CLAUSE_BOUNDARIES.finditer(text[:max_chars + 30]))):
            if match.end() <= max_chars:
                return match.end()

        # Fall back to max_chars but try to avoid breaking words
        if max_chars < len(text):
            # Find last space before max_chars
            last_space = text[:max_chars].rfind(' ')
            if last_space > max_chars * 0.8:  # Only if we're not losing too much
                return last_space + 1

        return max_chars


class CharacterCountStrategy(ChunkingStrategy):
    """Simple chunking by character count."""

    def chunk(self, text: str, config: ChunkerConfig) -> list[TextChunk]:
        """Split text by character count."""
        chunks = []
        text = text.strip()
        index = 0

        while len(text) > config.max_chars:
            chunk_text = text[:config.max_chars].strip()

            # Try to break at word boundary
            last_space = chunk_text.rfind(' ')
            if last_space > config.min_chars:
                chunk_text = chunk_text[:last_space].strip()
                text = text[last_space + 1:]
            else:
                text = text[config.max_chars:]

            chunks.append(TextChunk(
                text=chunk_text,
                index=index,
                char_count=len(chunk_text),
                estimated_duration=len(chunk_text) / config.chars_per_second
            ))
            index += 1

        # Add remaining text
        if text:
            chunks.append(TextChunk(
                text=text,
                index=index,
                char_count=len(text),
                estimated_duration=len(text) / config.chars_per_second
            ))

        return chunks


class TextChunker:
    """
    Text chunking service for TTS generation.

    Splits long text into manageable chunks for models with limited context.
    Generates audio for each chunk and concatenates results.
    """

    # Default configurations for known extensions
    DEFAULT_CONFIGS = {
        "kokoro": ChunkerConfig(
            max_chars=500,
            min_chars=50,
            prefer_sentence_boundary=True,
            chars_per_second=15.0
        ),
        "index_tts": ChunkerConfig(
            max_chars=200,  # IndexTTS has shorter context
            min_chars=30,
            prefer_sentence_boundary=True,
            chars_per_second=14.0
        ),
        "gpt_sovits": ChunkerConfig(
            max_chars=300,
            min_chars=50,
            prefer_sentence_boundary=True,
            chars_per_second=16.0
        ),
        "maya1": ChunkerConfig(
            max_chars=400,
            min_chars=50,
            prefer_sentence_boundary=True,
            chars_per_second=18.0
        ),
        "vibevoice": ChunkerConfig(
            max_chars=250,
            min_chars=30,
            prefer_sentence_boundary=True,
            chars_per_second=15.0
        ),
        "espeak_ng": ChunkerConfig(
            max_chars=5000,  # espeak can handle very long text
            min_chars=100,
            prefer_sentence_boundary=True,
            chars_per_second=13.0
        ),
    }

    def __init__(self, extension_name: str = "default"):
        """Initialize chunker for a specific extension."""
        self.extension_name = extension_name
        self.config = self.DEFAULT_CONFIGS.get(
            extension_name,
            ChunkerConfig()  # Default config
        )

        # Choose strategy based on config
        if self.config.prefer_sentence_boundary:
            self.strategy = SentenceBoundaryStrategy()
        else:
            self.strategy = CharacterCountStrategy()

    def chunk_text(self, text: str) -> list[TextChunk]:
        """
        Split text into chunks for TTS generation.

        Args:
            text: The input text to chunk

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # If text is short enough, return as single chunk
        if len(text) <= self.config.max_chars:
            return [TextChunk(
                text=text.strip(),
                index=0,
                char_count=len(text.strip()),
                estimated_duration=len(text.strip()) / self.config.chars_per_second
            )]

        # Otherwise, apply chunking strategy
        logger.info(f"Chunking text ({len(text)} chars) for {self.extension_name}")
        chunks = self.strategy.chunk(text, self.config)

        total_chars = sum(c.char_count for c in chunks)
        estimated_total_duration = sum(c.estimated_duration for c in chunks)

        logger.info(
            f"Split into {len(chunks)} chunks "
            f"(total {total_chars} chars, ~{estimated_total_duration:.1f}s)"
        )

        return chunks

    async def generate_chunked_audio(
        self,
        text: str,
        generator: Callable[[str], tuple[bytes, float]],
        progress_callback: Callable[[int, int, float], None] | None = None
    ) -> tuple[bytes, float]:
        """
        Generate audio for text by chunking if necessary.

        Args:
            text: The text to convert to speech
            generator: Async function that generates audio from text
                       Takes (text) and returns (audio_bytes, duration)
            progress_callback: Optional callback for progress updates
                             Args: (current_chunk, total_chunks, progress_percent)

        Returns:
            Tuple of (concatenated_audio_bytes, total_duration)
        """

        chunks = self.chunk_text(text)

        if len(chunks) == 1:
            # No chunking needed, generate directly
            logger.info("Text is short enough, generating without chunking")
            return await generator(chunks[0].text)

        # Generate audio for each chunk
        logger.info(f"Generating audio for {len(chunks)} chunks")
        start_time = time.time()

        audio_segments = []
        total_duration = 0.0

        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            logger.info(f"Generating chunk {i + 1}/{len(chunks)}: {chunk.char_count} chars")

            audio_bytes, duration = await generator(chunk.text)

            audio_segments.append(audio_bytes)
            total_duration += duration

            chunk_time = time.time() - chunk_start
            logger.info(
                f"Chunk {i + 1} generated in {chunk_time:.2f}s "
                f"(audio: {duration:.2f}s, ratio: {chunk_time/duration:.2f}x)"
            )

            # Report progress
            if progress_callback:
                progress = (i + 1) / len(chunks) * 100
                progress_callback(i + 1, len(chunks), progress)

        # Concatenate audio segments
        logger.info("Concatenating audio segments...")
        concatenated = self._concatenate_audio(audio_segments)

        total_time = time.time() - start_time
        logger.info(
            f"Chunked generation complete: "
            f"{total_duration:.2f}s audio in {total_time:.2f}s "
            f"(ratio: {total_time/total_duration:.2f}x)"
        )

        return concatenated, total_duration

    def _concatenate_audio(self, audio_segments: list[bytes]) -> bytes:
        """
        Concatenate audio segments into a single audio file.

        Args:
            audio_segments: List of audio bytes (WAV format)

        Returns:
            Concatenated audio bytes
        """
        import io
        import wave

        import numpy as np

        if not audio_segments:
            raise ValueError("No audio segments to concatenate")

        if len(audio_segments) == 1:
            return audio_segments[0]

        # Read first segment to get format info
        first_wav = wave.open(io.BytesIO(audio_segments[0]), 'rb')
        sample_rate = first_wav.getframerate()
        channels = first_wav.getnchannels()
        sample_width = first_wav.getsampwidth()
        first_wav.close()

        # Read all segments and concatenate
        all_samples = []
        for segment in audio_segments:
            wav = wave.open(io.BytesIO(segment), 'rb')
            frames = wav.readframes(wav.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
            all_samples.append(samples)
            wav.close()

        # Concatenate
        concatenated = np.concatenate(all_samples)

        # Write to new WAV
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(concatenated.tobytes())

        return output.getvalue()


def get_chunker_for_extension(extension_name: str) -> TextChunker:
    """Get a TextChunker instance configured for a specific extension."""
    return TextChunker(extension_name)
