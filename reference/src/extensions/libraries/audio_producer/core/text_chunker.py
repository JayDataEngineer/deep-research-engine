"""
Text Chunking Service for TTS models with limited context.
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
    estimated_duration: float


@dataclass
class ChunkerConfig:
    """Configuration for text chunking."""
    max_chars: int = 500
    min_chars: int = 50
    prefer_sentence_boundary: bool = True
    overlap_chars: int = 0
    chars_per_second: float = 15.0


class ChunkingStrategy:
    """Base class for chunking strategies."""

    def chunk(self, text: str, config: ChunkerConfig) -> list:
        """Split text into chunks."""
        raise NotImplementedError


class SentenceBoundaryStrategy(ChunkingStrategy):
    """Chunk text at sentence boundaries when possible."""

    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    CLAUSE_BOUNDARIES = re.compile(r'(?<=[,;])\s+')

    def chunk(self, text: str, config: ChunkerConfig) -> list:
        """Split text at sentence or clause boundaries."""
        chunks = []
        text = text.strip()

        while len(text) > config.max_chars:
            break_point = self._find_break_point(text, config.max_chars)

            chunk_text = text[:break_point].strip()
            if len(chunk_text) >= config.min_chars:
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    char_count=len(chunk_text),
                    estimated_duration=len(chunk_text) / config.chars_per_second
                ))

            if config.overlap_chars > 0:
                text = text[break_point - config.overlap_chars:].strip()
            else:
                text = text[break_point:].strip()

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
        for match in reversed(list(self.SENTENCE_ENDINGS.finditer(text[:max_chars + 50]))):
            if match.end() <= max_chars:
                return match.end()

        for match in reversed(list(self.CLAUSE_BOUNDARIES.finditer(text[:max_chars + 30]))):
            if match.end() <= max_chars:
                return match.end()

        if max_chars < len(text):
            last_space = text[:max_chars].rfind(' ')
            if last_space > max_chars * 0.8:
                return last_space + 1

        return max_chars


class CharacterCountStrategy(ChunkingStrategy):
    """Simple chunking by character count."""

    def chunk(self, text: str, config: ChunkerConfig) -> list:
        """Split text by character count."""
        chunks = []
        text = text.strip()
        index = 0

        while len(text) > config.max_chars:
            chunk_text = text[:config.max_chars].strip()

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

        if text:
            chunks.append(TextChunk(
                text=text,
                index=index,
                char_count=len(text),
                estimated_duration=len(text) / config.chars_per_second
            ))

        return chunks


class TextChunker:
    """Text chunking service for TTS generation."""

    DEFAULT_CONFIGS = {
        "kokoro": ChunkerConfig(max_chars=500, min_chars=50, prefer_sentence_boundary=True, chars_per_second=15.0),
        "index_tts": ChunkerConfig(max_chars=200, min_chars=30, prefer_sentence_boundary=True, chars_per_second=14.0),
        "gpt_sovits": ChunkerConfig(max_chars=300, min_chars=50, prefer_sentence_boundary=True, chars_per_second=16.0),
        "maya1": ChunkerConfig(max_chars=400, min_chars=50, prefer_sentence_boundary=True, chars_per_second=18.0),
        "vibevoice": ChunkerConfig(max_chars=250, min_chars=30, prefer_sentence_boundary=True, chars_per_second=15.0),
        "espeak_ng": ChunkerConfig(max_chars=5000, min_chars=100, prefer_sentence_boundary=True, chars_per_second=13.0),
    }

    def __init__(self, extension_name: str = "default"):
        """Initialize chunker for a specific extension."""
        self.extension_name = extension_name
        self.config = self.DEFAULT_CONFIGS.get(extension_name, ChunkerConfig())

        if self.config.prefer_sentence_boundary:
            self.strategy = SentenceBoundaryStrategy()
        else:
            self.strategy = CharacterCountStrategy()

    def chunk_text(self, text: str) -> list:
        """Split text into chunks for TTS generation."""
        if not text or not text.strip():
            return []

        if len(text) <= self.config.max_chars:
            return [TextChunk(
                text=text.strip(),
                index=0,
                char_count=len(text.strip()),
                estimated_duration=len(text.strip()) / self.config.chars_per_second
            )]

        logger.info(f"Chunking text ({len(text)} chars) for {self.extension_name}")
        chunks = self.strategy.chunk(text, self.config)

        total_chars = sum(c.char_count for c in chunks)
        estimated_total_duration = sum(c.estimated_duration for c in chunks)

        logger.info(f"Split into {len(chunks)} chunks (total {total_chars} chars, ~{estimated_total_duration:.1f}s)")

        return chunks


def get_chunker_for_extension(extension_name: str) -> TextChunker:
    """Get a TextChunker instance configured for a specific extension."""
    return TextChunker(extension_name)
