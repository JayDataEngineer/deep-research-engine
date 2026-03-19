"""Audio Producer API - Emotion Tagging and Audio Generation.

Provides API endpoints for emotion detection and audio generation,
using the centralized content_formatter emotion module.
"""

import logging
from pathlib import Path
from typing import List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.config import settings
from src.constants import DATA_OUTPUTS_DIR, VOICE_SAMPLES_DIR
from src.shared.factory import get_llm

# Import from centralized content_formatter emotion module
from src.extensions.content.content_formatter.emotion import (
    DialogueLine,
    EmotionVector as CFEmotionVector,
    detect_emotions,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio-producer", tags=["audio-producer"])

# Ensure directories exist
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Enums
# ============================================================================

class EmotionProcessingMode(str, Enum):
    """Emotion detection processing modes."""
    SINGLE_CALL = "single_call"
    ROLLING_WINDOW = "rolling_window"
    BATCH = "batch"
    DEEP_AGENT = "deep_agent"

class ChunkingStrategy(str, Enum):
    """Chunking strategies."""
    SENTENCE_BOUNDARY = "sentence_boundary"
    MID_SENTENCE = "mid_sentence"
    FIXED = "fixed"

# ============================================================================
# Data Models
# ============================================================================

class EmotionVector(BaseModel):
    """8-dimensional emotion vector."""
    happy: float = Field(default=0.0, ge=0.0, le=1.0)
    angry: float = Field(default=0.0, ge=0.0, le=1.0)
    sad: float = Field(default=0.0, ge=0.0, le=1.0)
    afraid: float = Field(default=0.0, ge=0.0, le=1.0)
    disgusted: float = Field(default=0.0, ge=0.0, le=1.0)
    melancholic: float = Field(default=0.0, ge=0.0, le=1.0)
    surprised: float = Field(default=0.0, ge=0.0, le=1.0)
    calm: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_list(self) -> List[float]:
        return [self.happy, self.angry, self.sad, self.afraid,
                self.disgusted, self.melancholic, self.surprised, self.calm]

class TextChunk(BaseModel):
    """A text chunk with emotion tagging."""
    index: int
    text: str
    emotion: EmotionVector


class EmotionDetectionRequest(BaseModel):
    """Request for emotion detection."""
    text: str = Field(..., min_length=1, max_length=10000)
    processing_mode: EmotionProcessingMode = Field(default=EmotionProcessingMode.DEEP_AGENT)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    chunking: ChunkingStrategy = Field(default=ChunkingStrategy.SENTENCE_BOUNDARY)
    max_chunk_chars: int = Field(default=500, ge=50, le=2000)


class EmotionDetectionResponse(BaseModel):
    """Response from emotion detection."""
    success: bool
    chunks: List[TextChunk]
    total_chunks: int
    processing_time: float
    message: str


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Audio Producer API",
        "version": "2.0.0",
        "description": "Emotion detection and audio generation using centralized content_formatter",
        "endpoints": {
            "POST /audio-producer/detect": "Detect emotions in text",
            "GET /audio-producer/modes": "List available emotion detection modes",
            "GET /audio-producer/voice-samples": "List voice samples",
        }
    }


@router.get("/modes")
async def list_modes():
    """List available emotion detection modes."""
    return {
        "modes": [mode.value for mode in EmotionProcessingMode],
        "descriptions": {
            "single_call": "Process entire text in one LLM call",
            "rolling_window": "Process with sliding context window",
            "batch": "Process in batches",
            "deep_agent": "Use deep_agent with structured output (recommended)"
        }
    }


@router.post("/detect", response_model=EmotionDetectionResponse)
async def detect_emotions_endpoint(request: EmotionDetectionRequest):
    """
    Detect emotions in text using centralized content_formatter emotion detection.

    Args:
        request: Text and processing configuration

    Returns:
        Chunks with emotion vectors
    """
    import time

    start_time = time.time()

    try:
        # Split text into chunks based on strategy
        chunks = _split_text(request.text, request.chunking, request.max_chunk_chars)

        # Create DialogueLines for detection
        dialogue_lines = [DialogueLine("USER", chunk) for chunk in chunks]

        # Detect emotions using centralized module
        detected_lines, llm_info = await detect_emotions(dialogue_lines)

        # Convert to response format
        detected_chunks = []
        for i, line in enumerate(detected_lines):
            if line.emotion:
                emo_list = line.emotion.to_list()
            else:
                emo_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

            detected_chunks.append(TextChunk(
                index=i,
                text=line.text,
                emotion=EmotionVector(
                    happy=emo_list[0],
                    angry=emo_list[1],
                    sad=emo_list[2],
                    afraid=emo_list[3],
                    disgusted=emo_list[4],
                    melancholic=emo_list[5],
                    surprised=emo_list[6],
                    calm=emo_list[7]
                )
            ))

        processing_time = time.time() - start_time

        return EmotionDetectionResponse(
            success=True,
            chunks=detected_chunks,
            total_chunks=len(detected_chunks),
            processing_time=processing_time,
            message=f"Processed {len(chunks)} chunks in {processing_time:.2f}s using deep_agent"
        )

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice-samples")
async def list_voice_samples():
    """List available voice samples."""
    samples = []
    for file in VOICE_SAMPLES_DIR.glob("*.wav"):
        samples.append({
            "name": file.stem,
            "filename": file.name,
            "size_bytes": file.stat().st_size
        })
    return {"samples": samples}


# ============================================================================
# Helper Functions
# ============================================================================

def _split_text(text: str, strategy: ChunkingStrategy, max_chars: int) -> List[str]:
    """Split text into chunks based on strategy."""
    if strategy == ChunkingStrategy.SENTENCE_BOUNDARY:
        # Split by sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return [c for c in chunks if c]

    elif strategy == ChunkingStrategy.MID_SENTENCE:
        # Split at fixed character intervals
        chunks = []
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i+max_chars])
        return [c for c in chunks if c]

    else:  # FIXED
        # Return entire text as one chunk
        return [text] if text else []
