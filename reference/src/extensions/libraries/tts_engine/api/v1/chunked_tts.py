"""
Chunked TTS API endpoints for long-form text generation.

This module provides endpoints for generating speech from long text
by automatically chunking it into appropriate segments.
"""

import json
import logging
import time
from datetime import datetime
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.extensions.libraries.tts_engine.api.schemas.tts import TTSResponse
from src.extensions.libraries.tts_engine.text_chunker import get_chunker_for_extension
from src.extensions.libraries.tts_engine.router import tts_service
from src.extensions.libraries.tts_engine.api.utils.error_handler import (
    validate_text_length,
    validate_text_not_empty,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chunked", tags=["chunked-tts"])

# SSE headers for streaming events (matching chat.py pattern)
SSE_HEADERS = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "text/event-stream",
}


class ChunkedTTSRequest(BaseModel):
    """Request model for chunked TTS generation."""

    text: str = Field(..., description="Text to convert to speech (up to 50000 characters)")
    voice: str = Field("", description="Voice ID to use (optional, uses extension default if empty)")
    extension_name: str = Field(..., description="Extension name to use for generation")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    pitch: float = Field(1.0, ge=0.5, le=2.0, description="Pitch adjustment")
    format: str = Field("wav", description="Audio output format (wav, mp3, ogg)")
    # Extension-specific parameters (renamed to avoid Pydantic namespace conflict)
    extension_params: dict = Field(default_factory=dict, description="Extension-specific parameters")

    class Config:
        extra = "allow"


class ChunkedTTSResponse(TTSResponse):
    """Response model for chunked TTS generation."""

    chunks: int = Field(..., description="Number of chunks generated")
    total_chars: int = Field(..., description="Total character count")
    generation_time: float = Field(..., description="Total generation time in seconds")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    max_chars: int | None = Field(None, description="Maximum characters per chunk")
    min_chars: int | None = Field(None, description="Minimum characters per chunk")
    prefer_sentence_boundary: bool = Field(
        True, description="Prefer splitting at sentence boundaries"
    )
    overlap_chars: int = Field(0, ge=0, le=100, description="Character overlap between chunks")


@router.post("/generate", response_model=ChunkedTTSResponse)
async def generate_chunked_speech(request: Request, chunked_request: ChunkedTTSRequest):
    """
    Generate speech from long text by automatic chunking.

    This endpoint handles long text (up to 50000 characters) by splitting it
    into appropriate chunks for the selected model, generating audio for each,
    and concatenating the results.

    Args:
        request: FastAPI Request object
        chunked_request: ChunkedTTSRequest with text and generation parameters

    Returns:
        ChunkedTTSResponse with audio URL and metadata
    """
    start_time = time.time()

    try:
        # Validate text using centralized validators
        validate_text_not_empty(chunked_request.text)
        validate_text_length(chunked_request.text, 50000)

        # Validate extension exists
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)

        if chunked_request.extension_name not in extension_manager.extensions:
            available = list(extension_manager.extensions.keys())
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extension '{chunked_request.extension_name}' not found. Available: {available}",
            )

        logger.info(
            f"Chunked TTS Request: extension={chunked_request.extension_name}, "
            f"text_length={len(chunked_request.text)}, voice={chunked_request.voice}"
        )

        # Get chunker for this extension
        chunker = get_chunker_for_extension(chunked_request.extension_name)

        # Prepare all parameters for generation
        all_params = {
            'voice_id': chunked_request.voice if chunked_request.voice else None,
            'speed': chunked_request.speed,
            'pitch': chunked_request.pitch,
            'audio_format': chunked_request.format,
        }

        # Add extension-specific parameters
        if chunked_request.extension_params:
            for key, value in chunked_request.extension_params.items():
                if not key.startswith('_'):
                    all_params[key] = value

        # Progress tracking
        progress_updates = []

        def progress_callback(current: int, total: int, percent: float):
            """Track generation progress."""
            progress_updates.append({
                'chunk': current,
                'total': total,
                'percent': percent,
                'timestamp': time.time()
            })
            logger.info(f"Progress: {current}/{total} chunks ({percent:.1f}%)")

        # Define the generator function for chunks
        async def generate_chunk(text: str) -> tuple[bytes, float]:
            """Generate audio for a single chunk."""
            params = {**all_params, 'text': text}
            audio_buffer, duration = await tts_service.generate_audio(**params)
            return audio_buffer.getvalue(), duration

        # Generate audio with chunking
        audio_bytes, total_duration = await chunker.generate_chunked_audio(
            chunked_request.text,
            generate_chunk,
            progress_callback=progress_callback
        )

        generation_time = time.time() - start_time

        logger.info(
            f"Chunked generation complete: "
            f"{len(progress_updates)} chunks, "
            f"{total_duration:.2f}s audio in {generation_time:.2f}s "
            f"(ratio: {generation_time/total_duration:.2f}x)"
        )

        # Save audio file
        from src.extensions.libraries.tts_engine.api.utils.storage import get_audio_url, save_audio_file

        audio_id = str(datetime.now().timestamp()).replace(".", "")
        save_audio_file(audio_bytes, audio_id, chunked_request.format)
        audio_url = get_audio_url(audio_id, chunked_request.format)

        # Get chunk count from progress updates
        chunk_count = len(progress_updates) if progress_updates else 1

        return ChunkedTTSResponse(
            success=True,
            message=f"Speech generated successfully from {chunk_count} chunks",
            audio_url=audio_url,
            duration=total_duration,
            chunks=chunk_count,
            total_chars=len(request.text),
            generation_time=generation_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chunked speech generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate speech: {str(e)}",
        ) from e


@router.post("/generate-stream")
async def generate_chunked_speech_stream(request: ChunkedTTSRequest):
    """
    Generate speech from long text with real-time progress streaming.

    This endpoint handles long text by splitting it into chunks and streams
    progress updates via Server-Sent Events (SSE). The frontend receives
    real-time updates as each chunk is processed.

    SSE Events:
        - tts-start: Initial event with total chunk count
        - tts-chunk-start: When a chunk begins processing
        - tts-progress: Progress update during chunk processing
        - tts-chunk-complete: When a chunk finishes
        - tts-complete: Final event with audio URL
        - tts-error: If an error occurs

    Args:
        request: ChunkedTTSRequest with text and generation parameters

    Returns:
        StreamingResponse with SSE events
    """
    start_time = time.time()

    async def stream_tts_progress() -> AsyncIterator[str]:
        """Stream TTS generation progress via SSE."""
        try:
            # Validate text using centralized validators
            validate_text_not_empty(request.text)
            validate_text_length(request.text, 50000)

            # Validate extension exists
            # Get extension_manager from FastAPI app state
            from src.api.router_utils import get_extension_manager
            extension_manager = get_extension_manager(request)

            if request.extension_name not in extension_manager.extensions:
                available = list(extension_manager.extensions.keys())
                yield _sse_event("tts-error", {
                    "error": f"Extension '{request.extension_name}' not found. Available: {available}",
                })
                return

            logger.info(
                f"Chunked TTS Stream Request: extension={request.extension_name}, "
                f"text_length={len(request.text)}, voice={request.voice}"
            )

            # Get chunker for this extension
            chunker = get_chunker_for_extension(request.extension_name)

            # First, analyze the text to get chunk count
            chunks = chunker.chunk_text(request.text)
            total_chunks = len(chunks)

            # Send start event
            yield _sse_event("tts-start", {
                "totalChunks": total_chunks,
                "textLength": len(request.text),
                "estimatedDuration": sum(c.estimated_duration for c in chunks),
            })

            # Prepare all parameters for generation
            all_params = {
                'voice_id': request.voice if request.voice else None,
                'speed': request.speed,
                'pitch': request.pitch,
                'audio_format': request.format,
            }

            # Add extension-specific parameters
            if request.extension_params:
                for key, value in request.extension_params.items():
                    if not key.startswith('_'):
                        all_params[key] = value

            # Track chunk completion times for ETA calculation
            chunk_times = []

            # Generate audio with chunking
            audio_segments = []
            total_duration = 0

            for i, chunk in enumerate(chunks):
                chunk_start_time = time.time()

                # Emit chunk start event
                text_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                yield _sse_event("tts-chunk-start", {
                    "chunkIndex": i,
                    "chunkText": text_preview,
                    "chunkLength": len(chunk.text),
                })

                # Generate audio for this chunk
                params = {**all_params, 'text': chunk.text}
                audio_buffer, duration = await tts_service.generate_audio(**params)
                audio_segments.append(audio_buffer.getvalue())
                total_duration += duration

                # Record chunk time
                chunk_time = time.time() - chunk_start_time
                chunk_times.append(chunk_time)

                # Calculate ETA
                avg_time = sum(chunk_times) / len(chunk_times)
                remaining = total_chunks - i - 1
                eta_seconds = avg_time * remaining

                # Emit progress event
                yield _sse_event("tts-progress", {
                    "chunkIndex": i,
                    "totalChunks": total_chunks,
                    "progress": (i + 1) / total_chunks,
                    "elapsedTime": time.time() - start_time,
                    "estimatedTimeRemaining": eta_seconds,
                })

                # Emit chunk complete event
                yield _sse_event("tts-chunk-complete", {
                    "chunkIndex": i,
                    "duration": duration,
                    "chunkTime": chunk_time,
                })

            # Concatenate all audio segments
            from io import BytesIO
            combined = BytesIO()
            for segment in audio_segments:
                combined.write(segment)
            audio_bytes = combined.getvalue()

            generation_time = time.time() - start_time

            # Save audio file
            from src.extensions.libraries.tts_engine.api.utils.storage import get_audio_url, save_audio_file

            audio_id = str(datetime.now().timestamp()).replace(".", "")
            save_audio_file(audio_bytes, audio_id, request.format)
            audio_url = get_audio_url(audio_id, request.format)

            # Send complete event
            yield _sse_event("tts-complete", {
                "audioUrl": audio_url,
                "totalDuration": total_duration,
                "totalChunks": total_chunks,
                "totalChars": len(request.text),
                "generationTime": generation_time,
                "format": request.format,
            })

            logger.info(
                f"Chunked generation complete: "
                f"{total_chunks} chunks, "
                f"{total_duration:.2f}s audio in {generation_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error in chunked speech streaming: {str(e)}", exc_info=True)
            yield _sse_event("tts-error", {
                "error": str(e),
                "detail": f"Failed to generate speech: {str(e)}",
            })

    return StreamingResponse(
        stream_tts_progress(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _sse_event(event_type: str, data: dict) -> str:
    """Format an SSE event."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


@router.post("/analyze")
async def analyze_text_chunking(
    text: str,
    extension_name: str,
    max_chars: int | None = None,
    min_chars: int | None = None,
):
    """
    Analyze how text would be chunked without generating audio.

    Useful for previewing chunk boundaries before generation.

    Args:
        text: Text to analyze
        extension_name: Extension to use for chunking configuration
        max_chars: Override max chunk size
        min_chars: Override min chunk size

    Returns:
        Analysis with chunk breakdown
    """
    try:
        # Validate text using centralized validator
        validate_text_not_empty(text)

        # Get chunker for this extension
        chunker = get_chunker_for_extension(extension_name)

        # Apply overrides if provided
        if max_chars is not None:
            chunker.config.max_chars = max_chars
        if min_chars is not None:
            chunker.config.min_chars = min_chars

        # Analyze chunks
        chunks = chunker.chunk_text(text)

        total_chars = sum(c.char_count for c in chunks)
        estimated_duration = sum(c.estimated_duration for c in chunks)

        return {
            "extension_name": extension_name,
            "text_length": len(text),
            "chunk_count": len(chunks),
            "total_chars": total_chars,
            "estimated_duration": estimated_duration,
            "chunks": [
                {
                    "index": c.index,
                    "text_preview": c.text[:100] + "..." if len(c.text) > 100 else c.text,
                    "char_count": c.char_count,
                    "estimated_duration": c.estimated_duration,
                }
                for c in chunks
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing text chunking: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze text: {str(e)}",
        ) from e


@router.get("/config/{extension_name}")
async def get_chunking_config(extension_name: str):
    """
    Get the default chunking configuration for an extension.

    Args:
        extension_name: Name of the extension

    Returns:
        Chunking configuration
    """
    try:
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(request)

        if extension_name not in extension_manager.extensions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extension '{extension_name}' not found.",
            )

        chunker = get_chunker_for_extension(extension_name)

        return {
            "extension_name": extension_name,
            "max_chars": chunker.config.max_chars,
            "min_chars": chunker.config.min_chars,
            "prefer_sentence_boundary": chunker.config.prefer_sentence_boundary,
            "overlap_chars": chunker.config.overlap_chars,
            "chars_per_second": chunker.config.chars_per_second,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunking config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunking config: {str(e)}",
        ) from e
