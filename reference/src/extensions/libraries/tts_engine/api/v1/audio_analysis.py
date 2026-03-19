"""
Audio Analysis API endpoints.

Provides endpoints for:
- Speaker diarization (identifying speakers in audio)
- Audio comparison (detecting duplicates/similar audio)
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# =============================================================================
# Request/Response Models
# =============================================================================


class DiarizationRequest(BaseModel):
    """Request for speaker diarization."""

    min_speakers: int = Field(default=1, ge=1, le=20, description="Minimum number of speakers")
    max_speakers: int = Field(default=10, ge=1, le=20, description="Maximum number of speakers")
    use_gpu: bool = Field(default=True, description="Use GPU if available")


class DiarizationResponse(BaseModel):
    """Response from speaker diarization."""

    speakers: list[dict] = Field(description="List of speaker segments with start/end times")
    num_speakers: int = Field(description="Total number of unique speakers detected")
    duration: float = Field(description="Audio duration in seconds")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ComparisonRequest(BaseModel):
    """Request for audio comparison."""

    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Similarity threshold")
    duplicate_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Duplicate threshold")
    use_gpu: bool = Field(default=True, description="Use GPU if available")


class ComparisonResponse(BaseModel):
    """Response from audio comparison."""

    file1: str = Field(description="First file path")
    file2: str = Field(description="Second file path")
    similarity: float = Field(description="Similarity score (0-1)")
    is_duplicate: bool = Field(description="Whether files are considered duplicates")
    is_similar: bool = Field(description="Whether files are considered similar")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class BatchComparisonResponse(BaseModel):
    """Response from batch audio comparison."""

    comparisons: list[ComparisonResponse] = Field(description="All pairwise comparisons")
    duplicates: list[tuple[str, str]] = Field(default_factory=list, description="Duplicate file pairs")
    similar_pairs: list[tuple[str, str]] = Field(default_factory=list, description="Similar file pairs")
    total_comparisons: int = Field(description="Total number of comparisons")
    duplicate_count: int = Field(description="Number of duplicate pairs found")
    similar_count: int = Field(description="Number of similar pairs found")


class TranscriptionResponse(BaseModel):
    """Response from audio transcription."""

    text: str = Field(description="Full transcript text")
    segments: list[dict] = Field(description="List of transcription segments with timestamps")
    language: str = Field(description="Detected language code")
    duration: float = Field(description="Audio duration in seconds")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class DiarizationWithTranscriptionResponse(BaseModel):
    """Response from combined speaker diarization + transcription."""

    segments: list[dict] = Field(description="Speaker-labeled transcript segments")
    num_speakers: int = Field(description="Total number of unique speakers detected")
    duration: float = Field(description="Audio duration in seconds")
    full_transcript: str = Field(description="Complete transcript text")
    language: str = Field(description="Detected language")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Speaker Diarization Endpoints
# =============================================================================


@router.post("/diarize", response_model=DiarizationResponse)
async def diarize_audio(
    file: UploadFile = File(...),
    min_speakers: int = Form(1),
    max_speakers: int = Form(10),
    use_gpu: bool = Form(True),
):
    """
    Analyze audio file for speaker diarization.

    Identifies speakers and their timestamps in the audio file.

    Args:
        file: Audio file to analyze
        min_speakers: Minimum number of speakers (default: 1)
        max_speakers: Maximum number of speakers (default: 10)
        use_gpu: Use GPU if available (default: True)

    Returns:
        DiarizationResponse with speaker segments and metadata

    **Note:** Requires pyannote.audio models to be installed and HF token configured.
    """
    from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory
    from src.extensions.libraries.audio_analysis.speaker_diarization.extension import (
        SpeakerDiarizationExtension,
    )

    # Save uploaded file
    temp_dir = get_temp_directory()
    file_path = temp_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Analyzing speakers in: {file.filename}")

        # Create extension instance
        config = {"use_gpu": use_gpu, "min_speakers": min_speakers, "max_speakers": max_speakers}
        extension = SpeakerDiarizationExtension(config)

        # Run diarization
        result = extension.analyze_audio(
            file_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        return DiarizationResponse(
            speakers=result.speakers,
            num_speakers=result.num_speakers,
            duration=result.duration,
            metadata=result.metadata,
        )

    except ImportError as e:
        logger.error(f"Diarization not available: {e}")
        raise HTTPException(
            status_code=501,
            detail="Speaker diarization not available. Install pyannote.audio and accept model licenses on HuggingFace.",
        )
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if file_path.exists():
            file_path.unlink()


@router.get("/diarize/stats")
async def get_speaker_stats_info():
    """
    Get information about the speaker diarization service.

    Returns information about available speaker diarization capabilities.
    To analyze an audio file, use POST with the audio file as multipart/form-data.
    """
    return {
        "service": "speaker_diarization",
        "description": "Audio speaker diarization and statistics",
        "endpoint": "/api/v1/tts/diarize/stats",
        "method": "POST",
        "content_type": "multipart/form-data",
        "parameters": {
            "file": "Audio file to analyze (required)",
            "min_speakers": "Minimum number of speakers (default: 1)",
            "max_speakers": "Maximum number of speakers (default: 10)",
            "use_gpu": "Use GPU if available (default: true)",
        },
        "response": {
            "speakers": "List of detected speakers with statistics",
            "total_speakers": "Total number of speakers detected",
            "total_duration": "Total audio duration in seconds",
        }
    }


@router.post("/diarize/stats")
async def get_speaker_stats(
    file: UploadFile = File(...),
    min_speakers: int = Form(1),
    max_speakers: int = Form(10),
    use_gpu: bool = Form(True),
):
    """
    Get detailed statistics for each speaker in audio file.

    Returns per-speaker statistics including:
    - Total speaking duration
    - Number of segments
    - Average segment length
    - All segments with timestamps

    Args:
        file: Audio file to analyze
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        use_gpu: Use GPU if available

    Returns:
        Dictionary with speaker statistics
    """
    from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory
    from src.extensions.libraries.audio_analysis.speaker_diarization.extension import (
        SpeakerDiarizationExtension,
    )

    temp_dir = get_temp_directory()
    file_path = temp_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        config = {"use_gpu": use_gpu, "min_speakers": min_speakers, "max_speakers": max_speakers}
        extension = SpeakerDiarizationExtension(config)

        result = extension.analyze_audio(file_path)
        stats = extension.get_speaker_stats(result)

        return {
            "speakers": stats,
            "num_speakers": result.num_speakers,
            "duration": result.duration,
        }

    except Exception as e:
        logger.error(f"Failed to get speaker stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()


@router.post("/diarize/transcribe", response_model=DiarizationWithTranscriptionResponse)
async def diarize_and_transcribe(
    file: UploadFile = File(...),
    min_speakers: int = Form(1),
    max_speakers: int = Form(10),
    language: str | None = Form(None),
    use_gpu: bool = Form(True),
):
    """
    Analyze audio file for both speaker diarization AND transcription.

    Combines faster-whisper (transcription) with pyannote.audio (speaker detection)
    to produce a complete transcript with speaker labels.

    Returns segments like:
    - SPEAKER_00: [00:00-00:05] "Hello world"
    - SPEAKER_01: [00:05-00:10] "How are you?"

    Args:
        file: Audio file to analyze
        min_speakers: Minimum number of speakers (default: 1)
        max_speakers: Maximum number of speakers (default: 10)
        language: Language code for transcription (default: auto-detect)
        use_gpu: Use GPU if available (default: True)

    Returns:
        DiarizationWithTranscriptionResponse with speaker-labeled transcript

    **Note:** Requires both pyannote.audio and faster-whisper.
    """
    from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory
    from src.extensions.libraries.audio_analysis.speaker_diarization.extension import (
        SpeakerDiarizationExtension,
    )

    temp_dir = get_temp_directory()
    file_path = temp_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Combined analysis: {file.filename}")

        config = {
            "use_gpu": use_gpu,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "whisper_model_size": "base",  # Use base model for good balance
        }
        extension = SpeakerDiarizationExtension(config)

        # Run combined analysis
        result = extension.analyze_with_transcription(
            file_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
        )

        return DiarizationWithTranscriptionResponse(
            segments=result.segments,
            num_speakers=result.num_speakers,
            duration=result.duration,
            full_transcript=result.full_transcript,
            language=result.language,
            metadata=result.metadata,
        )

    except ImportError as e:
        logger.error(f"Combined analysis not available: {e}")
        raise HTTPException(
            status_code=501,
            detail="Combined analysis requires faster-whisper and pyannote.audio",
        )
    except Exception as e:
        logger.error(f"Combined analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_only(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    use_gpu: bool = Form(True),
):
    """
    Transcribe audio file using faster-whisper (no speaker labels).

    Use this when you only need the text transcript without speaker identification.
    Faster than full diarization + transcription.

    Args:
        file: Audio file to transcribe
        language: Language code (default: auto-detect)
        use_gpu: Use GPU if available

    Returns:
        TranscriptionResponse with full transcript and segments
    """
    from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory
    from src.extensions.libraries.audio_analysis.speaker_diarization.extension import (
        SpeakerDiarizationExtension,
    )

    temp_dir = get_temp_directory()
    file_path = temp_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Transcribing: {file.filename}")

        config = {
            "use_gpu": use_gpu,
            "whisper_model_size": "base",
        }
        extension = SpeakerDiarizationExtension(config)

        # Run transcription only
        result = extension.transcribe_audio(file_path, language)

        return TranscriptionResponse(
            text=result.text,
            segments=result.segments,
            language=result.language,
            duration=result.duration,
            metadata=result.metadata,
        )

    except ImportError as e:
        logger.error(f"Transcription not available: {e}")
        raise HTTPException(
            status_code=501,
            detail="Transcription requires faster-whisper",
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()


# =============================================================================
# Audio Comparison Endpoints
# =============================================================================


@router.post("/compare", response_model=ComparisonResponse)
async def compare_audio(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    similarity_threshold: float = Form(0.85),
    duplicate_threshold: float = Form(0.95),
    use_gpu: bool = Form(True),
):
    """
    Compare two audio files for similarity.

    Uses CLAP audio embeddings to compute similarity between files.
    Can detect exact duplicates and similar audio.

    Args:
        file1: First audio file
        file2: Second audio file
        similarity_threshold: Threshold for "similar" (default: 0.85)
        duplicate_threshold: Threshold for "duplicate" (default: 0.95)
        use_gpu: Use GPU if available

    Returns:
        ComparisonResponse with similarity score and classification
    """
    from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory
    from src.extensions.libraries.audio_analysis.audio_comparison.extension import (
        AudioComparisonExtension,
    )

    temp_dir = get_temp_directory()
    file1_path = temp_dir / file1.filename
    file2_path = temp_dir / file2.filename

    try:
        # Save files
        with open(file1_path, "wb") as f:
            f.write(await file1.read())
        with open(file2_path, "wb") as f:
            f.write(await file2.read())

        logger.info(f"Comparing: {file1.filename} vs {file2.filename}")

        # Create extension instance
        config = {
            "use_gpu": use_gpu,
            "similarity_threshold": similarity_threshold,
            "duplicate_threshold": duplicate_threshold,
        }
        extension = AudioComparisonExtension(config)

        # Run comparison
        result = extension.compare_audio(
            file1_path,
            file2_path,
            similarity_threshold,
            duplicate_threshold,
        )

        return ComparisonResponse(
            file1=file1.filename,
            file2=file2.filename,
            similarity=result.similarity,
            is_duplicate=result.is_duplicate,
            is_similar=result.is_similar,
            metadata=result.metadata,
        )

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file1_path.exists():
            file1_path.unlink()
        if file2_path.exists():
            file2_path.unlink()


@router.post("/compare/batch", response_model=BatchComparisonResponse)
async def compare_audio_batch(
    files: list[UploadFile] = File(...),
    similarity_threshold: float = Form(0.85),
    duplicate_threshold: float = Form(0.95),
    use_gpu: bool = Form(True),
):
    """
    Compare multiple audio files against each other.

    Performs pairwise comparison of all uploaded files.

    Args:
        files: List of audio files to compare
        similarity_threshold: Threshold for "similar"
        duplicate_threshold: Threshold for "duplicate"
        use_gpu: Use GPU if available

    Returns:
        BatchComparisonResponse with all comparisons and duplicate/similar pairs
    """
    from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory
    from src.extensions.libraries.audio_analysis.audio_comparison.extension import (
        AudioComparisonExtension,
    )

    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 files to compare")

    temp_dir = get_temp_directory()
    file_paths = []

    try:
        # Save all files
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)

        logger.info(f"Batch comparing {len(files)} files")

        config = {
            "use_gpu": use_gpu,
            "similarity_threshold": similarity_threshold,
            "duplicate_threshold": duplicate_threshold,
        }
        extension = AudioComparisonExtension(config)

        result = extension.compare_batch(file_paths, similarity_threshold, duplicate_threshold)

        return BatchComparisonResponse(
            comparisons=[c.to_dict() for c in result.comparisons],
            duplicates=result.duplicates,
            similar_pairs=result.similar_pairs,
            total_comparisons=result.total_comparisons,
            duplicate_count=result.duplicate_count,
            similar_count=result.similar_count,
        )

    except Exception as e:
        logger.error(f"Batch comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        for path in file_paths:
            if path.exists():
                path.unlink()
