import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from loguru import logger
from openai import OpenAI

# LiteLLM (ASR/VLM)
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "fake-key")

# Face Recognition (CompreFace)
COMPREFACE_BASE_URL = os.environ.get("COMPREFACE_BASE_URL", "http://localhost:8080")
COMPREFACE_API_KEY = os.environ.get("COMPREFACE_API_KEY", "")

# Models
ASR_MODEL = os.environ.get("ASR_MODEL", "vibevoice-asr")
VLM_MODEL = os.environ.get("VLM_MODEL", "gpt-4o-mini")

# VibeVoice settings
VIBEVOICE_MIN_SPEAKERS = int(os.environ.get("VIBEVOICE_MIN_SPEAKERS", "1"))
VIBEVOICE_MAX_SPEAKERS = int(os.environ.get("VIBEVOICE_MAX_SPEAKERS", "10"))


async def asr_transcribe(
    audio_path: Path,
    language: Optional[str] = None,
    include_diarization: bool = True,
) -> Union[str, Dict[str, Any]]:
    """Transcribe audio file using LiteLLM ASR endpoint via OpenAI client.

    Supports both Whisper and VibeVoice ASR models:
    - Whisper: Returns simple text transcription
    - VibeVoice ASR-7B: Returns transcription with built-in diarization and timestamps

    Args:
        audio_path: Path to audio file
        language: Optional language hint (e.g., "en", "zh")
        include_diarization: If True and using VibeVoice, return structured output
            with speaker segments and timestamps. If False, return plain text.

    Returns:
        If include_diarization=True and using VibeVoice:
            Dict with keys: text, segments, speakers, duration, diarization
        Otherwise:
            Plain transcription text (str)

    Example VibeVoice output:
        {
            "text": "Speaker 0: Hello. Speaker 1: Hi there.",
            "segments": [
                {"speaker": "Speaker 0", "start": 0.0, "end": 2.5, "text": "Hello."},
                {"speaker": "Speaker 1", "start": 3.0, "end": 5.0, "text": "Hi there."}
            ],
            "speakers": 2,
            "duration": 120.5,
            "diarization": "native"
        }
    """
    logger.info(f"Transcribing audio: {audio_path} (diarization={include_diarization})")

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return "" if not include_diarization else {"text": "", "segments": [], "speakers": 0}

    try:
        client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
        )

        # Determine if we're using VibeVoice
        is_vibevoice = "vibevoice" in ASR_MODEL.lower()

        with open(audio_path, "rb") as audio_file:
            kwargs = {
                "model": ASR_MODEL,
                "file": audio_file,
            }
            if language:
                kwargs["language"] = language

            # Request verbose_json for VibeVoice to get diarization
            if is_vibevoice and include_diarization:
                kwargs["response_format"] = "verbose_json"

            # Call ASR API
            result = await asyncio.to_thread(client.audio.transcriptions.create, **kwargs)

        # Parse response based on model
        if is_vibevoice and include_diarization:
            # VibeVoice returns enhanced response with segments
            return _parse_vibevoice_response(result)
        else:
            # Whisper or simple transcription
            transcription = result.text
            logger.info(f"Transcription complete: {len(transcription)} chars")
            return transcription

    except Exception as e:
        logger.error(f"ASR failed for {audio_path}: {e}")
        if include_diarization:
            return {
                "text": f"[Audio Transcript failed: {e}]",
                "segments": [],
                "speakers": 0,
                "error": str(e)
            }
        return f"[Audio Transcript for {audio_path.name} failed: {e}]"


def _parse_vibevoice_response(result: Any) -> Dict[str, Any]:
    """Parse VibeVoice ASR response into structured format with diarization.

    Args:
        result: Raw response from VibeVoice ASR API

    Returns:
        Dict with transcription, segments, speaker count, and metadata
    """
    # Extract full text
    full_text = getattr(result, "text", "")

    # Extract segments with speaker information
    segments_raw = getattr(result, "segments", [])
    segments: List[Dict[str, Any]] = []
    speakers_set = set()

    for seg in segments_raw:
        # Handle both dict and object segments
        if hasattr(seg, "__dict__"):
            seg_dict = seg.__dict__
        else:
            seg_dict = seg

        speaker = seg_dict.get("speaker", "Speaker 0")
        speakers_set.add(speaker)

        segments.append({
            "speaker": speaker,
            "start": float(seg_dict.get("start", 0.0)),
            "end": float(seg_dict.get("end", 0.0)),
            "duration": float(seg_dict.get("duration", 0.0)),
            "text": seg_dict.get("text", ""),
            "confidence": seg_dict.get("confidence", 1.0),
        })

    # Count unique speakers
    num_speakers = len(speakers_set)
    if num_speakers == 0:
        num_speakers = 1

    # Get duration
    duration = getattr(result, "duration", 0.0)
    if duration == 0.0 and segments:
        duration = max(seg["end"] for seg in segments)

    return {
        "text": full_text,
        "segments": segments,
        "speakers": num_speakers,
        "speaker_list": sorted(list(speakers_set)),
        "duration": duration,
        "diarization": "native",
        "model": ASR_MODEL,
    }


async def vlm_analyze(
    image_paths: List[Path] | Path,
    prompt: str = "Describe this image in detail.",
) -> str:
    """Analyze one or more images using LiteLLM VLM endpoint."""
    if isinstance(image_paths, Path):
        image_paths = [image_paths]

    logger.info(f"Analyzing {len(image_paths)} images with prompt: {prompt[:50]}...")

    try:
        client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for path in image_paths:
            if not path.exists():
                logger.warning(f"Image not found: {path}")
                continue

            with open(path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type:
                mime_type = "image/jpeg"

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        if len(content) == 1:
            return "[No valid images found for analysis]"

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
        )

        description = response.choices[0].message.content
        logger.info(f"VLM analysis complete: {len(description)} chars")
        return description

    except Exception as e:
        logger.error(f"VLM failed: {e}")
        return f"[Image analysis failed: {e}]"


async def face_recognize(image_path: Path, min_similarity: float = 0.8) -> List[Dict[str, Any]]:
    """Recognize faces in image using CompreFace API."""
    if not COMPREFACE_API_KEY:
        logger.debug("CompreFace API key not set, skipping face recognition")
        return []

    logger.info(f"Recognizing faces in: {image_path}")

    url = f"{COMPREFACE_BASE_URL}/api/v1/recognition/recognize"
    headers = {"x-api-key": COMPREFACE_API_KEY}
    params = {"face_plugins": "embedding"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(image_path, "rb") as f:
                files = {"file": (image_path.name, f, "image/jpeg")}
                response = await client.post(
                    url,
                    files=files,
                    headers=headers,
                    params=params,
                )

        if response.status_code != 200:
            logger.warning(f"CompreFace failed with status {response.status_code}")
            return []

        data = response.json()
        results = []

        for face_data in data.get("result", []):
            subjects = face_data.get("subjects", [])
            subject = subjects[0]["subject"] if subjects else None
            similarity = subjects[0]["similarity"] if subjects else 0.0

            if similarity >= min_similarity:
                results.append(
                    {
                        "name": subject or "Unknown",
                        "confidence": similarity,
                        "box": face_data.get("box", {}),
                        "embedding": face_data.get("embedding"),
                    }
                )

        logger.info(f"Detected {len(results)} faces in {image_path.name}")
        return results

    except Exception as e:
        logger.error(f"Face recognition error for {image_path}: {e}")
        return []


async def video_analyze(video_path: Path, fps: float = 0.5) -> str:
    """Analyze video by extracting frames and using VLM."""
    logger.info(f"Analyzing video: {video_path}")

    try:
        import cv2

        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if video_fps == 0:
            logger.error(f"Could not read video FPS: {video_path}")
            return "[Video analysis failed: Could not read FPS]"

        interval = int(video_fps / fps) if fps > 0 else 1
        frames = []
        frame_count = 0

        # Create temporary directory for frames if needed, or just keep in memory if small
        # For now, let's just encode them directly
        while len(frames) < 10:  # Limit to 10 frames for analysis
            ret, frame = cap.read()
            if not ret:
                break

            if int(frame_count) % int(interval) == 0:
                _, buffer = cv2.imencode(".jpg", frame)
                frames.append(base64.b64encode(buffer).decode("utf-8"))

            frame_count = int(frame_count) + 1

        cap.release()

        if not frames:
            return "[No frames extracted from video]"

        # Call VLM with frames
        client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
        )

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Analyze these video frames and describe what's happening."}
        ]
        for frame_b64 in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                }
            )

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
        )

        description = response.choices[0].message.content

        # Optionally also try ASR on video audio
        # TODO: Implement audio extraction from video for ASR

        logger.info(f"Video analysis complete: {len(description)} chars")
        return description

    except ImportError:
        logger.warning("opencv-python not installed, video analysis limited")
        return "[Video analysis requires opencv-python]"
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        return f"[Video analysis failed: {e}]"

