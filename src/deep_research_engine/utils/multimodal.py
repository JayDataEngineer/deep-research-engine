import os
import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import functools
import httpx
from openai import OpenAI
from loguru import logger

# LiteLLM (ASR/VLM)
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "fake-key")

# Face Recognition (CompreFace)
COMPREFACE_BASE_URL = os.environ.get("COMPREFACE_BASE_URL", "http://localhost:8080")
COMPREFACE_API_KEY = os.environ.get("COMPREFACE_API_KEY", "")

# Models
ASR_MODEL = os.environ.get("ASR_MODEL", "whisper-1")
VLM_MODEL = os.environ.get("VLM_MODEL", "gpt-4o-mini")


async def asr_transcribe(audio_path: Path, language: Optional[str] = None) -> str:
    """Transcribe audio file using LiteLLM ASR endpoint via OpenAI client."""
    logger.info(f"Transcribing audio: {audio_path}")

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return ""

    try:
        client = OpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
        )

        with open(audio_path, "rb") as audio_file:
            kwargs = {
                "model": ASR_MODEL,
                "file": audio_file,
            }
            if language:
                kwargs["language"] = language

            # OpenAI client create is normally synchronous, but LiteLLM might be slow.
            # We run it in a thread if needed, or just call it directly if it's fast enough.
            # To keep it async-friendly:
            # To keep it async-friendly:
            result = await asyncio.to_thread(client.audio.transcriptions.create, **kwargs)

        transcription = result.text
        logger.info(f"Transcription complete: {len(transcription)} chars")
        return transcription

    except Exception as e:
        logger.error(f"ASR failed for {audio_path}: {e}")
        return f"[Audio Transcript for {audio_path.name} failed: {e}]"


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
        try:
            # We would need to extract audio first. For now, just return description.
            pass
        except:
            pass

        logger.info(f"Video analysis complete: {len(description)} chars")
        return description

    except ImportError:
        logger.warning("opencv-python not installed, video analysis limited")
        return "[Video analysis requires opencv-python]"
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        return f"[Video analysis failed: {e}]"

