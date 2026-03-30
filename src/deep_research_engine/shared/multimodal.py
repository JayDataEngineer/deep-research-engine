"""Multimodal and Hardware-Constraint Shared Utilities.

Centralizes logic for:
1. Temporal Normalization
2. FFmpeg Scene Detection
3. Acoustic Mega-Files (Concatenation + Un-stitching)
4. GPU VRAM Management
5. Biometric Identity Resolution
6. ASR Transcription (VibeVoice via InfraClient)
7. VLM Image Analysis (via InfraClient)
8. Face Recognition (CompreFace)
"""

import asyncio
import base64
import gc
import httpx
import mimetypes
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytz
import torch
from loguru import logger
from sklearn.cluster import DBSCAN

# Face Recognition (CompreFace)
COMPREFACE_BASE_URL = os.environ.get("COMPREFACE_BASE_URL", "http://localhost:8080")
COMPREFACE_API_KEY = os.environ.get("COMPREFACE_API_KEY", "")

from ..prompts import PromptRegistry, get_prompt
from . import IdentityMapping
from .infra_client import InfraClient, InfraError, InfraUnreachable


# ---------------------------------------------------------------------------
# InfraClient singleton
# ---------------------------------------------------------------------------

_infra: InfraClient | None = None


def _get_infra() -> InfraClient:
    """Lazy singleton for InfraClient."""
    global _infra
    if _infra is None:
        _infra = InfraClient()
    return _infra


def _extract_chat_text(response: dict) -> str:
    """Extract text content from InfraClient chat response dict."""
    return response["choices"][0]["message"]["content"]


# =============================================================================
# ASR TRANSCRIPTION (Whisper/VibeVoice)
# =============================================================================


async def asr_transcribe(
    audio_path: Path,
    language: Optional[str] = None,
    include_diarization: bool = True,
) -> Union[str, Dict[str, Any]]:
    """Transcribe audio file using VibeVoice ASR via InfraClient.

    Auto-loads the ASR model and handles GPU management.
    """
    logger.info(f"Transcribing audio: {audio_path} (diarization={include_diarization})")

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return "" if not include_diarization else {"text": "", "segments": [], "speakers": 0}

    try:
        result = await asyncio.to_thread(
            _get_infra().transcribe,
            audio=audio_path,
            diarize=include_diarization,
            language=language,
            response_format="verbose_json" if include_diarization else "json",
        )

        if include_diarization:
            return _parse_vibevoice_response(result)

        text = result.get("text", "")
        logger.info(f"Transcription complete: {len(text)} chars")
        return text

    except InfraError as e:
        logger.error(f"ASR failed for {audio_path}: {e}")
        if include_diarization:
            return {"text": f"[ASR failed: {e}]", "segments": [], "speakers": 0, "error": str(e)}
        return f"[ASR failed for {audio_path.name}: {e}]"
    except Exception as e:
        logger.error(f"ASR failed for {audio_path}: {e}")
        if include_diarization:
            return {"text": f"[ASR failed: {e}]", "segments": [], "speakers": 0, "error": str(e)}
        return f"[ASR failed for {audio_path.name}: {e}]"


def _parse_vibevoice_response(result: dict) -> Dict[str, Any]:
    """Parse VibeVoice ASR response dict into structured format with diarization."""
    full_text = result.get("text", "")
    segments_raw = result.get("segments", [])
    segments: List[Dict[str, Any]] = []
    speakers_set = set()

    for seg in segments_raw:
        speaker = seg.get("speaker", "Speaker 0")
        speakers_set.add(speaker)
        segments.append({
            "speaker": speaker,
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "duration": float(seg.get("duration", 0.0)),
            "text": seg.get("text", ""),
            "confidence": seg.get("confidence", 1.0),
        })

    num_speakers = len(speakers_set) or 1
    duration = result.get("duration", 0.0)
    if duration == 0.0 and segments:
        duration = max(seg["end"] for seg in segments)

    return {
        "text": full_text,
        "segments": segments,
        "speakers": num_speakers,
        "speaker_list": sorted(list(speakers_set)),
        "duration": duration,
        "diarization": "native",
        "model": result.get("model", "vibevoice-asr"),
    }


# =============================================================================
# VLM IMAGE ANALYSIS
# =============================================================================


async def vlm_analyze(
    image_paths: List[Path] | Path,
    prompt: str = "Describe this image in detail.",
) -> str:
    """Analyze one or more images using VLM via InfraClient."""
    if isinstance(image_paths, Path):
        image_paths = [image_paths]

    logger.info(f"Analyzing {len(image_paths)} images with prompt: {prompt[:50]}...")

    try:
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

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            })

        if len(content) == 1:
            return "[No valid images found for analysis]"

        response = await asyncio.to_thread(
            _get_infra().chat,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
        )

        description = _extract_chat_text(response)
        logger.info(f"VLM analysis complete: {len(description)} chars")
        return description

    except InfraUnreachable as e:
        raise RuntimeError("VLM connection failed - endpoint unreachable") from e
    except InfraError as e:
        error_msg = str(e)
        if "image input is not supported" in error_msg or "mmproj" in error_msg:
            raise RuntimeError(f"VLM model does not support image input") from e
        logger.error(f"VLM failed: {e}")
        return f"[Image analysis failed: {e}]"
    except Exception as e:
        error_msg = str(e)
        if "image input is not supported" in error_msg or "mmproj" in error_msg:
            raise RuntimeError(f"VLM model does not support image input") from e
        logger.error(f"VLM failed: {e}")
        return f"[Image analysis failed: {e}]"


# =============================================================================
# FACE RECOGNITION (CompreFace)
# =============================================================================


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
                response = await client.post(url, files=files, headers=headers, params=params)

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
                results.append({
                    "name": subject or "Unknown",
                    "confidence": similarity,
                    "box": face_data.get("box", {}),
                    "embedding": face_data.get("embedding"),
                })

        logger.info(f"Detected {len(results)} faces in {image_path.name}")
        return results

    except Exception as e:
        logger.error(f"Face recognition error for {image_path}: {e}")
        return []


# =============================================================================
# VIDEO ANALYSIS (VLM + optional ASR)
# =============================================================================


async def video_analyze(video_path: Path, fps: float = 0.5) -> str:
    """Analyze video by extracting frames and using VLM via InfraClient."""
    logger.info(f"Analyzing video: {video_path}")

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if video_fps == 0:
            logger.error(f"Could not read video FPS: {video_path}")
            return "[Video analysis failed: Could not read FPS]"

        interval = int(video_fps / fps) if fps > 0 else 1
        frames = []
        frame_count = 0

        while len(frames) < 10:
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

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Analyze these video frames and describe what's happening."}
        ]
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            })

        response = await asyncio.to_thread(
            _get_infra().chat,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
        )

        description = _extract_chat_text(response)
        logger.info(f"Video analysis complete: {len(description)} chars")
        return description

    except ImportError:
        logger.warning("opencv-python not installed, video analysis limited")
        return "[Video analysis requires opencv-python]"
    except InfraUnreachable as e:
        raise RuntimeError("VLM connection failed - endpoint unreachable") from e
    except InfraError as e:
        error_msg = str(e)
        if "image input is not supported" in error_msg or "mmproj" in error_msg:
            raise RuntimeError("VLM model does not support image input") from e
        logger.error(f"Video analysis failed: {e}")
        return f"[Video analysis failed: {e}]"
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        return f"[Video analysis failed: {e}]"


# =============================================================================
# RESOURCE & MEMORY MANAGEMENT (PHASE 4)
# =============================================================================


def purge_vram():
    """Explicitly purge VRAM and collect garbage."""
    logger.info("Purging VRAM...")
    # Step 1: Garbage Collection
    gc.collect()
    # Step 2: Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"VRAM cleared. Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")


# =============================================================================
# TEMPORAL NORMALIZATION (PHASE 1)
# =============================================================================

TARGET_TZ = pytz.timezone("America/New_York")


def normalize_timestamp(timestamp: Union[float, str]) -> str:
    """Normalize timestamp to America/New_York with proper DST handling."""
    if isinstance(timestamp, (int, float)):
        utc_dt = datetime.fromtimestamp(timestamp, pytz.utc)
    else:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                utc_dt = pytz.utc.localize(dt)
            else:
                utc_dt = dt.astimezone(pytz.utc)
        except ValueError:
            # Try Telegram HTML format: "13.03.2026 08:59:37 UTC-05:00"
            try:
                cleaned = timestamp.strip()
                # Remove "UTC" prefix from timezone offset
                cleaned = re.sub(r'UTC([+-])', r'\1', cleaned)
                dt = datetime.strptime(cleaned, "%d.%m.%Y %H:%M:%S %z")
                utc_dt = dt.astimezone(pytz.utc)
            except (ValueError, AttributeError):
                logger.warning(f"Could not parse timestamp: {timestamp}")
                return str(timestamp)

    localized_dt = TARGET_TZ.normalize(utc_dt.astimezone(TARGET_TZ))
    return localized_dt.isoformat()


# =============================================================================
# VISUAL PROCESSING (PHASE 2)
# =============================================================================


async def extract_scene_frames(
    video_path: Path, output_dir: Path, threshold: float = 0.3
) -> list[dict[str, Any]]:
    """Extract visually distinct frames using FFmpeg scene detection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "time.txt"

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-filter_complex",
        f"select='gt(scene,{threshold})',metadata=print:file={metadata_file}",
        "-vsync",
        "vfr",
        str(output_dir / "frame_%04d.png"),
    ]

    logger.info(f"Extracting scenes from {video_path.name} (threshold={threshold})")
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _stdout, stderr = await process.communicate()

    if process.returncode != 0:
        logger.error(f"FFmpeg failed: {stderr.decode()}")
        return []

    frames = []
    pts_times = {}

    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                content = f.read()
                # Split by "frame:" to handle each frame block separately for robustness
                blocks = re.split(r"frame:\d+", content)
                idx = 1
                for block in blocks:
                    if not block.strip():
                        continue
                    
                    pts_match = re.search(r"pts_time:([\d\.]+)", block)
                    score_match = re.search(r"lavfi\.scene_score=([\d\.]+)", block)
                    
                    if pts_match:
                        pts_times[idx] = {
                            "pts_time": float(pts_match.group(1)),
                            "scene_score": float(score_match.group(1)) if score_match else 0.0,
                        }
                        idx += 1
        except Exception as e:
            logger.warning(f"Failed to parse FFmpeg metadata: {e}")

    for i, f in enumerate(sorted(output_dir.glob("frame_*.png")), 1):
        meta = pts_times.get(i, {"pts_time": 0.0, "scene_score": 0.0})
        frames.append(
            {
                "path": str(f),
                "filename": f.name,
                "pts_time": meta["pts_time"],
                "scene_score": meta["scene_score"],
            }
        )

    return frames


async def batch_vlm_video_analysis(
    frame_items: list[dict[str, Any]],
    prompt: str = "Generate a cohesive narrative for this video segment based on these key frames. Describe the actions, characters, and environment.",
    batch_size: int = 15
) -> str:
    """Analyze multiple frames using a VLM to generate a cohesive narrative."""
    if not frame_items:
        return "[No frames available for video analysis]"

    # Sort frames by timestamp to ensure chronological narrative
    sorted_frames = sorted(frame_items, key=lambda x: x.get("pts_time", 0))
    
    # If there are too many frames, batch them to avoid context window/token limits
    all_descriptions = []
    
    # Simple batching logic
    vlm_unavailable = False
    for i in range(0, len(sorted_frames), batch_size):
        if vlm_unavailable:
            all_descriptions.append("[VLM unavailable - image analysis skipped]")
            continue

        batch = sorted_frames[i : i + batch_size]
        logger.info(f"Analyzing VLM batch {i//batch_size + 1} ({len(batch)} frames)")

        try:
            content = [{"type": "text", "text": prompt}]

            for frame in batch:
                frame_path = Path(frame["path"])
                if not frame_path.exists():
                    continue

                with open(frame_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")

                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })

            response = await asyncio.to_thread(
                _get_infra().chat,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=2000,
            )
            all_descriptions.append(_extract_chat_text(response))

        except InfraUnreachable as e:
            logger.warning("Infra unreachable - skipping all remaining batches")
            vlm_unavailable = True
            all_descriptions.append(f"[Batch analysis failed: {e}]")
        except InfraError as e:
            error_msg = str(e)
            logger.error(f"Batch VLM analysis failed for batch starting at {i}: {e}")
            if "image input is not supported" in error_msg or "mmproj" in error_msg:
                logger.warning("VLM model does not support images - skipping all remaining batches")
                vlm_unavailable = True
            all_descriptions.append(f"[Batch analysis failed: {e}]")
        except Exception as e:
            logger.error(f"Batch VLM analysis failed for batch starting at {i}: {e}")
            all_descriptions.append(f"[Batch analysis failed: {e}]")

    if len(all_descriptions) == 1:
        return all_descriptions[0]

    # If multiple batches, synthesize into a final narrative
    if len(all_descriptions) > 1:
        logger.info("Synthesizing multi-batch video narrative...")
        synth_prompt = "Combine the following sequential segment descriptions into a single, cohesive narrative for the entire video:\n\n" + \
                       "\n\n---\n\n".join(all_descriptions)

        response = await asyncio.to_thread(
            _get_infra().chat,
            messages=[{"role": "user", "content": synth_prompt}],
            temperature=0.3,
        )
        return _extract_chat_text(response)

    return "[VLM analysis returned no results]"


def cluster_identities(
    embeddings: np.ndarray, eps: float = 0.6, min_samples: int = 2
) -> np.ndarray:
    """Cluster facial embeddings using DBSCAN with Cosine similarity."""
    if len(embeddings) == 0:
        return np.array([])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return dbscan.fit_predict(embeddings)


# =============================================================================
# ACOUSTIC PROCESSING (PHASE 3 & 5)
# =============================================================================


async def create_mega_files(
    audio_paths: list[Path], output_dir: Path, max_duration_sec: float = 3600.0
) -> list[dict[str, Any]]:
    """Concatenate audio fragments into multiple Mega-Files with resampling."""
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = []
    current_batch = []
    current_duration = 0.0

    for path in audio_paths:
        duration = get_audio_duration(path)
        
        # If a single file is longer than max_duration, it will be its own batch
        if current_duration + duration > max_duration_sec and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_duration = 0.0

        current_batch.append((path, duration))
        current_duration += duration

    if current_batch:
        batches.append(current_batch)

    boundary_maps = []

    for i, batch in enumerate(batches):
        mega_file_path = output_dir / f"mega_audio_batch_{i:02d}.wav"
        concat_list_file = mega_file_path.with_suffix(".txt")

        boundary_map = {"mega_file_path": str(mega_file_path), "files": []}
        current_offset = 0.0

        with open(concat_list_file, "w") as f:
            for path, duration in batch:
                f.write(f"file '{path.absolute()}'\n")
                boundary_map["files"].append(
                    {
                        "original_path": str(path),
                        "name": path.name,
                        "start": current_offset,
                        "end": current_offset + duration,
                        "duration": duration,
                    }
                )
                current_offset += duration

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_file),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(mega_file_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        await process.wait()

        if process.returncode == 0:
            boundary_maps.append(boundary_map)
        else:
            logger.error(f"FFmpeg concat failed for batch {i}")

    return boundary_maps


def get_audio_duration(path: Path) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except Exception:
        return 0.0


async def extract_audio_snippet(
    input_audio: Path, start_sec: float, duration_sec: float, output_path: Path
) -> bool:
    """Extract a snippet of audio using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(input_audio),
        "-c",
        "copy",
        str(output_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    await process.wait()
    return process.returncode == 0


async def extract_speaker_embedding(audio_segment_path: Path) -> np.ndarray:
    """Extract ECAPA-TDNN speaker embedding using SpeechBrain."""
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    signal, _fs = classifier.load_audio(str(audio_segment_path))
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze().cpu().numpy()


def calculate_speaker_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between speaker embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


async def unify_speakers(
    mega_file_results: list[tuple[Path, dict[str, Any]]], output_dir: Path, threshold: float = 0.8
) -> dict[str, str]:
    """Unify Speaker IDs across multiple Mega-Files using acoustic embeddings."""
    global_speaker_map: dict[str, str] = {}
    speaker_embeddings: dict[str, np.ndarray] = {}
    next_global_id = 0

    snippet_dir = output_dir / "speaker_snippets"
    snippet_dir.mkdir(parents=True, exist_ok=True)

    for mf_idx, (audio_path, asr_output) in enumerate(mega_file_results):
        segments = asr_output.get("segments", [])
        speakers = asr_output.get("speaker_list", [])
        for speaker_id in speakers:
            # Enhancement #1: Extract high-confidence 10-30s snippets
            # Prioritize segments with duration between 10-30s and high confidence
            candidates = [
                s for s in segments 
                if s["speaker"] == speaker_id 
                and s.get("duration", 0) >= 10.0
            ]
            
            # Sort by confidence if available, then by duration (favoring 20s-ish)
            if candidates:
                # Prefer confidence > 0.9 if possible, then duration closest to 20s
                candidates.sort(key=lambda x: (x.get("confidence", 1.0), -abs(x.get("duration", 20) - 20)), reverse=True)
                target_seg = candidates[0]
            else:
                # Fallback to longest segment
                candidates = [s for s in segments if s["speaker"] == speaker_id]
                candidates.sort(key=lambda x: x.get("duration", 0), reverse=True)
                target_seg = candidates[0] if candidates else None

            if not target_seg:
                continue

            snippet_path = snippet_dir / f"mf{mf_idx}_{speaker_id}.wav"
            # Limit snippet to 30s
            snippet_duration = min(target_seg["duration"], 30.0)
            
            if await extract_audio_snippet(
                audio_path, target_seg["start"], snippet_duration, snippet_path
            ):
                emb = await extract_speaker_embedding(snippet_path)
                match_id = None
                for gid, gemb in speaker_embeddings.items():
                    if calculate_speaker_similarity(emb, gemb) >= threshold:
                        match_id = gid
                        break
                if match_id:
                    global_speaker_map[f"mf{mf_idx}:{speaker_id}"] = match_id
                else:
                    new_gid = f"Global_Speaker_{next_global_id}"
                    speaker_embeddings[new_gid] = emb
                    global_speaker_map[f"mf{mf_idx}:{speaker_id}"] = new_gid
                    next_global_id += 1
    return global_speaker_map


def unstitch_transcript(
    asr_results: dict[str, Any], boundary_map: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Map Mega-File ASR segments back to original individual files."""
    segments = asr_results.get("segments", [])
    files = boundary_map.get("files", [])
    unstitched: dict[str, list[dict[str, Any]]] = {f["original_path"]: [] for f in files}
    for seg in segments:
        for f in files:
            if seg["start"] < f["end"] and seg["end"] > f["start"]:
                new_seg = seg.copy()
                new_seg["start"] = max(0.0, seg["start"] - f["start"])
                new_seg["end"] = seg["end"] - f["start"]
                new_seg["original_file"] = f["original_path"]
                unstitched[f["original_path"]].append(new_seg)
    return unstitched


# =============================================================================
# IDENTITY RESOLUTION (PHASE 6)
# =============================================================================


async def resolve_identities(
    asr_segments: list[dict[str, Any]],
    visual_events: list[dict[str, Any]],
    text_messages: list[dict[str, Any]],
) -> dict[str, str]:
    """Use a two-pass Research-then-Format approach to resolve identities."""
    logger.info("Resolving identities using Research-then-Format pattern...")
    try:
        research_prompt_template = await get_prompt(PromptRegistry.IDENTITY_RESEARCH)
        format_prompt_template = await get_prompt(PromptRegistry.IDENTITY_FORMAT)

        asr_summary = "\n".join(
            [f"[{s['start']:.1f}s] {s['speaker']}: {s['text']}" for s in asr_segments[:50]]
        )
        visual_summary = "\n".join(
            [f"[{v.get('timestamp', 'N/A')}] {v['cluster']}" for v in visual_events[:50]]
        )
        text_summary = "\n".join(
            [f"[{t['timestamp']}] {t['sender']}: {t['text']}" for t in text_messages[:50]]
        )

        research_prompt = research_prompt_template.compile(
            asr_summary=asr_summary, visual_summary=visual_summary, text_summary=text_summary
        )

        # Pass 1: Research
        research_response = await asyncio.to_thread(
            _get_infra().chat,
            messages=[
                {"role": "system", "content": "You are a master identity resolution researcher."},
                {"role": "user", "content": research_prompt},
            ],
            temperature=0.0,
        )
        research_report = _extract_chat_text(research_response)

        # Pass 2: Format as structured JSON
        format_prompt = format_prompt_template.compile(research_report=research_report)
        format_response = await asyncio.to_thread(
            _get_infra().chat,
            messages=[{"role": "user", "content": format_prompt}],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "IdentityMapping",
                    "schema": IdentityMapping.model_json_schema(),
                    "strict": True,
                }
            },
        )
        content = _extract_chat_text(format_response)
        mapping_result = IdentityMapping.model_validate_json(content)
        return mapping_result.mapping

    except Exception as e:
        logger.error(f"Identity resolution failed: {e}")
        return {}


# =============================================================================
# VIDEO AUDIO EXTRACTION (For ASR)
# =============================================================================


async def extract_audio_from_video(
    video_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """Extract audio track from video and convert to ASR-compatible format.
    
    Extracts the audio track from a video file and converts it to 16kHz mono
    WAV format, which is compatible with VibeVoice ASR and other speech models.
    
    Args:
        video_path: Path to input video file (.mp4, .mkv, .avi, .mov, etc.)
        output_path: Optional output path (default: same name with .wav extension)
        sample_rate: Target sample rate in Hz (default: 16kHz for ASR)
        channels: Number of audio channels (default: 1 for mono)
    
    Returns:
        Path to extracted audio file
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        subprocess.CalledProcessError: If FFmpeg fails
        RuntimeError: If output file is not created
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command for audio extraction
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", str(video_path),  # Input video
        "-vn",  # No video output
        "-acodec", "pcm_s16le",  # PCM 16-bit codec
        "-ar", str(sample_rate),  # Sample rate
        "-ac", str(channels),  # Channels (mono)
        str(output_path)  # Output file
    ]
    
    logger.info(f"Extracting audio from {video_path.name}...")
    
    # Run FFmpeg
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
        logger.error(f"Audio extraction failed: {error_msg}")
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
            stdout,
            stderr
        )
    
    # Verify output
    if not output_path.exists():
        raise RuntimeError(f"Audio extraction failed: output file not created")
    
    file_size_kb = output_path.stat().st_size / 1024
    logger.info(f"Audio extracted: {output_path} ({file_size_kb:.1f} KB)")
    
    return output_path


async def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using FFprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds (0.0 if failed)
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, _ = await process.communicate()
    
    if process.returncode != 0:
        return 0.0
    
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0
