"""Multimodal and Hardware-Constraint Shared Utilities.

Centralizes logic for:
1. Temporal Normalization
2. FFmpeg Scene Detection
3. Acoustic Mega-Files (Concatenation + Un-stitching)
4. GPU VRAM Management
5. Biometric Identity Resolution
6. Model Loading through LiteLLM/Research Orchestration
"""

import asyncio
import base64
import gc
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np
import pytz
import torch
from deepagents import create_deep_agent
from loguru import logger
from sklearn.cluster import DBSCAN

from ..prompts import PromptRegistry, get_prompt
from . import IdentityMapping
from .chat_llm import get_chat_llm

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
    for i in range(0, len(sorted_frames), batch_size):
        batch = sorted_frames[i : i + batch_size]
        logger.info(f"Analyzing VLM batch {i//batch_size + 1} ({len(batch)} frames)")
        
        try:
            llm = await get_chat_llm(temperature=0.0)
            
            content = [{"type": "text", "text": prompt}]
            
            for frame in batch:
                frame_path = Path(frame["path"])
                if not frame_path.exists():
                    continue
                    
                with open(frame_path, "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
                
                # Assume PNG as per extract_scene_frames
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
            
            response = await llm.ainvoke([{"role": "user", "content": content}])
            all_descriptions.append(response.content)
            
        except Exception as e:
            logger.error(f"Batch VLM analysis failed for batch starting at {i}: {e}")
            all_descriptions.append(f"[Batch analysis failed: {e}]")

    if len(all_descriptions) == 1:
        return all_descriptions[0]
    
    # If multiple batches, synthesize into a final narrative
    if len(all_descriptions) > 1:
        logger.info("Synthesizing multi-batch video narrative...")
        synthesis_llm = await get_chat_llm(temperature=0.3)
        synth_prompt = "Combine the following sequential segment descriptions into a single, cohesive narrative for the entire video:\n\n" + \
                       "\n\n---\n\n".join(all_descriptions)
        
        response = await synthesis_llm.ainvoke([{"role": "user", "content": synth_prompt}])
        return response.content

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
        research_agent = create_deep_agent(
            model=research_prompt_template.config.get("model", "openai:gpt-4o"),
            system_prompt="You are a master identity resolution researcher.",
        )
        research_result = await research_agent.ainvoke({"messages": [("user", research_prompt)]})
        research_report = research_result["messages"][-1].content

        format_model = format_prompt_template.config.get("model", "qwen-35-27-fast")
        llm = await get_chat_llm(model=format_model, temperature=0.0)
        structured_llm = llm.with_structured_output(IdentityMapping)
        mapping_result = await structured_llm.ainvoke(
            [
                {
                    "role": "user",
                    "content": format_prompt_template.compile(research_report=research_report),
                }
            ]
        )

        return mapping_result.mapping if isinstance(mapping_result, IdentityMapping) else {}
    except Exception as e:
        logger.error(f"Identity resolution failed: {e}")
        return {}
