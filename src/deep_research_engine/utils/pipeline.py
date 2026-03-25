"""Advanced Ingestion Pipeline - Multimodal processing on hardware-constrained environments.

Implements the phased ingestion approach:
1. Temporal Normalization (America/New_York + DST)
2. Visual Processing (FFmpeg scene detection + DBSCAN clustering)
3. Acoustic Mega-File Strategy (Concatenation + Boundary Map)
4. GPU Memory Lifecycle Hooks (del, gc.collect, empty_cache)
5. Cross-Mega-File Acoustic Clustering (ECAPA-TDNN)
6. Multimodal Synthesis (LLM Identity Resolution)
"""

import asyncio
import gc
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytz
import torch
from loguru import logger
from sklearn.cluster import DBSCAN

# Timezone Constants
TARGET_TZ = pytz.timezone('America/New_York')

# --- Phase 1: Temporal Normalization ---

def normalize_timestamp(timestamp: float | str) -> str:
    """Normalize timestamp to America/New_York with proper DST handling.
    
    Args:
        timestamp: UNIX epoch (float) or ISO string
        
    Returns:
        ISO format localized timestamp string
    """
    if isinstance(timestamp, (int, float)):
        utc_dt = datetime.fromtimestamp(timestamp, pytz.utc)
    else:
        # Try parsing string
        try:
            # Assume UTC if not specified
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                utc_dt = pytz.utc.localize(dt)
            else:
                utc_dt = dt.astimezone(pytz.utc)
        except ValueError:
            logger.warning(f"Could not parse timestamp: {timestamp}")
            return str(timestamp)
            
    # Convert and normalize for DST
    localized_dt = TARGET_TZ.normalize(utc_dt.astimezone(TARGET_TZ))
    return localized_dt.isoformat()


# --- Phase 2: Visual Processing & Spatial Identity Clustering ---

async def extract_scene_frames(video_path: Path, output_dir: Path, threshold: float = 0.3) -> list[dict[str, Any]]:
    """Extract visually distinct frames using FFmpeg scene detection.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        threshold: Scene change threshold (0.0 to 1.0)
        
    Returns:
        List of frame metadata (path, timestamp, pts)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "time.txt"
    
    # ffmpeg -i input.mp4 -filter_complex "select='gt(scene,0.3)',metadata=print:file=time.txt" -vsync vfr frame_%03d.png
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-filter_complex", f"select='gt(scene,{threshold})',metadata=print:file={metadata_file}",
        "-vsync", "vfr",
        str(output_dir / "frame_%04d.png")
    ]
    
    logger.info(f"Extracting scenes from {video_path.name} (threshold={threshold})")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    _stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        logger.error(f"FFmpeg failed: {stderr.decode()}")
        return []
        
    # Parse metadata_file to match frames to timestamps
    frames = []
    pts_times = {}
    
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                content = f.read()
                # Find frames and their timestamps
                # Format: 
                # [Parsed_metadata_1 @ 0x...] pts:108859 pts_time:1.20954
                # lavfi.scene_score=0.436456
                
                # Match pts_time and the following scene score
                matches = re.finditer(r"pts_time:([\d\.]+).*?lavfi\.scene_score=([\d\.]+)", content, re.DOTALL)
                for i, match in enumerate(matches):
                    pts_times[i + 1] = {
                        "pts_time": float(match.group(1)),
                        "scene_score": float(match.group(2))
                    }
        except Exception as e:
            logger.warning(f"Failed to parse FFmpeg metadata: {e}")
            
    # List files in output_dir and map to metadata
    for i, f in enumerate(sorted(output_dir.glob("frame_*.png")), 1):
        meta = pts_times.get(i, {"pts_time": 0.0, "scene_score": 0.0})
        frames.append({
            "path": str(f),
            "filename": f.name,
            "pts_time": meta["pts_time"],
            "scene_score": meta["scene_score"]
        })
        
    return frames


def cluster_identities(embeddings: np.ndarray, eps: float = 0.6, min_samples: int = 2) -> np.ndarray:
    """Cluster facial embeddings using DBSCAN with Cosine similarity.
    
    Args:
        embeddings: (N, D) array of embeddings
        eps: Epsilon threshold for cosine distance
        min_samples: Minimum samples for a cluster
        
    Returns:
        Cluster labels for each embedding
    """
    if len(embeddings) == 0:
        return np.array([])
        
    # metric='cosine' is crucial for high-dimensional embeddings
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings)
    return labels


# --- Phase 3: Acoustic Mega-File Strategy ---

async def create_mega_files(
    audio_paths: list[Path], 
    output_dir: Path, 
    max_duration_sec: float = 3600.0
) -> list[dict[str, Any]]:
    """Concatenate audio fragments into multiple Mega-Files if duration exceeds limit.
    
    Args:
        audio_paths: List of paths to audio files, sorted chronologically
        output_dir: Directory to save the concatenated mega-files
        max_duration_sec: Maximum duration for a single mega-file (default 1hr)
        
    Returns:
        List of Boundary Map dictionaries, one for each mega-file created
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batches = []
    current_batch = []
    current_duration = 0.0
    
    for path in audio_paths:
        duration = get_audio_duration(path)
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
                boundary_map["files"].append({
                    "original_path": str(path),
                    "name": path.name,
                    "start": current_offset,
                    "end": current_offset + duration,
                    "duration": duration
                })
                current_offset += duration
        
        # Run FFmpeg concatenation
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", str(concat_list_file), 
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(mega_file_path)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
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
        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except Exception:
        return 0.0


# --- Phase 4: GPU Memory Management ---

def purge_vram():
    """Explicitly purge VRAM and collect garbage."""
    logger.info("Purging VRAM...")
    # Step 1: Garbage Collection
    gc.collect()
    # Step 2: Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"VRAM cleared. Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")


# --- Phase 5: Acoustic Clustering ---

async def extract_speaker_embedding(audio_segment_path: Path) -> np.ndarray:
    """Extract ECAPA-TDNN speaker embedding using SpeechBrain."""
    from speechbrain.inference.speaker import EncoderClassifier
    
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    signal, _fs = classifier.load_audio(str(audio_segment_path))
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze().cpu().numpy()


async def extract_audio_snippet(
    input_audio: Path, 
    start_sec: float, 
    duration_sec: float, 
    output_path: Path
) -> bool:
    """Extract a snippet of audio using FFmpeg."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(duration_sec),
        "-i", str(input_audio), "-c", "copy", str(output_path)
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    await process.wait()
    return process.returncode == 0


def calculate_speaker_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two speaker embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


async def unify_speakers(
    mega_file_results: list[tuple[Path, dict[str, Any]]], 
    output_dir: Path,
    threshold: float = 0.8
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
            # Find a good segment (e.g., 5-10s)
            target_seg = next(
                (s for s in segments if s["speaker"] == speaker_id and s["duration"] > 5.0), 
                None
            )
            if not target_seg:
                target_seg = next(
                    (s for s in segments if s["speaker"] == speaker_id), 
                    None
                )
            
            if not target_seg: continue
            
            snippet_path = snippet_dir / f"mf{mf_idx}_{speaker_id}.wav"
            success = await extract_audio_snippet(
                audio_path, 
                target_seg["start"], 
                min(target_seg["duration"], 30.0), 
                snippet_path
            )
            
            if success:
                emb = await extract_speaker_embedding(snippet_path)
                
                # Compare with existing global speakers
                match_id = None
                for gid, gemb in speaker_embeddings.items():
                    sim = calculate_speaker_similarity(emb, gemb)
                    if sim >= threshold:
                        match_id = gid
                        break
                
                if match_id is not None:
                    global_speaker_map[f"mf{mf_idx}:{speaker_id}"] = match_id
                else:
                    new_gid = f"Global_Speaker_{next_global_id}"
                    speaker_embeddings[new_gid] = emb
                    global_speaker_map[f"mf{mf_idx}:{speaker_id}"] = new_gid
                    next_global_id += 1
            
    return global_speaker_map


# --- Phase 6: Multimodal Synthesis ---

from pydantic import BaseModel, Field
from deepagents import create_deep_agent
from ..prompts import get_prompt, PromptRegistry

# LLM Constants (Fallbacks if not in LangFuse config)
LLM_MODEL_RESEARCH = os.environ.get("LLM_MODEL_RESEARCH", "openai:gpt-4o")
LLM_MODEL_FORMAT = os.environ.get("LLM_MODEL_FORMAT", "openai:gpt-4o-mini")


class IdentityMapping(BaseModel):
    """Mapping from generic IDs to names."""
    mapping: dict[str, str] = Field(
        description="Mapping from Speaker ID or Visual Cluster ID to real name"
    )


async def resolve_identities(
    asr_segments: list[dict[str, Any]], 
    visual_events: list[dict[str, Any]], 
    text_messages: list[dict[str, Any]]
) -> dict[str, str]:
    """Use a two-pass approach to resolve identities across modalities.
    
    1. Research: DeepAgent deduces identities across modalities (Text output).
    2. Format: Standard LLM formats the research into a structured JSON mapping.
    
    Args:
        asr_segments: Transcription segments with speaker IDs
        visual_events: Visual cluster events with timestamps
        text_messages: Text messages with senders and timestamps
        
    Returns:
        Dictionary mapping IDs to resolved names
    """
    logger.info("Resolving identities using Research-then-Format pattern...")
    
    try:
        # Load prompts from LangFuse
        research_prompt_template = await get_prompt(PromptRegistry.IDENTITY_RESEARCH)
        format_prompt_template = await get_prompt(PromptRegistry.IDENTITY_FORMAT)

        # Prepare context summaries
        asr_summary = "\n".join([
            f"[{s['start']:.1f}s] {s['speaker']}: {s['text']}" 
            for s in asr_segments[:50] # Limit for prompt
        ])
        
        visual_summary = "\n".join([
            f"[{v.get('timestamp', 'N/A')}] {v['cluster']}" 
            for v in visual_events[:50]
        ])
        
        text_summary = "\n".join([
            f"[{t['timestamp']}] {t['sender']}: {t['text']}" 
            for t in text_messages[:50]
        ])

        # Pass 1: Research (Deductive Reasoning via DeepAgent)
        research_prompt = research_prompt_template.compile(
            asr_summary=asr_summary,
            visual_summary=visual_summary,
            text_summary=text_summary
        )
        
        research_agent = create_deep_agent(
            model=research_prompt_template.config.get("model", LLM_MODEL_RESEARCH),
            system_prompt=(
                "You are a master identity resolution researcher. "
                "Your goal is to cross-reference temporal metadata (audio, visual, text) "
                "to deduce the real names of the participants."
            )
        )
        
        research_result = await research_agent.ainvoke({"messages": [("user", research_prompt)]})
        research_report = research_result["messages"][-1].content
        
        logger.info("Research pass complete. Starting formatting pass...")

        # Pass 2: Format (Structured Output via standard LLM)
        try:
            from shared.chat_llm import get_chat_llm
            format_model = format_prompt_template.config.get("model", LLM_MODEL_FORMAT)
            llm = await get_chat_llm(model=format_model, temperature=0.0)
        except (ImportError, ModuleNotFoundError):
            # Fallback to direct langchain
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=LLM_MODEL_FORMAT.split(":")[-1], temperature=0.0)

        structured_llm = llm.with_structured_output(IdentityMapping)
        
        format_prompt = format_prompt_template.compile(research_report=research_report)
        
        mapping_result = await structured_llm.ainvoke(format_prompt)
        
        if isinstance(mapping_result, IdentityMapping):
            logger.info(f"Identity resolution complete: {len(mapping_result.mapping)} mappings found")
            return mapping_result.mapping
        
        return {}

    except Exception as e:
        logger.error(f"Identity resolution failed: {e}")
        return {}


def unstitch_transcript(
    asr_results: dict[str, Any], 
    boundary_map: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Map Mega-File ASR segments back to original individual files.
    
    Args:
        asr_results: VibeVoice ASR output with segments
        boundary_map: Map of offsets created during Phase 3
        
    Returns:
        Dictionary mapping original file names to their respective segments
    """
    segments = asr_results.get("segments", [])
    files = boundary_map.get("files", [])
    
    unstitched: dict[str, list[dict[str, Any]]] = {f["name"]: [] for f in files}
    
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        # Find which file(s) this segment belongs to
        for f in files:
            f_start = f["start"]
            f_end = f["end"]
            
            # Check for overlap
            if seg_start < f_end and seg_end > f_start:
                # Calculate local segment times
                local_start = max(0.0, seg_start - f_start)
                local_end = seg_end - f_start
                
                new_seg = seg.copy()
                new_seg["start"] = local_start
                new_seg["end"] = local_end
                new_seg["original_file"] = f["name"]
                
                unstitched[f["name"]].append(new_seg)
                
    return unstitched
