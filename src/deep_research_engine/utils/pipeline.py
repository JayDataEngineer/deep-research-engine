"""Advanced Ingestion Pipeline - Multimodal processing on hardware-constrained environments.

Implements the phased ingestion approach:
1. Temporal Normalization (America/New_York + DST)
2. Visual Processing (FFmpeg scene detection + DBSCAN clustering)
3. Acoustic Mega-File Strategy (Concatenation + Boundary Map)
4. GPU Memory Lifecycle Hooks (del, gc.collect, empty_cache)
5. Cross-Mega-File Acoustic Clustering (ECAPA-TDNN)
6. Multimodal Synthesis (LLM Identity Resolution)

Note: Core multimodal functions are in shared/multimodal.py to avoid duplication.
This module contains only pipeline-specific utilities.
"""

import re
import subprocess
from pathlib import Path
from typing import Any

import pytz
from loguru import logger

# Import shared multimodal functions instead of duplicating

# Timezone Constants
TARGET_TZ = pytz.timezone('America/New_York')

# --- Phase 1: Temporal Normalization ---
# (normalize_timestamp imported from shared/multimodal.py)


# --- Phase 2: Visual Processing & Spatial Identity Clustering ---
# (extract_scene_frames, cluster_identities imported from shared/multimodal.py)

async def parse_ffmpeg_metadata(time_file: Path) -> list[dict[str, Any]]:
    """Parse FFmpeg time.txt metadata to extract precise timestamps.
    
    This is pipeline-specific and not needed in shared/multimodal.py.
    
    FFmpeg generates time.txt like:
    frame=    1 pts=      1 time=00:00:01.23
    frame=    2 pts=     45 time=00:00:04.50
    
    Or in the newer format:
    [Parsed_metadata_1 @ 0x...] pts:108859 pts_time:1.20954
    lavfi.scene_score=0.436456
    
    Args:
        time_file: Path to FFmpeg metadata file
        
    Returns:
        List of frame metadata with precise timestamps
    """
    frames = []
    
    if not time_file.exists():
        logger.warning(f"FFmpeg metadata file not found: {time_file}")
        return frames
    
    with open(time_file) as f:
        content = f.read()
    
    # Try new format first: pts_time with scene score
    new_format_matches = re.finditer(
        r"pts_time:([\d\.]+).*?lavfi\.scene_score=([\d\.]+)",
        content,
        re.DOTALL
    )
    
    for i, match in enumerate(new_format_matches, 1):
        pts_time = float(match.group(1))
        scene_score = float(match.group(2))
        
        # Calculate milliseconds
        total_ms = int(pts_time * 1000)
        hours = total_ms // 3600000
        minutes = (total_ms % 3600000) // 60000
        seconds = (total_ms % 60000) // 1000
        milliseconds = total_ms % 1000
        
        frames.append({
            "frame_num": i,
            "pts_time": pts_time,
            "timestamp_ms": total_ms,
            "timestamp_formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}",
            "scene_score": scene_score
        })
    
    # If new format didn't work, try old format
    if not frames:
        old_format_pattern = r"frame=\s*(\d+)\s+pts=\s*(\d+)\s+time=(\d+):(\d+):(\d+)\.(\d+)"
        old_matches = re.finditer(old_format_pattern, content)
        
        for match in old_matches:
            frame_num = int(match.group(1))
            pts = int(match.group(2))
            hours = int(match.group(3))
            minutes = int(match.group(4))
            seconds = int(match.group(5))
            milliseconds = int(match.group(6).ljust(3, '0')[:3])
            
            total_ms = (
                hours * 3600000 +
                minutes * 60000 +
                seconds * 1000 +
                milliseconds
            )
            
            frames.append({
                "frame_num": frame_num,
                "pts": pts,
                "timestamp_ms": total_ms,
                "timestamp_formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}",
                "scene_score": 0.0
            })
    
    logger.info(f"Parsed {len(frames)} frame timestamps from {time_file}")
    return frames


# --- Phase 3: Acoustic Mega-File Strategy ---
# (create_mega_files imported from shared/multimodal.py)


# --- Phase 4: GPU Memory Management ---
# (purge_vram imported from shared/multimodal.py)


# --- Phase 5: Acoustic Clustering ---
# (extract_speaker_embedding, extract_audio_snippet, etc. imported)


# --- Phase 6: Multimodal Synthesis ---
# (resolve_identities, unstitch_transcript imported)


# =============================================================================
# HELPER FUNCTIONS (Pipeline-specific)
# =============================================================================

def get_audio_duration_ffprobe(path: Path) -> float:
    """Get audio duration using ffprobe (alternative to shared function).
    
    This is kept here for pipeline-specific use cases.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        return 0.0
    
    try:
        return float(process.stdout.strip())
    except ValueError:
        return 0.0
