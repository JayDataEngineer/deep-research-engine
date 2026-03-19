"""
Audio QA API endpoints for analyzing TTS-generated audio.

Provides REST API endpoints for:
- File upload
- Fast DSP-based audio quality analysis (no model required)
- Results retrieval

This uses fast signal processing analysis only - no LLM models needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


def _makedirs_safe(path: Path) -> None:
    """Thread-safe directory creation."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if not path.is_dir():
            raise

# Supported audio file extensions
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma",
    ".aiff", ".aif", ".pcm", ".raw", ".opus", ".mp4", ".webm"
}

from src.constants import DATA_INPUTS_DIR, DATA_OUTPUTS_DIR

# Optional imports for audio analysis (will use simple mock if not available)
try:
    from src.extensions.libraries.audio_analysis.graph import (
        run_analysis_modular,
        run_analysis_with_mock,
    )
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False
    run_analysis_modular = None

    # Simple mock functions when langgraph is not installed
    def run_analysis_with_mock(*args, **kwargs):
        """Simple mock analysis when langgraph is not installed."""
        return {
            "status": "complete",
            "overall_quality": "good",
            "score": 8.5,
            "issues": [],
            "analysis": "Mock analysis - audio QA dependencies not installed",
            # Required keys for job processing
            "chunks_processed_count": 1,
            "identified_errors": [],
            "critical_errors_found": 0,
            "job_status": "complete",
        }

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/audio-qa", tags=["audio-qa"])

# Configuration - paths using constants (data/inputs and data/outputs)
UPLOAD_DIR = DATA_INPUTS_DIR / "audio_qa" / "uploads"
OUTPUT_DIR = DATA_OUTPUTS_DIR / "audio_qa" / "outputs"

# Ensure directories exist (thread-safe)
_makedirs_safe(UPLOAD_DIR)
_makedirs_safe(OUTPUT_DIR)


class AnalysisRequest(BaseModel):
    """Request body for analysis."""
    mode: str = "modular"  # "mock" or "modular"


class AnalysisStatus(BaseModel):
    """Status of an ongoing analysis."""
    job_id: str
    status: str  # "pending", "processing", "complete", "error"
    progress: float  # 0-100
    message: str


# In-memory job storage (in production, use Redis or database)
jobs: dict = {}


def get_default_config() -> dict:
    """Get default configuration for audio QA."""
    return {
        "audio": {
            "chunk_length_seconds": 60,  # 60s chunks for efficient streaming processing
            "silence_threshold_db": -40,
            "min_silence_duration": 0.5,
        },
        "extended_analysis": {
            "enabled": True,
            "volume_threshold_db": 15.0,  # Increased from 6.0 - 15dB catches real issues, not normal speech
            "clip_threshold": 0.95,
            "noise_floor_db": -50,
            "min_silence_duration": 2.0,
            "max_silence_duration": 10.0,
        },
        "output_dir": str(OUTPUT_DIR),
    }


async def run_analysis_job(
    job_id: str,
    audio_path: Path,
    mode: str = "modular",
):
    """
    Run analysis in background task using the specified mode.

    Modes:
    - "mock": Fast mock mode for testing
    - "modular": CLAP + Whisper + Librosa analysis (~5-10 min)
    """
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Loading audio and creating manifest..."

        config = get_default_config()
        config["output_dir"] = str(OUTPUT_DIR / job_id)

        # Route to appropriate analysis mode
        if mode == "mock":
            logger.info(f"Running analysis for job {job_id} in MOCK mode")
            jobs[job_id]["message"] = "Running audio quality analysis (mock mode)..."
            jobs[job_id]["progress"] = 20

            try:
                result = run_analysis_with_mock(
                    input_path=str(audio_path),
                    config=config,
                    reference_text=None,
                )
            except Exception as e:
                logger.error(f"Mock analysis failed: {e}")
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = f"Analysis error: {str(e)}"
                return

        elif mode == "modular":
            # Use modular analysis (CLAP + Whisper + Librosa)
            logger.info(f"Running analysis for job {job_id} in MODULAR mode (CLAP + Whisper + Librosa)")
            jobs[job_id]["message"] = "Running audio quality analysis (modular mode - CLAP + Whisper + DSP)..."
            jobs[job_id]["progress"] = 20

            try:
                if run_analysis_modular is None:
                    raise ImportError("run_analysis_modular not available")

                logger.info(f"Calling run_analysis_modular with input_path={audio_path}")
                result = run_analysis_modular(
                    input_path=str(audio_path),
                    config=config,
                    reference_text=None,
                )
                logger.info(f"run_analysis_modular completed, chunks_processed={result.get('chunks_processed_count', 0)}")
            except Exception as e:
                logger.error(f"Modular analysis failed: {e}")
                import traceback
                traceback.print_exc()
                jobs[job_id]["status"] = "error"
                jobs[job_id]["message"] = f"Analysis error: {str(e)}"
                return

        else:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = f"Unknown mode: {mode}. Use 'mock' or 'modular'"
            return

        # Extract results
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Analysis complete"

        # Store results
        jobs[job_id]["result"] = {
            "total_chunks": result["chunks_processed_count"],
            "chunks_processed": result["chunks_processed_count"],
            "total_errors": len(result["identified_errors"]),
            "critical_errors": result["critical_errors_found"],
            "job_status": result["job_status"],
            "mode": mode,
            "errors": [
                {
                    "timestamp_start": e["timestamp_start"],
                    "timestamp_end": e["timestamp_end"],
                    "error_type": e["error_type"],
                    "severity": e["severity"],
                    "description": e["description"],
                    "proposed_fix": e["proposed_fix"],
                }
                for e in result["identified_errors"]
            ],
        }

        # Clean up uploaded file
        audio_path.unlink(missing_ok=True)

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = f"Analysis failed: {str(e)}"
        jobs[job_id]["progress"] = 0


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Audio QA Automation API",
        "version": "0.2.0",
        "description": "TTS audio quality assurance using CLAP + Whisper + Librosa",
        "endpoints": {
            "POST /audio-qa/analyze": "Upload and analyze audio file",
            "GET /audio-qa/status/{job_id}": "Get analysis status",
            "GET /audio-qa/result/{job_id}": "Get analysis results",
            "GET /audio-qa/download/{job_id}": "Download QA report",
        }
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "audio-qa",
        "has_graph": HAS_GRAPH,
        "backend": "modular",
    }


@router.post("/analyze")
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = "modular",
):
    """
    Upload an audio file and start analysis.

    Returns a job_id for tracking progress.

    Modes:
    - **modular** (default): CLAP + Whisper + Librosa analysis (~5-10 min)
    - **mock**: Fast mock mode for testing

    - **file**: Audio file to analyze (MP3, WAV, M4A, etc.)
    - **mode**: Analysis mode (default: "modular")
    """
    # Validate file type - check both content_type and file extension
    is_valid_content_type = file.content_type and file.content_type.startswith("audio/")
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    is_valid_extension = file_ext in AUDIO_EXTENSIONS

    if not is_valid_content_type and not is_valid_extension:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Please upload an audio file. Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    # Validate mode
    valid_modes = ["mock", "modular"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Use one of: {', '.join(valid_modes)}"
        )

    # Create job ID
    job_id = f"job_{asyncio.get_event_loop().time()}"

    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}") from e

    # Initialize job
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": f"Job queued (mode: {mode})",
        "file": file.filename,
        "mode": mode,
    }

    # Start background analysis
    background_tasks.add_task(
        run_analysis_job,
        job_id,
        file_path,
        mode,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,
        "message": f"Analysis job started (mode: {mode})"
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of an analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
    }


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get the analysis results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job['status']}"
        )

    if "result" not in job:
        raise HTTPException(status_code=500, detail="Result not available")

    return job["result"]


@router.get("/download/{job_id}")
async def download_report(job_id: str):
    """Download the QA report as a text file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job['status']}"
        )

    report_path = OUTPUT_DIR / job_id / "qa_report.txt"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        path=report_path,
        filename=f"qa_report_{job_id}.txt",
        media_type="text/plain"
    )
