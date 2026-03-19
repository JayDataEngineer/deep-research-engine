"""
Batch TTS API endpoints for processing multiple texts.

This module provides endpoints for queueing and processing multiple TTS jobs.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.extensions.libraries.tts_engine.router import tts_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])


class JobStatus(str, Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """A single job in a batch."""
    id: str
    text: str
    voice: str
    extension_name: str
    speed: float
    pitch: float
    audio_format: str
    params: dict
    status: JobStatus = JobStatus.PENDING
    audio_url: str | None = None
    duration: float | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class BatchRequest:
    """A batch processing request."""
    id: str
    jobs: list[BatchJob]
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    cancelled: bool = False


# Global batch storage
active_batches: dict[str, BatchRequest] = {}
batch_lock = asyncio.Lock()


# Pydantic models for API
class BatchJobItem(BaseModel):
    """Item in a batch job."""
    text: str = Field(..., description="Text to convert")
    voice: str = Field("", description="Voice ID")
    speed: float = Field(1.0, ge=0.25, le=4.0)
    pitch: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", description="Audio format")
    params: dict = Field(default_factory=dict, description="Extra parameters")


class CreateBatchRequest(BaseModel):
    """Request to create a batch."""
    extension_name: str = Field(..., description="Extension to use")
    jobs: list[BatchJobItem] = Field(..., description="Jobs to process", min_items=1, max_items=100)


class BatchStatusResponse(BaseModel):
    """Response for batch status."""
    batch_id: str
    status: JobStatus
    total_jobs: int
    completed: int
    failed: int
    running: int
    pending: int
    jobs: list[dict]
    created_at: float
    started_at: float | None
    completed_at: float | None


class BatchResponse(BaseModel):
    """Response after batch creation."""
    batch_id: str
    status: JobStatus
    total_jobs: int
    message: str


async def process_single_job(job: BatchJob) -> None:
    """Process a single TTS job."""
    job.status = JobStatus.RUNNING
    job.started_at = time.time()

    try:
        # Prepare parameters
        params = {
            'text': job.text,
            'voice_id': job.voice if job.voice else None,
            'speed': job.speed,
            'pitch': job.pitch,
            'audio_format': job.audio_format,
            **job.params
        }

        # Generate audio
        audio_buffer, duration = await tts_service.generate_audio(**params)

        # Save audio
        from src.extensions.libraries.tts_engine.api.utils.storage import get_audio_url, save_audio_file

        audio_id = str(datetime.now().timestamp()).replace(".", "")
        audio_bytes = audio_buffer.getvalue()
        save_audio_file(audio_bytes, audio_id, job.audio_format)
        job.audio_url = get_audio_url(audio_id, job.audio_format)
        job.duration = duration
        job.status = JobStatus.COMPLETED

    except Exception as e:
        logger.error(f"Job {job.id} failed: {str(e)}")
        job.status = JobStatus.FAILED
        job.error = str(e)

    finally:
        job.completed_at = time.time()


async def process_batch(batch: BatchRequest) -> None:
    """Process all jobs in a batch."""
    batch.status = JobStatus.RUNNING
    batch.started_at = time.time()

    try:
        for job in batch.jobs:
            if batch.cancelled:
                # Mark remaining jobs as cancelled
                job.status = JobStatus.CANCELLED
                continue

            await process_single_job(job)

        batch.status = JobStatus.COMPLETED

    except Exception as e:
        logger.error(f"Batch {batch.id} failed: {str(e)}")
        # Don't fail the whole batch, individual jobs have their own status

    finally:
        batch.completed_at = time.time()


@router.post("/create", response_model=BatchResponse)
async def create_batch(http_request: Request, request: CreateBatchRequest):
    """
    Create a new batch processing job.

    Args:
        http_request: FastAPI Request object
        request: Batch creation request with jobs

    Returns:
        BatchResponse with batch ID and status
    """
    try:
        # Validate extension exists
        # Get extension_manager from FastAPI app state
        from src.api.router_utils import get_extension_manager
        extension_manager = get_extension_manager(http_request)

        if request.extension_name not in extension_manager.extensions:
            available = list(extension_manager.extensions.keys())
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extension '{request.extension_name}' not found. Available: {available}",
            )

        # Create batch
        batch_id = str(uuid.uuid4())[:8]
        jobs = []

        for i, job_item in enumerate(request.jobs):
            job = BatchJob(
                id=f"{batch_id}-{i}",
                text=job_item.text,
                voice=job_item.voice,
                extension_name=request.extension_name,
                speed=job_item.speed,
                pitch=job_item.pitch,
                audio_format=job_item.format,
                params=job_item.params
            )
            jobs.append(job)

        batch = BatchRequest(id=batch_id, jobs=jobs)

        async with batch_lock:
            active_batches[batch_id] = batch

        # Start processing in background
        asyncio.create_task(process_batch(batch))

        logger.info(f"Created batch {batch_id} with {len(jobs)} jobs")

        return BatchResponse(
            batch_id=batch_id,
            status=JobStatus.PENDING,
            total_jobs=len(jobs),
            message=f"Batch created with {len(jobs)} jobs"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create batch: {str(e)}",
        ) from e


@router.get("/status/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """
    Get the status of a batch job.

    Args:
        batch_id: Batch ID

    Returns:
        BatchStatusResponse with current status
    """
    async with batch_lock:
        if batch_id not in active_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_id}' not found",
            )

        batch = active_batches[batch_id]

    # Count jobs by status
    completed = sum(1 for j in batch.jobs if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in batch.jobs if j.status == JobStatus.FAILED)
    running = sum(1 for j in batch.jobs if j.status == JobStatus.RUNNING)
    pending = sum(1 for j in batch.jobs if j.status == JobStatus.PENDING)

    # Convert jobs to dict for response
    jobs_data = []
    for job in batch.jobs:
        jobs_data.append({
            "id": job.id,
            "status": job.status.value,
            "audio_url": job.audio_url,
            "duration": job.duration,
            "error": job.error,
            "text_preview": job.text[:100] + "..." if len(job.text) > 100 else job.text,
        })

    return BatchStatusResponse(
        batch_id=batch_id,
        status=batch.status,
        total_jobs=len(batch.jobs),
        completed=completed,
        failed=failed,
        running=running,
        pending=pending,
        jobs=jobs_data,
        created_at=batch.created_at,
        started_at=batch.started_at,
        completed_at=batch.completed_at,
    )


@router.post("/cancel/{batch_id}")
async def cancel_batch(batch_id: str):
    """
    Cancel a running batch.

    Args:
        batch_id: Batch ID to cancel

    Returns:
        Cancellation confirmation
    """
    async with batch_lock:
        if batch_id not in active_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_id}' not found",
            )

        batch = active_batches[batch_id]

        if batch.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel batch in {batch.status.value} status",
            )

        batch.cancelled = True
        batch.status = JobStatus.CANCELLED

    logger.info(f"Cancelled batch {batch_id}")

    return {
        "batch_id": batch_id,
        "status": "cancelled",
        "message": "Batch cancellation requested"
    }


@router.get("/jobs/{batch_id}/download")
async def download_batch_results(batch_id: str):
    """
    Get download URLs for all completed jobs in a batch.

    Args:
        batch_id: Batch ID

    Returns:
        List of download URLs
    """
    async with batch_lock:
        if batch_id not in active_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_id}' not found",
            )

        batch = active_batches[batch_id]

    results = []
    for job in batch.jobs:
        if job.status == JobStatus.COMPLETED:
            results.append({
                "job_id": job.id,
                "audio_url": job.audio_url,
                "duration": job.duration,
                "text": job.text,
            })

    return {
        "batch_id": batch_id,
        "total_results": len(results),
        "results": results
    }


@router.delete("/{batch_id}")
async def delete_batch(batch_id: str):
    """
    Delete a batch and free up memory.

    Args:
        batch_id: Batch ID to delete

    Returns:
        Deletion confirmation
    """
    async with batch_lock:
        if batch_id not in active_batches:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch '{batch_id}' not found",
            )

        batch = active_batches[batch_id]

        # Don't delete running batches
        if batch.status == JobStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete running batch. Cancel it first.",
            )

        del active_batches[batch_id]

    logger.info(f"Deleted batch {batch_id}")

    return {
        "batch_id": batch_id,
        "status": "deleted",
        "message": "Batch deleted successfully"
    }


@router.get("/list")
async def list_batches():
    """
    List all active batches.

    Returns:
        List of batch summaries
    """
    async with batch_lock:
        batches = []

        for batch_id, batch in active_batches.items():
            completed = sum(1 for j in batch.jobs if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in batch.jobs if j.status == JobStatus.FAILED)

            batches.append({
                "batch_id": batch_id,
                "status": batch.status.value,
                "total_jobs": len(batch.jobs),
                "completed": completed,
                "failed": failed,
                "created_at": batch.created_at,
            })

    return {
        "total_batches": len(batches),
        "batches": batches
    }
