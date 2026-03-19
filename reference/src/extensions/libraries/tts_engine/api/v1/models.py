"""
Model management API endpoints for handling model downloads and validation.
"""

import logging
import os
import threading

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.extensions.libraries.tts_engine.api.utils.download_models import (
    download_gpt_sovits,
    download_indextts,
    download_kokoro,
    download_maya1,
    download_vibevoice_1_5b,
    download_vibevoice_7b,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global variable to store download progress (in a real app, consider using Redis or similar for scalability)
download_progress: dict[str, dict[str, float]] = {}


class ModelDownloadRequest(BaseModel):
    model_name: str  # "kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits"
    force_download: bool = False  # Allow re-downloading existing models


class ModelDownloadResponse(BaseModel):
    success: bool
    message: str
    model_name: str


class ModelStatusResponse(BaseModel):
    model_name: str
    exists: bool
    download_in_progress: bool
    progress: float | None = None


def check_model_exists(model_name: str) -> tuple[bool, str | None]:
    """Check if a model exists in the filesystem."""
    from src.extensions.libraries.tts_engine.api.utils.model_validator import model_validator

    model_path = model_validator._get_model_path(model_name)
    if model_path is None:
        return False, None

    exists = (
        os.path.exists(model_path)
        and os.path.isdir(model_path)
        and len(os.listdir(model_path)) > 0
    )
    return exists, model_path if exists else None


def download_model_with_progress(model_name: str):
    """Download model with progress tracking."""
    global download_progress

    if (
        model_name in download_progress
        and download_progress[model_name].get("progress", 0) < 100
    ):
        # Already downloading
        return False, f"Download already in progress for {model_name}"

    try:
        # Initialize progress tracking
        download_progress[model_name] = {"progress": 0.0, "status": "starting"}

        # Create a progress callback function
        def progress_callback(progress_percent: float):
            download_progress[model_name] = {
                "progress": progress_percent,
                "status": "downloading" if progress_percent < 100 else "completed",
            }

        # Call the appropriate download function with progress callback
        if model_name == "kokoro":
            download_kokoro(progress_callback)
        elif model_name == "vibevoice":
            download_vibevoice_1_5b(progress_callback)
        elif model_name == "vibevoice-7b":
            download_vibevoice_7b(progress_callback)
        elif model_name == "indextts":
            download_indextts(progress_callback)
        elif model_name == "gptsovits":
            download_gpt_sovits(progress_callback)
        elif model_name == "maya1":
            download_maya1(progress_callback)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Update progress to 100% to indicate completion
        download_progress[model_name] = {"progress": 100.0, "status": "completed"}

        return True, f"Successfully downloaded {model_name}"

    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        download_progress[model_name] = {
            "progress": 0.0,
            "status": "failed",
            "error": str(e),
        }
        return False, f"Failed to download {model_name}: {str(e)}"


@router.post("/models/download", response_model=ModelDownloadResponse)
async def download_model(request: ModelDownloadRequest):
    """Download a specific model in the background."""
    try:
        model_name = request.model_name.lower()

        # Validate model name
        valid_models = ["kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits", "maya1"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name: {model_name}. Valid models are: {valid_models}",
            )

        # Check if model already exists
        if check_model_exists(model_name):
            # Allow re-download if explicitly requested
            if request.force_download:
                logger.info(f"Force re-downloading model {model_name}")
            else:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Model {model_name} already exists",
                )

        # Start download in a separate thread to avoid blocking
        def run_download():
            download_model_with_progress(model_name)

        thread = threading.Thread(target=run_download)
        thread.start()

        return ModelDownloadResponse(
            success=True,
            message=f"Starting download for {model_name}",
            model_name=model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating model download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initiating model download: {str(e)}",
        ) from e


@router.get("/models/status/{model_name}", response_model=ModelStatusResponse)
async def get_model_status(model_name: str):
    """Get the status of a specific model."""
    try:
        model_name = model_name.lower()

        # Validate model name
        valid_models = ["kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits", "maya1"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name: {model_name}. Valid models are: {valid_models}",
            )

        result = check_model_exists(model_name)
        # Handle both single boolean and tuple returns
        if isinstance(result, tuple):
            exists, _ = result
        else:
            exists = result

        # Check download progress
        download_in_progress = False
        progress = None

        if model_name in download_progress:
            status_info = download_progress[model_name]
            if status_info.get("progress", 0) < 100:
                download_in_progress = True
                progress = status_info.get("progress", 0)

        # If download completed, clean up progress tracking
        if (
            model_name in download_progress
            and download_progress[model_name].get("progress") == 100
        ):
            del download_progress[model_name]

        return ModelStatusResponse(
            model_name=model_name,
            exists=exists,
            download_in_progress=download_in_progress,
            progress=progress,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model status: {str(e)}",
        ) from e


@router.get("/models/checksum/{model_name}")
async def validate_model_checksum(model_name: str):
    """Validate the checksum of a specific model."""
    try:
        from src.extensions.libraries.tts_engine.api.utils.model_validator import model_validator

        model_name = model_name.lower()

        # Validate model name
        valid_models = ["kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits", "maya1"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name: {model_name}. Valid models are: {valid_models}",
            )

        result = model_validator.validate_model(model_name)

        if not result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} validation failed: {result.get('error', 'Model not found')}",
            )

        return {
            "model_name": model_name,
            "checksum": result["checksum"],
            "valid": result["valid"],
            "message": f"Model {model_name} is valid",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating model checksum: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating model checksum: {str(e)}",
        ) from e


@router.post("/models/validate/{model_name}")
async def validate_model_endpoint(model_name: str, use_hf_checksums: bool = False):
    """Validate a model and return detailed information including checksum.

    Args:
        model_name: Name of the model to validate
        use_hf_checksums: If True, use Hugging Face checksums for validation
    """
    try:
        from src.extensions.libraries.tts_engine.model_validation_service import model_validation_service
        from src.extensions.libraries.tts_engine.api.utils.model_validator import model_validator

        model_name = model_name.lower()

        # Validate model name
        valid_models = ["kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits", "maya1"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name: {model_name}. Valid models are: {valid_models}",
            )

        # Check if model exists first
        if not check_model_exists(model_name):
            return {
                "model_name": model_name,
                "valid": False,
                "exists": False,
                "error": f"Model {model_name} does not exist",
                "checksum": None,
            }

        # Perform validation
        if use_hf_checksums:
            result = model_validation_service.validate_model_with_hf_checksums(model_name)
        else:
            result = model_validator.validate_model(model_name)

        # Add existence flag
        result["exists"] = True

        # If validation failed and not using HF checksums, trigger background validation
        if not result.get("valid", False) and not use_hf_checksums:
            model_validation_service.validate_model_now(model_name)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating model: {str(e)}",
        ) from e


@router.post("/models/redownload/{model_name}")
async def redownload_model(model_name: str):
    """Re-download a model, typically used when corruption is detected."""
    try:
        model_name = model_name.lower()

        # Validate model name
        valid_models = ["kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits", "maya1"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name: {model_name}. Valid models are: {valid_models}",
            )

        # Start download in a separate thread to avoid blocking
        def run_download():
            download_model_with_progress(model_name)

        thread = threading.Thread(target=run_download)
        thread.start()

        return ModelDownloadResponse(
            success=True,
            message=f"Starting re-download for {model_name}",
            model_name=model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating model re-download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initiating model re-download: {str(e)}",
        ) from e


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model directory."""
    try:
        from src.extensions.libraries.tts_engine.api.utils.download_models import delete_model_directory

        model_name = model_name.lower()

        # Validate model name
        valid_models = ["kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits", "maya1"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name: {model_name}. Valid models are: {valid_models}",
            )

        # Check if model exists
        if not check_model_exists(model_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} does not exist",
            )

        # Delete the model
        if delete_model_directory(model_name):
            return {
                "success": True,
                "message": f"Successfully deleted model {model_name}",
                "model_name": model_name,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete model {model_name}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}",
        ) from e


@router.post("/models/validate-and-redownload/{model_name}")
async def validate_and_redownload_model(model_name: str):
    """Validate a model using Hugging Face checksums and re-download if needed."""
    try:
        from src.extensions.libraries.tts_engine.api.utils.download_models import validate_and_redownload_if_needed

        model_name = model_name.lower()

        # Validate model name
        valid_models = ["kokoro", "vibevoice", "vibevoice-7b", "indextts", "gptsovits", "maya1"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name: {model_name}. Valid models are: {valid_models}",
            )

        # Start validation and potential re-download in a separate thread
        def validate_and_redownload():
            try:
                success = validate_and_redownload_if_needed(model_name)
                if success:
                    logger.info(f"Model {model_name} validation and re-download completed successfully")
                else:
                    logger.error(f"Model {model_name} validation and re-download failed")
            except Exception as e:
                logger.error(f"Error during model validation and re-download: {str(e)}")

        thread = threading.Thread(target=validate_and_redownload)
        thread.start()

        return {
            "success": True,
            "message": f"Started validation and re-download process for {model_name}",
            "model_name": model_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating model validation and re-download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initiating model validation and re-download: {str(e)}",
        ) from e
