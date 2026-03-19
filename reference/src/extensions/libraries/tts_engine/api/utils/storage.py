"""
Storage utilities for managing audio files and user data.
Based on ComfyUI's folder_paths.py approach.
"""

import os
from datetime import datetime

from src.config import settings


def ensure_directories_exist():
    """Ensure all required storage directories exist."""
    directories = [
        settings.storage_base_dir,
        settings.user_directory,
        settings.output_directory,
        settings.temp_directory,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_output_directory() -> str:
    """Get the output directory for generated audio files."""
    return settings.output_directory


def get_user_directory() -> str:
    """Get the user directory for user-specific data."""
    return settings.user_directory


def get_temp_directory() -> str:
    """Get the temp directory for temporary files."""
    return settings.temp_directory


def get_audio_path(audio_id: str, format: str = "wav") -> str:
    """Get the full path for an audio file."""
    ensure_directories_exist()  # Ensure directories exist

    # Using default user for now, in the future we could have actual user management
    get_user_directory()
    output_dir = get_output_directory()

    # Create a subfolder based on the date for organization
    date_subfolder = datetime.now().strftime("%Y-%m-%d")

    # Create the full path
    full_path = os.path.join(output_dir, date_subfolder)
    os.makedirs(full_path, exist_ok=True)

    # Generate the filename
    filename = f"audio_{audio_id}.{format}"

    return os.path.join(full_path, filename)


def save_audio_file(audio_bytes: bytes, audio_id: str, format: str = "wav") -> str:
    """Save audio bytes to a file and return the relative path."""
    full_path = get_audio_path(audio_id, format)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "wb") as f:
        f.write(audio_bytes)

    return full_path


def get_audio_url(audio_id: str, format: str = "wav") -> str:
    """Get the URL for an audio file."""
    # This creates a URL that will be served by the API endpoint
    # Using relative path so it works with proxying from frontend to backend
    # Note: The route is mounted at /api/v1/tts/audio/{audio_id_with_format}
    return f"/api/v1/tts/audio/{audio_id}.{format}"


def get_audio_by_id(audio_id_with_format: str) -> str | None:
    """Get the full path to an audio file by its ID which includes the format."""
    # audio_id_with_format is something like "1234567890.wav" or "1234567890.mp3"
    if "." not in audio_id_with_format:
        return None

    audio_id, format = audio_id_with_format.rsplit(".", 1)

    # Try to find the file by searching through dated subdirectories
    output_dir = get_output_directory()

    # First check root output directory (used by AudioService.save_audio_file())
    direct_path = os.path.join(output_dir, f"{audio_id}.{format}")
    if os.path.exists(direct_path):
        return direct_path

    # Check today's date folder first
    date_subfolder = datetime.now().strftime("%Y-%m-%d")
    dated_path = os.path.join(output_dir, date_subfolder, f"audio_{audio_id}.{format}")

    if os.path.exists(dated_path):
        return dated_path

    # If not in today's folder, search all dated subfolders
    # The audio file may have been generated on a different date
    for dated_folder in os.listdir(output_dir):
        dated_folder_path = os.path.join(output_dir, dated_folder)
        if os.path.isdir(dated_folder_path):
            potential_path = os.path.join(dated_folder_path, f"audio_{audio_id}.{format}")
            if os.path.exists(potential_path):
                return potential_path

    # File not found in any location
    return None


# Ensure directories exist when the module is imported
ensure_directories_exist()
