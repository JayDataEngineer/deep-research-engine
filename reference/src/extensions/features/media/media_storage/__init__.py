"""Media Storage Extension.

Handles media file uploads (images, audio, video) with validation and storage.
"""

from .extension import (
    MediaStorageExtension,
    register_extension,
)

__all__ = [
    'MediaStorageExtension',
    'register_extension',
]
