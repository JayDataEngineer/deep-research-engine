"""Image Storage Extension.

Handles image extraction and storage from user messages for image editing.
"""

from .extension import (
    ImageStorageExtension,
    get_image_storage,
    register_extension,
)

__all__ = [
    'ImageStorageExtension',
    'get_image_storage',
    'register_extension',
]
