"""Qwen Image Edit extension using factory pattern.

DRY Implementation - eliminates boilerplate code.
"""

from pathlib import Path
from src.extensions.libraries.comfyui.factory import create_extension_from_config


def register_extension():
    """
    Register the Qwen Image Edit extension using the factory pattern.
    """
    return create_extension_from_config(Path(__file__).parent)
