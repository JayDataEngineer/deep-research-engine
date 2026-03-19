"""LTX Video extension using factory pattern.

LTX-2 video generation with IC-LoRA support for:
- Text-to-video
- Image-to-video
- Video-to-video
- Canny/Depth/Pose control
"""

from pathlib import Path
from src.extensions.libraries.comfyui.factory import create_extension_from_config


def register_extension():
    """
    Register the LTX Video extension using the factory pattern.
    """
    return create_extension_from_config(Path(__file__).parent)
