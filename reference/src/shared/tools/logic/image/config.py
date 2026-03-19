"""Configuration for image generation service."""

import sys
from pathlib import Path
from pydantic import BaseModel

# Import from src.constants (the main constants file)
src_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to src/
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from constants import DEFAULT_COMFYUI_URL, IMAGES_DIR, ENV_COMFYUI_ADDRESS
except ImportError:
    # Fallback defaults - go up to project root (backend/ -> project root)
    DEFAULT_COMFYUI_URL = "http://127.0.0.1:18188"
    IMAGES_DIR = Path(__file__).parent.parent.parent.parent.parent.parent / "data" / "outputs" / "generated_images"
    ENV_COMFYUI_ADDRESS = "COMFYUI_ADDRESS"


class ImageGenConfig(BaseModel):
    """Configuration for image generation service."""
    comfy_url: str = DEFAULT_COMFYUI_URL
    output_dir: str = ""  # Will be set to IMAGES_DIR in __init__

    def __init__(self, **data):
        if not data.get("output_dir"):
            data["output_dir"] = str(IMAGES_DIR)
        super().__init__(**data)

    class Config:
        env_prefix = "IMG_GEN"
