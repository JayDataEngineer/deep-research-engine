"""Image Generation Extension.

Provides image generation and character editing endpoints via ComfyUI service extensions.
Uses the new ComfyUI extension-based architecture instead of direct HTTP calls.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class ImagesExtension(LogicExtension):
    """Image generation extension via ComfyUI service extensions."""

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._router: APIRouter = None
        # Cache for ComfyUI extensions
        self._z_image_turbo_ext = None
        self._qwen_image_edit_ext = None
        # Character reference storage (session_id -> image_data)
        self._character_references: dict[str, bytes] = {}

    async def initialize(self, context):
        """Initialize the images extension."""
        await super().initialize(context)
        logger.info("Images extension initialized")

    async def shutdown(self):
        """Shutdown the extension."""
        pass

    async def process(self, input_data: dict, **kwargs):
        """Process image generation request."""
        action = input_data.get('action', 'generate')

        if action == 'generate':
            return await self.generate_image(input_data)
        elif action == 'edit':
            return await self.edit_image(input_data)
        elif action == 'set_reference':
            return await self.set_reference(input_data)
        elif action == 'clear_reference':
            return await self.clear_reference(input_data)
        else:
            return {"error": f"Unknown action: {action}"}

    @property
    def router(self) -> APIRouter:
        """Get the API router for this extension."""
        if self._router is None:
            from .routes import router
            self._router = router
        return self._router

    def _get_z_image_turbo_extension(self) -> Optional[Any]:
        """Get the z_image_turbo ComfyUI extension."""
        if self._z_image_turbo_ext is None and self.context:
            self._z_image_turbo_ext = self.context.get_extension("z_image_turbo")
        return self._z_image_turbo_ext

    def _get_qwen_image_edit_extension(self) -> Optional[Any]:
        """Get the qwen_image_edit ComfyUI extension."""
        if self._qwen_image_edit_ext is None and self.context:
            self._qwen_image_edit_ext = self.context.get_extension("qwen_image_edit")
        return self._qwen_image_edit_ext

    def _image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 data URL."""
        image_bytes = Path(image_path).read_bytes()
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    async def generate_image(self, input_data: dict) -> dict:
        """Generate a new character image using z_image_turbo extension."""
        try:
            ext = self._get_z_image_turbo_extension()
            if not ext:
                return {"error": "z_image_turbo extension not available"}

            # Generate image via extension
            result = await ext.send_request("generate", {
                "prompt": input_data.get("prompt", ""),
                "width": input_data.get("width", 1024),
                "height": input_data.get("height", 1024),
                "steps": input_data.get("steps", 9),
                "negative_prompt": input_data.get("negative_prompt", ""),
                "seed": input_data.get("seed", -1),
            })

            if not result.get("success"):
                return {"error": result.get("error", "Generation failed")}

            # Convert output path to base64 for frontend
            output_path = result.get("output_path")
            image_base64 = self._image_to_base64(output_path)

            return {
                "image": image_base64,
                "prompt": input_data.get("prompt", ""),
                "message": "Character image generated successfully",
            }
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return {"error": str(e)}

    async def edit_image(self, input_data: dict) -> dict:
        """Edit a character image using qwen_image_edit extension."""
        try:
            ext = self._get_qwen_image_edit_extension()
            if not ext:
                return {"error": "qwen_image_edit extension not available"}

            session_id = input_data.get("session_id")

            # Get reference image
            ref_image_data = self._character_references.get(session_id)
            if not ref_image_data:
                return {"error": f"No character reference set for session: {session_id}"}

            # Convert reference image to base64 for transmission
            ref_image_base64 = base64.b64encode(ref_image_data).decode('utf-8')

            # Edit image via extension
            result = await ext.send_request("edit", {
                "image": ref_image_base64,
                "prompt": input_data.get("prompt", ""),
                "strength": input_data.get("strength", 0.5),
                "negative_prompt": input_data.get("negative_prompt", ""),
            })

            if not result.get("success"):
                return {"error": result.get("error", "Edit failed")}

            # Convert output path to base64 for frontend
            output_path = result.get("output_path")
            image_base64 = self._image_to_base64(output_path)

            return {
                "image": image_base64,
                "prompt": input_data.get("prompt", ""),
                "message": "Character image edited successfully",
            }
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Image editing error: {e}")
            return {"error": str(e)}

    async def set_reference(self, input_data: dict) -> dict:
        """Set character reference image for editing."""
        try:
            # Decode base64 image
            image_data = input_data.get("image_data", "")
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data)

            # Store reference
            session_id = input_data.get("session_id")
            self._character_references[session_id] = image_bytes

            return {
                "success": True,
                "message": "Character reference image saved",
            }
        except Exception as e:
            logger.error(f"Set reference error: {e}")
            return {"error": str(e)}

    async def clear_reference(self, input_data: dict) -> dict:
        """Clear character reference image."""
        try:
            session_id = input_data.get("session_id")
            self._character_references.pop(session_id, None)

            return {
                "success": True,
                "message": "Character reference cleared",
            }
        except Exception as e:
            logger.error(f"Clear reference error: {e}")
            return {"error": str(e)}

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(ImagesExtension)
