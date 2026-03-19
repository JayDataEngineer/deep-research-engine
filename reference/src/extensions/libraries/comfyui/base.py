"""
Base extension class for ComfyUI extensions.

This module provides a simplified base class that handles common functionality
for ComfyUI extensions, reducing boilerplate code and ensuring consistent behavior.

ComfyUI extensions extend ServiceExtension from the core system and connect
to an external ComfyUI server for image generation and editing.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from src.extensions.core.base import ServiceExtension
from src.extensions.integrations.comfyui.shared import ComfyUIEngine

logger = logging.getLogger(__name__)


class BaseComfyUIExtension(ServiceExtension):
    """Base class for ComfyUI workflow extensions that reduces boilerplate.

    This class handles common functionality like:
    - ComfyUI server communication via shared engine
    - Workflow loading and execution
    - Image generation and editing
    - Status checking

    ComfyUI workflow extensions should extend this class and implement
    workflow-specific configuration in their config.yaml files.
    """

    def __init__(self, config: dict[str, Any], root_path: Path):
        """Initialize the base ComfyUI extension.

        Args:
            config: Extension configuration dictionary
            root_path: Path to the extension directory
        """
        super().__init__(config, root_path)

        # Service configuration
        service_config = config.get("service", {})
        self.comfy_address = service_config.get(
            "default_address", "http://127.0.0.1:18188"
        )

        # Workflow configuration
        workflow_config = config.get("workflow", {})
        self.workflow_name = workflow_config.get("name", "default")
        self.workflow_file = workflow_config.get("file")
        self.workflow_type = workflow_config.get("type", "text_to_image")

        # Paths
        from src.constants import PROJECT_ROOT
        self.project_root = PROJECT_ROOT
        self.workflows_dir = self.project_root / "data" / "inputs" / "workflows"

        # Determine outputs directory based on workflow type
        if self.workflow_type in ["text_to_video", "image_to_video", "video_to_video"]:
            self.outputs_dir = self.project_root / "data" / "outputs" / "generated_videos"
        else:
            self.outputs_dir = self.project_root / "data" / "outputs" / "generated_images"

        if self.workflow_file:
            self.workflow_path = self.workflows_dir / self.workflow_file
        else:
            self.workflow_path = None

        # Shared engine (initialized on first use)
        self._engine: Optional[ComfyUIEngine] = None

    # ==========================================================================
    # SERVICE EXTENSION METHODS
    # ==========================================================================

    async def initialize(self, context: Any = None) -> None:
        """Initialize the extension.

        Args:
            context: Extension context from ExtensionManager
        """
        self.context = context
        self._status = self._verify_installation()

    async def shutdown(self) -> None:
        """Shutdown the extension.

        ComfyUI server is managed separately, so nothing to shutdown here.
        """
        pass

    async def start_service(self) -> None:
        """Start the service (ComfyUI is managed separately).

        The ComfyUI server is an external service that should already be running.
        """
        self.port = None
        self.base_url = self.comfy_address

    async def stop_service(self) -> None:
        """Stop the service (ComfyUI is managed separately).

        The ComfyUI server is an external service, nothing to stop here.
        """
        pass

    async def send_request(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the extension.

        Args:
            endpoint: API endpoint (e.g., "generate", "edit", "status", "generate_video")
            data: Request data

        Returns:
            Response dictionary

        Raises:
            ValueError: If endpoint is not supported
        """
        if self.workflow_type == "text_to_image":
            if endpoint == "generate":
                return await self._generate(**data)
            elif endpoint == "status":
                return self._status()
        elif self.workflow_type == "image_edit":
            if endpoint == "edit":
                return await self._edit(**data)
            elif endpoint == "status":
                return self._status()
        elif self.workflow_type in ["text_to_video", "image_to_video", "video_to_video"]:
            if endpoint == "generate":
                return await self._generate_video(**data)
            elif endpoint == "status":
                return self._status()

        raise ValueError(f"Unknown endpoint: {endpoint}")

    # ==========================================================================
    # CORE FUNCTIONALITY
    # ==========================================================================

    @property
    def engine(self) -> ComfyUIEngine:
        """Get the shared ComfyUI engine (lazy initialization)."""
        if self._engine is None:
            self._engine = ComfyUIEngine(
                comfy_address=self.comfy_address,
                workflow_path=self.workflow_path,
                outputs_dir=self.outputs_dir,
            )
        return self._engine

    async def _generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        negative_prompt: str = "",
        seed: int = -1,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate an image from text prompt.

        Args:
            prompt: Image description
            width: Image width
            height: Image height
            steps: Sampling steps
            negative_prompt: Negative prompt
            seed: Random seed (-1 for random)
            output_path: Optional output file path

        Returns:
            Result dict with success status and output info
        """
        return self.engine.generate(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            negative_prompt=negative_prompt,
            seed=seed,
            output_path=output_path,
        )

    async def _edit(
        self,
        image: str,
        prompt: str,
        strength: float = 0.5,
        negative_prompt: str = "",
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Edit an image.

        Args:
            image: Path to source image or base64 data
            prompt: Edit instruction
            strength: Edit strength (0-1)
            negative_prompt: Negative prompt
            output_path: Optional output file path

        Returns:
            Result dict with success status and output info
        """
        return self.engine.edit(
            image=image,
            prompt=prompt,
            strength=strength,
            negative_prompt=negative_prompt,
            output_path=output_path,
        )

    async def _generate_video(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 121,
        fps: int = 24,
        steps: int = 50,
        cfg_scale: float = 7.0,
        negative_prompt: str = "blurry, distorted, bad quality, unnatural movements, stiff pose",
        seed: int = -1,
        input_image: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a video from text prompt or input image.

        Args:
            prompt: Video description
            width: Video width
            height: Video height
            num_frames: Number of frames to generate
            fps: Frames per second
            steps: Sampling steps
            cfg_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt
            seed: Random seed (-1 for random)
            input_image: Optional input image path for image-to-video
            output_path: Optional output file path

        Returns:
            Result dict with success status and output info
        """
        return self.engine.generate_video(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            input_image=input_image,
            output_path=output_path,
        )

    def _status(self) -> dict[str, Any]:
        """Get ComfyUI server status.

        Returns:
            Status dict
        """
        return self.engine.get_status()

    def _verify_installation(self) -> dict[str, Any]:
        """Verify installation status.

        Returns:
            Status dict with installation state
        """
        result = {
            "workflow_exists": self.workflow_path.exists() if self.workflow_path else False,
            "server_running": False,
        }

        try:
            status = self.engine.get_status()
            result["server_running"] = status.get("connected", False)
        except Exception:
            result["server_running"] = False

        return result

    # ==========================================================================
    # CAPABILITIES
    # ==========================================================================

    def get_capabilities(self) -> dict[str, Any]:
        """Return extension capabilities.

        Returns:
            Capabilities dict
        """
        workflow_config = self.config.get("workflow", {})
        parameters = workflow_config.get("parameters", {})

        # Determine input/output types based on workflow type
        if self.workflow_type == "text_to_image":
            input_type = ["text"]
            output_type = ["image"]
        elif self.workflow_type == "image_edit":
            input_type = ["image", "text"]
            output_type = ["image"]
        elif self.workflow_type == "text_to_video":
            input_type = ["text"]
            output_type = ["video"]
        elif self.workflow_type in ["image_to_video", "video_to_video"]:
            input_type = ["image", "text"]
            output_type = ["video"]
        else:
            input_type = ["text"]
            output_type = ["image"]

        return {
            "name": self.workflow_name,
            "type": self.workflow_type,
            "display_name": self.config.get("extension", {}).get("display_name", self.workflow_name),
            "description": self.config.get("extension", {}).get("description", ""),
            "input_type": input_type,
            "output_type": output_type,
            "parameters": parameters,
            "server_address": self.comfy_address,
            "workflow_file": self.workflow_file,
        }
