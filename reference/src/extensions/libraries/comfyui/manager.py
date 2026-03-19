"""ComfyUI Manager

Manages ComfyUI workflows, image generation, and models.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests

from src.constants import DEFAULT_COMFYUI_URL, BACKEND_ROOT, PROJECT_ROOT

logger = logging.getLogger(__name__)


class ComfyUIManager:
    """Manages ComfyUI workflows and image generation."""

    def __init__(self, comfy_url: Optional[str] = None):
        """Initialize the ComfyUI manager.

        Args:
            comfy_url: ComfyUI server URL (default from constants)
        """
        self.comfy_url = comfy_url or os.getenv("COMFYUI_ADDRESS", DEFAULT_COMFYUI_URL)
        self._session = requests.Session()

        # Paths
        self.project_root = PROJECT_ROOT
        self.backend_root = BACKEND_ROOT
        self.workflows_dir = self.project_root / "data" / "inputs" / "workflows"
        self.outputs_dir = self.project_root / "data" / "outputs" / "generated_images"
        self.comfy_dir = self.backend_root / "vendor" / "ComfyUI"

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows.

        Returns:
            List of workflow info dicts
        """
        workflows = []

        # Check unified workflows directory
        if self.workflows_dir.exists():
            for workflow_file in self.workflows_dir.glob("*.json"):
                try:
                    with open(workflow_file) as f:
                        data = json.load(f)

                    # Extract metadata
                    name = workflow_file.stem
                    nodes_count = len(data.get("nodes", [])) if "nodes" in data else len(data)

                    workflows.append({
                        "name": name,
                        "path": str(workflow_file),
                        "nodes": nodes_count,
                        "type": self._detect_workflow_type(name, data),
                    })
                except Exception as e:
                    logger.warning(f"Failed to read workflow {workflow_file}: {e}")

        return sorted(workflows, key=lambda x: x["name"])

    def _detect_workflow_type(self, name: str, data: dict) -> str:
        """Detect workflow type from content."""
        nodes = data.get("nodes", [])
        if not nodes:
            return "unknown"

        node_types = {node.get("type", "") for node in nodes}

        if "CLIPLoader" in node_types and "UNETLoader" in node_types:
            if "Qwen" in name or "qwen" in name.lower():
                return "image_edit"
            return "image_gen"

        return "unknown"

    def get_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a workflow by name.

        Args:
            name: Workflow filename (without .json)

        Returns:
            Workflow dict or None
        """
        workflow_path = self.workflows_dir / f"{name}.json"
        if not workflow_path.exists():
            return None

        with open(workflow_path) as f:
            return json.load(f)

    def generate(
        self,
        prompt: str,
        workflow: str = "z_image_turbo",
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        negative_prompt: str = "",
        seed: int = -1,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an image.

        Args:
            prompt: Image description
            workflow: Workflow name to use
            width: Image width
            height: Image height
            steps: Sampling steps
            negative_prompt: Negative prompt
            seed: Random seed (-1 for random)
            output_path: Optional output file path

        Returns:
            Result dict with status and output info
        """
        try:
            from src.shared.tools.logic.image.generation import (
                get_image_service_http,
                ImageGenConfig
            )

            config = ImageGenConfig(
                comfy_url=self.comfy_url,
                output_dir=str(self.outputs_dir),
            )

            # Import service
            from src.shared.tools.logic.image.generation import ImageGenerationServiceHTTP
            service = ImageGenerationServiceHTTP(config)

            # Use sync wrapper
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                image_bytes = loop.run_until_complete(
                    service.generate_character_image(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        sample_steps=steps,
                        seed=seed,
                    )
                )
            finally:
                loop.close()

            # Save image
            self.outputs_dir.mkdir(parents=True, exist_ok=True)

            if output_path:
                image_path = Path(output_path)
            else:
                import time
                timestamp = int(time.time())
                image_path = self.outputs_dir / f"gen_{timestamp}.png"

            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)

            return {
                "success": True,
                "output_path": str(image_path),
                "size": len(image_bytes),
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def edit_image(
        self,
        image_path: str,
        prompt: str,
        strength: float = 1.0,
        workflow: str = "qwen_gguf_api",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Edit an image.

        Args:
            image_path: Path to source image
            prompt: Edit instruction
            strength: Edit strength (0-1)
            workflow: Workflow name to use
            output_path: Optional output file path

        Returns:
            Result dict with status and output info
        """
        try:
            from src.shared.tools.logic.image.generation import (
                get_image_service_http,
                ImageGenConfig
            )

            config = ImageGenConfig(
                comfy_url=self.comfy_url,
                output_dir=str(self.outputs_dir),
            )

            from src.shared.tools.logic.image.generation import ImageGenerationServiceHTTP
            service = ImageGenerationServiceHTTP(config)

            # Read source image
            image_bytes = Path(image_path).read_bytes()

            # Run async edit
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                edited_bytes = loop.run_until_complete(
                    service.edit_image_qwen(
                        image_data=image_bytes,
                        prompt=prompt,
                        strength=strength,
                    )
                )
            finally:
                loop.close()

            # Save edited image
            if output_path:
                out_path = Path(output_path)
            else:
                import time
                timestamp = int(time.time())
                out_path = self.outputs_dir / f"edit_{timestamp}.png"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(edited_bytes)

            return {
                "success": True,
                "output_path": str(out_path),
                "size": len(edited_bytes),
            }

        except Exception as e:
            logger.error(f"Edit failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def list_models(self) -> Dict[str, List[str]]:
        """List available models in ComfyUI.

        Returns:
            Dict mapping model types to lists of models
        """
        models = {
            "checkpoints": [],
            "loras": [],
            "vae": [],
            "upscale": [],
        }

        comfy_models = self.comfy_dir / "models"

        # Checkpoints
        checkpoints_dir = comfy_models / "checkpoints"
        if checkpoints_dir.exists():
            models["checkpoints"] = [
                f.name for f in checkpoints_dir.glob("*.safetensors")
            ]

        # LoRAs
        loras_dir = comfy_models / "loras"
        if loras_dir.exists():
            models["loras"] = [
                f.name for f in loras_dir.glob("*.safetensors")
            ]

        # VAE
        vae_dir = comfy_models / "vae"
        if vae_dir.exists():
            models["vae"] = [
                f.name for f in vae_dir.glob("*.safetensors")
            ]

        return models

    def get_status(self) -> Dict[str, Any]:
        """Get ComfyUI server status.

        Returns:
            Status dict
        """
        try:
            response = self._session.get(f"{self.comfy_url}/system_stats", timeout=5)
            response.raise_for_status()
            return {
                "connected": True,
                "url": self.comfy_url,
                "system": response.json(),
            }
        except Exception as e:
            return {
                "connected": False,
                "url": self.comfy_url,
                "error": str(e),
            }

    def list_outputs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent generated images.

        Args:
            limit: Max number of outputs

        Returns:
            List of output info dicts
        """
        outputs = []

        if not self.outputs_dir.exists():
            return outputs

        for image_path in sorted(self.outputs_dir.glob("*.png"), reverse=True)[:limit]:
            stat = image_path.stat()
            outputs.append({
                "path": str(image_path),
                "name": image_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })

        return outputs
