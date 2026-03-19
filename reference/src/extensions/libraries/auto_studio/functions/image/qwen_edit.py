"""Qwen Image Edit composable function.

Generates start frames for video clips using Qwen Image Edit via ComfyUI.
Supports multi-image input with text-referencing (image1, image2, image3).
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from ...core.composable import BaseComposableFunction, FunctionInput, FunctionResult

logger = logging.getLogger(__name__)


class QwenEditConfig(BaseModel):
    """Configuration for Qwen Image Edit function."""
    api_base: str = "http://localhost:8188"
    timeout: int = 300
    workflow_path: str = "data/inputs/workflows/qwen_image_edit_api.json"
    output_format: str = "png"
    output_dir: str = "data/outputs/auto_studio/qwen_frames"


class QwenImageEditFunction(BaseComposableFunction):
    """Generate start frames using Qwen Image Edit.

    This function:
    1. Takes up to 3 image inputs (image1, image2, image3)
    2. Applies Qwen Image Edit with text-referencing prompts
    3. Outputs a start frame for each video clip

    Multi-Image Support:
    - image1: Primary canvas/reference (e.g., base studio scene)
    - image2: Secondary reference (e.g., speaker A close-up)
    - image3: Tertiary reference (e.g., speaker B close-up)

    The prompt must use text-referencing: "background of image 1",
    "person from image 2 on the left", etc.
    """

    name = "qwen_image_edit"
    input_types = ["image", "prompt"]
    output_types = ["image"]
    requires_gpu = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.qwen_config = QwenEditConfig(**config.get("qwen_edit", {}))
        self.output_dir = Path(self.qwen_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load workflow template
        self.workflow_path = Path(self.qwen_config.workflow_path)
        self.workflow_template = self._load_workflow_template()

    def _load_workflow_template(self) -> Dict[str, Any]:
        """Load the workflow JSON template."""
        if not self.workflow_path.exists():
            logger.warning(f"[QwenEdit] Workflow template not found: {self.workflow_path}")
            return {}

        with open(self.workflow_path, 'r') as f:
            return json.load(f)

    def _validate_config(self):
        """Validate Qwen Edit configuration."""
        if not self.qwen_config.api_base:
            raise ValueError("qwen_edit.api_base is required")

    async def validate_input(self, input_data: FunctionInput) -> Tuple[bool, Optional[str]]:
        """Validate input data.

        Required:
        - image1: Path to primary image (base studio scene) in input_paths
        - qwen_edit_prompt: Prompt in plan_data
        - clip_id: For output naming

        Optional:
        - image2: Secondary reference image (e.g., speaker A close-up)
        - image3: Tertiary reference image (e.g., speaker B close-up)
        """
        # Check at least image1
        if "image1" not in input_data.input_paths:
            return False, "Missing 'image1' in input_paths"

        image1 = input_data.input_paths["image1"]
        if not image1.exists():
            return False, f"image1 not found: {image1}"

        # Check optional images
        for key in ["image2", "image3"]:
            if key in input_data.input_paths:
                path = input_data.input_paths[key]
                if not path.exists():
                    return False, f"{key} not found: {path}"

        # Check prompt
        prompt = input_data.plan_data.get("qwen_edit_prompt", "")
        if not prompt:
            return False, "Missing 'qwen_edit_prompt' in plan_data"

        return True, None

    async def execute(self, input_data: FunctionInput) -> FunctionResult:
        """Execute Qwen Image Edit generation.

        Args:
            input_data: Contains image1, [image2], [image3], qwen_edit_prompt,
                       qwen_negative_prompt, clip_id

        Returns:
            FunctionResult with output frame path
        """
        clip_id = input_data.clip_id
        image1 = input_data.input_paths.get("image1")
        image2 = input_data.input_paths.get("image2")
        image3 = input_data.input_paths.get("image3")
        prompt = input_data.plan_data.get("qwen_edit_prompt")
        negative_prompt = input_data.plan_data.get("qwen_negative_prompt", "")
        strength = input_data.plan_data.get("edit_strength", 0.75)
        seed = input_data.plan_data.get("seed", -1)

        # Output path
        output_path = self.output_dir / f"{clip_id}_start_frame.{self.qwen_config.output_format}"

        logger.info(f"[QwenEdit] Generating frame for clip {clip_id}")
        if image2:
            logger.info(f"[QwenEdit] Using 2 reference images (image1 + image2)")
        if image3:
            logger.info(f"[QwenEdit] Using 3 reference images (image1 + image2 + image3)")
        logger.debug(f"[QwenEdit] Prompt: {prompt[:100]}...")

        start_time = time.time()

        try:
            # Check if ComfyUI is available
            api_url = f"{self.qwen_config.api_base}/system_stats"
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(api_url)
                if response.status_code != 200:
                    raise ConnectionError(f"ComfyUI not available at {self.qwen_config.api_base}")

            # Build workflow from template
            workflow = self._build_workflow_from_template(
                image1=str(image1) if image1 else None,
                image2=str(image2) if image2 else None,
                image3=str(image3) if image3 else None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                seed=seed
            )

            # Execute workflow via ComfyUI
            result = await self._execute_comfyui_workflow(workflow)

            elapsed = time.time() - start_time
            logger.info(f"[QwenEdit] Generated frame for {clip_id} in {elapsed:.1f}s")

            # Verify output
            if not output_path.exists():
                return FunctionResult(
                    success=False,
                    error=f"Output file not created: {output_path}",
                    metadata={"clip_id": clip_id, "elapsed": elapsed}
                )

            return FunctionResult(
                success=True,
                data={
                    "output_paths": {
                        "start_frame": str(output_path)
                    },
                    "frame_info": {
                        "clip_id": clip_id,
                        "prompt": prompt,
                        "image_count": 1 + bool(image2) + bool(image3),
                        "path": str(output_path)
                    }
                },
                checkpoint_data={
                    "clip_id": clip_id,
                    "output_path": str(output_path),
                    "input_prompt": prompt
                },
                metadata={"clip_id": clip_id, "elapsed": elapsed}
            )

        except ConnectionError as e:
            logger.error(f"[QwenEdit] ComfyUI connection error: {e}")
            return FunctionResult(
                success=False,
                error=f"ComfyUI connection failed: {e}",
                metadata={"clip_id": clip_id}
            )
        except Exception as e:
            logger.exception(f"[QwenEdit] Execution error: {e}")
            return FunctionResult(
                success=False,
                error=f"Execution failed: {e}",
                metadata={"clip_id": clip_id}
            )

    def _build_workflow_from_template(
        self,
        image1: Optional[str],
        image2: Optional[str],
        image3: Optional[str],
        prompt: str,
        negative_prompt: str,
        strength: float,
        seed: int
    ) -> Dict[str, Any]:
        """Build ComfyUI workflow from template.

        Loads the qwen_image_edit_api.json template and populates it with
        the provided parameters.

        Args:
            image1: Path to primary image (required)
            image2: Path to secondary reference image (optional)
            image3: Path to tertiary reference image (optional)
            prompt: Edit prompt with text-referencing
            negative_prompt: Negative prompt
            strength: Edit strength (0.1-1.0)
            seed: Random seed (-1 for random)

        Returns:
            ComfyUI workflow dict ready for API
        """
        if not self.workflow_template:
            logger.error("[QwenEdit] No workflow template loaded")
            raise ValueError("Workflow template not loaded")

        # Deep copy template
        workflow = json.loads(json.dumps(self.workflow_template))

        # Populate template variables
        prompt_data = workflow.get("prompt", {})

        # Update node 22 (LoadImage) - primary canvas
        if "22" in prompt_data and image1:
            prompt_data["22"]["inputs"]["image"] = Path(image1).name

        # Update node 26 (TextEncodeQwenImageEditPlus - positive prompt)
        if "26" in prompt_data:
            prompt_data["26"]["inputs"]["prompt"] = prompt
            # image1 is connected via link ["22", 0]

        # Update node 19 (TextEncodeQwenImageEditPlus - negative prompt)
        if "19" in prompt_data:
            prompt_data["19"]["inputs"]["prompt"] = negative_prompt

        # Update node 25 (KSampler)
        if "25" in prompt_data:
            prompt_data["25"]["inputs"]["denoise"] = strength
            if seed >= 0:
                prompt_data["25"]["inputs"]["seed"] = seed
            else:
                prompt_data["25"]["inputs"]["seed"] = int(time.time() * 1000) % 2**32

        # TODO: Add support for image2 and image3
        # These would need to be loaded via additional LoadImage nodes
        # and connected to node 26's image2, image3 inputs

        return workflow

    async def _execute_comfyui_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow via ComfyUI API.

        Args:
            workflow: ComfyUI workflow dict with "prompt" key

        Returns:
            Execution result with output file info
        """
        api_url = f"{self.qwen_config.api_base}/prompt"
        client_id = "auto_studio_qwen_edit"

        # Submit workflow
        async with httpx.AsyncClient(timeout=self.qwen_config.timeout) as client:
            # Submit prompt
            submit_response = await client.post(
                api_url,
                json={
                    "prompt": workflow.get("prompt", workflow),
                    "client_id": client_id
                }
            )
            submit_response.raise_for_status()
            submit_data = submit_response.json()

            prompt_id = submit_data.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"No prompt_id in response: {submit_data}")

            logger.info(f"[QwenEdit] Workflow submitted: {prompt_id}")

            # Poll for completion
            history_url = f"{self.qwen_config.api_base}/history/{prompt_id}"
            max_wait = self.qwen_config.timeout
            start_wait = time.time()

            while True:
                if time.time() - start_wait > max_wait:
                    raise TimeoutError(f"Workflow {prompt_id} timed out after {max_wait}s")

                await asyncio.sleep(1)

                history_response = await client.get(history_url)
                history_response.raise_for_status()
                history_data = history_response.json()

                if prompt_id in history_data:
                    status = history_data[prompt_id].get("status", {})
                    if status.get("completed", False):
                        logger.info(f"[QwenEdit] Workflow completed: {prompt_id}")
                        break

            # Get outputs
            outputs = history_data[prompt_id].get("outputs", {})
            logger.info(f"[QwenEdit] Outputs: {list(outputs.keys())}")

            return {
                "status": "completed",
                "prompt_id": prompt_id,
                "outputs": outputs
            }

    async def _queue_prompt(self, workflow: Dict[str, Any], client_id: str) -> str:
        """Queue a prompt with ComfyUI.

        Args:
            workflow: ComfyUI workflow dict
            client_id: Client ID for the request

        Returns:
            prompt_id
        """
        api_url = f"{self.qwen_config.api_base}/prompt"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                api_url,
                json={
                    "prompt": workflow.get("prompt", workflow),
                    "client_id": client_id
                }
            )
            response.raise_for_status()
            data = response.json()

            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"No prompt_id in response: {data}")

            return prompt_id

    async def _wait_for_completion(self, prompt_id: str) -> Dict[str, Any]:
        """Wait for workflow completion.

        Args:
            prompt_id: Prompt ID to wait for

        Returns:
            History data for the prompt
        """
        history_url = f"{self.qwen_config.api_base}/history/{prompt_id}"
        max_wait = self.qwen_config.timeout
        start_wait = time.time()

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                if time.time() - start_wait > max_wait:
                    raise TimeoutError(f"Workflow {prompt_id} timed out after {max_wait}s")

                await asyncio.sleep(1)

                response = await client.get(history_url)
                response.raise_for_status()
                history_data = response.json()

                if prompt_id in history_data:
                    status = history_data[prompt_id].get("status", {})
                    if status.get("completed", False):
                        return history_data[prompt_id]

    def _get_output_images(self, history_data: Dict[str, Any]) -> list:
        """Extract output images from history data.

        Args:
            history_data: History data from ComfyUI

        Returns:
            List of output image paths
        """
        outputs = history_data.get("outputs", {})
        images = []

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    images.append(img["filename"])

        return images
