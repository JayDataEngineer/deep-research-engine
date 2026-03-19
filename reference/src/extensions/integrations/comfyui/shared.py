"""ComfyUI Shared Engine

Shared engine for all ComfyUI-based extensions.
Provides HTTP API communication and workflow execution.
"""

import asyncio
import base64
import json
import logging
import random
import re
import time
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from src.constants import IMAGES_DIR

logger = logging.getLogger(__name__)


class ComfyUIEngine:
    """Shared engine for ComfyUI workflow execution.

    This engine:
    - Communicates with ComfyUI HTTP API
    - Loads and executes workflows
    - Handles both image generation and editing
    """

    def __init__(
        self,
        comfy_address: str = "http://127.0.0.1:18188",
        workflow_path: Optional[Path] = None,
        outputs_dir: Optional[Path] = None,
    ):
        """Initialize the ComfyUI engine.

        Args:
            comfy_address: ComfyUI server URL
            workflow_path: Path to workflow JSON file
            outputs_dir: Directory for generated images
        """
        self.comfy_address = comfy_address
        self.workflow_path = workflow_path
        self.outputs_dir = outputs_dir or IMAGES_DIR
        self._session = requests.Session()

        # Ensure output directory exists
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def _load_workflow(self) -> dict:
        """Load the workflow from JSON file.

        Returns:
            Workflow dict

        Raises:
            FileNotFoundError: If workflow file not found
        """
        if not self.workflow_path or not self.workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {self.workflow_path}")

        with open(self.workflow_path) as f:
            return json.load(f)

    def _queue_prompt(self, workflow: dict) -> str:
        """Send workflow to ComfyUI and return prompt ID.

        Args:
            workflow: Workflow dict (API format)

        Returns:
            Prompt ID string

        Raises:
            RuntimeError: If queueing fails
        """
        data = json.dumps({"prompt": workflow}).encode("utf-8")

        for attempt in range(3):
            try:
                response = self._session.post(
                    f"{self.comfy_address}/prompt",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                return result["prompt_id"]
            except Exception as e:
                logger.error(f"Queue attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise RuntimeError(f"Failed to queue prompt: {e}")

    def _wait_for_completion(self, prompt_id: str, timeout: int = 180) -> dict:
        """Wait for prompt to complete and return history.

        Args:
            prompt_id: Prompt ID from queue response
            timeout: Timeout in seconds

        Returns:
            History dict

        Raises:
            RuntimeError: If timeout or error
        """
        start = time.time()

        while True:
            try:
                response = self._session.get(
                    f"{self.comfy_address}/history/{prompt_id}",
                    timeout=5,
                )
                response.raise_for_status()
                history = response.json()

                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("completed", False) or history[prompt_id].get("outputs"):
                        return history[prompt_id]

                elapsed = time.time() - start
                if elapsed > timeout:
                    raise RuntimeError(f"Generation timeout after {timeout}s")

                time.sleep(1)

            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start
                if elapsed > timeout:
                    raise RuntimeError(f"Generation timeout after {timeout}s")
                time.sleep(2)

    def _get_output_image(self, history: dict) -> bytes:
        """Get the generated image from ComfyUI output.

        Args:
            history: History dict from _wait_for_completion

        Returns:
            Image bytes

        Raises:
            RuntimeError: If no image found
        """
        outputs = history.get("outputs", {})

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img in node_output["images"]:
                    params = urllib.parse.urlencode({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
                    url = f"{self.comfy_address}/view?{params}"

                    response = self._session.get(url, timeout=30)
                    response.raise_for_status()
                    return response.content

        raise RuntimeError("No image found in outputs")

    def _convert_workflow_to_api(self, workflow: dict) -> dict:
        """Convert workflow JSON to API format.

        Args:
            workflow: Workflow dict (from file)

        Returns:
            Workflow dict in API format
        """
        # If already in API format (dict of node_id -> {class_type, inputs})
        if "nodes" not in workflow:
            return workflow

        # Convert from node array format to API dict format
        nodes_dict = {}
        auxiliary_node_types = {"MarkdownNote", "Note", "Comment", "PrimitiveNode", "Reroute", "CM_FloatToInt", "PrimitiveInt", "PrimitiveFloat", "EmptyImage", "CreateVideo", "SaveVideo", "LatentUpscaleModelLoader", "GetNode", "SetNode"}

        for node in workflow.get("nodes", []):
            node_id = node["id"]
            node_type = node["type"]

            # Skip auxiliary node types
            if node_type in auxiliary_node_types:
                continue

            # Skip subgraph nodes (UUID class types like "3eaa20c4-5842-4fe4-87df-c0a7e83a6a78")
            if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', node_type):
                continue

            node_id_str = str(node_id)
            inputs = self._extract_node_inputs(node, workflow)

            nodes_dict[node_id_str] = {
                "class_type": node["type"],
                "inputs": inputs,
            }

        return nodes_dict

    def _extract_node_inputs(self, node: dict, workflow: dict) -> dict:
        """Extract inputs from a node for the API format.

        Args:
            node: Node dict
            workflow: Full workflow dict

        Returns:
            Inputs dict
        """
        inputs = {}

        if "widgets_values" in node:
            widgets = node["widgets_values"]
            node_type = node["type"]

            # Handle different node types
            if node_type == "CLIPLoader":
                if len(widgets) > 0:
                    inputs["clip_name"] = widgets[0]
                if len(widgets) > 1:
                    inputs["type"] = widgets[1]
            elif node_type == "VAELoader":
                if len(widgets) > 0:
                    inputs["vae_name"] = widgets[0]
            elif node_type == "UNETLoader":
                if len(widgets) > 0:
                    inputs["unet_name"] = widgets[0]
                if len(widgets) > 1:
                    inputs["weight_dtype"] = widgets[1]
            elif node_type == "EmptySD3LatentImage":
                if len(widgets) > 0:
                    inputs["width"] = widgets[0]
                if len(widgets) > 1:
                    inputs["height"] = widgets[1]
            elif node_type == "CLIPTextEncode":
                if len(widgets) > 0:
                    inputs["text"] = widgets[0]
            elif node_type == "KSampler":
                seed_val = widgets[0]
                if len(widgets) > 1 and widgets[1] == "randomize":
                    seed_val = random.randint(0, 2**32 - 1)
                inputs["seed"] = int(seed_val)
                if len(widgets) > 2:
                    inputs["steps"] = widgets[2]
            elif node_type == "SaveImage":
                if len(widgets) > 0:
                    inputs["filename_prefix"] = widgets[0]
            elif node_type == "LoadImage":
                if len(widgets) > 0:
                    inputs["image"] = widgets[0]
                if len(widgets) > 1:
                    inputs["upload"] = widgets[1]
            elif node_type == "LTXVAudioVAELoader":
                if len(widgets) > 0:
                    inputs["ckpt_name"] = widgets[0]
            elif node_type == "LTXVGemmaCLIPModelLoader":
                # Node expects gemma_path, ltxv_path (not model_path, ltx_model_path)
                if len(widgets) > 0:
                    inputs["gemma_path"] = widgets[0]
                if len(widgets) > 1:
                    inputs["ltxv_path"] = widgets[1]
                if len(widgets) > 2:
                    inputs["max_length"] = widgets[2]
            elif node_type == "LTXVGemmaEnhancePrompt":
                if len(widgets) > 1:
                    inputs["system_prompt"] = widgets[1]
                if len(widgets) > 2:
                    inputs["max_tokens"] = widgets[2]
                if len(widgets) > 3:
                    inputs["bypass_i2v"] = widgets[3]
                if len(widgets) > 4:
                    inputs["seed"] = widgets[4]
                if len(widgets) > 5:
                    inputs["seed_mode"] = widgets[5]
                    inputs["bypass_i2v"] = widgets[4]
            elif node_type == "PrimitiveStringMultiline":
                if len(widgets) > 0:
                    inputs["value"] = widgets[0]
            elif node_type == "LTXVConditioning":
                if len(widgets) > 0:
                    inputs["shift"] = widgets[0]
            elif node_type == "CheckpointLoaderSimple":
                if len(widgets) > 0:
                    inputs["ckpt_name"] = widgets[0]
            elif node_type == "LoraLoaderModelOnly":
                if len(widgets) > 0:
                    inputs["lora_name"] = widgets[0]
                if len(widgets) > 1:
                    inputs["strength_model"] = widgets[1]

        # Handle input connections (links)
        if workflow and "inputs" in node:
            links_map = {}
            for link in workflow.get("links", []):
                if len(link) >= 5:
                    link_id = link[0]
                    source_node_id = link[1]
                    source_output_index = link[2]
                    links_map[link_id] = (source_node_id, source_output_index)

            for input_def in node["inputs"]:
                if "link" in input_def:
                    link_id = input_def["link"]
                    if link_id in links_map:
                        source_node_id, source_output_index = links_map[link_id]
                        inputs[input_def["name"]] = [str(source_node_id), source_output_index]

        return inputs

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        negative_prompt: str = "",
        seed: int = -1,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an image from text prompt.

        Args:
            prompt: Image description
            width: Image width
            height: Image height
            steps: Sampling steps
            negative_prompt: Negative prompt
            seed: Random seed
            output_path: Optional output file path

        Returns:
            Result dict with status and output info
        """
        try:
            # Load and convert workflow
            workflow_data = self._load_workflow()
            api_workflow = self._convert_workflow_to_api(workflow_data)
            api_workflow = self._update_generation_params(
                api_workflow, prompt, width, height, steps, seed
            )

            # Execute in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                prompt_id = loop.run_until_complete(self._queue_prompt(api_workflow))
                logger.info(f"Queued prompt: {prompt_id}")

                history = loop.run_until_complete(self._wait_for_completion(prompt_id))
                image_bytes = loop.run_until_complete(self._get_output_image(history))
            finally:
                loop.close()

            # Save image
            if output_path:
                image_path = Path(output_path)
            else:
                timestamp = int(time.time())
                image_path = self.outputs_dir / f"gen_{timestamp}.png"

            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)

            logger.info(f"Generated image: {len(image_bytes)} bytes")

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

    def edit(
        self,
        image: str,
        prompt: str,
        strength: float = 0.5,
        negative_prompt: str = "",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Edit an image.

        Args:
            image: Path to source image or base64 data
            prompt: Edit instruction
            strength: Edit strength (0-1)
            negative_prompt: Negative prompt
            output_path: Optional output file path

        Returns:
            Result dict with status and output info
        """
        try:
            # Load image bytes
            if image.startswith("data:image"):
                # Base64 data URL
                image_bytes = base64.b64decode(image.split(",", 1)[1])
            else:
                # File path
                image_bytes = Path(image).read_bytes()

            # Load and convert workflow
            workflow_data = self._load_workflow()
            api_workflow = self._convert_workflow_to_api(workflow_data)
            api_workflow = self._update_edit_params(api_workflow, prompt, image_bytes)

            # Upload image to ComfyUI
            self._upload_image(image_bytes, "comfyui_edit_input.png")

            # Execute in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                prompt_id = loop.run_until_complete(self._queue_prompt(api_workflow))
                logger.info(f"Queued edit prompt: {prompt_id}")

                history = loop.run_until_complete(self._wait_for_completion(prompt_id))
                edited_bytes = loop.run_until_complete(self._get_output_image(history))
            finally:
                loop.close()

            # Save edited image
            if output_path:
                out_path = Path(output_path)
            else:
                timestamp = int(time.time())
                out_path = self.outputs_dir / f"edit_{timestamp}.png"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(edited_bytes)

            logger.info(f"Edited image: {len(edited_bytes)} bytes")

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

    def _update_generation_params(
        self,
        workflow: dict,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        seed: int = -1,
    ) -> dict:
        """Update workflow with generation parameters.

        Args:
            workflow: Workflow dict (API format)
            prompt: Text prompt
            width: Image width
            height: Image height
            steps: Sampling steps
            seed: Random seed

        Returns:
            Updated workflow dict
        """
        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})

            if class_type == "CLIPTextEncode":
                node_data["inputs"]["text"] = prompt
            elif class_type == "EmptySD3LatentImage":
                node_data["inputs"]["width"] = width
                node_data["inputs"]["height"] = height
            elif class_type == "KSampler":
                if "seed" in inputs:
                    if inputs.get("seed", 0) == -1 or seed != -1:
                        node_data["inputs"]["seed"] = seed if seed != -1 else random.randint(0, 2**32 - 1)
                if "steps" in inputs:
                    node_data["inputs"]["steps"] = steps

        return workflow

    def _update_edit_params(
        self,
        workflow: dict,
        prompt: str,
        image_data: bytes,
    ) -> dict:
        """Update workflow for image editing.

        Args:
            workflow: Workflow dict (API format)
            prompt: Edit instruction
            image_data: Source image bytes

        Returns:
            Updated workflow dict
        """
        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type", "")

            if class_type == "CLIPTextEncode":
                node_data["inputs"]["text"] = prompt
            elif class_type == "LoadImage":
                node_data["inputs"]["image"] = "comfyui_edit_input.png"

        return workflow

    def _upload_image(self, image_bytes: bytes, filename: str):
        """Upload an image to ComfyUI's input directory.

        Args:
            image_bytes: Image data
            filename: Target filename

        Raises:
            RuntimeError: If upload fails
        """
        upload_url = f"{self.comfy_address}/upload/image"
        files = {
            "image": (filename, image_bytes, "image/png"),
            "overwrite": "true",
        }

        try:
            response = self._session.post(upload_url, files=files, timeout=30)
            response.raise_for_status()
            logger.info(f"Uploaded image to ComfyUI: {filename}")
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get ComfyUI server status.

        Returns:
            Status dict
        """
        try:
            response = self._session.get(f"{self.comfy_address}/system_stats", timeout=5)
            response.raise_for_status()
            return {
                "connected": True,
                "url": self.comfy_address,
                "system": response.json(),
            }
        except Exception as e:
            return {
                "connected": False,
                "url": self.comfy_address,
                "error": str(e),
            }

    def generate_video(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 121,
        fps: int = 24,
        steps: int = 50,
        cfg_scale: float = 7.0,
        negative_prompt: str = "blurry, distorted, bad quality",
        seed: int = -1,
        input_image: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            seed: Random seed
            input_image: Optional input image path for image-to-video
            output_path: Optional output file path

        Returns:
            Result dict with status and output info
        """
        try:
            # Load and convert workflow
            workflow_data = self._load_workflow()
            api_workflow = self._convert_workflow_to_api(workflow_data)
            api_workflow = self._update_video_params(
                api_workflow, prompt, width, height, num_frames, fps, steps, cfg_scale, seed
            )

            # Upload input image if provided
            if input_image:
                self._upload_image_for_video(input_image)
                # Update LoadImage nodes to use the uploaded image filename
                image_filename = Path(input_image).name
                for node_id, node_data in api_workflow.items():
                    if node_data.get("class_type") == "LoadImage":
                        if "inputs" not in node_data:
                            node_data["inputs"] = {}
                        node_data["inputs"]["image"] = image_filename
                        print(f"DEBUG: Updated LoadImage node {node_id} to use {image_filename}")


            # Execute in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                prompt_id = self._queue_prompt(api_workflow)
                logger.info(f"Queued video prompt: {prompt_id}")

                history = self._wait_for_completion(prompt_id, timeout=600)
                video_bytes = loop.run_until_complete(self._get_output_video(history))
            finally:
                loop.close()

            # Save video
            if output_path:
                video_path = Path(output_path)
            else:
                timestamp = int(time.time())
                video_path = self.outputs_dir / f"video_{timestamp}.mp4"

            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(video_bytes)

            logger.info(f"Generated video: {len(video_bytes)} bytes")

            return {
                "success": True,
                "output_path": str(video_path),
                "size": len(video_bytes),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Video generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _update_video_params(
        self,
        workflow: dict,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 121,
        fps: int = 24,
        steps: int = 50,
        cfg_scale: float = 7.0,
        seed: int = -1,
    ) -> dict:
        """Update workflow with video generation parameters.

        Args:
            workflow: Workflow dict (API format)
            prompt: Text prompt
            width: Video width
            height: Video height
            num_frames: Number of frames
            fps: Frames per second
            steps: Sampling steps
            cfg_scale: CFG scale
            seed: Random seed

        Returns:
            Updated workflow dict
        """
        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})

            if class_type == "CLIPTextEncode":
                # Update positive/negative prompts
                if "positive" in str(inputs).lower() or "Positive" in class_type:
                    node_data["inputs"]["text"] = prompt
                elif "negative" in str(inputs).lower() or "Negative" in class_type:
                    node_data["inputs"]["text"] = inputs.get("text", "")
            elif class_type in ["EmptyLTXLatentVideo", "EmptySD3LatentImage"]:
                node_data["inputs"]["width"] = width
                node_data["inputs"]["height"] = height
                if "num_frames" in inputs:
                    node_data["inputs"]["num_frames"] = num_frames
                if "fps" in inputs:
                    node_data["inputs"]["fps"] = fps
            elif class_type in ["LTXVSampler", "KSampler"]:
                if "seed" in inputs:
                    if inputs.get("seed", 0) == -1 or seed != -1:
                        node_data["inputs"]["seed"] = seed if seed != -1 else random.randint(0, 2**32 - 1)
                if "steps" in inputs:
                    node_data["inputs"]["steps"] = steps
                if "cfg_scale" in inputs or "cfg" in inputs:
                    cfg_key = "cfg_scale" if "cfg_scale" in inputs else "cfg"
                    node_data["inputs"][cfg_key] = cfg_scale

        return workflow

    def _upload_image_for_video(self, image_path: str):
        """Upload an image for video generation.

        Args:
            image_path: Path to input image (can be just filename if already uploaded)

        Raises:
            RuntimeError: If upload fails
        """
        # Check if it's an absolute path or an existing relative path
        path = Path(image_path)
        if path.is_absolute() or path.exists():
            image_bytes = path.read_bytes()
            filename = path.name
            self._upload_image(image_bytes, filename)
        # Otherwise, assume it's just a filename that's already in ComfyUI's input directory

    async def _get_output_video(self, history: dict) -> bytes:
        """Get the generated video from ComfyUI output.

        Args:
            history: History dict from _wait_for_completion

        Returns:
            Video bytes

        Raises:
            RuntimeError: If no video found
        """
        outputs = history.get("outputs", {})

        for node_id, node_output in outputs.items():
            if "videos" in node_output:
                for video in node_output["videos"]:
                    params = urllib.parse.urlencode({
                        "filename": video["filename"],
                        "subfolder": video.get("subfolder", ""),
                        "type": video.get("type", "output"),
                    })
                    url = f"{self.comfy_address}/view?{params}"

                    response = self._session.get(url, timeout=60)
                    response.raise_for_status()
                    return response.content

            # Fallback to images for frame-based video
            if "images" in node_output:
                # Collect all frames
                frames = []
                for img in node_output["images"]:
                    params = urllib.parse.urlencode({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
                    url = f"{self.comfy_address}/view?{params}"

                    response = self._session.get(url, timeout=30)
                    response.raise_for_status()
                    frames.append(response.content)

                # Combine frames into video (requires ffmpeg)
                # For now, return the first frame as a placeholder
                if frames:
                    return frames[0]

        raise RuntimeError("No video found in outputs")
