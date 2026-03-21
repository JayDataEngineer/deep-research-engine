"""Image generation using ComfyUI HTTP API with z-image-turbo."""

import asyncio
import base64
import io
import json
from .logging_config import get_logger
import os
import random
import urllib.parse
from pathlib import Path
from typing import Optional
from collections import defaultdict

import requests

# Import from src.constants (the main constants file)
# Note: We need to add src to path for this import
import sys
from pathlib import Path
src_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to src/
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

try:
    from constants import DEFAULT_COMFYUI_URL, BACKEND_ROOT, PROJECT_ROOT, IMAGES_DIR, ENV_COMFYUI_ADDRESS
except ImportError:
    # Fallback for when running from different context
    DEFAULT_COMFYUI_URL = "http://127.0.0.1:18188"
    BACKEND_ROOT = Path(__file__).parent.parent.parent.parent
    PROJECT_ROOT = BACKEND_ROOT.parent
    IMAGES_DIR = PROJECT_ROOT / "data" / "outputs" / "generated_images"
    ENV_COMFYUI_ADDRESS = "COMFYUI_ADDRESS"

from .config import ImageGenConfig

logger = get_logger(__name__)


class ImageGenerationServiceHTTP:
    """Service for image generation using ComfyUI HTTP API."""

    def __init__(self, config: ImageGenConfig, workflow_path: Optional[Path] = None):
        """Initialize the image generation service.

        Args:
            config: Image generation configuration
            workflow_path: Path to workflow JSON file
        """
        self.config = config
        self.workflow_path = workflow_path or self._find_workflow()
        self._session = requests.Session()
        self._character_references: dict[str, bytes] = defaultdict(bytes)

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _find_workflow(self) -> Path:
        """Find the z-image-turbo workflow JSON."""
        backend_dir = BACKEND_ROOT
        project_root = PROJECT_ROOT

        # Search in unified data/inputs/workflows directory
        workflow_path = project_root / "data" / "inputs" / "workflows" / "image_z_image_turbo.json"
        if workflow_path.exists():
            return workflow_path

        # Also try z_image_turbo.json
        workflow_path = project_root / "data" / "inputs" / "workflows" / "z_image_turbo.json"
        if workflow_path.exists():
            return workflow_path

        # Search in backend/workflows directory (legacy)
        workflow_path = backend_dir / "workflows" / "z_image_turbo.json"
        if workflow_path.exists():
            return workflow_path

        # Search in project root
        workflow_path = project_root / "image_z_image_turbo.json"
        if workflow_path.exists():
            return workflow_path

        # Search in backend dir
        workflow_path = backend_dir / "image_z_image_turbo.json"
        if workflow_path.exists():
            return workflow_path

        # Fallback to ComfyUI examples
        comfy_path = backend_dir / "vendor" / "ComfyUI"
        for path in comfy_path.rglob("*.json"):
            if "z-image-turbo" in path.name.lower() or "z_image_turbo" in path.name.lower():
                if "3d" not in str(path).lower() and "hunyuan" not in str(path).lower():
                    return path

        raise FileNotFoundError(f"No workflow JSON found")

    def _find_qwen_workflow(self) -> Path:
        """Find the Qwen GGUF image edit workflow JSON."""
        project_root = PROJECT_ROOT

        workflow_path = project_root / "data" / "inputs" / "workflows" / "Qwen_GGUF_API.json"
        if workflow_path.exists():
            return workflow_path

        workflow_path = project_root / "data" / "inputs" / "workflows" / "qwen_gguf_api.json"
        if workflow_path.exists():
            return workflow_path

        raise FileNotFoundError(f"No Qwen GGUF workflow JSON found")

    def _load_workflow(self) -> dict:
        """Load the ComfyUI workflow from JSON."""
        with open(self.workflow_path) as f:
            return json.load(f)

    def _update_prompt(self, workflow: dict, prompt: str, width: int = 1024, height: int = 1024, steps: int = 9) -> dict:
        """Update the workflow with our parameters."""
        for node in workflow['nodes']:
            if node['type'] == 'CLIPTextEncode':
                node['widgets_values'][0] = prompt
            elif node['type'] == 'EmptySD3LatentImage':
                node['widgets_values'][0] = width
                node['widgets_values'][1] = height
            elif node['type'] == 'KSampler':
                node['widgets_values'][2] = steps
                if len(node['widgets_values']) > 1 and node['widgets_values'][1] == "randomize":
                    node['widgets_values'][0] = random.randint(0, 2**32 - 1)
        return workflow

    async def _queue_prompt(self, workflow: dict) -> str:
        """Send workflow to ComfyUI and return prompt ID."""
        # Build link lookup for remapping
        links_map = {}
        for link in workflow.get('links', []):
            if len(link) >= 5:
                link_id = link[0]
                source_node_id = link[1]
                source_output_index = link[2]
                target_node_id = link[3]
                target_input_index = link[4]
                links_map[(target_node_id, target_input_index)] = (source_node_id, source_output_index)

        # Find LoraLoaderModelOnly nodes that need to be bypassed
        lora_nodes_to_skip = set()
        backend_dir = BACKEND_ROOT
        loras_dir = backend_dir / "vendor" / "ComfyUI" / "models" / "loras"

        for node in workflow.get('nodes', []):
            if node['type'] == 'LoraLoaderModelOnly':
                lora_name = node['widgets_values'][0] if node['widgets_values'] else None
                if lora_name:
                    lora_path = loras_dir / lora_name
                    if not lora_path.exists():
                        logger.info(f"LoRA file not found: {lora_name}, bypassing node {node['id']}")
                        lora_nodes_to_skip.add(node['id'])

        # Convert nodes array to dict format for API
        if 'nodes' in workflow:
            nodes_dict = {}
            auxiliary_node_types = {'MarkdownNote', 'Note', 'Comment', 'PrimitiveNode', 'Reroute'}

            for node in workflow['nodes']:
                node_id = node['id']
                node_type = node['type']

                if node_type in auxiliary_node_types:
                    continue

                if node_id in lora_nodes_to_skip:
                    continue

                node_id_str = str(node_id)
                inputs = self._get_node_inputs(node, workflow)

                # Remap inputs from skipped LoraLoader nodes
                for input_name, input_value in list(inputs.items()):
                    if isinstance(input_value, list) and len(input_value) >= 2:
                        source_node_id = input_value[0]
                        if int(source_node_id) in lora_nodes_to_skip:
                            for link in workflow.get('links', []):
                                if len(link) >= 5 and link[3] == int(source_node_id):
                                    if link[3] in lora_nodes_to_skip:
                                        inputs[input_name] = [str(link[1]), link[2]]
                                        break

                nodes_dict[node_id_str] = {
                    'class_type': node['type'],
                    'inputs': inputs,
                }
            prompt = nodes_dict
        else:
            prompt = workflow

        data = json.dumps({"prompt": prompt}).encode('utf-8')

        logger.info(f"[IMG_GEN] Sending prompt to ComfyUI: {self.config.comfy_url}/prompt")

        for attempt in range(3):
            try:
                response = self._session.post(
                    f"{self.config.comfy_url}/prompt",
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                return result['prompt_id']
            except Exception as e:
                logger.error(f"[IMG_GEN] Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                raise RuntimeError(f"Failed to queue prompt: {e}")

    def _get_node_inputs(self, node: dict, workflow: dict = None) -> dict:
        """Extract inputs from a node for the API format."""
        inputs = {}

        if 'widgets_values' in node:
            widgets = node['widgets_values']
            if node['type'] == 'CLIPLoader':
                if len(widgets) > 0:
                    inputs['clip_name'] = widgets[0]
                if len(widgets) > 1:
                    inputs['type'] = widgets[1]
            elif node['type'] == 'VAELoader':
                if len(widgets) > 0:
                    inputs['vae_name'] = widgets[0]
            elif node['type'] == 'UNETLoader':
                if len(widgets) > 0:
                    inputs['unet_name'] = widgets[0]
                if len(widgets) > 1:
                    inputs['weight_dtype'] = widgets[1]
            elif node['type'] == 'EmptySD3LatentImage':
                if len(widgets) > 0:
                    inputs['width'] = widgets[0]
                if len(widgets) > 1:
                    inputs['height'] = widgets[1]
                if len(widgets) > 2:
                    inputs['batch_size'] = widgets[2]
            elif node['type'] == 'CLIPTextEncode':
                if len(widgets) > 0:
                    inputs['text'] = widgets[0]
            elif node['type'] == 'KSampler':
                seed_val = widgets[0]
                if len(widgets) > 1 and widgets[1] == "randomize":
                    seed_val = random.randint(0, 2**32 - 1)
                inputs['seed'] = int(seed_val)
                if len(widgets) > 2:
                    inputs['steps'] = widgets[2]
                if len(widgets) > 3:
                    inputs['cfg'] = widgets[3]
                if len(widgets) > 4:
                    inputs['sampler_name'] = widgets[4]
                if len(widgets) > 5:
                    inputs['scheduler'] = widgets[5]
                if len(widgets) > 6:
                    inputs['denoise'] = widgets[6]
            elif node['type'] == 'ModelSamplingAuraFlow':
                if len(widgets) > 0:
                    inputs['shift'] = widgets[0]
            elif node['type'] == 'LoraLoaderModelOnly':
                if len(widgets) > 0:
                    inputs['lora_name'] = widgets[0]
                if len(widgets) > 1:
                    inputs['strength_model'] = widgets[1]
            elif node['type'] == 'SaveImage':
                if len(widgets) > 0:
                    inputs['filename_prefix'] = widgets[0]

        if workflow and 'inputs' in node:
            links_map = {}
            for link in workflow.get('links', []):
                if len(link) >= 5:
                    link_id = link[0]
                    source_node_id = link[1]
                    source_output_index = link[2]
                    links_map[link_id] = (source_node_id, source_output_index)

            for input_def in node['inputs']:
                if 'link' in input_def:
                    link_id = input_def['link']
                    if link_id in links_map:
                        source_node_id, source_output_index = links_map[link_id]
                        inputs[input_def['name']] = [str(source_node_id), source_output_index]

        return inputs

    async def _wait_for_completion(self, prompt_id: str, timeout: int = 180) -> dict:
        """Wait for prompt to complete and return history."""
        start = asyncio.get_event_loop().time()

        while True:
            try:
                response = self._session.get(
                    f"{self.config.comfy_url}/history/{prompt_id}",
                    timeout=5
                )
                response.raise_for_status()
                history = response.json()

                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('completed', False) or history[prompt_id].get('outputs'):
                        return history[prompt_id]

                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > timeout:
                    raise RuntimeError(f"Generation timeout after {timeout}s")

                await asyncio.sleep(1)

            except requests.exceptions.RequestException as e:
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed > timeout:
                    raise RuntimeError(f"Generation timeout after {timeout}s")
                await asyncio.sleep(2)

    async def _get_output_image(self, history: dict) -> bytes:
        """Get the generated image from ComfyUI output."""
        outputs = history.get('outputs', {})

        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for img in node_output['images']:
                    params = urllib.parse.urlencode({
                        'filename': img['filename'],
                        'subfolder': img.get('subfolder', ''),
                        'type': img.get('type', 'output')
                    })
                    url = f"{self.config.comfy_url}/view?{params}"

                    response = self._session.get(url, timeout=30)
                    response.raise_for_status()
                    return response.content

        raise RuntimeError("No image found in outputs")

    async def generate_character_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        sample_steps: int = 9,
        seed: int = -1,
    ) -> bytes:
        """Generate a new character image using z-image-turbo.

        Args:
            prompt: Character description
            negative_prompt: Not used by z-image-turbo (ignored)
            width: Image width (default 1024)
            height: Image height (default 1024)
            sample_steps: Number of sampling steps (default 9 for turbo)
            seed: Random seed (not used, z-image-turbo uses its own)

        Returns:
            Generated image bytes (PNG format)
        """
        workflow = self._load_workflow()
        workflow = self._update_prompt(workflow, prompt, width, height, sample_steps)

        prompt_id = await self._queue_prompt(workflow)
        logger.info(f"Queued prompt: {prompt_id}")

        history = await self._wait_for_completion(prompt_id)
        image_bytes = await self._get_output_image(history)

        logger.info(f"Generated image: {len(image_bytes)} bytes")
        return image_bytes

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
    ) -> bytes:
        """Synchronous wrapper for image generation.

        This is a convenience method for synchronous code.
        Runs the async method in an event loop.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_character_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    seed=seed,
                )
            )
        finally:
            loop.close()

    def set_character_reference(self, session_id: str, image_data: bytes):
        """Store a character reference image for a session."""
        self._character_references[session_id] = image_data

    def get_character_reference(self, session_id: str) -> Optional[bytes]:
        """Get the character reference image for a session."""
        return self._character_references.get(session_id)

    def clear_character_reference(self, session_id: str):
        """Remove the character reference for a session."""
        self._character_references.pop(session_id, None)

    def image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 data URL."""
        return f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"

    def base64_to_image(self, data_url: str) -> bytes:
        """Convert base64 data URL to image bytes."""
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        return base64.b64decode(data_url)

    async def edit_image_qwen(
        self,
        image_data: bytes,
        prompt: str,
        strength: float = 1.0,
        negative_prompt: str = "",
    ) -> bytes:
        """Edit an image using the Qwen Image Edit GGUF model.

        Args:
            image_data: Source image bytes to edit
            prompt: Edit instruction
            strength: Edit strength (0-1). 1.0 = full edit.
            negative_prompt: Things to avoid in the edit

        Returns:
            Edited image bytes (PNG format)
        """
        workflow_path = self._find_qwen_workflow()
        with open(workflow_path) as f:
            workflow = json.load(f)

        logger.info(f"[QWEN_EDIT] Using workflow: {workflow_path}")

        workflow = self._update_qwen_workflow(workflow, prompt, negative_prompt, image_data)

        prompt_id = await self._queue_prompt(workflow)
        logger.info(f"[QWEN_EDIT] Queued prompt: {prompt_id}")

        history = await self._wait_for_completion(prompt_id)
        image_bytes = await self._get_output_image(history)

        logger.info(f"[QWEN_EDIT] Generated edited image: {len(image_bytes)} bytes")
        return image_bytes

    def _update_qwen_workflow(
        self,
        workflow: dict,
        prompt: str,
        negative_prompt: str,
        image_data: bytes,
    ) -> dict:
        """Update Qwen GGUF workflow with prompt and image."""
        image_b64 = base64.b64encode(image_data).decode('utf-8')

        # Clean workflow: remove _meta keys
        clean_workflow = {}
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and "class_type" in node_data:
                clean_workflow[node_id] = {
                    "class_type": node_data["class_type"],
                    "inputs": node_data.get("inputs", {})
                }

        workflow = clean_workflow

        # Update node 26 - positive prompt
        if "26" in workflow:
            workflow["26"]["inputs"]["prompt"] = prompt

        # Update node 19 - negative prompt
        if "19" in workflow:
            workflow["19"]["inputs"]["prompt"] = negative_prompt

        # Upload image to ComfyUI
        self._upload_image_to_comfyui(image_b64, "qwen_edit_input.png")

        # Update node 22 to reference the uploaded image
        if "22" in workflow:
            workflow["22"]["inputs"]["image"] = "qwen_edit_input.png"

        return workflow

    def _upload_image_to_comfyui(self, image_b64: str, filename: str):
        """Upload a base64 image to ComfyUI's input directory."""
        image_bytes = base64.b64decode(image_b64)

        upload_url = f"{self.config.comfy_url}/upload/image"
        files = {
            'image': (filename, image_bytes, 'image/png'),
            'overwrite': 'true'
        }

        try:
            response = self._session.post(upload_url, files=files, timeout=30)
            response.raise_for_status()
            logger.info(f"[QWEN_EDIT] Uploaded image to ComfyUI: {filename}")
        except Exception as e:
            logger.error(f"[QWEN_EDIT] Failed to upload image: {e}")
            raise

    async def edit_character_image(
        self,
        session_id: str,
        prompt: str,
        strength: float = 0.5,
        negative_prompt: str = "",
    ) -> bytes:
        """Edit a character image using img2img.

        Args:
            session_id: Session ID with stored character reference
            prompt: Edit instruction
            strength: Edit strength (0-1)
            negative_prompt: Things to avoid in the edit

        Returns:
            Edited image bytes (PNG format)
        """
        init_image = self.get_character_reference(session_id)
        if not init_image:
            raise ValueError(f"No character reference set for session: {session_id}")

        # For now, use Qwen editing
        return await self.edit_image_qwen(
            image_data=init_image,
            prompt=prompt,
            strength=strength,
            negative_prompt=negative_prompt,
        )


# Singleton instance
_image_service_http: Optional[ImageGenerationServiceHTTP] = None


def get_image_service_http() -> ImageGenerationServiceHTTP:
    """Get the singleton image generation service instance (HTTP API)."""
    global _image_service_http
    if _image_service_http is None:
        config = ImageGenConfig(
            comfy_url=os.getenv(ENV_COMFYUI_ADDRESS, DEFAULT_COMFYUI_URL),
            output_dir=os.getenv("COMFY_OUTPUT_DIR", str(IMAGES_DIR)),
        )
        _image_service_http = ImageGenerationServiceHTTP(config)
        logger.info("Created new ComfyUI HTTP image service")
    return _image_service_http


# =============================================================================
# Wrapper functions for backward compatibility
# =============================================================================

async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
) -> bytes:
    """Generate an image from text prompt.

    Args:
        prompt: Image description
        negative_prompt: What to avoid
        width: Image width
        height: Image height
        seed: Random seed (-1 for random)

    Returns:
        Generated image as bytes
    """
    service = get_image_service_http()
    return await service.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
    )


async def generate_character_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
) -> bytes:
    """Generate a character image (alias for generate_image).

    Args:
        prompt: Character description
        negative_prompt: What to avoid
        width: Image width
        height: Image height
        seed: Random seed (-1 for random)

    Returns:
        Generated character image as bytes
    """
    return await generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
    )
