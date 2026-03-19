"""LTX2 Video Generation composable function.

Generates video clips using LTX-2 via ComfyUI.
Each video is generated from a start frame (from Qwen Edit).

QUALITY MODE - LTX-2 Audio Generation:
========================================
LTX-2 can generate BOTH video and audio together when the prompt contains
QUOTED DIALOGUE. This is a key feature for Quality Mode.

Example prompt with audio:
    'Emma smiles warmly: "Welcome back to the show!" she speaks with
    genuine enthusiasm, her eyes bright as she looks toward the guest.'

The quoted text "Welcome back to the show!" triggers LTX-2 to generate
corresponding audio with lip-sync.

For Quality Mode pipeline:
1. generate_video_audio(): Image → Video + Audio (with quoted dialogue)
2. IndexTTS generates professional TTS (separate, with emotion vectors)
3. upscale_with_lipsync(): Video + TTS Audio → Upscaled Video (lipsync refined)
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from ...core.composable import BaseComposableFunction, FunctionInput, FunctionResult

logger = logging.getLogger(__name__)


class LTX2Config(BaseModel):
    """Configuration for LTX2 video generation function."""
    api_base: str = "http://localhost:8188"
    timeout: int = 600
    fps: int = 24
    guidance_scale: float = 7.0
    num_inference_steps: int = 50
    resolution: str = "1080p"  # Options: 720p, 1080p, 4k
    output_format: str = "mp4"
    output_dir: str = "data/outputs/auto_studio/ltx2_videos"

    # Generation mode
    mode: str = "generation"  # Options: generation, upscale_lipsync

    # Control options
    control_type: str = "none"  # Options: none, canny, depth, pose


class LTX2VideoFunction(BaseComposableFunction):
    """Generate video clips using LTX-2.

    LTX-2 supports TWO modes for Quality Mode pipeline:

    **Generation Mode (Image + Text → Video + Audio):**
    1. Takes a start frame (from Qwen Edit)
    2. Uses LTX-2 with quoted dialogue in prompt to generate Video + Audio
    3. Prompt format: 'Emma smiles: "Spoken text here" she continues...'
    4. Outputs video clip + AI-generated audio

    **Upscale/Lipsync Mode (Video + Audio → Video):**
    1. Takes video clip + professional TTS audio (from IndexTTS)
    2. Uses LTX-2 upscaling (temporal + spatial)
    3. Refines lipsync to match new TTS audio
    4. Outputs enhanced video with perfect lipsync

    QUALITY MODE WORKFLOW:
    =====================
    Fast Mode: Script → Deep Agent → Qwen Edit → LTX-2 Video → Stitch
    Quality Mode: Script → Deep Agent → Qwen Edit → LTX-2 Video+AI-Audio
               → IndexTTS Professional Audio
               → LTX-2 Upscale+Lipsync → Stitch

    Integrates with the ComfyUI LTXVideo extension.
    """

    name = "ltx2_video_generation"
    input_types = ["image", "video", "audio", "prompt"]
    output_types = ["video", "audio"]
    requires_gpu = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ltx2_config = LTX2Config(**config.get("ltx2", {}))
        self.output_dir = Path(self.ltx2_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolution mapping
        self._resolution_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160)
        }

    def _validate_config(self):
        """Validate LTX2 configuration."""
        if not self.ltx2_config.api_base:
            raise ValueError("ltx2.api_base is required")
        if self.ltx2_config.fps not in [24, 30, 60]:
            raise ValueError("ltx2.fps must be 24, 30, or 60")
        if self.ltx2_config.resolution not in ["720p", "1080p", "4k"]:
            raise ValueError("ltx2.resolution must be 720p, 1080p, or 4k")

    async def validate_input(self, input_data: FunctionInput) -> Tuple[bool, Optional[str]]:
        """Validate input data.

        Required:
        - mode: "generation" or "upscale_lipsync"

        For "generation" mode:
        - start_frame: Path to start frame in input_paths
        - ltx2_prompt: Prompt in plan_data
        - duration_frames: Number of frames in plan_data

        For "upscale_lipsync" mode:
        - input_video: Path to video in input_paths
        - input_audio: Path to audio in input_paths
        - ltx2_prompt: Optional refinement prompt
        """
        mode = input_data.plan_data.get("mode", "generation")

        if mode == "generation":
            # Check start frame
            if "start_frame" not in input_data.input_paths:
                return False, "Missing 'start_frame' in input_paths for generation mode"

            start_frame = input_data.input_paths["start_frame"]
            if not start_frame.exists():
                return False, f"Start frame not found: {start_frame}"

            # Check prompt
            prompt = input_data.plan_data.get("ltx2_prompt", "")
            if not prompt:
                return False, "Missing 'ltx2_prompt' in plan_data"

            # Check duration
            duration_frames = input_data.plan_data.get("duration_frames")
            if not duration_frames or duration_frames <= 0:
                return False, f"Invalid duration_frames: {duration_frames}"

        elif mode == "upscale_lipsync":
            # Check input video
            if "input_video" not in input_data.input_paths:
                return False, "Missing 'input_video' in input_paths for upscale_lipsync mode"

            input_video = input_data.input_paths["input_video"]
            if not Path(input_video).exists():
                return False, f"Input video not found: {input_video}"

            # Check input audio
            if "input_audio" not in input_data.input_paths:
                return False, "Missing 'input_audio' in input_paths for upscale_lipsync mode"

            input_audio = input_data.input_paths["input_audio"]
            if not Path(input_audio).exists():
                return False, f"Input audio not found: {input_audio}"

        else:
            return False, f"Invalid mode: {mode}. Must be 'generation' or 'upscale_lipsync'"

        return True, None

    async def execute(self, input_data: FunctionInput) -> FunctionResult:
        """Execute LTX2 video generation via ComfyUI.

        Args:
            input_data: Contains start_frame, ltx2_prompt, duration_frames, clip_id

        Returns:
            FunctionResult with output video path
        """
        clip_id = input_data.clip_id
        start_frame = input_data.input_paths["start_frame"]
        prompt = input_data.plan_data.get("ltx2_prompt")
        negative_prompt = input_data.plan_data.get("ltx2_negative_prompt",
                                                      "blurry, distorted, bad quality, unnatural movements, stiff pose")
        duration_frames = input_data.plan_data.get("duration_frames")
        camera_angle = input_data.plan_data.get("camera_angle", "medium_close_up")

        # Calculate resolution
        width, height = self._resolution_map.get(self.ltx2_config.resolution, (1920, 1080))

        # Output path
        output_path = self.output_dir / f"{clip_id}_video.{self.ltx2_config.output_format}"

        logger.info(f"[LTX2] Generating video for clip {clip_id}")
        logger.debug(f"[LTX2] Duration: {duration_frames} frames ({duration_frames/self.ltx2_config.fps:.1f}s @ {self.ltx2_config.fps}fps)")
        logger.debug(f"[LTX2] Resolution: {self.ltx2_config.resolution} ({width}x{height})")
        logger.debug(f"[LTX2] Prompt: {prompt[:100]}...")

        start_time = time.time()

        try:
            # Build and execute ComfyUI workflow
            result = await self._execute_comfyui_workflow(
                start_frame=str(start_frame),
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=duration_frames,
                fps=self.ltx2_config.fps,
                width=width,
                height=height,
                guidance_scale=self.ltx2_config.guidance_scale,
                num_inference_steps=self.ltx2_config.num_inference_steps,
                output_path=str(output_path)
            )

            if not result.get("success"):
                return FunctionResult(
                    success=False,
                    error=result.get("error", "Unknown error"),
                    metadata={"clip_id": clip_id}
                )

            elapsed = time.time() - start_time
            logger.info(f"[LTX2] Generated video for {clip_id} in {elapsed:.1f}s")

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
                        "video": str(output_path)
                    },
                    "video_info": {
                        "clip_id": clip_id,
                        "duration_frames": duration_frames,
                        "duration_seconds": duration_frames / self.ltx2_config.fps,
                        "fps": self.ltx2_config.fps,
                        "resolution": self.ltx2_config.resolution,
                        "width": width,
                        "height": height,
                        "path": str(output_path)
                    }
                },
                checkpoint_data={
                    "clip_id": clip_id,
                    "output_path": str(output_path),
                    "input_prompt": prompt,
                    "duration_frames": duration_frames
                },
                metadata={"clip_id": clip_id, "elapsed": elapsed}
            )

        except Exception as e:
            logger.exception(f"[LTX2] Execution error: {e}")
            return FunctionResult(
                success=False,
                error=f"Execution failed: {e}",
                metadata={"clip_id": clip_id}
            )

    async def _execute_comfyui_workflow(
        self,
        start_frame: str,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        fps: int,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
        output_path: str
    ) -> Dict[str, Any]:
        """Execute LTX-2 video generation via ComfyUI API.

        Args:
            start_frame: Path to start frame image
            prompt: Video generation prompt
            negative_prompt: Negative prompt
            num_frames: Number of frames to generate
            fps: Frames per second
            width: Video width
            height: Video height
            guidance_scale: CFG scale
            num_inference_steps: Number of denoising steps
            output_path: Output file path

        Returns:
            Execution result dict
        """
        # Import here to avoid circular dependency
        from src.extensions.integrations.comfyui.shared import ComfyUIEngine
        from src.constants import PROJECT_ROOT

        # Initialize ComfyUI engine
        workflow_path = PROJECT_ROOT / "data" / "inputs" / "workflows" / "video_ltx_video.json"
        outputs_dir = PROJECT_ROOT / "data" / "outputs" / "generated_videos"

        engine = ComfyUIEngine(
            comfy_address=self.ltx2_config.api_base,
            workflow_path=workflow_path,
            outputs_dir=outputs_dir
        )

        # Upload start frame image
        logger.debug(f"[LTX2] Uploading start frame: {start_frame}")
        start_frame_path = Path(start_frame)
        image_bytes = start_frame_path.read_bytes()
        engine._upload_image(image_bytes, start_frame_path.name)

        # Generate video using the ComfyUI engine
        result = engine.generate_video(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=num_inference_steps,
            cfg_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=-1,  # Random seed
            input_image=start_frame_path.name,  # Use uploaded image
            output_path=output_path
        )

        return result

    def get_checkpoint_data(self, input_data: FunctionInput, result: FunctionResult) -> Dict[str, Any]:
        """Get checkpoint data for resuming execution.

        Args:
            input_data: Original input data
            result: Result from successful execution

        Returns:
            Checkpoint data dict
        """
        if result.success:
            return result.checkpoint_data or {}
        return {}
