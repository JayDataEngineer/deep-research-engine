"""Auto Studio Extension - Automated Multi-Modal Content Generation.

This extension provides a composable system for building complex AI media pipelines:
- Video Podcast: Markdown script → video with lip-synced audio
- Deep Agent Planning: Single intelligent agent plans clips, emotions, prompts
- Multi-Modal Support: Text, Audio, Image, Video generation workflows

Key Innovation: Uses a Deep Agent with thinking mode to plan everything in one pass.
Prompts are modularly defined in config.yaml and assembled using Jinja2.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class AutoStudioExtension(LogicExtension):
    """Auto Studio extension - Automated multi-modal content generation.

    This extension supports two modes:
    - Config Mode: Full Deep Agent planning for video podcasts
    - Fast Mode: Real-time conversational video generation

    Configuration is done via config.yaml with modular prompt sections.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._planner = None
        self._fast_mode_planner = None
        self._mode = config.get("mode", "config")  # Default to config mode

    async def initialize(self, context):
        """Initialize the Auto Studio extension."""
        await super().initialize(context)
        logger.info(f"Auto Studio extension initialized (mode: {self._mode})")

    async def shutdown(self):
        """Cleanup resources."""
        if self._planner:
            # Cleanup if needed
            pass

    def _get_base_scene_path(self, base_scene_path: Optional[Path] = None) -> Path:
        """Resolve base scene path from argument or config.

        Args:
            base_scene_path: Optional explicit base scene path

        Returns:
            Path to base scene image

        Raises:
            ValueError: If base scene not found
        """
        if base_scene_path is None:
            base_scene_path = Path(self.config.get("base_scene", {}).get("path", ""))
            if not base_scene_path.exists():
                raise ValueError(
                    f"Base scene not found at {base_scene_path}. "
                    "Please create a base studio scene image or provide base_scene_path."
                )
        return base_scene_path

    def _prepare_context(self, context: Optional[dict] = None) -> dict:
        """Prepare context with base scene description from config.

        Args:
            context: Optional context dict

        Returns:
            Context dict with base_scene_description if not already present
        """
        if context is None:
            context = {}
        if "base_scene_description" not in context:
            context["base_scene_description"] = self.config.get("base_scene", {}).get("description", "")
        return context

    async def plan_video_podcast(
        self,
        script: str,
        base_scene_path: Optional[Path] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """Plan video podcast using Deep Agent.

        Args:
            script: Markdown script content
            base_scene_path: Path to base studio scene image
            context: Additional context (show info, speaker profiles, etc.)

        Returns:
            Dict with plan JSON and markdown
        """
        from src.extensions.libraries.auto_studio.functions.planning.deep_agent_planner import DeepAgentPlanner
        from src.extensions.libraries.auto_studio.functions.planning.json_to_markdown import plan_to_markdown
        from src.extensions.libraries.auto_studio.core.prompt_manager import ConfigPromptManager

        # Initialize prompt manager with config
        prompt_manager = ConfigPromptManager(self.config)

        # Initialize planner
        self._planner = DeepAgentPlanner(
            config=self.config.get("deep_agent", {}),
            prompt_manager=prompt_manager
        )

        # Resolve base scene path
        base_scene_path = self._get_base_scene_path(base_scene_path)

        # Prepare context with base scene description
        context = self._prepare_context(context)

        # Execute planning
        plan = await self._planner.plan_video_podcast(
            script=script,
            base_scene_path=str(base_scene_path),
            context=context,
        )

        # Convert to markdown
        markdown = plan_to_markdown(plan.model_dump())

        return {
            "plan": plan.model_dump(),
            "markdown": markdown,
        }

    async def generate_video_podcast(
        self,
        script: str,
        output_path: Optional[Path] = None,
        plan: Optional[dict] = None,
    ) -> dict:
        """Generate video podcast from script.

        Args:
            script: Markdown script content
            output_path: Optional path for output video
            plan: Optional pre-generated plan (skips Deep Agent planning)

        Returns:
            Dict with video_path, processing_time, etc.
        """
        # If no plan provided, generate one first
        if plan is None:
            plan_result = await self.plan_video_podcast(script)
            plan = plan_result["plan"]

        # TODO: Implement full generation pipeline
        # Phase 2: Qwen Image Edit
        # Phase 3: LTX2 Video Generation
        # Phase 4: Stitch Video
        # Phase 5: WhisperX → IndexTTS → Lip Sync

        yield json.dumps({"status": "processing", "message": "Generation pipeline not yet implemented"})

    # Process method for JobExtension compatibility
    async def process(self, input_data: dict, **kwargs):
        """Process text to generate video.

        Args:
            input_data: Dict with 'text', 'file_path', or 'plan' key
            **kwargs: Additional parameters including:
                - base_scene_path: Path to base studio scene
                - output_path: Optional path for output video
                - context: Additional context for planning

        Yields:
            Progress updates and final result
        """
        text = input_data.get("text")
        file_path_str = input_data.get("file_path")
        plan = input_data.get("plan")

        # Get input text
        if text:
            script = text
        elif file_path_str:
            file_path = Path(file_path_str)
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
            script = file_path.read_text()
        elif plan:
            # Use pre-generated plan, need script from it
            script = plan.get("script", "")
        else:
            raise ValueError("input_data must contain 'text', 'file_path', or 'plan'")

        base_scene_path = kwargs.get("base_scene_path")
        context = kwargs.get("context", {})

        yield json.dumps({"status": "planning", "message": "Planning video production..."})

        # Plan
        plan_result = await self.plan_video_podcast(script, base_scene_path, context)

        yield json.dumps({
            "status": "planned",
            "plan": plan_result["plan"],
            "markdown": plan_result["markdown"],
        })

        # TODO: Generate video
        # result = await self.generate_video_podcast(script, plan=plan_result["plan"])

        yield json.dumps({
            "status": "complete",
            # "video_path": result["video_path"],
            "plan": plan_result["plan"],
            "markdown": plan_result["markdown"],
        })

    async def generate_fast_prompt(
        self,
        user_message: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Generate LTX2 prompt from user message (Fast Mode).

        Args:
            user_message: The user's input message
            context: Additional context (speaker info, scene description, etc.)

        Returns:
            Dict with FastModePrompt data
        """
        from src.extensions.libraries.auto_studio.functions.planning.fast_mode_planner import FastModePlanner
        from src.extensions.libraries.auto_studio.core.prompt_manager import ConfigPromptManager

        # Initialize prompt manager with config
        prompt_manager = ConfigPromptManager(self.config)

        # Initialize fast mode planner
        if self._fast_mode_planner is None:
            self._fast_mode_planner = FastModePlanner(
                config=self.config.get("fast_mode", {}),
                prompt_manager=prompt_manager
            )

        # Prepare context with base scene description
        context = self._prepare_context(context)

        # Execute fast mode planning
        prompt = await self._fast_mode_planner.generate_prompt(
            user_message=user_message,
            context=context,
        )

        return {
            "prompt": prompt.model_dump(),
        }

    async def generate_fast_video(
        self,
        user_message: str,
        base_scene_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        context: Optional[dict] = None,
    ):
        """Generate video from user message (Fast Mode).

        Args:
            user_message: The user's input message
            base_scene_path: Path to base scene image
            output_path: Optional path for output video
            context: Additional context

        Yields:
            Progress updates and final result
        """
        # Resolve base scene path
        base_scene_path = self._get_base_scene_path(base_scene_path)

        # Prepare context
        context = self._prepare_context(context)

        # Generate LTX2 prompt
        yield json.dumps({"status": "generating_prompt", "message": "Generating LTX2 prompt..."})
        prompt_result = await self.generate_fast_prompt(user_message, context)

        yield json.dumps({
            "status": "prompt_ready",
            "prompt": prompt_result["prompt"],
        })

        # TODO: Generate video using LTX2
        # Phase 2: LTX2 Video Generation from base scene + prompt

        yield json.dumps({
            "status": "complete",
            "prompt": prompt_result["prompt"],
            # "video_path": result["video_path"],
        })

    # Process method for JobExtension compatibility


from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(AutoStudioExtension)
