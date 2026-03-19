"""Fast Mode - Real-time conversational video generation.

Direct LLM → LTX2 pipeline for minimal latency.
User message → LLM generates LTX2 prompt → LTX2 generates video.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ...core.prompt_manager import ConfigPromptManager

logger = logging.getLogger(__name__)


class FastModePrompt(BaseModel):
    """Simple prompt for fast mode video generation."""
    ltx2_prompt: str = Field(description="Prompt for LTX2 video generation")
    ltx2_negative_prompt: str = Field(default="", description="Negative prompt for LTX2")
    duration_frames: int = Field(default=120, ge=1, description="Duration in frames")
    camera_angle: str = Field(default="medium_close_up", description="Camera angle")


class FastModePlanner:
    """
    Fast Mode planner for real-time conversational video generation.

    Bypasses Deep Agent planning for minimal latency:
    1. User sends message
    2. LLM converts message to LTX2 prompt
    3. LTX2 generates video+audio from base image

    No clip breaks, no emotion vectors, no complex planning.
    """

    def __init__(self, config: dict, prompt_manager: ConfigPromptManager):
        """Initialize the Fast Mode planner.

        Args:
            config: Configuration dict with model settings
            prompt_manager: ConfigPromptManager instance
        """
        self.config = config
        self.prompt_manager = prompt_manager
        self.max_retries = config.get("max_retries", 2)

        # Load defaults from config
        defaults = config.get("defaults", {})
        self.default_duration = defaults.get("duration_frames", 120)
        self.default_camera = defaults.get("camera_angle", "medium_close_up")
        self.default_negative = defaults.get("negative_prompt", "blurry, distorted, bad quality")
        self.fallback_template = defaults.get("fallback_prompt_template", "{user_message}")

        logger.info("Fast Mode planner initialized")

    async def generate_prompt(
        self,
        user_message: str,
        context: dict,
    ) -> FastModePrompt:
        """
        Generate LTX2 prompt from user message.

        Args:
            user_message: The user's input message
            context: Additional context (speaker info, scene description, etc.)

        Returns:
            FastModePrompt with LTX2 prompt and parameters
        """
        from src.shared.chat_llm import get_chat_llm

        # Prepare variables for template rendering
        variables = {
            "user_message": user_message,
            "scene_description": context.get("scene_description", ""),
            "speaker_info": context.get("speaker_info", ""),
            "style_guidelines": context.get("style_guidelines", ""),
        }

        # Assemble prompt from config (fast mode)
        assembled = self.prompt_manager.assemble_prompt(variables, mode="fast")

        # Get LLM from backend system
        llm = get_chat_llm(
            model=self.config.get("model", "glm-4.7"),
            temperature=self.config.get("temperature", 0.7),
            enable_thinking=self.config.get("thinking_enabled", False)  # Faster without thinking
        )

        logger.info("Fast Mode: Generating LTX2 prompt from user message...")

        # Build messages
        messages = [
            {"role": "system", "content": assembled.system},
            {"role": "user", "content": assembled.user},
        ]

        # Try with validation and retry
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"[FastMode] Retry attempt {attempt}/{self.max_retries}")
                    # Add error feedback
                    error_msg = self._format_validation_error(last_error)
                    messages.append({
                        "role": "user",
                        "content": f"**VALIDATION ERROR** (Attempt {attempt}/{self.max_retries})\n\n{error_msg}\n\nPlease fix and return only valid JSON."
                    })

                # Call LLM
                response = await llm.ainvoke(messages)

                # Parse response
                content = response.content
                logger.debug(f"[FastMode] LLM response length: {len(content)} chars")

                # Extract JSON
                data = self._extract_json(content)

                # Validate with Pydantic
                prompt = FastModePrompt.model_validate(data)

                logger.info(f"Fast Mode: Generated prompt ({prompt.duration_frames} frames)")
                return prompt

            except Exception as e:
                last_error = e
                logger.warning(f"[FastMode] Attempt {attempt + 1} failed: {e}")

        # All retries exhausted - return a fallback prompt
        logger.warning(f"Fast Mode: All retries failed, using fallback prompt")
        fallback_prompt = self.fallback_template.format(user_message=user_message)
        return FastModePrompt(
            ltx2_prompt=fallback_prompt,
            ltx2_negative_prompt=self.default_negative,
            duration_frames=self.default_duration,
            camera_angle=self.default_camera
        )

    def _extract_json(self, content: str) -> dict:
        """Extract JSON from LLM response."""
        content = content.strip()

        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                json_part = parts[1]
                if json_part.startswith("json"):
                    json_part = json_part[4:]
                content = json_part.strip()

        return json.loads(content)

    def _format_validation_error(self, error: Exception) -> str:
        """Format validation error for LLM feedback."""
        return f"Error: {str(error)}\n\nPlease return a valid JSON response with ltx2_prompt field."
