"""Deep Agent Planner for video podcast production.

Uses the backend's LLM system with structured output for planning:
- Clip breaks
- Emotion vectors
- Qwen Image Edit prompts
- LTX2 prompts
- Additional parameters (num_frames, camera_angle, etc.)

Includes Pydantic validation with automatic retry (up to 2 attempts).
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ...core.prompt_manager import ConfigPromptManager

logger = logging.getLogger(__name__)


class EmotionVector(BaseModel):
    """Emotion vector for IndexTTS."""
    happy: float = Field(ge=0.0, le=1.0, description="Happiness (0.0-1.0)")
    angry: float = Field(ge=0.0, le=1.0, description="Anger (0.0-1.0)")
    sad: float = Field(ge=0.0, le=1.0, description="Sadness (0.0-1.0)")
    afraid: float = Field(ge=0.0, le=1.0, description="Fear (0.0-1.0)")
    disgusted: float = Field(ge=0.0, le=1.0, description="Disgust (0.0-1.0)")
    melancholic: float = Field(ge=0.0, le=1.0, description="Melancholy (0.0-1.0)")
    surprised: float = Field(ge=0.0, le=1.0, description="Surprise (0.0-1.0)")
    calm: float = Field(ge=0.0, le=1.0, description="Calmness (0.0-1.0)")
    intensity: float = Field(ge=0.0, le=1.0, description="Overall intensity (0.0-1.0)")


class ClipPlan(BaseModel):
    """Plan for a single clip."""
    clip_id: str = Field(description="Unique clip identifier (e.g., clip_001, clip_002)")
    start_line: int = Field(description="First dialogue line number (1-indexed)")
    end_line: int = Field(description="Last dialogue line number (1-indexed)")
    speakers: list[str] = Field(description="Speakers present in this clip")
    camera_break_type: str = Field(
        description="Why this clip ends: speaker_change, scene_change, pause, or natural_end"
    )
    duration_frames: int = Field(description="Number of frames for LTX2 at 24fps", ge=1)
    emotion_vectors: dict[str, EmotionVector] = Field(description="Emotion vectors per speaker")
    qwen_edit_prompt: str = Field(description="Prompt for Qwen Image Edit (zoom from base scene)")
    qwen_negative_prompt: str = Field(default="", description="Negative prompt for Qwen Edit")
    ltx2_prompt: str = Field(description="Prompt for LTX2 video generation")
    ltx2_negative_prompt: str = Field(default="", description="Negative prompt for LTX2")
    camera_angle: str = Field(
        description="Camera angle: wide, medium, close_up, medium_close_up, extreme_close_up"
    )
    additional_params: dict[str, Any] = Field(default_factory=dict, description="Any additional parameters")


class ScriptInfo(BaseModel):
    """Script metadata."""
    title: str = Field(description="Script title")
    total_lines: int = Field(description="Total number of dialogue lines", ge=0)
    num_clips: int = Field(description="Number of clips in the plan", ge=0)
    estimated_duration_seconds: float = Field(description="Estimated total duration in seconds", ge=0.0)


class VideoPodcastPlan(BaseModel):
    """Complete plan for video podcast generation."""
    script_info: ScriptInfo = Field(description="Script metadata")
    base_scene_reference: str = Field(description="Path to base studio scene image")
    clips: list[ClipPlan] = Field(description="Plan for each clip", min_length=1)


class DeepAgentPlanner:
    """
    Deep Agent for planning video podcast generation.

    Uses the backend's LLM system (get_chat_llm) with structured output:
    - Automatic Pydantic validation
    - Built-in retry on validation errors (up to 2 attempts)

    Planning happens in one pass:
    - Clip breaks
    - Emotion vectors
    - Qwen Image Edit prompts
    - LTX2 prompts
    - Additional parameters
    """

    def __init__(self, config: dict, prompt_manager: ConfigPromptManager):
        """
        Initialize the Deep Agent planner.

        Args:
            config: Configuration dict with model settings
            prompt_manager: ConfigPromptManager instance
        """
        self.config = config
        self.prompt_manager = prompt_manager
        self.max_retries = config.get("max_retries", 2)

        logger.info(f"Deep Agent planner initialized, max_retries: {self.max_retries}")

    async def plan_video_podcast(
        self,
        script: str,
        base_scene_path: str,
        context: dict,
    ) -> VideoPodcastPlan:
        """
        Generate complete plan for video podcast with backend LLM and validation.

        Args:
            script: Markdown script content
            base_scene_path: Path to base studio scene image
            context: Additional context (show info, speaker profiles, etc.)

        Returns:
            VideoPodcastPlan with all clip information
        """
        from src.shared.chat_llm import get_chat_llm

        # Prepare variables for template rendering
        variables = {
            "script": script,
            "base_scene_description": context.get("base_scene_description", ""),
            "show_context": context.get("show_context", ""),
            "speaker_profiles": context.get("speaker_profiles", ""),
            "style_guidelines": context.get("style_guidelines", ""),
        }

        # Assemble prompt from config sections
        assembled = self.prompt_manager.assemble_prompt(variables)

        # Get LLM from backend system
        llm = get_chat_llm(
            model=self.config.get("model"),
            temperature=self.config.get("temperature", 0.7),
            enable_thinking=self.config.get("thinking_enabled", True)
        )

        logger.info("Executing Deep Agent planner with backend LLM...")

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
                    logger.info(f"[DeepAgent] Retry attempt {attempt}/{self.max_retries}")
                    # Add error feedback to messages
                    error_msg = self._format_validation_error(last_error)
                    messages.append({
                        "role": "user",
                        "content": f"**VALIDATION ERROR** (Attempt {attempt}/{self.max_retries})\n\n{error_msg}\n\nPlease fix these errors and try again. Return the complete corrected JSON response only."
                    })

                # Call LLM
                response = await llm.ainvoke(messages)

                # Parse response
                content = response.content
                logger.debug(f"[DeepAgent] LLM response length: {len(content)} chars")

                # Extract JSON from response
                data = self._extract_json(content)

                # Validate with Pydantic
                plan = VideoPodcastPlan.model_validate(data)

                # Success!
                plan.base_scene_reference = base_scene_path
                logger.info(f"Deep Agent planning complete! Generated {len(plan.clips)} clips")
                return plan

            except ValidationError as e:
                last_error = e
                error_count = len(e.errors())
                logger.warning(f"[DeepAgent] Validation failed with {error_count} errors (attempt {attempt + 1}/{self.max_retries + 1})")

            except Exception as e:
                logger.error(f"[DeepAgent] Unexpected error: {e}")
                raise

        # All retries exhausted
        raise ValueError(
            f"Deep Agent planning failed after {self.max_retries + 1} attempts. "
            f"The LLM could not produce a valid plan. Last error: {last_error}"
        )

    def _extract_json(self, content: str) -> dict:
        """Extract JSON from LLM response.

        Args:
            content: Raw LLM response content

        Returns:
            Parsed JSON dict
        """
        content = content.strip()

        # Check for code block
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                json_part = parts[1]
                if json_part.startswith("json"):
                    json_part = json_part[4:]
                content = json_part.strip()

        return json.loads(content)

    def _format_validation_error(self, error: ValidationError) -> str:
        """Format ValidationError for LLM feedback.

        Args:
            error: The ValidationError to format

        Returns:
            Formatted error message
        """
        errors = error.errors()
        output = ["Your response failed validation:\n"]

        # Group errors by field
        by_field: dict = {}
        for err in errors:
            loc = " -> ".join(str(l) for l in err['loc'])
            if loc not in by_field:
                by_field[loc] = []
            by_field[loc].append(err['msg'])

        # Format grouped errors
        for field, messages in sorted(by_field.items()):
            output.append(f"  - {field}:")
            for msg in messages:
                output.append(f"      {msg}")

        return "\n".join(output)
