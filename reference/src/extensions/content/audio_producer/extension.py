"""Audio Producer Extension - Thin wrapper for Audio Producer Library.

This extension provides a simple interface for audio production workflows:
- Podcast: Multi-speaker markdown to audio
- Simple: Text to speech with emotion
- Audiobook: (Future)

It uses the Audio Producer Library's workflows.
For advanced usage, use the Audio Producer Library directly.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.extensions.core.base import LogicExtension

logger = logging.getLogger(__name__)


class AudioProducerExtension(LogicExtension):
    """Audio Producer extension - Thin wrapper for Audio Producer Library.

    This extension uses workflows from the Audio Producer Library:
    - PodcastDirector: Multi-speaker podcast production
    - Future workflows can be added as needed

    All the heavy lifting is done by the library workflows.
    Configuration is done via config.yaml which gets converted to AudioJobConfig.
    """

    def __init__(self, config: dict, root_path: Path):
        super().__init__(config, root_path)
        self._director = None
        self._router = None

    async def initialize(self, context):
        """Initialize the audio producer extension."""
        await super().initialize(context)
        logger.info("Audio Producer extension initialized")

    async def shutdown(self):
        """Cleanup resources."""
        if self._director:
            await self._director.cleanup()

    @property
    def router(self):
        """Get the API router for this extension."""
        if self._router is None:
            from .api import router
            self._router = router
        return self._router

    def _load_config(self) -> Any:
        """Load and convert config to AudioJobConfig."""
        from src.extensions.libraries.audio_producer.config_schemas import AudioJobConfig, VoiceConfig, VoiceMap

        # Convert legacy config format to AudioJobConfig
        processing = self.config.get("processing", {})

        # Determine workflow from config or default to podcast
        workflow_id = processing.get("workflow", "podcast")

        # Build AudioJobConfig
        config_data = {
            "workflow_id": workflow_id,
            "emotion": {
                **processing.get("emotion_detection", {}),
            },
            "sampling": {
                **processing.get("generation", {}),
            },
            "chunking": {
                **processing.get("chunking", {}),
            },
            "output": {
                **self.config.get("output", {}),
            },
        }

        # Convert voices config - directly create a VoiceMap
        voices = processing.get("voices", {})
        voice_map_dict = {}
        for speaker, voice_config in voices.items():
            if isinstance(voice_config, str):
                voice_map_dict[speaker] = VoiceConfig(voice_id=voice_config)
            elif isinstance(voice_config, dict):
                voice_map_dict[speaker] = VoiceConfig(**voice_config)

        config_data["voices"] = VoiceMap(voices=voice_map_dict)

        return AudioJobConfig(**config_data)

    async def process_audio(
        self,
        input_text: str,
        output_audio_path: Optional[Path] = None,
    ) -> dict:
        """Process text to generate audio using the configured workflow.

        Args:
            input_text: Text to process (markdown for podcast, plain text for simple)
            output_audio_path: Optional path for output audio file

        Returns:
            Dict with audio_path, duration, num_lines, processing_time
        """
        from src.extensions.libraries.audio_producer import PodcastDirector, create_podcast_director

        # Load config
        config = self._load_config()

        # Create appropriate director based on workflow
        prompts_dir = self.root_path / "prompts"

        if config.workflow_id == "podcast":
            self._director = create_podcast_director(config, str(prompts_dir))
        else:
            # Default to podcast for now
            self._director = create_podcast_director(config, str(prompts_dir))

        # Execute workflow
        result = await self._director.execute(input_text, output_audio_path)

        return result

    # Process method for JobExtension compatibility
    async def process(self, input_data: dict, **kwargs):
        """Process text to generate audio.

        Args:
            input_data: Dict with 'text' or 'file_path' key
            **kwargs: Additional parameters including:
                - output_audio_path: Optional path for output audio
                - workflow: Optional workflow override

        Yields:
            Progress updates and final result
        """
        # Get input text
        text = input_data.get("text")
        file_path_str = input_data.get("file_path")

        if text:
            input_text = text
        elif file_path_str:
            file_path = Path(file_path_str)
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
            input_text = file_path.read_text()
        else:
            raise ValueError("input_data must contain 'text' or 'file_path'")

        output_audio_path = kwargs.get("output_audio_path")

        yield json.dumps({"status": "processing", "message": "Processing audio..."})

        result = await self.process_audio(input_text, output_audio_path)

        yield json.dumps({
            "status": "complete",
            "audio_path": result["audio_path"],
            "duration": result["duration"],
            "num_lines": result["num_lines"],
            "processing_time": result["processing_time"],
        })

from src.extensions.core.factory import register_extension_factory
register_extension = register_extension_factory(AudioProducerExtension)
