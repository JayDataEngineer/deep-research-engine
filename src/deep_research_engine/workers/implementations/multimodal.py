"""Multimodal Worker - Process audio, images, and video content."""

from pathlib import Path
from typing import Any

from ...shared.multimodal import asr_transcribe, face_recognize, video_analyze, vlm_analyze
from ...state import EngineState, WorkerResult
from ...utils.logging import get_loguru_logger
from ...utils.nlp import extract_entities
from ..base import BaseWorker, WorkerConfig, WorkerRegistry

logger = get_loguru_logger(__name__)


@WorkerRegistry.register("multimodal")
class MultimodalWorker(BaseWorker):
    """Multimodal content processor using LiteLLM/CompreFace."""

    worker_type = "multimodal"
    description = "Processes audio, images, and video with ASR/VLM/face recognition"

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        self._start_timing()
        
        media_path_str = config.custom.get("media_path") or state.get("context", {}).get("media_path")
        if not media_path_str:
            return self._create_result(success=False, error="No media path provided")
            
        media_path = Path(media_path_str)
        media_type = config.custom.get("media_type", "auto")
        
        if media_type == "auto":
            media_type = self._detect_media_type(media_path)
            
        logger.info(f"Processing {media_type}: {media_path}")
        
        try:
            result: dict[str, Any] = {
                "media_path": str(media_path),
                "media_type": media_type,
            }
            
            if media_type == "audio":
                transcript = await asr_transcribe(media_path)
                result["transcription"] = transcript
                # Extract entities from transcript
                entities = await extract_entities(transcript)
                result.update(entities)
                
            elif media_type == "image":
                desc = await vlm_analyze(media_path)
                faces = await face_recognize(media_path)
                result["description"] = desc
                result["faces"] = faces
                # Extract entities from description
                entities = await extract_entities(desc)
                result.update(entities)
                
            elif media_type == "video":
                video_desc = await video_analyze(media_path)
                result["description"] = video_desc
                # Extract entities from description
                entities = await extract_entities(video_desc)
                result.update(entities)
            
            else:
                return self._create_result(success=False, error=f"Unsupported media type: {media_type}")

            return self._create_result(
                success=True,
                output=result,
                output_type=f"multimodal_{media_type}_analysis"
            )

        except Exception as e:
            logger.error(f"MultimodalWorker failed: {e}")
            return self._create_result(success=False, error=str(e))

    def _detect_media_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in {".mp3", ".wav", ".ogg", ".m4a"}: return "audio"
        if ext in {".jpg", ".jpeg", ".png", ".webp"}: return "image"
        if ext in {".mp4", ".mov", ".webm"}: return "video"
        return "unknown"
