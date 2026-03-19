"""Multimodal Worker - Process audio, images, and video content.

This worker handles multimodal content processing:
1. Audio: Transcription with ASR (Automatic Speech Recognition)
2. Images: Analysis with VLM (Vision Language Model)
3. Video: Frame extraction + analysis
4. Face: Face recognition and entity extraction

Key Features:
- ASR transcription for audio files
- VLM analysis for images
- Video frame extraction and analysis
- Face recognition for entity extraction
- Supports multiple file formats

Usage:
    worker = MultimodalWorker(config)
    result = await worker.execute(state, config)
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from ...state import EngineState, WorkerResult
from ..base import BaseWorker, WorkerConfig, WorkerRegistry


@WorkerRegistry.register("multimodal")
class MultimodalWorker(BaseWorker):
    """Multimodal content processor.

    This worker:
    1. Receives a media item (audio, image, video)
    2. Processes based on media type:
       - Audio: ASR transcription
       - Image: VLM analysis
       - Video: Frame extraction + VLM analysis
    3. Returns structured analysis results

    Media Types Supported:
    - audio: mp3, wav, m4a, flac, ogg
    - image: jpg, png, gif, webp, bmp
    - video: mp4, mov, avi, webm, mkv
    """

    worker_type = "multimodal"
    description = "Processes audio, images, and video with ASR/VLM/face recognition"
    required_tools = ["asr_transcribe", "vlm_analyze", "face_recognize"]

    def __init__(self, config: WorkerConfig):
        super().__init__(config)
        self.media_item = config.custom.get("media_item", {})
        self.media_path = config.custom.get("media_path", "")
        self.media_type = config.custom.get("media_type", "auto")

        # Processing options
        self.enable_asr = config.custom.get("enable_asr", True)
        self.enable_vlm = config.custom.get("enable_vlm", True)
        self.enable_face_recognition = config.custom.get("enable_face_recognition", False)
        self.video_frame_interval = config.custom.get("video_frame_interval", 5.0)  # seconds

        # Storage for results
        self.storage_backend = config.custom.get("storage_backend", "mongodb")

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Process multimodal content.

        Args:
            state: Current engine state
            config: Worker configuration

        Returns:
            WorkerResult with analysis results
        """
        self._start_timing()

        try:
            # Get media info from config or state
            media_path = self.media_path or state.get("input_data", {}).get("media_path", "")
            media_type = self.media_type or state.get("input_data", {}).get("media_type", "auto")

            if not media_path:
                return self._create_result(
                    success=False,
                    output=None,
                    error="No media path provided",
                )

            # Auto-detect media type if needed
            if media_type == "auto":
                media_type = self._detect_media_type(media_path)

            # Get YAML config
            yaml_config = state.get("context", {}).get("config", {})

            # Process based on media type
            if media_type == "audio":
                result = await self._process_audio(
                    media_path=media_path,
                    yaml_config=yaml_config,
                )
            elif media_type == "image":
                result = await self._process_image(
                    media_path=media_path,
                    yaml_config=yaml_config,
                )
            elif media_type == "video":
                result = await self._process_video(
                    media_path=media_path,
                    yaml_config=yaml_config,
                )
            else:
                return self._create_result(
                    success=False,
                    output=None,
                    error=f"Unsupported media type: {media_type}",
                )

            self._log(f"Multimodal processing complete: {media_type}")

            return self._create_result(
                success=True,
                output=result,
                output_type=f"multimodal_{media_type}_analysis",
                confidence=result.get("confidence", 0.8),
                sources=[{"path": media_path, "type": media_type}],
            )

        except Exception as e:
            self._log(f"Multimodal processing failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )

    async def _process_audio(
        self,
        media_path: str,
        yaml_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Process audio file with ASR.

        Args:
            media_path: Path to audio file
            yaml_config: LLM configuration

        Returns:
            Transcription and metadata
        """
        result = {
            "media_type": "audio",
            "media_path": media_path,
            "processed_at": datetime.now().isoformat(),
        }

        # Transcribe with ASR
        if self.enable_asr:
            transcription = await self._transcribe_audio(media_path)
            result["transcription"] = transcription
            result["confidence"] = transcription.get("confidence", 0.8)

        return result

    async def _process_image(
        self,
        media_path: str,
        yaml_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Process image with VLM.

        Args:
            media_path: Path to image file
            yaml_config: LLM configuration

        Returns:
            Image analysis and metadata
        """
        result = {
            "media_type": "image",
            "media_path": media_path,
            "processed_at": datetime.now().isoformat(),
        }

        # Analyze with VLM
        if self.enable_vlm:
            analysis = await self._analyze_image(media_path)
            result["analysis"] = analysis
            result["confidence"] = analysis.get("confidence", 0.85)

        # Face recognition
        if self.enable_face_recognition:
            faces = await self._recognize_faces(media_path)
            result["faces"] = faces

        return result

    async def _process_video(
        self,
        media_path: str,
        yaml_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Process video by extracting frames and analyzing.

        Args:
            media_path: Path to video file
            yaml_config: LLM configuration

        Returns:
            Video analysis and metadata
        """
        result = {
            "media_type": "video",
            "media_path": media_path,
            "processed_at": datetime.now().isoformat(),
        }

        # Extract frames
        frames = await self._extract_video_frames(
            media_path,
            interval=self.video_frame_interval,
        )
        result["frames_extracted"] = len(frames)
        result["frame_paths"] = frames

        # Analyze frames with VLM
        if self.enable_vlm and frames:
            frame_analyses = await asyncio.gather(
                *[
                    self._analyze_image(frame_path)
                    for frame_path in frames[:10]  # Limit to 10 frames
                ]
            )
            result["frame_analyses"] = frame_analyses

            # Combine analyses into summary
            combined_description = "\n".join(
                [a.get("description", "") for a in frame_analyses if a]
            )
            result["combined_description"] = combined_description
            result["confidence"] = 0.8

        return result

    async def _transcribe_audio(
        self,
        media_path: str,
    ) -> dict[str, Any]:
        """Transcribe audio file with ASR.

        Args:
            media_path: Path to audio file

        Returns:
            Transcription dict
        """
        try:
            # Use whisper or other ASR service
            from src.shared.tools.multimodal.asr import transcribe_audio

            transcription = await transcribe_audio(media_path)

            return {
                "text": transcription.get("text", ""),
                "language": transcription.get("language", "en"),
                "duration": transcription.get("duration", 0),
                "confidence": transcription.get("confidence", 0.85),
                "segments": transcription.get("segments", []),
            }

        except ImportError:
            # Fallback: return placeholder
            self._log("ASR not available, returning placeholder", level="warning")
            return {
                "text": "[ASR transcription not available]",
                "confidence": 0.0,
            }

        except Exception as e:
            self._log(f"ASR transcription failed: {e}", level="warning")
            return {
                "text": "",
                "error": str(e),
                "confidence": 0.0,
            }

    async def _analyze_image(
        self,
        media_path: str,
    ) -> dict[str, Any]:
        """Analyze image with VLM.

        Args:
            media_path: Path to image file

        Returns:
            Analysis dict
        """
        try:
            # Use VLM for image analysis
            from src.shared.tools.multimodal.vlm import analyze_image

            analysis = await analyze_image(
                media_path,
                prompt="Describe this image in detail, including any text, people, objects, and context.",
            )

            return {
                "description": analysis.get("description", ""),
                "objects": analysis.get("objects", []),
                "text": analysis.get("extracted_text", ""),
                "confidence": analysis.get("confidence", 0.85),
            }

        except ImportError:
            # Fallback: return placeholder
            self._log("VLM not available, returning placeholder", level="warning")
            return {
                "description": "[VLM analysis not available]",
                "confidence": 0.0,
            }

        except Exception as e:
            self._log(f"VLM analysis failed: {e}", level="warning")
            return {
                "description": "",
                "error": str(e),
                "confidence": 0.0,
            }

    async def _recognize_faces(
        self,
        media_path: str,
    ) -> list[dict[str, Any]]:
        """Recognize faces in image.

        Args:
            media_path: Path to image file

        Returns:
            List of face recognition results
        """
        try:
            from src.shared.tools.multimodal.face_recognition import recognize_faces

            faces = await recognize_faces(media_path)

            return [
                {
                    "name": face.get("name", "Unknown"),
                    "confidence": face.get("confidence", 0.0),
                    "bounding_box": face.get("bounding_box", {}),
                }
                for face in faces
            ]

        except ImportError:
            self._log("Face recognition not available", level="warning")
            return []

        except Exception as e:
            self._log(f"Face recognition failed: {e}", level="warning")
            return []

    async def _extract_video_frames(
        self,
        media_path: str,
        interval: float = 5.0,
    ) -> list[str]:
        """Extract frames from video.

        Args:
            media_path: Path to video file
            interval: Seconds between frames

        Returns:
            List of frame paths
        """
        try:
            from src.shared.tools.multimodal.video import extract_frames

            frames = await extract_frames(media_path, interval=interval)

            return frames

        except ImportError:
            self._log("Video frame extraction not available", level="warning")
            return []

        except Exception as e:
            self._log(f"Frame extraction failed: {e}", level="warning")
            return []

    def _detect_media_type(self, path: str) -> str:
        """Detect media type from file extension.

        Args:
            path: File path

        Returns:
            Media type string
        """
        ext = Path(path).suffix.lower()

        audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
        video_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"}

        if ext in audio_exts:
            return "audio"
        elif ext in image_exts:
            return "image"
        elif ext in video_exts:
            return "video"
        else:
            return "unknown"
