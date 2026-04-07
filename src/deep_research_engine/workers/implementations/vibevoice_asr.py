"""VibeVoice ASR Worker - Advanced speech recognition with built-in diarization.

VibeVoice ASR-7B provides:
- Transcription + Speaker Diarization + Timestamps in a single pass
- No need for separate pyannote.audio or alignment steps
- Accurate speaker separation with precise temporal alignment
- Support for audio up to 60 minutes

Reference: VibeVoice ASR-7B via LiteLLM OpenAI-compatible endpoint
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

from ...state import WorkerResult
from ..base import BaseWorker, WorkerConfig, WorkerRegistry

# Configuration from environment
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://litellm.local/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-litellm-master")
ASR_MODEL = os.environ.get("ASR_MODEL", "vibevoice-asr")
VIBEVOICE_MIN_SPEAKERS = int(os.environ.get("VIBEVOICE_MIN_SPEAKERS", "1"))
VIBEVOICE_MAX_SPEAKERS = int(os.environ.get("VIBEVOICE_MAX_SPEAKERS", "10"))


@WorkerRegistry.register("vibevoice_asr")
class VibeVoiceASRWorker(BaseWorker):
    """VibeVoice ASR worker for transcription with native diarization.

    This worker leverages VibeVoice ASR-7B's built-in capabilities to provide:
    - High-quality transcription
    - Speaker diarization (who spoke when)
    - Precise timestamps for each segment
    - All in a single API call

    Args:
        config: Worker configuration with optional custom parameters:
            - min_speakers: Minimum number of speakers to detect (default: 1)
            - max_speakers: Maximum number of speakers to detect (default: 10)
            - language: Optional language hint for ASR
            - response_format: "verbose_json" for full diarization (default)
    """

    worker_type = "vibevoice_asr"
    description = "Advanced ASR with built-in speaker diarization via VibeVoice ASR-7B"

    async def execute(self, state: dict[str, Any], config: WorkerConfig) -> WorkerResult:
        """Execute VibeVoice ASR on audio file.

        Args:
            state: Current workflow state
            config: Worker configuration with audio_path and optional parameters

        Returns:
            WorkerResult with transcription, diarization, and timestamps
        """
        self._start_timing()

        # Get audio path from config or state context
        context = state.get("context", {})
        audio_path_str = config.custom.get("audio_path") or context.get("audio_path")
        if not audio_path_str:
            return self._create_result(
                success=False,
                error="No audio_path provided in config.custom or state.context"
            )

        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return self._create_result(
                success=False,
                error=f"Audio file not found: {audio_path}"
            )

        # Get configuration parameters
        min_speakers = config.custom.get("min_speakers", VIBEVOICE_MIN_SPEAKERS)
        max_speakers = config.custom.get("max_speakers", VIBEVOICE_MAX_SPEAKERS)
        language = config.custom.get("language")
        response_format = config.custom.get("response_format", "verbose_json")

        logger.info(
            f"Processing audio with VibeVoice ASR: {audio_path.name} "
            f"(speakers: {min_speakers}-{max_speakers}, language: {language or 'auto'})"
        )

        try:
            # Initialize OpenAI client for LiteLLM endpoint
            client = OpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
            )

            # Call VibeVoice ASR via LiteLLM
            # VibeVoice returns enhanced verbose_json with diarization
            result = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model=ASR_MODEL,
                file=open(audio_path, "rb"),
                response_format=response_format,
                language=language,
            )

            # Parse the response
            # VibeVoice returns segments with speaker information
            transcription_data = self._parse_vibevoice_response(
                result,
                min_speakers,
                max_speakers
            )

            logger.info(
                f"VibeVoice ASR complete: {len(transcription_data['segments'])} segments, "
                f"{transcription_data['speakers']} speakers detected"
            )

            return self._create_result(
                success=True,
                output=transcription_data,
                output_type="vibevoice_asr_transcription",
                metadata={
                    "duration": transcription_data.get("duration", 0),
                    "speakers": transcription_data["speakers"],
                    "segments_count": len(transcription_data["segments"]),
                    "model": ASR_MODEL,
                    "diarization": "native",
                }
            )

        except Exception as e:
            logger.error(f"VibeVoice ASR failed for {audio_path}: {e}")
            return self._create_result(
                success=False,
                error=str(e),
                metadata={"audio_path": str(audio_path), "model": ASR_MODEL}
            )

    def _parse_vibevoice_response(
        self,
        result: Any,
        min_speakers: int,
        max_speakers: int
    ) -> dict[str, Any]:
        """Parse VibeVoice ASR response into structured format.

        VibeVoice returns a response similar to OpenAI's verbose_json format
        with additional diarization information in segments.

        Expected response structure:
        {
            "text": "Full transcription text",
            "segments": [
                {
                    "speaker": "Speaker 0",
                    "start": 0.0,
                    "end": 2.5,
                    "duration": 2.5,
                    "text": "Transcript for this segment"
                },
                ...
            ],
            "speakers": 2,
            "duration": 120.5
        }

        Args:
            result: Raw response from VibeVoice ASR API
            min_speakers: Minimum speakers configuration
            max_speakers: Maximum speakers configuration

        Returns:
            Structured transcription data with diarization
        """
        # Extract full text
        full_text = getattr(result, "text", "")

        # Extract segments with speaker information
        segments_raw = getattr(result, "segments", [])
        segments: list[dict[str, Any]] = []

        speakers_set = set()

        for seg in segments_raw:
            # Handle both dict and object segments
            if hasattr(seg, "__dict__"):
                seg_dict = seg.__dict__
            else:
                seg_dict = seg

            speaker = seg_dict.get("speaker", "Speaker 0")
            speakers_set.add(speaker)

            segments.append({
                "speaker": speaker,
                "start": float(seg_dict.get("start", 0.0)),
                "end": float(seg_dict.get("end", 0.0)),
                "duration": float(seg_dict.get("duration", 0.0)),
                "text": seg_dict.get("text", ""),
                "confidence": seg_dict.get("confidence", 1.0),
            })

        # Count unique speakers
        num_speakers = len(speakers_set)
        if num_speakers == 0:
            num_speakers = 1  # At least one speaker

        # Get duration
        duration = getattr(result, "duration", 0.0)
        if duration == 0.0 and segments:
            # Calculate from segments if not provided
            duration = max(seg["end"] for seg in segments)

        return {
            "text": full_text,
            "segments": segments,
            "speakers": num_speakers,
            "speaker_list": sorted(list(speakers_set)),
            "duration": duration,
            "diarization": "native",
            "model": ASR_MODEL,
        }


@WorkerRegistry.register("asr_worker")
class ASRWorker(VibeVoiceASRWorker):
    """Backward-compatible ASR worker that uses VibeVoice by default.

    This is an alias for VibeVoiceASRWorker to maintain compatibility
    with existing configs that use worker_type: asr_worker.

    To use Whisper instead of VibeVoice, set ASR_MODEL=whisper-1 in .env
    or configure the worker with custom.model = "whisper-1".
    """

    worker_type = "asr_worker"
    description = "ASR worker (defaults to VibeVoice ASR with diarization)"

    async def execute(self, state: dict[str, Any], config: WorkerConfig) -> WorkerResult:
        """Execute ASR using configured model (VibeVoice by default)."""
        # Allow model override via config
        custom_model = config.custom.get("model")
        if custom_model:
            global ASR_MODEL
            original_model = ASR_MODEL
            ASR_MODEL = custom_model
            try:
                result = await super().execute(state, config)
            finally:
                ASR_MODEL = original_model
            return result

        return await super().execute(state, config)
