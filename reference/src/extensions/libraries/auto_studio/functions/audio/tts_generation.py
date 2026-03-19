"""TTS Audio Generation composable function.

Generates podcast audio using IndexTTS vLLM with voice cloning.
Each dialogue line gets emotion-based TTS generation.

Uses the shared ExtensionProcessManager for DRY code - same as Audio Producer.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field

from ...core.composable import BaseComposableFunction, FunctionInput, FunctionResult
from src.extensions.libraries.tts_engine.extension_process_manager import ExtensionProcessManager

logger = logging.getLogger(__name__)


class TTSConfig(BaseModel):
    """Configuration for TTS audio generation function."""
    # ExtensionProcessManager handles port management internally (TTS_EXTENSION_PORT)
    timeout: int = 120
    output_format: str = "wav"
    sample_rate: int = 22050
    output_dir: str = "data/outputs/auto_studio/audio"

    # Generation parameters
    speed: float = 1.0
    pitch: float = 1.0

    # IndexTTS vLLM specific
    diffusion_steps: int = 25
    num_beams: int = 3
    cfg_rate: float = 0.7

    # Parallel processing (vLLM has 16x internal concurrency, so 4 workers is sufficient)
    parallel_workers: int = 4


class TTSGenerationFunction(BaseComposableFunction):
    """Generate podcast audio using IndexTTS vLLM.

    This function:
    1. Takes dialogue lines with speaker info and emotion vectors
    2. Uses IndexTTS vLLM with voice cloning reference audio
    3. Applies emotion vectors for expressive speech
    4. Outputs audio chunks with timing metadata

    Uses the shared ExtensionProcessManager for DRY code - same as Audio Producer.
    """

    name = "tts_generation"
    input_types = ["text", "prompt"]
    output_types = ["audio"]
    requires_gpu = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tts_config = TTSConfig(**config.get("tts", {}))
        self.output_dir = Path(self.tts_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Shared ExtensionProcessManager for DRY code
        self._tts_manager: Optional[ExtensionProcessManager] = None

    def _validate_config(self):
        """Validate TTS configuration (ExtensionProcessManager handles URL internally)."""
        # No validation needed - ExtensionProcessManager manages everything
        pass

    async def _get_tts_manager(self) -> ExtensionProcessManager:
        """Get or create the shared TTS manager (lazy initialization)."""
        if self._tts_manager is None:
            self._tts_manager = ExtensionProcessManager()
        return self._tts_manager

    async def _ensure_extension_loaded(self) -> None:
        """Ensure the IndexTTS vLLM extension is loaded."""
        manager = await self._get_tts_manager()
        # Only start if not already running the correct extension
        if manager.active_extension != "index_tts_vllm":
            if manager.process is not None:
                await manager.stop_current()
            start_result = await manager.start_extension("index_tts_vllm")
            if start_result.get("status") != "ready":
                raise RuntimeError(f"Failed to start IndexTTS vLLM: {start_result.get('message', 'Unknown error')}")

    async def validate_input(self, input_data: FunctionInput) -> Tuple[bool, Optional[str]]:
        """Validate input data.

        Required in plan_data:
        - dialogue_lines: List of dialogue with speaker, text, emotion vectors
        - voice_references: Dict mapping speaker names to reference audio paths
        """
        dialogue_lines = input_data.plan_data.get("dialogue_lines")
        if not dialogue_lines:
            return False, "Missing 'dialogue_lines' in plan_data"

        voice_references = input_data.plan_data.get("voice_references")
        if not voice_references:
            return False, "Missing 'voice_references' in plan_data"

        # Verify voice reference files exist
        for speaker, ref_path in voice_references.items():
            if not Path(ref_path).exists():
                return False, f"Voice reference not found: {ref_path}"

        return True, None

    async def execute(self, input_data: FunctionInput) -> FunctionResult:
        """Execute TTS audio generation.

        Args:
            input_data: Contains dialogue_lines, voice_references

        Returns:
            FunctionResult with audio chunks and timing info
        """
        dialogue_lines = input_data.plan_data.get("dialogue_lines", [])
        voice_references = input_data.plan_data.get("voice_references", {})

        logger.info(f"[TTS] Generating audio for {len(dialogue_lines)} dialogue lines")

        start_time = time.time()

        try:
            # Ensure IndexTTS vLLM extension is loaded
            await self._ensure_extension_loaded()

            # Generate audio chunks
            audio_chunks = await self._generate_audio_parallel(
                dialogue_lines=dialogue_lines,
                voice_references=voice_references
            )

            elapsed = time.time() - start_time
            total_duration = sum(chunk["duration"] for chunk in audio_chunks)

            logger.info(f"[TTS] Generated {len(audio_chunks)} audio chunks in {elapsed:.1f}s (total: {total_duration:.1f}s)")

            # Save audio chunks
            output_paths = {}
            for chunk in audio_chunks:
                output_path = self.output_dir / f"{chunk['clip_id']}_audio.wav"
                chunk_path = self.output_dir / f"{chunk['index']:03d}_{chunk['speaker']}.wav"

                # Save the audio
                with open(chunk_path, 'wb') as f:
                    f.write(chunk["audio_data"])

                output_paths[chunk["clip_id"]] = str(chunk_path)

            return FunctionResult(
                success=True,
                data={
                    "output_paths": output_paths,
                    "audio_chunks": audio_chunks,
                    "timing_info": {
                        "total_chunks": len(audio_chunks),
                        "total_duration": total_duration,
                        "clips_with_timing": [
                            {
                                "clip_id": chunk["clip_id"],
                                "duration": chunk["duration"],
                                "speaker": chunk["speaker"],
                                "text": chunk["text"]
                            }
                            for chunk in audio_chunks
                        ]
                    }
                },
                checkpoint_data={
                    "audio_chunks": audio_chunks,
                    "output_paths": output_paths
                },
                metadata={
                    "num_chunks": len(audio_chunks),
                    "total_duration": total_duration,
                    "elapsed": elapsed
                }
            )

        except Exception as e:
            logger.exception(f"[TTS] Execution error: {e}")
            return FunctionResult(
                success=False,
                error=f"TTS generation failed: {e}",
                metadata={}
            )

    async def _generate_audio_parallel(
        self,
        dialogue_lines: list,
        voice_references: dict
    ) -> list:
        """Generate audio for all dialogue lines in parallel.

        Uses shared ExtensionProcessManager for DRY code - same as Audio Producer.

        Args:
            dialogue_lines: List of dialogue with speaker, text, emotion_vectors
            voice_references: Speaker -> reference audio path mapping

        Returns:
            List of audio chunk dicts with metadata
        """
        manager = await self._get_tts_manager()

        async def generate_single(line: dict, idx: int) -> dict:
            """Generate audio for a single line using ExtensionProcessManager."""
            speaker = line["speaker"]
            text = line["text"]
            emotion_vector = line.get("emotion_vectors", {})
            clip_id = line.get("clip_id", f"clip_{idx:03d}")

            # Get voice reference
            voice_ref = voice_references.get(speaker)
            if not voice_ref:
                logger.warning(f"[TTS] No voice reference for {speaker}, using default")
                voice_ref = ""

            # Convert emotion vector to IndexTTS format [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_vector = self._emotion_to_index_tts(emotion_vector)

            # Build generation params for ExtensionProcessManager
            generation_params = {
                "ref_audio_path": voice_ref,
                "emo_vector": emo_vector,
                "diffusion_steps": self.tts_config.diffusion_steps,
                "num_beams": self.tts_config.num_beams,
                "inference_cfg_rate": self.tts_config.cfg_rate,
            }

            # Call TTS via shared ExtensionProcessManager (DRY)
            audio_data, duration = await manager.generate_audio(
                text=text,
                voice_id=speaker,
                speed=self.tts_config.speed,
                pitch=self.tts_config.pitch,
                output_format=self.tts_config.output_format,
                **generation_params
            )

            logger.debug(f"[TTS] Generated {speaker}: {text[:30]}... ({duration:.2f}s)")

            return {
                "index": idx,
                "clip_id": clip_id,
                "speaker": speaker,
                "text": text,
                "audio_data": audio_data,
                "duration": duration,
                "emotion_used": emotion_vector,
                "voice_reference": voice_ref
            }

        # Generate in parallel with limited concurrency
        tasks = [
            generate_single(line, idx)
            for idx, line in enumerate(dialogue_lines)
        ]

        # Use semaphore to limit concurrent requests (vLLM has internal batching)
        semaphore = asyncio.Semaphore(self.tts_config.parallel_workers)

        async def bounded_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[bounded_task(t) for t in tasks])
        return results

    def _emotion_to_index_tts(self, emotion_vector: dict) -> list:
        """Convert emotion vector to IndexTTS format.

        IndexTTS expects: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

        Args:
            emotion_vector: Dict with emotion values

        Returns:
            List of 8 float values
        """
        return [
            emotion_vector.get("happy", 0.0),
            emotion_vector.get("angry", 0.0),
            emotion_vector.get("sad", 0.0),
            emotion_vector.get("afraid", 0.0),
            emotion_vector.get("disgusted", 0.0),
            emotion_vector.get("melancholic", 0.0),
            emotion_vector.get("surprised", 0.0),
            emotion_vector.get("calm", 0.0),
        ]

    async def cleanup(self) -> None:
        """Cleanup resources - stop the TTS extension if running."""
        if self._tts_manager is not None:
            try:
                await self._tts_manager.stop_current()
                logger.info("[TTS] ExtensionProcessManager stopped")
            except Exception as e:
                logger.warning(f"[TTS] Error stopping extension: {e}")
