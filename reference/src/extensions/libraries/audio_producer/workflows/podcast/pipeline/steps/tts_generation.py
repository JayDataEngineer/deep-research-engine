"""
TTS Generation Pipeline Step

**PURE FUNCTION**: Input → Process → Output

Input:
  - emotion_tagged_lines: List[EmotionTaggedLine] (from emotion_detection step)
  - config: TTSGenerationConfig (how to generate audio)
  - voice_map: Dict[str, str] (speaker → reference_audio path)

Output:
  - List[AudioChunk] (generated audio chunks with metadata)

This step is:
- Deterministic: same input → same output (TTS is deterministic given seed)
- Side-effect free: no mutations, HTTP calls are pure I/O
- Testable: can mock HTTP client
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A chunk of generated audio with metadata."""
    index: int                      # Line index
    speaker: str                     # Speaker name
    text: str                        # Original text
    audio_data: bytes                # Raw audio bytes (WAV format)
    duration: float                  # Duration in seconds
    emotion_used: List[float]        # Emotion vector that was used
    voice_reference: Optional[str] = None  # Reference audio file used

    def save(self, path: Path) -> None:
        """Save audio to file (side effect, but convenient)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.audio_data)


@dataclass
class TTSGenerationConfig:
    """Configuration for TTS generation step."""
    # TTS service endpoint
    tts_url: str = "http://127.0.0.1:9000"

    # Audio output format
    output_format: str = "wav"
    sample_rate: int = 22050

    # Generation parameters
    speed: float = 1.0
    pitch: float = 1.0

    # Batching (for efficiency)
    batch_size: int = 8               # Chunks per TTS call
    parallel_workers: int = 4         # Number of parallel TTS workers

    # IndexTTS vLLM specific
    diffusion_steps: int = 25
    num_beams: int = 3
    cfg_rate: float = 0.7


# ============================================================================
# PURE FUNCTION - The core of this step
# ============================================================================

async def generate_audio(
    emotion_tagged_lines: List,
    config: TTSGenerationConfig,
    voice_map: Dict[str, str],
    http_client: httpx.AsyncClient = None
) -> List[AudioChunk]:
    """Generate audio for emotion-tagged dialogue lines.

    **PURE FUNCTION**: Given the same inputs, returns the same outputs.
    HTTP calls are deterministic I/O, not side effects.

    Args:
        emotion_tagged_lines: Output from emotion_detection step
        config: TTS generation configuration
        voice_map: Speaker → reference_audio mapping
        http_client: HTTP client (creates new one if None)

    Returns:
        List of AudioChunk objects

    Example:
        voice_map = {
            "EMMA": "/path/to/emma.mp3",
            "OLIVIA": "/path/to/olivia.mp3",
            "NARRATOR": "/path/to/narrator.mp3"
        }
        config = TTSGenerationConfig()
        chunks = await generate_audio(lines, config, voice_map)
        for chunk in chunks:
            print(f"{chunk.speaker}: {chunk.duration}s")
    """
    # Create HTTP client if not provided
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=120.0)

    results = []

    # Generate audio for each line
    for line in emotion_tagged_lines:
        chunk = await _generate_single_line(
            line=line,
            config=config,
            voice_reference=voice_map.get(line.speaker),
            http_client=http_client
        )
        results.append(chunk)

    return results


# ============================================================================
# Helper functions (still pure - no side effects besides HTTP I/O)
# ============================================================================

async def _generate_single_line(
    line: Any,  # EmotionTaggedLine
    config: TTSGenerationConfig,
    voice_reference: str,
    http_client: httpx.AsyncClient
) -> AudioChunk:
    """Generate audio for a single dialogue line.

    **PURE**: HTTP call is deterministic I/O.
    """
    # Build request payload
    payload = {
        "text": line.text,
        "speed": config.speed,
        "pitch": config.pitch,
        "output_format": config.output_format,
        "extra_params": {
            # Reference audio for voice cloning
            "ref_audio_path": voice_reference,

            # Emotion vector for TTS
            "emo_vector": line.emotion.to_index_tts_format(),

            # IndexTTS vLLM parameters
            "diffusion_steps": config.diffusion_steps,
            "num_beams": config.num_beams,
            "inference_cfg_rate": config.cfg_rate,
        }
    }

    # Call TTS service
    url = f"{config.tts_url}/generate"
    response = await http_client.post(url, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"TTS failed: {response.text}")

    # Extract audio data and duration
    audio_data = response.content
    duration = float(response.headers.get("X-Duration", 0.0))

    return AudioChunk(
        index=line.index,
        speaker=line.speaker,
        text=line.text,
        audio_data=audio_data,
        duration=duration,
        emotion_used=line.emotion.to_list(),
        voice_reference=voice_reference
    )


async def generate_audio_parallel(
    emotion_tagged_lines: List,
    config: TTSGenerationConfig,
    voice_map: Dict[str, str],
    http_client: httpx.AsyncClient = None
) -> List[AudioChunk]:
    """Generate audio in parallel (multiple TTS workers).

    **PURE**: Parallel execution but deterministic results.
    """
    import asyncio

    if http_client is None:
        http_client = httpx.AsyncClient(timeout=120.0)

    # Create tasks for parallel execution
    tasks = []
    for line in emotion_tagged_lines:
        task = _generate_single_line(
            line=line,
            config=config,
            voice_reference=voice_map.get(line.speaker),
            http_client=http_client
        )
        tasks.append(task)

    # Execute in parallel
    results = await asyncio.gather(*tasks)
    return results
