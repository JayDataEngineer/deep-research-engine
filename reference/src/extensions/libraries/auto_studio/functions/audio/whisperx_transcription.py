"""WhisperX Transcription Function for Auto Studio.

Provides word-level timestamp transcription for lip-sync alignment.

Usage:
    from src.extensions.libraries.auto_studio.functions.audio.whisperx_transcription import WhisperXTranscriptionFunction

    func = WhisperXTranscriptionFunction(config)
    result = await func.execute(FunctionInput(
        audio_path="output.mp3",
        output_dir="outputs/"
    ))
"""

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ...core.composable import BaseComposableFunction, FunctionInput, FunctionResult

logger = logging.getLogger(__name__)


class WordTimestamp(BaseModel):
    """Single word with timestamp."""
    word: str = Field(description="The word text")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    score: float = Field(default=1.0, description="Confidence score")


class SegmentTimestamp(BaseModel):
    """Segment (phrase) with word-level timestamps."""
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Segment text (full phrase)")
    speaker: Optional[str] = Field(default=None, description="Speaker label if diarization enabled")
    words: list[WordTimestamp] = Field(default_factory=list, description="Word-level timestamps")


class TranscriptionOutput(BaseModel):
    """Output from WhisperX transcription."""
    full_text: str = Field(description="Full transcript text")
    language: str = Field(description="Detected language code")
    duration: float = Field(description="Audio duration in seconds")
    segments: list[SegmentTimestamp] = Field(description="Segments with word-level timestamps")

    # Optional speaker info
    num_speakers: Optional[int] = Field(default=None, description="Number of speakers detected")
    speaker_segments: Optional[list[dict]] = Field(default=None, description="Speaker diarization segments")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()


class WhisperXTranscriptionFunction(BaseComposableFunction):
    """
    WhisperX transcription for word-level timestamps.

    Features:
    - Word-level timestamps for lip-sync
    - Optional speaker diarization
    - Multi-language support

    Input: Audio file path
    Output: Transcription with word-level timestamps
    """

    def __init__(self, config: dict):
        """
        Initialize WhisperX transcription function.

        Args:
            config: Configuration dict with:
                - model: Whisper model size (default: large-v3)
                - device: cuda or cpu
                - compute_type: float16, int8, etc.
                - batch_size: Batch size for processing
                - diarization: Enable speaker diarization
                - min_speakers: Minimum speakers
                - max_speakers: Maximum speakers
                - hf_token: HuggingFace token for diarization model
        """
        self.config = config
        self.model_name = config.get("model", "large-v3")
        self.device = config.get("device", "cuda")
        self.compute_type = config.get("compute_type", "float16")
        self.batch_size = config.get("batch_size", 16)
        self.enable_diarization = config.get("diarization", False)
        self.min_speakers = config.get("min_speakers", None)
        self.max_speakers = config.get("max_speakers", None)

    async def execute(self, input_data: FunctionInput) -> FunctionResult:
        """
        Transcribe audio with WhisperX.

        Args:
            input_data: FunctionInput with:
                - audio_path: Path to audio file
                - output_dir: Optional output directory

        Returns:
            FunctionResult with TranscriptionOutput
        """
        audio_path = input_data.get("audio_path")
        output_dir = input_data.get("output_dir", ".")

        if not audio_path:
            raise ValueError("audio_path is required")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting WhisperX transcription: {audio_path.name}")

        try:
            import whisperx

            # Step 1: Transcribe with WhisperX
            logger.info("Step 1/3: Transcribing audio...")
            device = self.device if self.device == "cuda" else "cpu"

            # Load audio
            import numpy as np

            audio = whisperx.load_audio(str(audio_path))

            # Transcribe
            model = whisperx.WhisperX(
                self.model_name,
                device=device,
                compute_type=self.compute_type,
            )

            result = model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=None,  # Auto-detect
            )

            detected_language = result["language"]
            full_text = " ".join([seg["text"] for seg in result["segments"]])
            duration = result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0

            logger.info(f"Language detected: {detected_language}")
            logger.info(f"Transcription: {len(result.get('segments', []))} segments")

            # Step 2: Align timestamps
            logger.info("Step 2/3: Aligning timestamps...")
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
            )

            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
            )

            # Convert to our format
            segments = []
            for seg in result["segments"]:
                words = []
                if "words" in seg:
                    for word_data in seg["words"]:
                        words.append(WordTimestamp(
                            word=word_data["word"],
                            start=word_data["start"],
                            end=word_data["end"],
                            score=word_data.get("score", 1.0),
                        ))

                segments.append(SegmentTimestamp(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    words=words,
                ))

            # Step 3: Speaker diarization (optional)
            speaker_segments = None
            num_speakers = None
            speaker_assignments = {}  # Maps segment index to speaker

            if self.enable_diarization:
                logger.info("Step 3/3: Running speaker diarization...")
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self._get_hf_token(),
                    device=device,
                )

                diarize_segments = diarize_model(
                    audio,
                    min_speakers=self.min_speakers,
                    max_speakers=self.max_speakers,
                )

                # Assign speakers to segments
                result = whisperx.assign_word_speakers(diarize_segments, result)
                speaker_segments = result.get("segments", [])

                # Update segments with speaker info
                speaker_set = set()
                for i, seg in enumerate(result.get("segments", [])):
                    if "speaker" in seg:
                        speaker_assignments[i] = seg["speaker"]
                        speaker_set.add(seg["speaker"])
                        if i < len(segments):
                            segments[i].speaker = seg["speaker"]

                num_speakers = len(speaker_set)
                logger.info(f"Diarization complete: {num_speakers} speakers")

            # Create output
            output = TranscriptionOutput(
                full_text=full_text,
                language=detected_language,
                duration=round(duration, 2),
                segments=segments,
                num_speakers=num_speakers,
                speaker_segments=speaker_segments,
            )

            # Save output
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{audio_path.stem}_transcription.json"
            output_path.write_text(output.model_dump_json(indent=2))

            logger.info(f"Transcription saved to: {output_path}")
            logger.info(f"Total: {len(segments)} segments, {sum(len(s.words) for s in segments)} words")

            return FunctionResult(
                data=output.to_dict(),
                output_files=[str(output_path)],
                metadata={
                    "language": detected_language,
                    "duration": duration,
                    "num_segments": len(segments),
                    "num_words": sum(len(s.words) for s in segments),
                    "num_speakers": num_speakers,
                },
            )

        except ImportError:
            raise RuntimeError(
                "whisperx is not installed. Install with: "
                "pip install whisperx"
            )
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            raise

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from config or environment."""
        # Check config first
        token = self.config.get("hf_token")
        if token:
            return token

        # Check environment
        import os
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
