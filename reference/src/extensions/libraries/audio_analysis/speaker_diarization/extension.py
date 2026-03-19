"""
Speaker Diarification Extension.

Identifies speakers and their timestamps in audio files using pyannote.audio.

Usage:
    from src.extensions.analysis.speaker_diarization.extension import register_extension
    extension = register_extension()
    result = extension.analyze_audio("audio.wav")
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DiarizationResult:
    """Result from speaker diarification analysis."""

    def __init__(
        self,
        speakers: list[dict],
        num_speakers: int,
        duration: float,
        metadata: dict | None = None,
    ):
        self.speakers = speakers  # List of {start, end, speaker, confidence}
        self.num_speakers = num_speakers
        self.duration = duration
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "speakers": self.speakers,
            "num_speakers": self.num_speakers,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class TranscriptionResult:
    """Result from audio transcription using faster-whisper."""

    def __init__(
        self,
        text: str,
        segments: list[dict],
        language: str,
        duration: float,
        metadata: dict | None = None,
    ):
        self.text = text  # Full transcript text
        self.segments = segments  # List of {start, end, text}
        self.language = language  # Detected language
        self.duration = duration  # Audio duration
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "segments": self.segments,
            "language": self.language,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class DiarizationWithTranscriptionResult:
    """Combined result from speaker diarization + transcription."""

    def __init__(
        self,
        segments: list[dict],
        num_speakers: int,
        duration: float,
        full_transcript: str,
        language: str,
        metadata: dict | None = None,
    ):
        # Each segment has: {start, end, speaker, text, confidence}
        self.segments = segments
        self.num_speakers = num_speakers
        self.duration = duration
        self.full_transcript = full_transcript
        self.language = language
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "segments": self.segments,
            "num_speakers": self.num_speakers,
            "duration": self.duration,
            "full_transcript": self.full_transcript,
            "language": self.language,
            "metadata": self.metadata,
        }

    def get_transcript_by_speaker(self) -> dict[str, str]:
        """Get transcript grouped by speaker."""
        speaker_transcripts = {}

        for segment in self.segments:
            speaker = segment["speaker"]
            text = segment["text"]

            if speaker not in speaker_transcripts:
                speaker_transcripts[speaker] = []

            speaker_transcripts[speaker].append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
            })

        # Join each speaker's segments
        return {
            speaker: " ".join([s["text"] for s in segments])
            for speaker, segments in speaker_transcripts.items()
        }


class SpeakerDiarizationExtension:
    """Speaker diarification extension using pyannote.audio."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._model = None
        self._pipeline = None
        self._whisper_model = None  # For transcription

    @property
    def name(self) -> str:
        return "speaker_diarization"

    @property
    def display_name(self) -> str:
        return "Speaker Diarification"

    def _load_pipeline(self):
        """Load the diarification pipeline (lazy loading)."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from pyannote.audio import Pipeline

            # Build pipeline from pretrained model
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._get_hf_token(),
            )

            # Move to GPU if available and configured
            if self.config.get("use_gpu", True):
                import torch

                if torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
                    logger.info("Diarification pipeline moved to GPU")

            self._pipeline = pipeline
            return self._pipeline

        except ImportError:
            logger.error("pyannote.audio not installed")
            raise RuntimeError("pyannote.audio is required for speaker diarification")
        except Exception as e:
            logger.error(f"Failed to load diarification pipeline: {e}")
            raise

    def _get_hf_token(self) -> str | None:
        """Get HuggingFace token from config or environment."""
        # Check config first
        token = self.config.get("hf_token")
        if token:
            return token

        # Check environment
        import os

        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def _load_whisper_model(self):
        """Load the faster-whisper model for transcription (lazy loading)."""
        if self._whisper_model is not None:
            return self._whisper_model

        try:
            from faster_whisper import WhisperModel

            model_size = self.config.get("whisper_model_size", "base")
            device = "cuda" if self.config.get("use_gpu", True) else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            logger.info(f"Loading Whisper model: {model_size} on {device}")
            self._whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            return self._whisper_model

        except ImportError:
            logger.error("faster-whisper not installed")
            raise RuntimeError("faster-whisper is required for transcription")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio file using faster-whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (default: auto-detect)

        Returns:
            TranscriptionResult with full transcript and segments
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting transcription: {audio_path.name}")

        try:
            import librosa

            # Get audio duration
            duration = librosa.get_duration(path=str(audio_path))

            # Load whisper model
            model = self._load_whisper_model()

            # Run transcription
            segments_gen = model.transcribe(
                str(audio_path),
                beam_size=5,
                language=language,
                vad_filter=True,  # Enable voice activity detection
            )

            # Collect segments and full text
            segments_list = []
            full_text_parts = []
            detected_language = "en"  # default

            for segment in segments_gen:
                segments_list.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                })
                full_text_parts.append(segment.text.strip())

            # Get detected language from segments info
            if hasattr(segments_gen, "language"):
                detected_language = segments_gen.language

            result = TranscriptionResult(
                text=" ".join(full_text_parts),
                segments=segments_list,
                language=detected_language,
                duration=round(duration, 2),
                metadata={
                    "model": f"faster-whisper-{self.config.get('whisper_model_size', 'base')}",
                },
            )

            logger.info(f"Transcription complete: {len(segments_list)} segments, language={detected_language}")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def analyze_with_transcription(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        language: str | None = None,
    ) -> DiarizationWithTranscriptionResult:
        """
        Analyze audio file for both speaker diarization AND transcription.

        Combines pyannote.audio (speaker detection) with faster-whisper (transcription)
        to produce a complete transcript with speaker labels.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers (default: from config)
            max_speakers: Maximum number of speakers (default: from config)
            language: Language code for transcription (default: auto-detect)

        Returns:
            DiarizationWithTranscriptionResult with speaker-labeled transcript
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use config defaults if not specified
        min_speakers = min_speakers or self.config.get("min_speakers", 1)
        max_speakers = max_speakers or self.config.get("max_speakers", 10)

        logger.info(f"Starting combined analysis: {audio_path.name}")
        logger.info(f"Speaker range: {min_speakers}-{max_speakers}")

        try:
            import librosa

            # Get audio duration
            duration = librosa.get_duration(path=str(audio_path))

            # Step 1: Run transcription with faster-whisper
            logger.info("Step 1/3: Running transcription...")
            transcription = self.transcribe_audio(audio_path, language)

            # Step 2: Run speaker diarization with pyannote
            logger.info("Step 2/3: Running speaker diarization...")
            diarization = self.analyze_audio(audio_path, min_speakers, max_speakers)

            # Step 3: Align transcription segments with speaker labels
            logger.info("Step 3/3: Aligning transcripts with speakers...")
            aligned_segments = self._align_transcription_with_speakers(
                transcription.segments,
                diarization.speakers,
            )

            # Get full transcript
            full_transcript = " ".join([s["text"] for s in aligned_segments])

            # Get unique speakers
            speaker_set = set(segment["speaker"] for segment in aligned_segments)

            result = DiarizationWithTranscriptionResult(
                segments=aligned_segments,
                num_speakers=len(speaker_set),
                duration=round(duration, 2),
                full_transcript=full_transcript,
                language=transcription.language,
                metadata={
                    "diarization_model": "pyannote/speaker-diarization-3.1",
                    "transcription_model": f"faster-whisper-{self.config.get('whisper_model_size', 'base')}",
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                },
            )

            logger.info(f"Combined analysis complete: {result.num_speakers} speakers, {len(aligned_segments)} segments")
            return result

        except Exception as e:
            logger.error(f"Combined analysis failed: {e}")
            raise

    def _align_transcription_with_speakers(
        self,
        transcription_segments: list[dict],
        speaker_segments: list[dict],
    ) -> list[dict]:
        """
        Align transcription segments with speaker labels.

        For each transcription segment, find the speaker who was speaking
        during that time window based on the diarization results.

        Args:
            transcription_segments: List of {start, end, text} from whisper
            speaker_segments: List of {start, end, speaker} from pyannote

        Returns:
            List of aligned segments with {start, end, speaker, text}
        """
        aligned = []

        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            trans_mid = (trans_start + trans_end) / 2

            # Find speaker segment that overlaps with the midpoint of this transcription segment
            best_speaker = "SPEAKER_UNKNOWN"
            max_overlap = 0

            for spk_seg in speaker_segments:
                spk_start = spk_seg["start"]
                spk_end = spk_seg["end"]

                # Check if transcription segment overlaps with speaker segment
                if trans_mid >= spk_start and trans_mid <= spk_end:
                    # Midpoint is inside this speaker segment
                    best_speaker = spk_seg["speaker"]
                    break

                # Calculate overlap
                overlap_start = max(trans_start, spk_start)
                overlap_end = min(trans_end, spk_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = spk_seg["speaker"]

            aligned.append({
                "start": trans_start,
                "end": trans_end,
                "speaker": best_speaker,
                "text": trans_seg["text"],
                "confidence": 1.0,  # Could be enhanced with actual confidence
            })

        return aligned

    def analyze_audio(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """
        Analyze audio file for speaker diarification.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers (default: from config)
            max_speakers: Maximum number of speakers (default: from config)

        Returns:
            DiarizationResult with speaker segments
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use config defaults if not specified
        min_speakers = min_speakers or self.config.get("min_speakers", 1)
        max_speakers = max_speakers or self.config.get("max_speakers", 10)

        logger.info(f"Starting diarification: {audio_path.name}")
        logger.info(f"Speaker range: {min_speakers}-{max_speakers}")

        try:
            import librosa

            # Get audio duration
            duration = librosa.get_duration(path=str(audio_path))

            # Load pipeline
            pipeline = self._load_pipeline()

            # Apply configuration
            pipeline.parameters = {
                "clustering": {
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                }
            }

            # Run diarification
            diarization = pipeline(str(audio_path))

            # Convert to our result format
            speakers = []
            speaker_set = set()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": str(speaker),
                    "confidence": 1.0,  # pyannote doesn't provide per-segment confidence
                }
                speakers.append(segment)
                speaker_set.add(str(speaker))

            result = DiarizationResult(
                speakers=speakers,
                num_speakers=len(speaker_set),
                duration=round(duration, 2),
                metadata={
                    "model": "pyannote/speaker-diarization-3.1",
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                },
            )

            logger.info(f"Diarification complete: {result.num_speakers} speakers, {len(speakers)} segments")
            return result

        except Exception as e:
            logger.error(f"Diarification failed: {e}")
            raise

    def get_speaker_timeline(
        self, result: DiarizationResult, resolution: float = 1.0
    ) -> dict[str, list]:
        """
        Convert diarification result to timeline format.

        Args:
            result: DiarizationResult from analyze_audio
            resolution: Time resolution in seconds (default: 1.0)

        Returns:
            Dict mapping speaker to list of (time, speaking) tuples
        """
        timeline = {}
        duration = result.duration

        # Create timeline with specified resolution
        for i in range(int(duration / resolution) + 1):
            time = i * resolution
            for segment in result.speakers:
                if segment["start"] <= time < segment["end"]:
                    speaker = segment["speaker"]
                    if speaker not in timeline:
                        timeline[speaker] = []
                    timeline[speaker].append(time)

        return timeline

    def get_speaker_stats(self, result: DiarizationResult) -> dict:
        """Calculate statistics for each speaker."""
        stats = {}

        for segment in result.speakers:
            speaker = segment["speaker"]
            duration = segment["end"] - segment["start"]

            if speaker not in stats:
                stats[speaker] = {
                    "total_duration": 0,
                    "segment_count": 0,
                    "segments": [],
                }

            stats[speaker]["total_duration"] += duration
            stats[speaker]["segment_count"] += 1
            stats[speaker]["segments"].append(
                {"start": segment["start"], "end": segment["end"], "duration": duration}
            )

        # Sort speakers by total duration
        for speaker in stats:
            stats[speaker]["segments"].sort(key=lambda x: x["start"])
            stats[speaker]["avg_segment_length"] = (
                stats[speaker]["total_duration"] / stats[speaker]["segment_count"]
            )

        return stats


def register_extension(config: dict | None = None) -> SpeakerDiarizationExtension:
    """Register the speaker diarification extension."""
    return SpeakerDiarizationExtension(config)


# Convenience function for quick usage
def analyze_speakers(
    audio_path: str | Path,
    min_speakers: int = 1,
    max_speakers: int = 10,
    use_gpu: bool = True,
) -> DiarizationResult:
    """
    Quick function to analyze speakers in audio.

    Args:
        audio_path: Path to audio file
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        use_gpu: Whether to use GPU if available

    Returns:
        DiarizationResult
    """
    config = {"use_gpu": use_gpu, "min_speakers": min_speakers, "max_speakers": max_speakers}
    extension = register_extension(config)
    return extension.analyze_audio(audio_path)
