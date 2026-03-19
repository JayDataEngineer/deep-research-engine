"""
Audio processing utilities for audio production.

This module provides common audio processing functionality used across
different audio production workflows.
"""

import io
import logging
import wave

import numpy as np

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = None


def _check_torch_available():
    """Check if PyTorch is available."""
    global TORCH_AVAILABLE
    if TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False
        except (OSError, ValueError):
            TORCH_AVAILABLE = False
    return TORCH_AVAILABLE


class AudioProcessor:
    """Utility class for common audio processing operations."""

    @staticmethod
    def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        try:
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)

            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data.tobytes())

            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error(f"Error converting numpy to WAV: {str(e)}")
            raise RuntimeError(f"Audio conversion failed: {str(e)}") from e

    @staticmethod
    def torch_to_wav_bytes(waveform, sample_rate: int) -> bytes:
        """Convert torch tensor to WAV bytes."""
        if not _check_torch_available():
            raise RuntimeError("PyTorch is not available for audio processing")

        try:
            import torch
            import torchaudio

            if not isinstance(waveform, torch.Tensor):
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)
                else:
                    raise ValueError("Input must be a torch tensor or numpy array")

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            buffer = io.BytesIO()
            torchaudio.save(buffer, waveform, sample_rate, format="wav")
            buffer.seek(0)

            return buffer.read()
        except Exception as e:
            logger.error(f"Error converting torch tensor to WAV: {str(e)}")
            raise RuntimeError(f"Audio conversion failed: {str(e)}") from e

    @staticmethod
    def get_wav_duration(wav_bytes: bytes) -> float:
        """Calculate duration of WAV audio in seconds."""
        try:
            buffer = io.BytesIO(wav_bytes)
            with wave.open(buffer, "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.error(f"Could not determine WAV duration: {str(e)}")
            return 1.0

    @staticmethod
    def concatenate_audio(audio_chunks: list) -> np.ndarray:
        """Concatenate multiple audio chunks."""
        try:
            if not audio_chunks:
                raise ValueError("No audio chunks provided")
            return np.concatenate(audio_chunks)
        except Exception as e:
            logger.error(f"Error concatenating audio: {str(e)}")
            raise RuntimeError(f"Audio concatenation failed: {str(e)}") from e
