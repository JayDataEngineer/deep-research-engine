"""
Audio processing utilities for TTS extensions.

This module provides common audio processing functionality used across
different TTS extensions to reduce code duplication.
"""

import io
import logging
import wave

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch will be imported dynamically in functions that need it to avoid loading issues
TORCH_AVAILABLE = None  # Will be set to True/False when needed


def _check_torch_available():
    """Check if PyTorch is available, importing it only when needed."""
    global TORCH_AVAILABLE
    if TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False
        except (OSError, ValueError):
            # Handle CUDA library loading errors
            TORCH_AVAILABLE = False
    return TORCH_AVAILABLE


class AudioProcessor:
    """Utility class for common audio processing operations."""

    @staticmethod
    def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes.

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            WAV audio as bytes
        """
        try:
            # Ensure audio is in the correct format (int16)
            if audio_data.dtype != np.int16:
                # Normalize to int16 range if needed
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # Assume float range is [-1.0, 1.0]
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    # Convert directly
                    audio_data = audio_data.astype(np.int16)

            # Write to buffer using wave module (more reliable than scipy with BytesIO)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(1)  # Mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data.tobytes())

            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error(f"Error converting numpy to WAV: {str(e)}")
            raise RuntimeError(f"Audio conversion failed: {str(e)}") from e

    @staticmethod
    def torch_to_wav_bytes(waveform, sample_rate: int) -> bytes:
        """Convert torch tensor to WAV bytes.

        Args:
            waveform: Audio data as torch tensor
            sample_rate: Sample rate in Hz

        Returns:
            WAV audio as bytes
        """
        if not _check_torch_available():
            raise RuntimeError("PyTorch is not available for audio processing")

        try:
            # Import torch/torchaudio only when needed
            import torch
            import torchaudio

            # Check that the input is a torch tensor
            if not isinstance(waveform, torch.Tensor):
                # Convert numpy array to tensor if needed
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)
                else:
                    raise ValueError("Input must be a torch tensor or numpy array")

            # Ensure tensor has the right shape (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension

            # Write to buffer
            buffer = io.BytesIO()
            torchaudio.save(buffer, waveform, sample_rate, format="wav")
            buffer.seek(0)

            return buffer.read()
        except Exception as e:
            logger.error(f"Error converting torch tensor to WAV: {str(e)}")
            raise RuntimeError(f"Audio conversion failed: {str(e)}") from e

    @staticmethod
    def get_wav_duration(wav_bytes: bytes) -> float:
        """Calculate duration of WAV audio in seconds.

        Args:
            wav_bytes: WAV audio data as bytes

        Returns:
            Duration in seconds
        """
        try:
            # Create a BytesIO object from the bytes
            buffer = io.BytesIO(wav_bytes)

            # Open as WAV file
            with wave.open(buffer, "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.error(f"Could not determine WAV duration: {str(e)}")
            # Return a default estimate if we can't calculate
            return 1.0

    @staticmethod
    def get_wav_duration_from_file(file_path: str) -> float:
        """Calculate duration of a WAV file in seconds.

        Args:
            file_path: Path to WAV file

        Returns:
            Duration in seconds
        """
        try:
            with wave.open(file_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.error(f"Could not determine WAV duration from file: {str(e)}")
            return 1.0

    @staticmethod
    def generate_envelope(
        num_samples: int,
        sample_rate: int,
        attack_time: float = 0.1,
        release_time: float = 0.3,
    ) -> np.ndarray:
        """Generate an ADSR envelope with attack and release.

        Args:
            num_samples: Number of samples in the audio
            sample_rate: Sample rate in Hz
            attack_time: Attack time in seconds
            release_time: Release time in seconds

        Returns:
            Envelope as numpy array
        """
        try:
            envelope = np.ones(num_samples)

            # Calculate sample counts
            attack_samples = int(attack_time * sample_rate)
            release_samples = int(release_time * sample_rate)

            # Ensure we don't exceed the audio length
            attack_samples = min(attack_samples, num_samples // 4)
            release_samples = min(release_samples, num_samples // 4)

            # Apply attack
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

            # Apply release
            if release_samples > 0:
                envelope[-release_samples:] = np.linspace(1, 0, release_samples)

            return envelope
        except Exception as e:
            logger.error(f"Error generating envelope: {str(e)}")
            return np.ones(num_samples)

    @staticmethod
    def generate_waveform(
        waveform_type: str,
        frequency: float,
        duration: float,
        sample_rate: int,
        amplitude: float = 0.3,
    ) -> np.ndarray:
        """Generate a basic waveform.

        Args:
            waveform_type: Type of waveform ('sine', 'square', 'sawtooth', 'triangle')
            frequency: Frequency in Hz
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            amplitude: Amplitude (0.0 to 1.0)

        Returns:
            Waveform as numpy array
        """
        try:
            num_samples = int(sample_rate * duration)
            t = np.linspace(0.0, duration, num_samples)

            # Convert amplitude to int16 range
            max_amplitude = np.iinfo(np.int16).max * amplitude

            if waveform_type == "sine":
                signal = max_amplitude * np.sin(2.0 * np.pi * frequency * t)
            elif waveform_type == "square":
                signal = max_amplitude * np.sign(np.sin(2.0 * np.pi * frequency * t))
            elif waveform_type == "sawtooth":
                signal = max_amplitude * (2 * (t * frequency % 1) - 1)
            elif waveform_type == "triangle":
                signal = max_amplitude * (2 * np.abs(2 * (t * frequency % 1) - 1) - 1)
            else:
                # Default to sine
                signal = max_amplitude * np.sin(2.0 * np.pi * frequency * t)

            return signal.astype(np.int16)
        except Exception as e:
            logger.error(f"Error generating waveform: {str(e)}")
            raise RuntimeError(f"Waveform generation failed: {str(e)}") from e

    @staticmethod
    def add_harmonics(
        signal: np.ndarray,
        fundamental_freq: float,
        sample_rate: int,
        num_harmonics: int,
        duration: float,
        harmonic_decay: float = 0.3,
    ) -> np.ndarray:
        """Add harmonics to a signal.

        Args:
            signal: Original signal
            fundamental_freq: Fundamental frequency in Hz
            sample_rate: Sample rate in Hz
            num_harmonics: Number of harmonics to add
            duration: Duration in seconds
            harmonic_decay: Decay factor for harmonics (0.0 to 1.0)

        Returns:
            Signal with harmonics added
        """
        try:
            t = np.linspace(0.0, duration, len(signal))
            enhanced_signal = signal.copy()

            for h in range(2, num_harmonics + 1):
                harmonic_freq = fundamental_freq * h
                harmonic_amp = np.iinfo(np.int16).max * harmonic_decay / h
                harmonic = harmonic_amp * np.sin(2.0 * np.pi * harmonic_freq * t)
                enhanced_signal += harmonic

            return enhanced_signal
        except Exception as e:
            logger.error(f"Error adding harmonics: {str(e)}")
            return signal

    @staticmethod
    def add_noise(signal: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add noise to a signal.

        Args:
            signal: Original signal
            noise_level: Noise level (0.0 to 1.0)

        Returns:
            Signal with noise added
        """
        try:
            max_amplitude = np.iinfo(np.int16).max
            noise = np.random.normal(0, max_amplitude * noise_level, len(signal))
            return signal + noise
        except Exception as e:
            logger.error(f"Error adding noise: {str(e)}")
            return signal

    @staticmethod
    def normalize_audio(
        audio_data: np.ndarray, target_level: float = 0.8
    ) -> np.ndarray:
        """Normalize audio to a target level.

        Args:
            audio_data: Audio data to normalize
            target_level: Target level (0.0 to 1.0)

        Returns:
            Normalized audio data
        """
        try:
            if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
                # Convert to float for normalization
                audio_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            else:
                audio_float = audio_data

            # Find peak amplitude
            peak = np.max(np.abs(audio_float))

            if peak > 0:
                # Normalize to target level
                normalized = audio_float * (target_level / peak)
            else:
                normalized = audio_float

            # Convert back to int16
            return (normalized * np.iinfo(np.int16).max).astype(np.int16)
        except Exception as e:
            logger.error(f"Error normalizing audio: {str(e)}")
            return audio_data

    @staticmethod
    def concatenate_audio(audio_chunks: list) -> np.ndarray:
        """Concatenate multiple audio chunks.

        Args:
            audio_chunks: List of audio arrays

        Returns:
            Concatenated audio array
        """
        try:
            if not audio_chunks:
                raise ValueError("No audio chunks provided")

            return np.concatenate(audio_chunks)
        except Exception as e:
            logger.error(f"Error concatenating audio: {str(e)}")
            raise RuntimeError(f"Audio concatenation failed: {str(e)}") from e
