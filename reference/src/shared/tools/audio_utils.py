"""
Audio Service

Provides unified audio file saving and URL generation.
"""

import asyncio
import io
from .logging_config import get_logger
from datetime import datetime
from pathlib import Path

logger = get_logger(__name__)


class AudioService:
    """Service for handling audio file operations."""

    @staticmethod
    def generate_audio_id() -> str:
        """Generate a unique audio ID based on timestamp.

        Returns:
            Unique audio ID string
        """
        return str(datetime.now().timestamp()).replace(".", "")

    @staticmethod
    def save_audio_file(
        audio_bytes: bytes,
        audio_id: str,
        format: str,
        storage_dir: Path = None
    ) -> Path:
        """Save audio bytes to storage.

        Args:
            audio_bytes: Raw audio data
            audio_id: Unique identifier for this audio
            format: Audio format (wav, mp3, ogg)
            storage_dir: Custom storage directory (uses default if None)

        Returns:
            Path to saved audio file
        """
        from src.extensions.libraries.tts_engine.api.utils.storage import get_output_directory

        if storage_dir is None:
            storage_dir = Path(get_output_directory())

        # Create date-based subfolder for organization
        from datetime import datetime
        date_subfolder = datetime.now().strftime("%Y-%m-%d")
        dated_storage_dir = storage_dir / date_subfolder
        dated_storage_dir.mkdir(parents=True, exist_ok=True)

        # Create audio filename with audio_ prefix to match storage.py convention
        filename = f"audio_{audio_id}.{format}"
        file_path = dated_storage_dir / filename

        # Write audio data
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        logger.debug(f"Saved audio file: {file_path}")
        return file_path

    @staticmethod
    def get_audio_url(audio_id: str, format: str) -> str:
        """Generate URL for accessing an audio file.

        Args:
            audio_id: Unique audio identifier
            format: Audio format

        Returns:
            URL string for accessing the audio
        """
        from src.extensions.libraries.tts_engine.api.utils.storage import get_audio_url
        return get_audio_url(audio_id, format)

    @staticmethod
    def get_audio_duration(audio_bytes: bytes, format: str) -> float:
        """Get duration of audio from bytes.

        Args:
            audio_bytes: Raw audio data
            format: Audio format

        Returns:
            Duration in seconds
        """
        try:
            import io as python_io
            import wave

            if format == "wav":
                wav_file = python_io.BytesIO(audio_bytes)
                with wave.open(wav_file, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                    return duration
            # For other formats, could use librosa or similar
            # For now, return a default
            return 0.0
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 0.0

    @staticmethod
    async def save_and_get_url(
        audio_buffer: io.BytesIO,
        format: str,
        return_duration: bool = True
    ) -> tuple[str, float | None]:
        """Save audio buffer and return URL with duration.

        This is a convenience method that combines saving, URL generation,
        and duration calculation. File operations are run in a thread pool
        to avoid blocking the event loop.

        Args:
            audio_buffer: Buffer containing audio data
            format: Audio format (wav, mp3, ogg)
            return_duration: Whether to calculate and return duration

        Returns:
            Tuple of (audio_url, duration)
        """
        audio_id = AudioService.generate_audio_id()
        audio_bytes = audio_buffer.getvalue()

        # Run file I/O operations in thread pool to avoid blocking event loop
        file_path = await asyncio.to_thread(AudioService.save_audio_file, audio_bytes, audio_id, format)

        # Add to audio history for LRU cache management
        try:
            from src.extensions.libraries.tts_engine.audio_history_manager import audio_history_manager
            audio_history_manager.initialize()
            audio_history_manager.add_entry(audio_id, format, str(file_path.absolute()))
        except Exception as e:
            # Log error but don't fail the save operation
            logger.warning(f"Failed to add audio to history: {e}")

        # Generate URL (fast, no I/O)
        audio_url = AudioService.get_audio_url(audio_id, format)

        # Get duration if requested (runs in thread pool for wav processing)
        duration = None
        if return_duration:
            duration = await asyncio.to_thread(AudioService.get_audio_duration, audio_bytes, format)

        logger.info(f"Audio saved: ID={audio_id}, format={format}, duration={duration}s")
        return audio_url, duration

    @staticmethod
    async def save_temp_audio(
        audio_bytes: bytes,
        filename: str,
        storage_dir: Path = None
    ) -> str:
        """Save temporary audio file (e.g., voice samples) and return path.

        Args:
            audio_bytes: Raw audio data
            filename: Name for the file
            storage_dir: Custom storage directory (uses temp dir if None)

        Returns:
            Path to saved audio file
        """
        from src.extensions.libraries.tts_engine.api.utils.storage import get_temp_directory

        if storage_dir is None:
            storage_dir = Path(get_temp_directory())

        # Ensure directory exists
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = storage_dir / filename
        await asyncio.to_thread(storage_dir.joinpath(filename).write_bytes, audio_bytes)

        logger.debug(f"Saved temp audio: {file_path}")
        return str(file_path)

    @staticmethod
    async def concatenate_audio_files(audio_files: list[Path]) -> bytes:
        """Concatenate multiple WAV audio files into a single audio byte array.

        Args:
            audio_files: List of paths to WAV audio files

        Returns:
            Combined audio data as WAV bytes
        """
        try:
            # Try using pydub first (faster)
            from pydub import AudioSegment

            def concat_with_pydub():
                combined = AudioSegment.empty()
                for audio_file in audio_files:
                    audio = AudioSegment.from_wav(audio_file)
                    combined += audio
                return combined.export(format="wav").read()

            return await asyncio.to_thread(concat_with_pydub)

        except ImportError:
            # Fallback: use wave and io (slower but no extra dependency)
            import wave
            import io

            def concat_with_wave():
                # Read all audio files and get parameters
                audio_data = []
                params = None
                total_frames = 0

                for audio_file in audio_files:
                    with wave.open(str(audio_file), 'rb') as wf:
                        if params is None:
                            params = wf.getparams()
                        frames = wf.readframes(wf.getnframes())
                        audio_data.append(frames)
                        total_frames += wf.getnframes()

                # Create output wave file
                output = io.BytesIO()
                with wave.open(output, 'wb') as wf_out:
                    wf_out.setparams(params)
                    for data in audio_data:
                        wf_out.writeframes(data)

                return output.getvalue()

            return await asyncio.to_thread(concat_with_wave)


# Singleton instance
audio_service = AudioService()
