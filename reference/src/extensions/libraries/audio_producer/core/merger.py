"""Core Merger Module - Audio file combining functionality.

This module provides audio concatenation and merging capabilities
that can be used by any workflow.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class AudioMerger:
    """Combine multiple audio files into one."""

    @staticmethod
    async def concatenate_wav_files(audio_files: List[Path]) -> bytes:
        """Concatenate multiple WAV files into a single WAV byte array.

        Args:
            audio_files: List of paths to WAV audio files

        Returns:
            Combined audio data as WAV bytes
        """
        try:
            # Try using pydub first (faster, easier)
            from pydub import AudioSegment

            def concat_with_pydub():
                combined = AudioSegment.empty()
                for audio_file in audio_files:
                    audio = AudioSegment.from_wav(audio_file)
                    combined += audio
                return combined.export(format="wav").read()

            return await asyncio.to_thread(concat_with_pydub)

        except ImportError:
            # Fallback: Manual WAV concatenation without pydub
            import struct
            import tempfile

            def concat_manual():
                """Manually concatenate WAV files by combining raw audio data."""
                if not audio_files:
                    raise ValueError("No audio files to concatenate")

                audio_data = []
                nchannels = None
                sampwidth = None
                framerate = None
                total_frames = 0

                # Read all WAV files
                for audio_file in audio_files:
                    with open(audio_file, 'rb') as f:
                        # Read RIFF header
                        riff_header = f.read(12)
                        if len(riff_header) < 12:
                            raise ValueError(f"Invalid WAV file (too small): {audio_file}")
                        if riff_header[:4] != b'RIFF' or riff_header[8:12] != b'WAVE':
                            raise ValueError(f"Invalid WAV file (bad header): {audio_file}")

                        # Parse chunks to find fmt and data
                        found_fmt = False
                        audio_bytes = None

                        while True:
                            chunk_header = f.read(8)
                            if len(chunk_header) < 8:
                                break  # End of file

                            chunk_id = chunk_header[:4]
                            chunk_size = struct.unpack('<I', chunk_header[4:8])[0]

                            if chunk_id == b'fmt ':
                                # Read format chunk
                                fmt_data = f.read(chunk_size)
                                if len(fmt_data) < 16:
                                    raise ValueError(f"Invalid fmt chunk in: {audio_file}")
                                # Extract format info
                                audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                                nchannels_local = struct.unpack('<H', fmt_data[2:4])[0]
                                framerate_local = struct.unpack('<I', fmt_data[4:8])[0]
                                byte_rate = struct.unpack('<I', fmt_data[8:12])[0]
                                block_align = struct.unpack('<H', fmt_data[12:14])[0]
                                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]

                                sampwidth_local = bits_per_sample // 8

                                if not found_fmt:
                                    nchannels = nchannels_local
                                    sampwidth = sampwidth_local
                                    framerate = framerate_local
                                    found_fmt = True
                                elif (nchannels_local != nchannels or
                                      sampwidth_local != sampwidth or
                                      framerate_local != framerate):
                                    logger.warning(
                                        f"Audio file {audio_file} has different parameters, "
                                        f"may cause issues"
                                    )

                            elif chunk_id == b'data':
                                # Read audio data
                                audio_bytes = f.read(chunk_size)
                                audio_data.append(audio_bytes)
                                if nchannels and sampwidth:
                                    total_frames += chunk_size // (nchannels * sampwidth)
                                break  # Found data, we're done

                            else:
                                # Skip unknown chunk
                                f.read(chunk_size)

                        if audio_bytes is None:
                            raise ValueError(f"No data chunk found in: {audio_file}")

                if nchannels is None or sampwidth is None or framerate is None:
                    raise ValueError("Could not determine audio format")

                # Build combined WAV file manually
                data_size = sum(len(d) for d in audio_data)
                total_size = 36 + data_size  # 36 = header size - 8

                # Build WAV header
                header = bytearray(44)
                header[0:4] = b'RIFF'
                header[4:8] = struct.pack('<I', total_size)
                header[8:12] = b'WAVE'
                header[12:16] = b'fmt '
                header[16:20] = struct.pack('<I', 16)  # fmt chunk size
                header[20:22] = struct.pack('<H', 1)  # PCM format
                header[22:24] = struct.pack('<H', nchannels)
                header[24:28] = struct.pack('<I', framerate)
                header[28:32] = struct.pack('<I', framerate * nchannels * sampwidth)  # byte rate
                header[32:34] = struct.pack('<H', nchannels * sampwidth)  # block align
                header[34:36] = struct.pack('<H', sampwidth * 8)  # bits per sample
                header[36:40] = b'data'
                header[40:44] = struct.pack('<I', data_size)

                # Use temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(bytes(header))
                    for data in audio_data:
                        tmp_file.write(data)
                    tmp_file.flush()

                    # Read back the temp file
                    with open(tmp_file.name, 'rb') as f:
                        return f.read()

            return await asyncio.to_thread(concat_manual)

    @staticmethod
    async def concatenate_with_silence(
        audio_files: List[Path],
        silence_duration: float = 0.5,
        sample_rate: int = 24000,
    ) -> bytes:
        """Concatenate WAV files with silence between each.

        Args:
            audio_files: List of paths to WAV audio files
            silence_duration: Duration of silence in seconds
            sample_rate: Sample rate for audio

        Returns:
            Combined audio data as WAV bytes
        """
        try:
            from pydub import AudioSegment
            from pydub.generators import silence

            def concat_with_pydub():
                # Create silence segment
                silence = AudioSegment.silent(
                    duration=int(silence_duration * 1000)  # pydub uses milliseconds
                )

                # Start with first file
                combined = AudioSegment.from_wav(audio_files[0])

                # Add remaining files with silence between
                for audio_file in audio_files[1:]:
                    combined += silence
                    combined += AudioSegment.from_wav(audio_file)

                return combined.export(format="wav").read()

            return await asyncio.to_thread(concat_with_pydub)

        except ImportError:
            # Fallback: concatenate without silence
            logger.warning("pydub not available, concatenating without silence")
            return await AudioMerger.concatenate_wav_files(audio_files)

    @staticmethod
    def extract_speaker_from_path(audio_path: Path) -> Optional[str]:
        """Extract speaker name from audio filename.

        Expected format: chunk_XXXX_SPEAKER.wav

        Args:
            audio_path: Path to audio file

        Returns:
            Speaker name or None if not found
        """
        match = re.match(r'chunk_\d+_(.+?)\.wav', audio_path.name)
        return match.group(1) if match else None

    @staticmethod
    async def concatenate_with_speaker_gaps(
        audio_files: List[Path],
        speaker_gap_duration: float = 0.3,
        normal_gap_duration: float = 0.1,
        normalize: bool = True,
        target_dBFS: float = -20.0,
        crossfade_same_speaker: float = 0.08,  # 80ms crossfade for same speaker
    ) -> bytes:
        """Concatenate WAV files with smart silence based on speaker changes.

        - Adds longer silence when speaker changes
        - Uses cross-fade for same speaker (smoother transitions)
        - Optionally normalizes audio levels

        Args:
            audio_files: List of paths to WAV audio files (sorted)
            speaker_gap_duration: Silence duration when speaker changes (seconds)
            normal_gap_duration: Silence duration between same speaker (seconds)
            normalize: Whether to normalize audio levels
            target_dBFS: Target volume level in dBFS (for normalization)
            crossfade_same_speaker: Cross-fade duration for same speaker (seconds)

        Returns:
            Combined audio data as WAV bytes
        """
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize

            def concat_with_pydub():
                # Create silence segments
                speaker_silence = AudioSegment.silent(duration=int(speaker_gap_duration * 1000))
                crossfade_ms = int(crossfade_same_speaker * 1000)

                # Start with first file (normalize if requested)
                combined = AudioSegment.from_wav(audio_files[0])
                if normalize:
                    combined = normalize(combined, headroom=0.1)

                # Track previous speaker
                prev_speaker = AudioMerger.extract_speaker_from_path(audio_files[0])

                # Add remaining files with smart silence/crossfade
                for audio_file in audio_files[1:]:
                    current_speaker = AudioMerger.extract_speaker_from_path(audio_file)

                    # Load and normalize audio
                    audio = AudioSegment.from_wav(audio_file)
                    if normalize:
                        audio = normalize(audio, headroom=0.1)

                    # Choose transition based on speaker change
                    if current_speaker and prev_speaker and current_speaker != prev_speaker:
                        # Speaker change: use silence gap (no crossfade between voices)
                        combined = combined + speaker_silence + audio
                        logger.debug(f"Speaker change: {prev_speaker} -> {current_speaker}, added {speaker_gap_duration}s silence")
                    else:
                        # Same speaker: use cross-fade for smooth transition
                        # Cross-fade the end of combined with start of new audio
                        combined = combined.append(audio, crossfade=crossfade_ms)
                        logger.debug(f"Same speaker: {current_speaker}, applied {crossfade_same_speaker}s crossfade")

                    prev_speaker = current_speaker

                return combined.export(format="wav").read()

            return await asyncio.to_thread(concat_with_pydub)

        except ImportError:
            # Fallback: concatenate without silence
            logger.warning("pydub not available, concatenating without silence/normalization")
            return await AudioMerger.concatenate_wav_files(audio_files)

    @staticmethod
    async def merge_audio_files(
        audio_files: List[Path],
        output_path: Path,
        add_silence: bool = True,
        silence_duration: float = 0.5,
        speaker_aware: bool = True,
        speaker_gap_duration: float = 0.3,
        normal_gap_duration: float = 0.1,
        normalize: bool = True,
        cleanup_source: bool = True,
    ) -> Path:
        """Merge audio files and save to disk.

        Args:
            audio_files: List of paths to audio files
            output_path: Where to save the merged file
            add_silence: Whether to add silence between segments
            silence_duration: Duration of silence in seconds (legacy, for non-speaker-aware mode)
            speaker_aware: Use speaker-aware gaps (longer on speaker change, shorter otherwise)
            speaker_gap_duration: Silence duration when speaker changes (seconds)
            normal_gap_duration: Silence duration between same speaker (seconds)
            normalize: Whether to normalize audio levels
            cleanup_source: Whether to delete source files after merging

        Returns:
            Path to the merged audio file
        """
        logger.info(f"Merging {len(audio_files)} audio files into {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge audio
        if add_silence and len(audio_files) > 1:
            if speaker_aware:
                audio_bytes = await AudioMerger.concatenate_with_speaker_gaps(
                    audio_files,
                    speaker_gap_duration=speaker_gap_duration,
                    normal_gap_duration=normal_gap_duration,
                    normalize=normalize,
                )
            else:
                audio_bytes = await AudioMerger.concatenate_with_silence(
                    audio_files,
                    silence_duration=silence_duration,
                )
        else:
            audio_bytes = await AudioMerger.concatenate_wav_files(audio_files)

        # Write output
        output_path.write_bytes(audio_bytes)
        logger.info(f"Merged audio saved to: {output_path}")

        # Cleanup source files if requested
        if cleanup_source:
            for audio_file in audio_files:
                try:
                    audio_file.unlink()
                    logger.debug(f"Deleted source file: {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete {audio_file}: {e}")

        return output_path

    @staticmethod
    async def get_audio_duration(file_path: Path) -> float:
        """Get duration of an audio file in seconds.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in seconds
        """
        try:
            from pydub import AudioSegment

            def get_duration_pydub():
                audio = AudioSegment.from_wav(file_path)
                return len(audio) / 1000.0  # pydub uses milliseconds

            return await asyncio.to_thread(get_duration_pydub)

        except ImportError:
            # Fallback: manual WAV parsing
            import struct

            def get_duration_manual():
                """Get duration by parsing WAV header manually."""
                with open(file_path, 'rb') as f:
                    header = f.read(44)

                    # Parse WAV header
                    if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
                        raise ValueError(f"Invalid WAV file: {file_path}")

                    # Extract format info
                    nchannels = struct.unpack('<H', header[22:24])[0]
                    framerate = struct.unpack('<I', header[24:28])[0]
                    sampwidth = struct.unpack('<H', header[34:36])[0] // 8

                    # Find data chunk
                    data_pos = 12
                    data_size = 0
                    while data_pos < 44:  # We already read 44 bytes
                        chunk_id = header[data_pos:data_pos+4]
                        chunk_size = struct.unpack('<I', header[data_pos+4:data_pos+8])[0]
                        if chunk_id == b'data':
                            data_size = chunk_size
                            break
                        data_pos += 8 + chunk_size

                    # Duration = data_size / (channels * bytes_per_sample * sample_rate)
                    duration = data_size / (nchannels * sampwidth * framerate)
                    return duration

            return await asyncio.to_thread(get_duration_manual)
