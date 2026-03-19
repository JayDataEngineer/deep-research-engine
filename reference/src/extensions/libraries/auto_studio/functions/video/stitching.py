"""Video Stitching composable function.

Stitches video clips together using FFmpeg.
Combines video with audio for final output.
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from pydantic import BaseModel, Field

from ...core.composable import BaseComposableFunction, FunctionInput, FunctionResult

logger = logging.getLogger(__name__)


class StitchingConfig(BaseModel):
    """Configuration for video stitching function."""
    ffmpeg_path: str = "ffmpeg"
    output_format: str = "mp4"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    output_dir: str = "data/outputs/auto_studio/final_videos"

    # Video settings
    crf: int = 23  # Quality (lower = better, 18-28 typical)
    preset: str = "medium"  # encoding speed vs compression
    pix_fmt: str = "yuv420p"  # pixel format for compatibility


class VideoStitchingFunction(BaseComposableFunction):
    """Stitch video clips together using FFmpeg.

    This function:
    1. Takes a list of video clips with timing info
    2. Stitches them together using FFmpeg concat demuxer
    3. Combines with audio track if provided
    4. Outputs final video file

    Uses FFmpeg for reliable, high-quality video stitching.
    """

    name = "video_stitching"
    input_types = ["video", "audio"]
    output_types = ["video"]
    requires_gpu = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.stitch_config = StitchingConfig(**config.get("stitching", {}))
        self.output_dir = Path(self.stitch_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_config(self):
        """Validate stitching configuration."""
        # Check if ffmpeg is available
        try:
            subprocess.run(
                [self.stitch_config.ffmpeg_path, "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ValueError(f"FFmpeg not found at: {self.stitch_config.ffmpeg_path}")

    async def validate_input(self, input_data: FunctionInput) -> Tuple[bool, Optional[str]]:
        """Validate input data.

        Required:
        - video_clips: List of video file paths in input_paths or plan_data
        - Optional: audio_track: Path to audio file to combine
        """
        clips = input_data.plan_data.get("video_clips", [])
        if not clips:
            # Check input_paths
            if "video_clips" not in input_data.input_paths:
                return False, "Missing 'video_clips' in plan_data or input_paths"
            clips = input_data.input_paths["video_clips"]

        if not clips or len(clips) == 0:
            return False, "No video clips provided"

        # Verify all clip files exist
        for clip_path in clips:
            if not Path(clip_path).exists():
                return False, f"Video clip not found: {clip_path}"

        return True, None

    async def execute(self, input_data: FunctionInput) -> FunctionResult:
        """Execute video stitching.

        Args:
            input_data: Contains video_clips list, optional audio_track

        Returns:
            FunctionResult with final video path
        """
        video_clips = input_data.plan_data.get("video_clips", [])
        audio_track = input_data.plan_data.get("audio_track")
        output_name = input_data.plan_data.get("output_name", "final_video")

        logger.info(f"[Stitch] Stitching {len(video_clips)} video clips")

        start_time = time.time()

        try:
            # Verify clips exist
            clip_paths = [Path(c) for c in video_clips]
            for clip_path in clip_paths:
                if not clip_path.exists():
                    raise FileNotFoundError(f"Clip not found: {clip_path}")

            # Stitch video clips
            output_path = self.output_dir / f"{output_name}.{self.stitch_config.output_format}"

            await self._stitch_videos(
                clip_paths=clip_paths,
                audio_track=Path(audio_track) if audio_track else None,
                output_path=output_path
            )

            elapsed = time.time() - start_time
            logger.info(f"[Stitch] Final video created in {elapsed:.1f}s: {output_path}")

            return FunctionResult(
                success=True,
                data={
                    "output_paths": {
                        "final_video": str(output_path)
                    },
                    "video_info": {
                        "num_clips": len(video_clips),
                        "has_audio": audio_track is not None,
                        "output_path": str(output_path),
                        "duration": self._get_video_duration(output_path)
                    }
                },
                checkpoint_data={
                    "output_path": str(output_path),
                    "num_clips": len(video_clips)
                },
                metadata={
                    "num_clips": len(video_clips),
                    "elapsed": elapsed
                }
            )

        except Exception as e:
            logger.exception(f"[Stitch] Execution error: {e}")
            return FunctionResult(
                success=False,
                error=f"Video stitching failed: {e}",
                metadata={}
            )

    async def _stitch_videos(
        self,
        clip_paths: List[Path],
        audio_track: Optional[Path],
        output_path: Path
    ):
        """Stitch video clips together using FFmpeg.

        Uses the concat demuxer for lossless stitching.

        Args:
            clip_paths: List of video clip paths
            audio_track: Optional audio file to add
            output_path: Output video path
        """
        # Create concat list file
        concat_file = self.output_dir / "concat_list.txt"
        with open(concat_file, 'w') as f:
            for clip_path in clip_paths:
                # Use absolute paths and escape backslashes
                safe_path = str(clip_path.absolute()).replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        # Build FFmpeg command
        cmd = [
            self.stitch_config.ffmpeg_path,
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "copy",  # Copy video stream (no re-encoding)
        ]

        # Add audio if provided
        if audio_track and audio_track.exists():
            cmd.extend([
                "-i", str(audio_track),
                "-c:a", self.stitch_config.audio_codec,
                "-map", "0:v:0",  # Use video from concat
                "-map", "1:a:0",  # Use audio from audio file
                "-shortest",  # Match shortest duration
            ])
        else:
            cmd.extend(["-c:a", "copy"])  # Copy audio from clips

        cmd.extend([
            "-y",  # Overwrite output
            str(output_path)
        ])

        logger.debug(f"[Stitch] FFmpeg command: {' '.join(cmd)}")

        # Run FFmpeg
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        # Clean up concat file
        concat_file.unlink()

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"[Stitch] Could not get duration: {e}")
            return 0.0
