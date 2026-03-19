"""IndexTTS vLLM Adapter for Audio Producer.

This adapter provides a unified interface for using IndexTTS vLLM
as a drop-in replacement for the old MultiIndexTTSManager.

Uses HTTP API to the index_tts_vllm extension service.

Format: Uses same IndexTTS2 emotion vectors [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
"""

import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Any, Optional

import httpx

# Suppress httpx INFO logs (HTTP request/response logs)
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.extensions.libraries.tts_engine.extension_process_manager import ExtensionProcessManager
from src.constants import BACKEND_ROOT, TTS_EXTENSIONS_DIR
from src.shared.utils.venv_manager import get_venv_python

logger = logging.getLogger(__name__)


class IndexTTSVLLMAdapter:
    """Adapter for IndexTTS vLLM with same interface as MultiIndexTTSManager.

    Uses HTTP API to communicate with index_tts_vllm extension service.
    """

    def __init__(self, port: int = 9000, config: dict[str, Any] | None = None):
        """Initialize the IndexTTS vLLM adapter.

        Args:
            port: Port for the IndexTTS vLLM service
            config: Additional configuration (not used currently)
        """
        self.port = port
        self.config = config or {}
        self.process_manager: Optional[ExtensionProcessManager] = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        self.is_initialized = False
        self.num_instances = 1  # vLLM has 16x concurrency internally

        # For multi-worker compatibility (maps worker_id to reference audio paths)
        self.worker_voice_refs: dict[int, str] = {}

        # Stats
        self.total_requests = 0
        self.total_chars = 0
        self.start_time = None

    async def start_all(self) -> bool:
        """Start IndexTTS vLLM service.

        Returns:
            True if started successfully
        """
        try:
            logger.info(f"Starting IndexTTS vLLM service on port {self.port}...")

            self.process_manager = ExtensionProcessManager()
            # The extension is at src/extensions/services/tts/index_tts_vllm
            ext_dir = BACKEND_ROOT / "src" / "extensions" / "services" / "tts" / "index_tts_vllm"
            python_exe = get_venv_python(backend_dir=BACKEND_ROOT, extension_dir=ext_dir)
            logger.info(f"Extension directory: {ext_dir}")
            logger.info(f"Python executable: {python_exe}")
            logger.info(f"Python exists: {python_exe and python_exe.exists()}")

            env = {
                **os.environ,
                'PYTHONPATH': str(BACKEND_ROOT),
            }

            # Debug: Verify Python version before starting subprocess
            import subprocess as sync_subprocess
            result = sync_subprocess.run([str(python_exe), "--version"], capture_output=True, text=True)
            logger.info(f"Python version check: {result.stdout.strip()}")
            logger.info(f"Starting subprocess with: {str(python_exe)}")

            # Start the subprocess
            # Note: stderr can be suppressed by setting INDEX_TTS_QUIET=1
            quiet_mode = os.environ.get('INDEX_TTS_QUIET', '0') == '1'
            self.process_manager.process = await asyncio.create_subprocess_exec(
                str(python_exe),
                "-m",
                "src.extensions.libraries.tts_engine.extension_service",
                "--extension", "index_tts_vllm",
                "--port", str(self.port),
                "--host", "127.0.0.1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL if quiet_mode else None,  # Suppress stderr if quiet mode
                cwd=str(BACKEND_ROOT),
                env=env
            )

            self.process_manager.active_extension = "index_tts_vllm"

            # Wait for service to be ready
            await self._wait_for_ready(timeout=60.0)

            # Load the model
            load_url = f"http://127.0.0.1:{self.port}/load"
            load_timeout = httpx.Timeout(600.0, connect=10.0)
            load_response = await self.client.post(load_url, timeout=load_timeout)
            load_data = load_response.json()

            if load_data.get("status") != "ready":
                raise RuntimeError(f"IndexTTS vLLM failed to load: {load_data}")

            self.is_initialized = True
            self.start_time = time.time()
            logger.info(f"IndexTTS vLLM service ready on port {self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start IndexTTS vLLM: {e}")
            raise RuntimeError(f"IndexTTS vLLM initialization FAILED: {e}. Process ABORTING.")

    async def _wait_for_ready(self, timeout: float = 30.0):
        """Wait for the service to be ready."""
        start = time.time()
        url = f"http://127.0.0.1:{self.port}/health"

        while time.time() - start < timeout:
            try:
                response = await self.client.get(url, timeout=2.0)
                if response.status_code == 200:
                    logger.info(f"IndexTTS vLLM health check passed")
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)

        raise RuntimeError(f"IndexTTS vLLM did not become ready within {timeout}s")

    async def generate_audio(
        self,
        worker_id: int,
        text: str,
        voice_id: str,
        emo_vector: List[float] | None = None,
        reference_audio: str | None = None,
        **kwargs
    ) -> Tuple[bytes, float]:
        """Generate audio for a single dialogue line.

        Compatible with MultiIndexTTSManager.generate_audio() interface.

        Args:
            worker_id: Worker ID for tracking (for compatibility)
            text: Text to synthesize
            voice_id: Voice identifier (for logging)
            emo_vector: 8-dim IndexTTS2 emotion vector
            reference_audio: Path to reference audio for voice cloning
            **kwargs: Additional parameters

        Returns:
            Tuple of (audio_bytes, duration_seconds)
        """
        if not self.is_initialized:
            raise RuntimeError("IndexTTS vLLM not initialized. Call start_all() first.")

        # CRITICAL FIX: Use the reference_audio directly, NOT cached by worker_id
        # Each line has its own speaker-specific reference audio (EMMA, OLIVIA, NARRATOR)
        # Caching by worker_id would cause all lines processed by the same worker to use the same voice
        ref_audio_path = reference_audio
        if not ref_audio_path:
            raise ValueError(f"reference_audio required for speaker (line: '{text[:50]}...')")

        # Update stats
        self.total_requests += 1
        self.total_chars += len(text)
        char_count = len(text)

        # Build request
        url = f"http://127.0.0.1:{self.port}/generate"
        payload = {
            "text": text,
            "voice_id": voice_id,
            "ref_audio_path": ref_audio_path,  # Engine expects "ref_audio_path"
            "output_format": "wav",
        }

        # Add emotion vector if provided
        if emo_vector:
            payload["emo_vector"] = emo_vector[:8]  # IndexTTS2 uses first 8 dims

        logger.info(f"IndexTTS vLLM [{voice_id}]: {char_count} chars -> '{text[:50]}...'")

        response = await self.client.post(url, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"TTS generation failed: {response.text}")

        # Get duration from response headers
        duration = float(response.headers.get("X-Duration", 0))

        # Return raw audio bytes
        return response.content, duration

    @property
    def working_ports(self) -> set:
        """For compatibility with MultiIndexTTSManager interface."""
        return {self.port} if self.is_initialized else set()

    async def stop_all(self):
        """Stop the service and free GPU memory, including all child processes."""
        import subprocess

        # First, try the normal cleanup through process_manager
        if self.process_manager and self.process_manager.active_extension:
            try:
                # Use stop_current() which is the correct method name
                await self.process_manager.stop_current()
            except Exception as e:
                logger.warning(f"Error stopping IndexTTS vLLM: {e}")
        elif self.process_manager and self.process_manager.process:
            # Fallback if process wasn't started through ExtensionProcessManager.start()
            try:
                self.process_manager.process.terminate()
                try:
                    await asyncio.wait_for(self.process_manager.process.wait(), timeout=5.0)
                except TimeoutError:
                    logger.warning("IndexTTS vLLM process did not terminate gracefully, forcing kill...")
                    self.process_manager.process.kill()
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error stopping IndexTTS vLLM: {e}")

        # AGGRESSIVE CLEANUP: Kill ALL vLLM-related processes to ensure GPU memory is freed
        # This handles orphaned child processes that don't get cleaned up properly
        # Method 1: Use pkill to kill all processes matching index_tts_vllm
        try:
            subprocess.run(["pkill", "-9", "-f", "index_tts_vllm"],
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            logger.info("Killed vLLM processes via pkill")
        except FileNotFoundError:
            pass  # pkill not available

        # Method 2: Also kill by port as final fallback
        try:
            subprocess.run(["fuser", "-k", "-9", "9000/tcp"],
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except FileNotFoundError:
            pass

        self.is_initialized = False
        if self.client:
            await self.client.aclose()

    async def restart_hung_instance(self, worker_id: int):
        """Compatibility method - no-op for vLLM (more stable than HTTP-based system)."""
        logger.warning(f"restart_hung_instance called for worker {worker_id}, but vLLM is more stable than the old HTTP system. This is a no-op.")

    def get_stats(self) -> dict[str, Any]:
        """Get generation statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        chars_per_sec = self.total_chars / elapsed if elapsed > 0 else 0
        return {
            "total_requests": self.total_requests,
            "total_chars": self.total_chars,
            "elapsed_time_sec": elapsed,
            "chars_per_sec": chars_per_sec,
        }

    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_all()
        self.worker_voice_refs.clear()
