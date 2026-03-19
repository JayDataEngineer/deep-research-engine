"""
Simplified process manager for running TTS extensions in isolated subprocesses.

Uses a single port (TTS_EXTENSION_PORT) - only one extension active at a time.

Note: Extension venv management now uses centralized VenvManager to eliminate
code duplication across the codebase.
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path

import httpx
from httpx import TimeoutException

from src.constants import BACKEND_ROOT, PROJECT_ROOT, TTS_EXTENSION_PORT, TTS_EXTENSION_URL, TTS_EXTENSIONS_DIR, MAIN_API_URL
from src.shared.utils.venv_manager import get_venv_python, get_extension_dir

logger = logging.getLogger(__name__)

class ExtensionProcessManager:
    """
    Manages TTS extension subprocesses with isolated environments.

    Only ONE extension runs at a time on a single port (TTS_EXTENSION_PORT).
    When switching extensions, we stop the current one and start the new one.
    """

    def __init__(self):
        self.active_extension: str | None = None
        self.process: asyncio.subprocess.Process | None = None
        # Configure HTTP client WITH connection pooling for better performance
        # We validate the extension matches before each request to prevent stale connections
        timeout = httpx.Timeout(
            connect=10.0,      # 10s to establish connection
            read=120.0,        # 120s for read operations (generation may take time)
            write=10.0,        # 10s for write operations
            pool=5.0           # 5s timeout for getting connection from pool
        )
        # OPTIMIZED: Enable connection pooling with limits - we validate extension per-request
        limits = httpx.Limits(
            max_connections=10,              # Allow multiple connections
            max_keepalive_connections=5,     # Keep some connections alive
            keepalive_expiry=30.0            # 30s expiry - balances freshness vs performance
        )
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits)
        self._shutdown_event = asyncio.Event()
        self._cached_voices: list | None = None  # Cache voices to avoid repeated requests

    async def cleanup_leftover_processes(self):
        """Clean up any leftover subprocesses on TTS_EXTENSION_PORT from previous runs.

        This is called on startup to prevent stale processes from blocking the port.

        IMPORTANT: This will skip the currently tracked process (if any) to avoid
        killing an active extension. The stop_current() method should be used to
        stop the tracked process before starting a new one.

        Uses multiple methods to ensure robust cleanup:
        1. lsof - find processes listening on the port
        2. psutil - verify all processes are found and killed
        3. Direct kill -9 as fallback
        """
        killed_pids = []
        tracked_pid = self.process.pid if self.process else None

        # Method 1: Try lsof first (most reliable on Linux)
        try:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-tP", "-iTCP", f":{TTS_EXTENSION_PORT}", "-sTCP:LISTEN",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

            if proc.returncode == 0 and stdout.strip():
                pids = [int(pid) for pid in stdout.strip().decode().split('\n') if pid]

                # Filter out the tracked process
                orphaned_pids = [pid for pid in pids if pid != tracked_pid]

                if orphaned_pids:
                    logger.warning(f"[CLEANUP] Found orphaned processes on port {TTS_EXTENSION_PORT}: {orphaned_pids}")
                    for pid in orphaned_pids:
                        try:
                            logger.info(f"[CLEANUP] Killing orphaned process {pid} (lsof)")
                            kill_proc = await asyncio.create_subprocess_exec(
                                "kill", "-9", str(pid),
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            await asyncio.wait_for(kill_proc.communicate(), timeout=5)
                            killed_pids.append(pid)
                        except Exception as e:
                            logger.warning(f"[CLEANUP] Failed to kill process {pid}: {e}")

        except (FileNotFoundError, asyncio.TimeoutError) as e:
            logger.warning(f"[CLEANUP] lsof method failed: {e}")

        # Method 2: Try psutil as backup (more portable)
        try:
            import psutil
            for conn in psutil.net_connections():
                if conn.laddr.port == TTS_EXTENSION_PORT and conn.status == 'LISTEN':
                    pid = conn.pid
                    if pid != tracked_pid and pid not in killed_pids:
                        try:
                            logger.info(f"[CLEANUP] Killing orphaned process {pid} (psutil)")
                            p = psutil.Process(pid)
                            p.kill()
                            killed_pids.append(pid)
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            logger.warning(f"[CLEANUP] psutil failed to kill {pid}: {e}")
        except ImportError:
            logger.debug("[CLEANUP] psutil not available, skipping")
        except Exception as e:
            logger.warning(f"[CLEANUP] psutil method failed: {e}")

        # Method 3: fuser as last resort (Linux-only)
        if not killed_pids:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "fuser", "-k", f"{TTS_EXTENSION_PORT}/tcp",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(proc.communicate(), timeout=5)
                logger.info("[CLEANUP] Ran fuser -k as last resort")
            except (FileNotFoundError, asyncio.TimeoutError):
                pass

        # Wait for processes to die
        if killed_pids:
            await asyncio.sleep(1)

        # Verify port is actually free
        port_free = True
        try:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-tP", "-iTCP", f":{TTS_EXTENSION_PORT}", "-sTCP:LISTEN",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode == 0 and stdout.strip():
                remaining_pids = [int(pid) for pid in stdout.strip().decode().split('\n') if pid]
                if remaining_pids:
                    port_free = False
                    # Check if only tracked process remains
                    if tracked_pid and remaining_pids == [tracked_pid]:
                        port_free = True
                        logger.info(f"[CLEANUP] Port {TTS_EXTENSION_PORT} has only tracked process (OK)")
                    else:
                        logger.error(f"[CLEANUP] Port {TTS_EXTENSION_PORT} STILL HAS PROCESSES: {remaining_pids}")
                        # Force kill again
                        for pid in remaining_pids:
                            if pid != tracked_pid:
                                kill_proc = await asyncio.create_subprocess_exec(
                                    "kill", "-9", str(pid),
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE
                                )
                                await asyncio.wait_for(kill_proc.communicate(), timeout=5)

        except (FileNotFoundError, asyncio.TimeoutError):
            pass

        if port_free:
            logger.info(f"[CLEANUP] Port {TTS_EXTENSION_PORT} is free")
        elif not killed_pids:
            logger.info(f"[CLEANUP] No processes found on port {TTS_EXTENSION_PORT}")

        return killed_pids

    def _is_port_in_use(self, port: int) -> bool:
        """Fast socket-based check if port is in use (no subprocess needed)."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                result = sock.connect_ex(('127.0.0.1', port))
                return result == 0
        except Exception:
            return False

    def _get_pids_on_port(self, port: int) -> list[int]:
        """Get PIDs listening on a port using lsof (cached result)."""
        try:
            proc = asyncio.create_subprocess_exec(
                "lsof", "-tP", "-iTCP", f":{port}", "-sTCP:LISTEN",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode == 0 and stdout.strip():
                return [int(pid) for pid in stdout.strip().decode().split('\n') if pid]
        except (FileNotFoundError, asyncio.TimeoutError):
            pass
        return []

    def _get_extension_dir(self, name: str) -> Path:
        """Get the directory for an extension using centralized VenvManager."""
        # Use centralized extension directory detection
        ext_dir = get_extension_dir()
        if ext_dir:
            # Called from within an extension context
            return ext_dir
        # Fallback to manual path resolution
        # TTS services are in backend/src/extensions/services/
        return TTS_EXTENSIONS_DIR / name

    def _get_venv_path(self, name: str) -> Path:
        """Get the venv path for running the extension service subprocess.

        Each TTS service has its own isolated venv at backend/src/extensions/services/{name}/.venv/.
        This is required because different TTS engines have incompatible torch/CUDA requirements.

        Uses centralized VengManager for venv detection.
        """
        # TTS services are in backend/src/extensions/services/
        extension_venv = TTS_EXTENSIONS_DIR / name / ".venv"

        # Fall back to root venv only if extension venv doesn't exist
        if extension_venv.exists():
            return extension_venv
        else:
            return PROJECT_ROOT / ".venv"

    def _has_isolated_venv(self, name: str) -> bool:
        """Check if the extension has an isolated venv."""
        # TTS services are in backend/src/extensions/services/
        extension_venv = TTS_EXTENSIONS_DIR / name / ".venv"
        return extension_venv.exists()

    def _get_extension_venv_python(self, name: str) -> Path:
        """Get the Python executable for an extension's venv.

        Uses centralized VenvManager for consistent venv detection across the codebase.
        """
        backend_dir = BACKEND_ROOT
        # TTS services are in backend/src/extensions/services/
        extension_dir = TTS_EXTENSIONS_DIR / name

        # Use centralized venv detection with keyword arguments to avoid confusion
        venv_python = get_venv_python(backend_dir=backend_dir, extension_dir=extension_dir)
        if venv_python:
            return venv_python

        # Fallback to extension venv
        extension_venv = extension_dir / ".venv"
        if extension_venv.exists():
            return extension_venv / "bin" / "python"

        # Fallback to root venv
        return PROJECT_ROOT / ".venv" / "bin" / "python"

    async def stop_current(self) -> dict:
        """Stop the currently running extension subprocess.

        IMPORTANT: This must complete successfully before starting a new extension.
        Raises an error if the process cannot be stopped.
        """
        import os
        import signal

        if not self.process:
            logger.info("[STOP] No extension subprocess to stop")
            return {"status": "no_extension", "message": "No extension running"}

        old_extension = self.active_extension
        pid = self.process.pid
        logger.info(f"[STOP] Stopping extension: {old_extension} (PID: {pid})")

        # Check if process has already exited
        if self.process.returncode is not None:
            logger.info(f"[STOP] Process {pid} already exited (exit code: {self.process.returncode})")
            self.process = None
            self.active_extension = None
            return {"status": "stopped", "extension": old_extension, "message": "Process already exited"}

        # CRITICAL: Find and kill vLLM EngineCore worker processes BEFORE killing parent
        # This prevents them from being orphaned and becoming untrackable
        try:
            # Find vLLM worker processes by searching for EngineCore in process list
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-f", "VLLM::EngineCore",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

            if proc.returncode == 0 and stdout.strip():
                worker_pids = [int(p) for p in stdout.strip().decode().split('\n') if p]
                logger.info(f"[STOP] Found {len(worker_pids)} vLLM EngineCore worker processes: {worker_pids}")

                # Kill each worker process directly
                for worker_pid in worker_pids:
                    try:
                        kill_proc = await asyncio.create_subprocess_exec(
                            "kill", "-9", str(worker_pid),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await asyncio.wait_for(kill_proc.communicate(), timeout=3)
                        logger.info(f"[STOP] Killed vLLM worker process {worker_pid}")
                    except Exception as e:
                        logger.warning(f"[STOP] Failed to kill worker {worker_pid}: {e}")
            else:
                logger.debug(f"[STOP] No vLLM EngineCore worker processes found")
        except (FileNotFoundError, asyncio.TimeoutError) as e:
            logger.warning(f"[STOP] Failed to find vLLM worker processes: {e}")
        except Exception as e:
            logger.warning(f"[STOP] Unexpected error finding vLLM workers: {e}")

        # Try graceful shutdown via unload endpoint
        try:
            logger.info(f"[STOP] Calling /unload endpoint on {TTS_EXTENSION_URL}...")
            response = await self.client.post(f"{TTS_EXTENSION_URL}/unload", timeout=5.0)
            logger.info(f"[STOP] /unload returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.warning(f"[STOP] Could not call unload endpoint: {e}")

        # Kill the ENTIRE process group (includes all children/multiprocessing resource trackers)
        try:
            logger.info(f"[STOP] Killing process group {pid}...")
            # Re-check process reference before terminating (defensive check)
            process = self.process
            if process is None:
                logger.warning(f"[STOP] Process became None before terminate - already stopped elsewhere")
                self.active_extension = None
                return {"status": "stopped", "extension": old_extension, "message": "Process already stopped"}

            # Kill the entire process group using killpg
            try:
                os.killpg(pid, signal.SIGKILL)
                logger.info(f"[STOP] Killed process group {pid} (including all children)")
            except (ProcessLookupError, PermissionError):
                # Fallback to regular kill if killpg fails
                process.kill()
                logger.info(f"[STOP] Killed process {pid} (fallback to kill())")

            # Wait for process to exit
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.info(f"[STOP] Process {pid} killed successfully (exit code: {process.returncode})")
            except TimeoutError:
                logger.warning(f"[STOP] Process {pid} still running after kill, forcing again...")
                # Try one more time
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    # Process already exited - this is OK
                    logger.debug(f"[STOP] Process {pid} already exited (ProcessLookupError during force kill)")
                except Exception as e:
                    # Log other errors during force kill
                    logger.warning(f"[STOP] Error during force kill of {pid}: {type(e).__name__}: {e}")
                await asyncio.sleep(1.0)
                if process.returncode is not None:
                    logger.info(f"[STOP] Process {pid} finally killed (exit code: {process.returncode})")
                else:
                    logger.error(f"[STOP] Process {pid} REFUSES TO DIE!")

        except ProcessLookupError:
            # Process already exited - this is OK
            logger.warning(f"[STOP] Process {pid} not found (ProcessLookupError) - likely already exited")
            if self.process and self.process.returncode is not None:
                logger.info(f"[STOP] Process exit code: {self.process.returncode}")
        except AttributeError as e:
            # Process became None - handle gracefully
            logger.warning(f"[STOP] Process became None during termination: {e}")
            self.process = None
            self.active_extension = None
            return {"status": "stopped", "extension": old_extension, "message": "Process already stopped"}
        except Exception as e:
            logger.error(f"[STOP] Error stopping process: {type(e).__name__}: {repr(e)}", exc_info=True)
            # Log subprocess stderr for debugging
            await self._log_stderr()
            raise RuntimeError(f"Failed to stop extension {old_extension}: {type(e).__name__}: {str(e) or repr(e)}")

        # FINAL CLEANUP: As a safety net, try pkill for any remaining vLLM processes
        # This handles any workers that might have been orphaned during the shutdown
        try:
            result = await asyncio.create_subprocess_exec(
                "pkill", "-9", "-f", "VLLM::EngineCore",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(result.communicate(), timeout=5)
            logger.info(f"[STOP] Final cleanup: killed any remaining vLLM EngineCore processes")
        except (FileNotFoundError, asyncio.TimeoutError):
            pass  # pkill not available or timeout - continue

        # Verify process is actually gone
        if self.process is not None and self.process.returncode is None:
            logger.error(f"[STOP] Process {pid} still running after terminate/kill!")
            await self._log_stderr()
            raise RuntimeError(f"Process {pid} refused to die")
        elif self.process is None:
            logger.info(f"[STOP] Process is None - already cleaned up")

        # Verify port is actually free
        await asyncio.sleep(0.5)  # Give OS time to release the port
        try:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-tP", "-iTCP", f":{TTS_EXTENSION_PORT}", "-sTCP:LISTEN",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            if proc.returncode == 0 and stdout.strip():
                remaining_pids = [int(p) for p in stdout.strip().decode().split('\n') if p]
                if remaining_pids:
                    logger.error(f"[STOP] Port {TTS_EXTENSION_PORT} STILL IN USE by PIDs: {remaining_pids}")
                    # Force kill any remaining processes
                    for remaining_pid in remaining_pids:
                        if remaining_pid != pid:  # Don't try to kill the already-dead process
                            logger.warning(f"[STOP] Force killing remaining process {remaining_pid}")
                            kill_proc = await asyncio.create_subprocess_exec(
                                "kill", "-9", str(remaining_pid),
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            await asyncio.wait_for(kill_proc.communicate(), timeout=5)
                    await asyncio.sleep(0.5)
        except (FileNotFoundError, asyncio.TimeoutError):
            pass

        logger.info(f"[STOP] Extension {old_extension} stopped successfully")
        self.process = None
        self.active_extension = None
        self._cached_voices = None  # Clear voice cache on extension switch

        # OPTIMIZED: Don't recreate HTTP client - connection pooling handles stale connections
        # Just verify port is free using fast socket check
        await asyncio.sleep(0.3)  # Brief wait for OS to release port

        if self._is_port_in_use(TTS_EXTENSION_PORT):
            logger.warning(f"[STOP] Port {TTS_EXTENSION_PORT} still in use after stop")
        else:
            logger.info(f"[STOP] Port {TTS_EXTENSION_PORT} confirmed free")

        return {"status": "stopped", "extension": old_extension}

    async def _unload_any_running_extension(self) -> bool:
        """
        Check if any TTS extension is already running on the shared port and unload it.

        This is critical because only ONE TTS extension can run at a time.
        Previous runs may have left a subprocess running on port TTS_EXTENSION_PORT.

        Returns:
            True if an extension was found and unloaded, False otherwise
        """
        # Check if port is in use by trying to connect to /status endpoint
        try:
            response = await self.client.get(
                f"{TTS_EXTENSION_URL}/status",
                timeout=2.0
            )
            if response.status_code == 200:
                # Something is running on the port
                logger.warning(f"[START] Detected extension already running on port {TTS_EXTENSION_PORT}")
                logger.warning(f"[START] Calling /unload to free the port before starting new extension...")

                # Call unload to gracefully shut it down
                try:
                    unload_response = await self.client.post(
                        f"{TTS_EXTENSION_URL}/unload",
                        timeout=10.0
                    )
                    logger.info(f"[START] /unload returned {unload_response.status_code}: {unload_response.text[:200]}")
                except Exception as e:
                    logger.warning(f"[START] /unload call failed: {e}")

                # Give the process time to shut down gracefully
                await asyncio.sleep(1.0)

                # Verify port is now free
                try:
                    verify_proc = await asyncio.create_subprocess_exec(
                        "lsof", "-tP", "-iTCP", f":{TTS_EXTENSION_PORT}", "-sTCP:LISTEN",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await asyncio.wait_for(verify_proc.communicate(), timeout=3)
                    if verify_proc.returncode == 0 and stdout.strip():
                        pids = [int(pid) for pid in stdout.strip().decode().split('\n') if pid]
                        if pids:
                            logger.warning(f"[START] Port still in use after /unload (PIDs: {pids}), force killing...")
                            for pid in pids:
                                try:
                                    kill_proc = await asyncio.create_subprocess_exec(
                                        "kill", "-9", str(pid),
                                        stdout=asyncio.subprocess.PIPE,
                                        stderr=asyncio.subprocess.PIPE
                                    )
                                    await asyncio.wait_for(kill_proc.communicate(), timeout=3)
                                except Exception:
                                    pass
                            await asyncio.sleep(0.5)
                except (FileNotFoundError, asyncio.TimeoutError):
                    pass

                return True  # Found and attempted to unload an extension

        except (httpx.ConnectError, httpx.ConnectTimeout, asyncio.TimeoutError):
            # Port is free - nothing running
            logger.debug(f"[START] Port {TTS_EXTENSION_PORT} is free - no extension running")
            return False
        except Exception as e:
            logger.warning(f"[START] Unexpected error checking port status: {e}")
            return False

    async def start_extension(self, name: str, config: dict = None) -> dict:
        """
        Start an extension subprocess on port TTS_EXTENSION_PORT.

        IMPORTANT: Only ONE TTS extension can run at a time on the shared port.
        If another extension is already running, we call /unload on it first to gracefully
        shut it down and free the port before starting the new one.

        Args:
            name: Extension name to start
            config: Optional config parameters to set before loading
        """
        # First, check if there's already an extension running on the port
        # This could be from a previous run that didn't clean up properly
        await self._unload_any_running_extension()

        # Then stop our tracked extension if we have one
        if self.process:
            await self.stop_current()

        ext_dir = self._get_extension_dir(name)

        # Normalize extension name: convert hyphens to underscores
        # Frontend may send "index-tts" but directory is "index_tts"
        extension_to_start = name.replace("-", "_")

        # Also check if the non-normalized directory exists first
        if not (TTS_EXTENSIONS_DIR / extension_to_start).exists():
            # Try the original name if normalized version doesn't exist
            if (TTS_EXTENSIONS_DIR / name).exists():
                extension_to_start = name

        # Get python executable using centralized venv detection
        python_exe = self._get_extension_venv_python(extension_to_start)

        logger.info(f"Starting extension: {extension_to_start} (requested: {name})")
        logger.info(f"  Python: {python_exe}")
        logger.info(f"  Working dir: {ext_dir}")

        try:
            # Set PYTHONPATH to include backend directory
            # PYTHONPATH = /path/to/backend so that module src.extensions.libraries.tts_engine.extension_service resolves correctly
            env = {
                **os.environ,
                'PYTHONPATH': str(BACKEND_ROOT),
                'MAIN_API_URL': MAIN_API_URL,  # For extension subprocess callbacks (emotion analysis, etc.)
            }

            # Start the subprocess with the correct registered extension name
            self.process = await asyncio.create_subprocess_exec(
                str(python_exe),
                "-m",
                "src.extensions.libraries.tts_engine.extension_service",
                "--extension", extension_to_start,
                "--port", str(TTS_EXTENSION_PORT),
                "--host", "127.0.0.1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BACKEND_ROOT),
                env=env
            )

            self.active_extension = extension_to_start

            # Wait for service to be ready (server should start quickly)
            await self._wait_for_ready(timeout=30.0)  # 30 seconds for server startup
            logger.info(f"Extension {extension_to_start} subprocess server is ready!")

            # Set config values before loading the extension
            if config:
                try:
                    await self.client.post(f"{TTS_EXTENSION_URL}/config", json=config)
                    logger.info(f"Set config for {extension_to_start}: {config}")
                except Exception as e:
                    logger.warning(f"Failed to set config for {extension_to_start}: {e}")

            # Load the extension model (this may take a long time for large models)
            logger.info(f"Loading extension {extension_to_start} model...")
            # Use longer timeout for model loading
            # IndexTTS vLLM can take 10-15 minutes to load the large model
            # Qwen3-TTS also takes significant time to load (first time downloads from HF)
            load_timeout_seconds = 1200.0 if extension_to_start in ("index_tts_vllm", "qwen3_tts") else 600.0
            load_timeout = httpx.Timeout(load_timeout_seconds, connect=10.0)
            load_response = await self.client.post(f"{TTS_EXTENSION_URL}/load", timeout=load_timeout)
            load_data = load_response.json()
            logger.info(f"Load response: {load_data}")

            # Check if load was successful
            if load_data.get("status") != "ready":
                error_msg = load_data.get("message", "Unknown error")
                logger.error(f"Failed to load extension {extension_to_start}: {error_msg}")
                # Capture stderr for debugging
                await self._log_stderr()
                await self._cleanup_subprocess()
                return {
                    "status": "error",
                    "message": f"Extension {extension_to_start} failed to load: {error_msg}"
                }

            logger.info(f"Extension {extension_to_start} is ready!")

            # Get voice count
            voices = await self._get_voice_count()

            return {
                "status": "ready",
                "extension": extension_to_start,
                "port": TTS_EXTENSION_PORT,
                "voices": voices
            }
        except (TimeoutError, TimeoutException) as e:
            logger.error(f"Extension {extension_to_start} timed out: {e}")
            # Capture stderr for debugging
            await self._log_stderr()
            await self._cleanup_subprocess()
            return {
                "status": "error",
                "message": f"Extension {extension_to_start} timed out during loading. Large models may take up to 10 minutes to load."
            }
        except Exception as e:
            logger.error(f"Extension {extension_to_start} failed to start: {e}")
            # Capture stderr for debugging
            await self._log_stderr()
            await self._cleanup_subprocess()
            return {
                "status": "error",
                "message": f"Extension {extension_to_start} failed to start: {str(e)}"
            }

    async def _log_stderr(self):
        """Log subprocess stderr for debugging."""
        if self.process and self.process.stderr:
            try:
                stderr_data = await asyncio.wait_for(self.process.stderr.read(), timeout=2.0)
                if stderr_data:
                    stderr_text = stderr_data.decode('utf-8', errors='replace')
                    if stderr_text.strip():
                        logger.error(f"Extension subprocess stderr:\n{stderr_text}")
            except asyncio.TimeoutError:
                logger.warning("Timeout reading subprocess stderr")
            except Exception as e:
                logger.warning(f"Error reading subprocess stderr: {e}")

    async def _cleanup_subprocess(self):
        """Clean up the current subprocess."""
        if self.process:
            try:
                # Try graceful shutdown first before reading stderr
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except TimeoutError:
                    logger.warning("Process did not terminate gracefully, killing...")
                    self.process.kill()
                    # Read any stderr after killing for diagnostics
                    if self.process.stderr:
                        try:
                            stderr_data = self.process.stderr.read()
                            if stderr_data:
                                stderr_text = stderr_data.decode('utf-8', errors='replace')
                                if stderr_text.strip():
                                    logger.error(f"Extension subprocess stderr:\n{stderr_text}")
                        except Exception:
                            pass  # Best effort stderr reading
                    await self.process.wait()
            except Exception as e:
                logger.error(f"Error cleaning up subprocess: {e}")
            finally:
                self.process = None
                self.active_extension = None

        # Efficiently wait for port TTS_EXTENSION_PORT to be free using exponential backoff
        logger.info(f"Waiting for port {TTS_EXTENSION_PORT} to be released...")
        import socket
        max_wait = 5.0  # Maximum wait time in seconds
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.05  # Start with 50ms intervals

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                result = sock.connect_ex(('127.0.0.1', TTS_EXTENSION_PORT))
                sock.close()
                if result != 0:  # Port is free
                    logger.info(f"Port {TTS_EXTENSION_PORT} is free")
                    return
            except Exception:
                pass
            await asyncio.sleep(check_interval)
            # Exponential backoff up to 0.5s
            check_interval = min(check_interval * 1.5, 0.5)

        logger.warning(f"Port {TTS_EXTENSION_PORT} still appears to be in use after waiting, proceeding anyway")

    async def _wait_for_ready(self, timeout: float = 300.0):
        """Wait for extension service to be ready."""
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                response = await self.client.get(f"{TTS_EXTENSION_URL}/status")
                if response.status_code == 200:
                    # Server is up - extension may still be loading in background
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)

        raise TimeoutError(f"Service not ready after {timeout}s")

    async def _get_voice_count(self) -> int:
        """Get the number of voices for the active extension."""
        try:
            response = await self.client.get(f"{TTS_EXTENSION_URL}/status")
            if response.status_code == 200:
                data = response.json()
                return data.get("voices", 0)
        except Exception:
            pass
        return 0

    async def get_status(self) -> dict:
        """Get the status of the active extension."""
        # If we have a tracked process, use it
        if self.process:
            # Check if process is still alive
            if self.process.returncode is not None:
                self.process = None
                self.active_extension = None
                return {"status": "died", "active_extension": None}

            try:
                response = await self.client.get(f"{TTS_EXTENSION_URL}/status")
                if response.status_code == 200:
                    data = response.json()
                    data["active_extension"] = self.active_extension
                    return data
            except Exception:
                pass

            return {
                "status": "running",
                "active_extension": self.active_extension,
                "model_loaded": self.model_loaded,
                "voices": await self._get_voice_count(),
                "service_ready": self.service_ready
            }

        # No tracked process - check if an extension is already running on port 9000
        # This handles the case where the backend restarted but the extension subprocess is still running
        await self._detect_running_extension()

        if self.process:
            # Successfully reconnected to running extension
            try:
                response = await self.client.get(f"{TTS_EXTENSION_URL}/status")
                if response.status_code == 200:
                    data = response.json()
                    data["active_extension"] = self.active_extension
                    return data
            except Exception:
                pass

            return {
                "status": "running",
                "active_extension": self.active_extension,
                "model_loaded": self.model_loaded,
                "voices": await self._get_voice_count(),
                "service_ready": self.service_ready
            }

        # Nothing running
        return {"status": "no_extension", "active_extension": None}

    async def _detect_running_extension(self) -> bool:
        """Detect if an extension is already running on port 9000 and reconnect to it.

        Returns True if an extension was detected and reconnected, False otherwise.
        """
        # Check if something is listening on port 9000
        try:
            proc = await asyncio.create_subprocess_exec(
                "lsof", "-tP", "-iTCP", f":{TTS_EXTENSION_PORT}", "-sTCP:LISTEN",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

            if proc.returncode != 0 or not stdout.strip():
                return False

            # Something is running - try to get its status
            try:
                response = await self.client.get(f"{TTS_EXTENSION_URL}/status", timeout=2.0)
                if response.status_code == 200:
                    data = response.json()

                    # Try to identify which extension this is
                    # The extension's status endpoint might not tell us the name directly
                    # So we need to infer it or check a metadata endpoint

                    # For now, we can at least reconnect to the process
                    # Get the PID to track it
                    pids = [int(pid) for pid in stdout.strip().decode().split('\n') if pid]
                    if pids:
                        import psutil
                        pid = pids[0]  # Take the first PID
                        try:
                            # Create a fake Process object for tracking
                            # We can't actually create a real asyncio.subprocess.Process for an existing PID
                            # But we can track the PID and check if it's alive
                            logger.info(f"[DETECT] Found running extension on port {TTS_EXTENSION_PORT}, PID: {pid}")

                            # Try to identify the extension by checking command line
                            p = psutil.Process(pid)
                            cmdline = p.cmdline()

                            # Look for extension name in command line
                            # Extensions are started with python -m src.extensions.services.{name}
                            extension_name = None
                            for part in cmdline:
                                if "services." in part:
                                    extension_name = part.split("services.")[-1]
                                    break

                            if extension_name:
                                logger.info(f"[DETECT] Identified extension as: {extension_name}")
                                self.active_extension = extension_name

                                # Create a wrapper to track this process
                                # We'll create a minimal object that has a pid attribute
                                class RunningProcessWrapper:
                                    def __init__(self, pid):
                                        self.pid = pid
                                        self.returncode = None

                                    def poll(self):
                                        """Check if process is still running."""
                                        try:
                                            p = psutil.Process(self.pid)
                                            if p.is_running():
                                                return None
                                            return self.returncode
                                        except psutil.NoSuchProcess:
                                            self.returncode = 0  # Mark as exited
                                            return 0

                                self.process = RunningProcessWrapper(pid)
                                logger.info(f"[DETECT] Reconnected to extension: {extension_name} (PID: {pid})")
                                return True
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            logger.warning(f"[DETECT] Could not access process {pid}: {e}")
            except Exception as e:
                logger.warning(f"[DETECT] Failed to query running extension status: {e}")
        except (FileNotFoundError, asyncio.TimeoutError) as e:
            pass

        return False

    async def generate_audio(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        pitch: float = 1.0,
        output_format: str = "wav",
        **extra_params
    ) -> tuple[bytes, float]:
        """Generate audio using the active extension."""
        if not self.process:
            raise RuntimeError("No extension is loaded")

        # Check if process is still alive before attempting generation
        if self.process.returncode is not None:
            exit_code = self.process.returncode
            extension_name = self.active_extension or "unknown"
            logger.error(f"Extension process {extension_name} died with exit code {exit_code}")
            await self._log_stderr()
            self.process = None
            self.active_extension = None
            raise RuntimeError(f"Extension process died (exit code: {exit_code})")

        # For qwen3_tts, check if we need to switch model types
        # Model switching requires restarting the extension with new config
        if self.active_extension == "qwen3_tts" and "model_type" in extra_params:
            requested_model_type = extra_params.get("model_type")
            requested_model_size = extra_params.get("model_size", "0.6B")

            # Get current model configuration from extension
            try:
                status_response = await self.client.get(f"{TTS_EXTENSION_URL}/status")
                status_data = status_response.json()
                current_model_type = status_data.get("config", {}).get("model_type", "CustomVoice")
                current_model_size = status_data.get("config", {}).get("model_size", "0.6B")

                # Check if we need to restart with different model
                if current_model_type != requested_model_type or current_model_size != requested_model_size:
                    logger.info(f"Model switch needed: {current_model_type}/{current_model_size} -> {requested_model_type}/{requested_model_size}")
                    logger.info("Restarting qwen3_tts extension with new model configuration...")

                    # Restart with new model configuration
                    await self.start_extension(
                        self.active_extension,
                        config={
                            "model_type": requested_model_type,
                            "model_size": requested_model_size
                        }
                    )
                    logger.info(f"Extension restarted with {requested_model_type} {requested_model_size} model")

                    # Remove model_type and model_size from extra_params since they're now in the config
                    extra_params = {k: v for k, v in extra_params.items() if k not in ("model_type", "model_size")}
            except Exception as e:
                logger.warning(f"Could not check model configuration: {e}, proceeding with current model")

        # Continue with generation (whether we restarted or not)
        logger.info(f"Building generate payload for {voice_id}...")
        # Build payload - extra_params go in a separate field
        payload = {
            "text": text,
            "voice_id": voice_id,
            "speed": speed,
            "pitch": pitch,
            "output_format": output_format,
            "extra_params": extra_params  # Wrap extra params in their own field
        }

        # Use longer timeout for qwen3_tts (voice clone prompt creation can be slow on first run)
        gen_timeout = 300.0 if self.active_extension == "qwen3_tts" else 120.0

        try:
            response = await self.client.post(
                f"{TTS_EXTENSION_URL}/generate",
                json=payload,
                timeout=gen_timeout
            )
        except httpx.RemoteProtocolError as e:
            # Server disconnected - process likely crashed
            logger.error(f"Server disconnected during generation: {e}")
            await self._log_stderr()
            # Check if process died
            if self.process.returncode is not None:
                exit_code = self.process.returncode
                logger.error(f"Extension process died after disconnect (exit code: {exit_code})")
                self.process = None
                self.active_extension = None
                raise RuntimeError(f"Extension crashed during generation (exit code: {exit_code})")
            raise

        if response.status_code != 200:
            raise RuntimeError(f"Generation failed: {response.text}")

        duration = float(response.headers.get("X-Duration", 0))
        return response.content, duration

    async def shutdown(self):
        """Stop the extension and cleanup."""
        await self.stop_current()
        await self.client.aclose()


# Global instance
extension_process_manager = ExtensionProcessManager()
