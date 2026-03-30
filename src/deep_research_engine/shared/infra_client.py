"""Shared Infra Client - Zero-config SDK for the AI platform.

Single-file client that auto-discovers your Docker infrastructure
and manages model lifecycle (GPU loading/unloading) transparently.

Usage:
    import sys; sys.path.insert(0, "../shared-docker-infra/sdk")
    from infra_client import InfraClient

    infra = InfraClient()

    # Chat (auto-loads LLM)
    reply = infra.chat("Explain quantum computing")

    # Transcribe with diarization (auto-loads ASR, unloads LLM)
    result = infra.transcribe("meeting.wav", diarize=True)

    # Text-to-speech (auto-loads TTS)
    audio = infra.synthesize("Hello world", voice="Emma")

    # Check what's running
    status = infra.status()
"""

from __future__ import annotations

import io
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import requests
except ImportError:
    raise ImportError(
        "infra_client requires 'requests'. "
        "Install with: pip install requests"
    )

logger = logging.getLogger("infra_client")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRAEFIK_PORTS = [18880, 8880, 80]
_DOCKER_HOSTNAMES = ["orchestrator:8000", "shared-orchestrator-orchestrator-1:8000"]
_DISCOVER_TIMEOUT = 2  # seconds per probe
_HEALTH_POLL_INTERVAL = 3  # seconds between health checks
_LOAD_TIMEOUT = 300  # max seconds to wait for model load
_REQUEST_TIMEOUT = 600  # max seconds for inference calls

# Valid slot types and their models (from orchestrator config)
SLOT_TYPES = ("llm", "asr", "tts", "image-gen")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InfraError(Exception):
    """Base exception for infra client errors."""


class InfraUnreachable(InfraError):
    """Cannot reach the orchestrator or any infra service."""


class ModelLoadError(InfraError):
    """Failed to load a model within the timeout."""


class ServiceError(InfraError):
    """Service returned an error response."""

    def __init__(self, message: str, status_code: int = 0, response: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class InfraClient:
    """Zero-config client for the shared Docker AI infrastructure.

    Auto-discovers the orchestrator endpoint by probing:
      1. INFRA_ORCHESTRATOR_URL env var
      2. Docker internal DNS (orchestrator:8000)
      3. Traefik on localhost (ports 18880, 8880, 80)

    All methods that require a GPU model automatically handle loading
    and unloading.  You never need to think about GPU management.
    """

    def __init__(self, url: Optional[str] = None, timeout: int = 30):
        self._timeout = timeout
        self._session = requests.Session()

        if url:
            self._base = url.rstrip("/")
        else:
            self._base = self._discover_endpoint()

        # If going through Traefik on localhost, set Host header on every request
        if "localhost" in self._base or "127.0.0.1" in self._base:
            self._session.headers.update({"Host": "orchestrator.local"})

        logger.info("InfraClient connected to %s", self._base)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_endpoint(self) -> str:
        """Find the orchestrator with zero config."""
        # 1. Explicit env var
        env_url = os.environ.get("INFRA_ORCHESTRATOR_URL")
        if env_url:
            return env_url.rstrip("/")

        # 2. Docker internal DNS (we're inside a container on shared-infra)
        for host in _DOCKER_HOSTNAMES:
            try:
                resp = requests.get(
                    f"http://{host}/health", timeout=_DISCOVER_TIMEOUT
                )
                if resp.ok:
                    return f"http://{host}"
            except requests.RequestException:
                pass

        # 3. Traefik on localhost (we're on the host machine)
        for port in _TRAEFIK_PORTS:
            try:
                resp = requests.get(
                    f"http://localhost:{port}/health",
                    headers={"Host": "orchestrator.local"},
                    timeout=_DISCOVER_TIMEOUT,
                )
                if resp.ok:
                    return f"http://localhost:{port}"
            except requests.RequestException:
                pass

        raise InfraUnreachable(
            "Cannot find the orchestrator. "
            "Is Docker running? Try: cd shared-docker-infra && task up"
        )

    # ------------------------------------------------------------------
    # Low-level API
    # ------------------------------------------------------------------

    def _get(self, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        resp = self._session.get(f"{self._base}{path}", **kwargs)
        return resp

    def _post(self, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        resp = self._session.post(f"{self._base}{path}", **kwargs)
        return resp

    @staticmethod
    def _raise_for_status(resp: requests.Response, context: str = ""):
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            msg = f"{context} failed (HTTP {resp.status_code}): {detail}"
            raise ServiceError(msg, status_code=resp.status_code, response=detail)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get full infrastructure status.

        Returns dict with:
            vram: {free_mb, used_mb, total_mb}
            slots: {llm: {...}, asr: {...}, tts: {...}, image-gen: {...}}
            available_models: {llm: [...], asr: [...], ...}
        """
        resp = self._get("/status")
        self._raise_for_status(resp, "status")
        return resp.json()

    def vram(self) -> Dict[str, Any]:
        """Get VRAM usage."""
        resp = self._get("/vram")
        self._raise_for_status(resp, "vram")
        return resp.json()

    def load(self, slot: str, model: str, wait: bool = True) -> Dict[str, Any]:
        """Load a model into a slot. Unloads conflicting models first.

        Args:
            slot: One of 'llm', 'asr', 'tts', 'image-gen'.
            model: Model key (e.g. 'vibevoice-asr', 'kokoro', 'qwen35-35b-a3b-vision').
            wait: If True, block until the model is healthy.

        Returns:
            Load response from orchestrator.
        """
        if slot not in SLOT_TYPES:
            raise ValueError(f"Invalid slot '{slot}'. Must be one of {SLOT_TYPES}")

        # Check if already loaded
        current = self._get_slot_info(slot)
        if current and current.get("current_model", {}).get("key") == model:
            if not current.get("loading", False):
                logger.info("Model %s already loaded in slot %s", model, slot)
                return {"status": "already_loaded", "slot": slot, "model": model}

        # Free GPU by unloading everything
        logger.info("Unloading all models to free GPU for %s/%s", slot, model)
        unload_resp = self._post("/unload/all", timeout=60)
        if not unload_resp.ok:
            logger.warning("Unload all returned %s", unload_resp.status_code)

        # Load the requested model
        # Use a long timeout — loading a 22GB model can take minutes.
        # If the POST itself times out, fall through to health polling.
        logger.info("Loading %s into slot %s", model, slot)
        try:
            resp = self._post(f"/load/{slot}/{model}", timeout=_LOAD_TIMEOUT)
            self._raise_for_status(resp, f"load {slot}/{model}")
            result = resp.json()
        except requests.exceptions.Timeout:
            logger.info("Load POST timed out (model still loading), polling for health...")
            result = {"status": "loading", "slot": slot, "model": model}

        if wait:
            self._wait_healthy(slot, model)

        return result

    def unload(self, slot: str) -> Dict[str, Any]:
        """Unload a model from a slot."""
        resp = self._post(f"/unload/{slot}", timeout=60)
        self._raise_for_status(resp, f"unload {slot}")
        return resp.json()

    def unload_all(self) -> Dict[str, Any]:
        """Unload all models from all slots."""
        resp = self._post("/unload/all", timeout=60)
        self._raise_for_status(resp, "unload all")
        return resp.json()

    def _get_slot_info(self, slot: str) -> Optional[Dict[str, Any]]:
        """Get current slot state from /status."""
        try:
            status = self.status()
            return status.get("slots", {}).get(slot)
        except Exception:
            return None

    def _wait_healthy(self, slot: str, model: str, timeout: int = _LOAD_TIMEOUT):
        """Poll until the model is loaded and healthy."""
        deadline = time.time() + timeout
        last_state = None
        while time.time() < deadline:
            slot_info = self._get_slot_info(slot)
            if slot_info:
                loading = slot_info.get("loading", False)
                current = slot_info.get("current_model")
                state = f"loading={loading}, current_model={current}"
                if state != last_state:
                    logger.info("Slot %s: %s", slot, state)
                    last_state = state

                if current and current.get("key") == model and not loading:
                    logger.info("Model %s is healthy in slot %s", model, slot)
                    return
            time.sleep(_HEALTH_POLL_INTERVAL)

        raise ModelLoadError(
            f"Model {model} in slot {slot} did not become healthy "
            f"within {timeout}s"
        )

    def _ensure_service(self, slot: str, model: str):
        """Ensure a model is loaded, handling GPU contention."""
        self.load(slot, model, wait=True)

    # ------------------------------------------------------------------
    # High-level API: Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Union[str, List[Dict[str, str]]],
        model: str = "qwen35-35b-a3b-vision",
        stream: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], Any]:
        """Chat with an LLM. Auto-loads the model if needed.

        Args:
            messages: A string (single user message) or OpenAI-format messages list.
            model: Model key. Defaults to the GPU LLM.
            stream: If True, returns a streaming response iterator.
            **kwargs: Passed to the OpenAI chat completions API.

        Returns:
            OpenAI-format chat completion response.
        """
        # Load model if it's a GPU model (skip for CPU/fallback models)
        if model and not model.endswith("-cpu"):
            self._ensure_service("llm", model)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        payload = {"model": model, "messages": messages, "stream": stream, **kwargs}
        resp = self._post(
            "/v1/chat/completions",
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        self._raise_for_status(resp, "chat")
        return resp.json()

    # ------------------------------------------------------------------
    # High-level API: Transcription (ASR)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, bytes],
        model: str = "vibevoice-asr",
        diarize: bool = True,
        language: Optional[str] = None,
        response_format: str = "verbose_json",
        num_speakers: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Transcribe audio with optional speaker diarization.

        Auto-loads the ASR model, unloading other models if needed.

        Args:
            audio: Path to audio file (wav/mp3/flac/opus) or raw bytes.
            model: ASR model key. Defaults to 'vibevoice-asr' (GPU, native diarization).
                   Use 'whisper-cpu' for CPU-only transcription.
            diarize: Enable speaker diarization.
            language: Language code (auto-detect if None).
            response_format: 'json' (text only) or 'verbose_json' (segments + speakers).
            num_speakers: Hint for number of speakers (auto-detect if None).
            **kwargs: Additional parameters passed to the API.

        Returns:
            Dict with 'text', 'segments' (if verbose), 'speakers', etc.
        """
        self._ensure_service("asr", model)

        data: Dict[str, Any] = {
            "model": model,
            "diarize": str(diarize).lower(),
            "response_format": response_format,
        }
        if language:
            data["language"] = language
        if num_speakers:
            data["num_speakers"] = str(num_speakers)
        data.update({k: str(v) for k, v in kwargs.items()})

        if isinstance(audio, (str, Path)):
            with open(audio, "rb") as f:
                resp = self._post(
                    "/v1/audio/transcriptions",
                    files={"file": f},
                    data=data,
                    timeout=_REQUEST_TIMEOUT,
                )
        elif isinstance(audio, bytes):
            resp = self._post(
                "/v1/audio/transcriptions",
                files={"file": ("audio.wav", io.BytesIO(audio))},
                data=data,
                timeout=_REQUEST_TIMEOUT,
            )
        else:
            raise TypeError(f"audio must be str, Path, or bytes, got {type(audio)}")

        self._raise_for_status(resp, "transcribe")
        return resp.json()

    # ------------------------------------------------------------------
    # High-level API: Text-to-Speech
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        model: str = "kokoro-cpu",
        voice: str = "Emma",
        output_format: str = "wav",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Synthesize speech from text. Auto-loads TTS model if needed.

        Args:
            text: Text to speak.
            model: TTS model key. Defaults to 'kokoro-cpu' (no GPU needed).
            voice: Voice preset name.
            output_format: 'wav', 'mp3', 'flac', or 'opus'.
            speed: Speech speed (0.25 to 4.0).
            **kwargs: Additional parameters passed to the API.

        Returns:
            Audio bytes in the requested format.
        """
        if not model.endswith("-cpu"):
            self._ensure_service("tts", model)

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": output_format,
            "speed": speed,
            **kwargs,
        }

        resp = self._post(
            "/v1/audio/speech",
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        self._raise_for_status(resp, "synthesize")
        return resp.content

    def synthesize_to_file(
        self,
        text: str,
        path: Union[str, Path],
        **kwargs,
    ) -> Path:
        """Synthesize speech and save to file.

        Args:
            text: Text to speak.
            path: Output file path.
            **kwargs: Passed to synthesize().

        Returns:
            Path to the written file.
        """
        audio = self.synthesize(text, **kwargs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        logger.info("Wrote %d bytes to %s", len(audio), path)
        return path

    # ------------------------------------------------------------------
    # High-level API: Image Generation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        model: str = "comfyui",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt via ComfyUI.

        Auto-loads the image-gen model, unloading other models if needed.

        Args:
            prompt: Text description of the desired image.
            model: Image gen model key. Currently only 'comfyui'.
            **kwargs: Additional ComfyUI parameters.

        Returns:
            Dict with image URL or base64 data.
        """
        self._ensure_service("image-gen", model)

        # ComfyUI has its own API - proxy through orchestrator or call directly
        resp = self._post(
            "/v1/images/generations",
            json={"prompt": prompt, "model": model, **kwargs},
            timeout=_REQUEST_TIMEOUT,
        )
        self._raise_for_status(resp, "generate_image")
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def available_models(self, slot: Optional[str] = None) -> Dict[str, List[Dict]]:
        """List available models, optionally filtered by slot type."""
        status = self.status()
        models = status.get("available_models", {})
        if slot:
            return {slot: models.get(slot, [])}
        return models

    def __repr__(self) -> str:
        return f"InfraClient(url={self._base})"
