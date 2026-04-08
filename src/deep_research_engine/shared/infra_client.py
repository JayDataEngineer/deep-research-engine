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
import logging
import os
import time
from pathlib import Path
from typing import Any, Union

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
_LOAD_TIMEOUT = 1800  # 30 minutes for model loading
_REQUEST_TIMEOUT = 300  # 5 min per request (general)
_ASR_TIMEOUT = 900  # 15 min for ASR (long voice notes need extended processing)

# Valid slot types and their models (from orchestrator config)
SLOT_TYPES = ("llm", "asr", "tts", "image-gen")

# LiteLLM cloud routing (cloud models route through litellm.local, not orchestrator)
_LITELLM_HOST = "litellm.local"
_LITELLM_KEY = os.environ.get("LITELLM_PROXY_KEY", "sk-litellm-e2e-test-master-key")

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

    def __init__(self, url: str | None = None, timeout: int = 30):
        self._timeout = timeout
        self._session = requests.Session()
        self._loaded_models: dict[str, str] = {}  # slot -> model key cache

        if url:
            self._base = url.rstrip("/")
        else:
            self._base = self._discover_endpoint()

        # If going through Traefik on localhost, set Host header on every request
        if "localhost" in self._base or "127.0.0.1" in self._base:
            self._session.headers.update({
                "Host": "api.orchestrator.local",
                "Authorization": f"Bearer {_LITELLM_KEY}",
            })

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
                    headers={"Host": "api.orchestrator.local"},
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

    def status(self) -> dict[str, Any]:
        """Get full infrastructure status.

        Returns dict with:
            vram: {free_mb, used_mb, total_mb}
            slots: {llm: {...}, asr: {...}, tts: {...}, image-gen: {...}}
            available_models: {llm: [...], asr: [...], ...}
        """
        resp = self._get("/status")
        self._raise_for_status(resp, "status")
        return resp.json()

    def vram(self) -> dict[str, Any]:
        """Get VRAM usage."""
        resp = self._get("/vram")
        self._raise_for_status(resp, "vram")
        return resp.json()

    def load(self, slot: str, model: str, timeout: int = 600, wait: bool = True) -> dict[str, Any]:
        """Load a model into a slot. Unloads conflicting models first.

        Args:
            slot: One of 'llm', 'asr', 'tts', 'image-gen'.
            model: Model key (e.g. 'vibevoice-asr', 'kokoro', 'qwen35-35b-a3b-vision').
            timeout: Max seconds to wait for model to become healthy (default 600).
            wait: If True, block until the model is healthy.

        Returns:
            Load response from orchestrator.
        """
        if slot not in SLOT_TYPES:
            raise ValueError(f"Invalid slot '{slot}'. Must be one of {SLOT_TYPES}")

        # Fast path: check in-memory cache (no network call)
        if self._loaded_models.get(slot) == model:
            logger.debug("Model %s already loaded in slot %s (cached)", model, slot)
            return {"status": "already_loaded", "slot": slot, "model": model}

        # Check if already loaded via status API
        current = self._get_slot_info(slot)
        if current:
            current_key = current.get("current_model", {}).get("key")
            is_loading = current.get("loading", False)

            # Model is already loaded and healthy
            if current_key == model and not is_loading:
                logger.info("Model %s already loaded in slot %s", model, slot)
                self._loaded_models[slot] = model
                return {"status": "already_loaded", "slot": slot, "model": model}

            # Slot is currently loading (model key may not be set yet during startup).
            # Wait for it to finish instead of triggering a duplicate load or unload.
            if is_loading:
                logger.info(
                    "Slot %s is already loading (current_key=%s), waiting for it...",
                    slot, current_key,
                )
                # Wait for loading to finish — poll until model key matches or timeout
                self._wait_healthy(slot, model, timeout=timeout)
                self._loaded_models[slot] = model
                return {"status": "already_loaded", "slot": slot, "model": model}

        # Check if VRAM is already sufficient (skip unload if enough VRAM)
        status = self.status()
        if status:
            free_mb = status.get("vram", {}).get("free_mb", 0)
            # Use known VRAM defaults (more reliable than slot info which may be stale)
            known_vram = {
                "gemma4-e4b-vision-fast": 6000,
                "gemma4-e4b-vision": 6000,
                "vibevoice-asr": 18500,
                "qwen35-35b-a3b-vision": 14000,
                "qwen35-35b-a3b-vision-fast": 14000,
            }
            needed_mb = known_vram.get(model, 6000)
            if free_mb >= needed_mb:
                logger.info("VRAM already sufficient (%dMB free, %dMB needed), skipping unload", free_mb, needed_mb)
            else:
                # Free GPU by unloading everything
                logger.info("Unloading all models to free GPU for %s/%s", slot, model)
                self._loaded_models.clear()
                unload_resp = self._post("/unload/all", timeout=300)
                if not unload_resp.ok:
                    logger.warning("Unload all returned %s", unload_resp.status_code)

                # Wait for VRAM to actually be freed (container stop is async)
                self._wait_vram_freed(model)

        # Load the requested model
        logger.info("Loading %s into slot %s", model, slot)
        try:
            resp = self._post(f"/load/{slot}/{model}", timeout=timeout)
            self._raise_for_status(resp, f"load {slot}/{model}")
            result = resp.json()
        except requests.exceptions.Timeout:
            logger.info("Load POST timed out (model still loading), polling for health...")
            result = {"status": "loading", "slot": slot, "model": model}

        if wait:
            self._wait_healthy(slot, model, timeout=timeout)

        self._loaded_models[slot] = model
        return result

    def unload(self, slot: str) -> dict[str, Any]:
        """Unload a model from a slot."""
        self._loaded_models.pop(slot, None)
        resp = self._post(f"/unload/{slot}", timeout=60)
        self._raise_for_status(resp, f"unload {slot}")
        return resp.json()

    def unload_all(self) -> dict[str, Any]:
        """Unload all models from all slots."""
        self._loaded_models.clear()
        resp = self._post("/unload/all", timeout=60)
        self._raise_for_status(resp, "unload all")
        return resp.json()

    def _wait_vram_freed(self, loading_model: str, timeout: int = 90):
        """Poll until VRAM is mostly freed after an unload.

        Container stop is async - the unload API returns immediately
        but the container may still hold GPU memory. This waits until
        enough VRAM is available for the next model.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                # Use /status which includes total_mb
                status = self.status()
                vram = status.get("vram", {})
                free_mb = vram.get("free_mb", 0)
                total_mb = vram.get("total_mb", 1)  # default 1 to avoid div-by-zero
                # Consider VRAM freed when >85% is available
                if free_mb > total_mb * 0.85:
                    logger.info("VRAM freed: %dMB/%dMB available", free_mb, total_mb)
                    return
            except Exception:
                pass
            logger.debug("Waiting for VRAM to be freed...")
            time.sleep(3)
        logger.warning("VRAM not fully freed after %ds, proceeding anyway", timeout)

    def _get_slot_info(self, slot: str) -> dict[str, Any] | None:
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
        messages: Union[str, list[dict[str, str]]],
        model: str = "qwen35-35b-a3b-vision",
        stream: bool = False,
        **kwargs,
    ) -> Union[dict[str, Any], Any]:
        """Chat with an LLM. Auto-loads the model if needed.

        Args:
            messages: A string (single user message) or OpenAI-format messages list.
            model: Model key. Defaults to the GPU LLM.
            stream: If True, returns a streaming response iterator.
            **kwargs: Passed to the OpenAI chat completions API.

        Returns:
            OpenAI-format chat completion response.
        """
        is_cloud = model and "-cloud" in model
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Map orchestrator load model names to LiteLLM model names
        model_name_map = {
            "gemma4-e4b-vision-fast": "gemma-4-e4b",
            "gemma4-e4b-vision": "gemma-4-e4b",
        }
        api_model = model_name_map.get(model, model)
        payload = {"model": api_model, "messages": messages, "stream": stream, **kwargs}

        # Disable thinking mode for text-only local models (prevents token waste on reasoning)
        # Skip for image/multimodal calls as extra_body breaks VLM
        has_images = any(
            isinstance(c, dict) and c.get("type") == "image_url"
            for m in messages
            for c in (m.get("content", []) if isinstance(m.get("content"), list) else [])
        )
        if not has_images and not is_cloud:
            payload["extra_body"] = {"thinking": {"type": "disabled"}}

        for attempt in range(2):
            # Load model if it's a GPU model (skip for CPU/cloud models)
            if model and not model.endswith("-cpu") and not is_cloud:
                self._ensure_service("llm", model)

            # Cloud models route through LiteLLM proxy, local models through orchestrator
            if is_cloud:
                resp = self._session.post(
                    f"{self._base}/v1/chat/completions",
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
                    headers={"Host": _LITELLM_HOST, "Authorization": f"Bearer {_LITELLM_KEY}"},
                )
            else:
                resp = self._post(
                    "/v1/chat/completions",
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
                )

            if resp.ok:
                return resp.json()

            # On 404/502/503, invalidate cache and retry once (orchestrator may have restarted / model busy)
            if resp.status_code in (404, 502, 503) and attempt == 0 and not is_cloud:
                logger.warning(
                    "chat got HTTP %d - invalidating model cache and retrying",
                    resp.status_code,
                )
                self._loaded_models.clear()
                continue

            self._raise_for_status(resp, "chat")

        # Should not reach here, but just in case
        self._raise_for_status(resp, "chat")

    # ------------------------------------------------------------------
    # High-level API: Transcription (ASR)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Union[str, Path, bytes],
        model: str = "vibevoice-asr",
        diarize: bool = True,
        language: str | None = None,
        response_format: str = "verbose_json",
        num_speakers: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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
        # Map orchestrator model key to VibeVoice API model name
        api_model = model
        if model == "vibevoice-asr":
            api_model = "vibevoice-asr-7b"

        data: dict[str, Any] = {
            "model": api_model,
            "diarize": str(diarize).lower(),
            "response_format": response_format,
        }
        if language:
            data["language"] = language
        if num_speakers:
            data["num_speakers"] = str(num_speakers)
        data.update({k: str(v) for k, v in kwargs.items()})

        for attempt in range(2):
            self._ensure_service("asr", model)

            if isinstance(audio, (str, Path)):
                with open(audio, "rb") as f:
                    resp = self._post(
                        "/v1/audio/transcriptions",
                        files={"file": f},
                        data=data,
                        timeout=_ASR_TIMEOUT,
                    )
            elif isinstance(audio, bytes):
                resp = self._post(
                    "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", io.BytesIO(audio))},
                    data=data,
                    timeout=_ASR_TIMEOUT,
                )
            else:
                raise TypeError(f"audio must be str, Path, or bytes, got {type(audio)}")

            if resp.ok:
                return resp.json()

            # On 404/503, invalidate cache and retry once
            if resp.status_code in (404, 503) and attempt == 0:
                logger.warning(
                    "transcribe got HTTP %d - invalidating model cache and retrying",
                    resp.status_code,
                )
                self._loaded_models.clear()
                continue

            self._raise_for_status(resp, "transcribe")

        # Should not reach here
        self._raise_for_status(resp, "transcribe")

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
    ) -> dict[str, Any]:
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

    def available_models(self, slot: str | None = None) -> dict[str, list[dict]]:
        """List available models, optionally filtered by slot type."""
        status = self.status()
        models = status.get("available_models", {})
        if slot:
            return {slot: models.get(slot, [])}
        return models

    def __repr__(self) -> str:
        return f"InfraClient(url={self._base})"
