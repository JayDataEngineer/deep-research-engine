"""
Base HTTP service wrapper for TTS extensions.

Each extension runs in its own subprocess with its own venv to avoid
dependency conflicts. This service provides a simple HTTP API for
load/unload/generate operations.

IMPORTANT: This file is outside the src.extensions package to avoid triggering
the import of all extensions when running in isolated venvs.
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# Determine paths directly without importing src.extensions
# This file is at: backend/src/services/tts_extension_service.py
# Going up 2 levels gets us to backend/, going up 3 gets to project root
_file_path = Path(__file__).resolve()
_services_dir = _file_path.parent
_backend_dir = _services_dir.parent  # backend/src/
_project_root = _backend_dir.parent  # backend/
_backend_root = _project_root  # Alias for clarity

# Add backend/src to path for imports
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Import what we need (avoiding src.extensions)
from pydantic import BaseModel, Field
import uvicorn

# Set up paths
# TTS_EXTENSIONS_DIR is backend/src/extensions/services/tts
TTS_EXTENSIONS_DIR = _backend_dir / "extensions" / "services" / "tts"
BACKEND_ROOT = _backend_root
MAIN_API_URL = os.environ.get("MAIN_API_URL", "http://localhost:8787")

# Get TTS port from constants or environment
try:
    from src.constants import TTS_EXTENSION_PORT
except Exception:
    TTS_EXTENSION_PORT = int(os.environ.get("TTS_EXTENSION_PORT", "9000"))

# Import shared logging config
from src.shared.logging_config import configure_logging

# CRITICAL: Add TTS extensions directory at BEGINNING of sys.path
# This ensures local extensions (like kokoro, maya1) take priority over pip-installed packages
# Backend imports use absolute paths (src.services.tts.xxx) so they won't be shadowed
if str(TTS_EXTENSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(TTS_EXTENSIONS_DIR))

# Also add subdirectories (tts/, etc.) to sys.path for extensions in subfolders
for subdir in TTS_EXTENSIONS_DIR.iterdir():
    if subdir.is_dir() and not subdir.name.startswith('_'):
        if str(subdir) not in sys.path:
            sys.path.insert(0, str(subdir))

# Configure logging with shared format
configure_logging()

logger = logging.getLogger(__name__)

# Debug: Log Python executable at startup
logger.info(f"TTSExtensionService starting with Python: {sys.executable}")
logger.info(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
logger.info(f"TTS_EXTENSIONS_DIR: {TTS_EXTENSIONS_DIR}")
logger.info(f"sys.path[0]: {sys.path[0] if sys.path else 'empty'}")


class GenerateRequest(BaseModel):
    """Request model for TTS generation."""
    text: str
    voice_id: str | None = None  # Optional - some extensions use voice cloning instead
    speed: float = 1.0
    pitch: float = 1.0
    output_format: str = "wav"
    extra_params: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        """Initialize and capture extra fields into extra_params."""
        # Known fields that should NOT go into extra_params
        known_fields = {
            "text", "voice_id", "speed", "pitch", "output_format", "extra_params"
        }
        # Extract extra fields
        extra = {k: v for k, v in data.items() if k not in known_fields}

        # MERGE with existing extra_params instead of replacing
        existing_params = data.get("extra_params", {})
        if isinstance(existing_params, dict):
            existing_params.update(extra)
            data["extra_params"] = existing_params
        else:
            data["extra_params"] = extra

        super().__init__(**data)


class GenerateResponse(BaseModel):
    """Response model for TTS generation."""
    audio_data: bytes
    duration: float


class ExtensionService:
    """HTTP service wrapper for a single TTS extension.

    IMPORTANT: Each subprocess must have completely independent extension state.
    We DO NOT use the shared extension_manager singleton - each subprocess
    imports and instantiates its own extension directly.
    """

    def __init__(self, extension_name: str, port: int):
        self.extension_name = extension_name
        self.port = port
        self.extension = None
        self.is_loaded = False
        self.is_loading = False  # Track if model is currently loading
        self.model_loaded = False  # Track if the TTS model is loaded
        self._extension_module = None  # Store the imported module

    def _ensure_discovered(self):
        """Import the extension module directly (independent discovery)."""
        if not self._extension_module:
            module_name = self.extension_name
            logger.info(f"[SUBPROCESS] Importing extension module: {module_name}")
            logger.info(f"[SUBPROCESS] sys.path[0] = {sys.path[0] if sys.path else 'empty'}")

            try:
                # Import the extension module directly from TTS_EXTENSIONS_DIR
                # Each extension has a register_extension() function
                import importlib

                # TTS_EXTENSIONS_DIR is already .../services/, so we import directly
                # The extension module is at TTS_EXTENSIONS_DIR/tts/<module_name>/
                # We import it like "kokoro" or "espeak_ng" (NOT "services.tts.kokoro")
                self._extension_module = importlib.import_module(module_name)

                logger.info(f"[SUBPROCESS] Successfully imported extension module: {module_name}")
                logger.info(f"[SUBPROCESS] Extension module has register_extension: {hasattr(self._extension_module, 'register_extension')}")
            except ImportError as e:
                logger.error(f"[SUBPROCESS] Failed to import extension module {module_name}: {e}")
                logger.error(f"[SUBPROCESS] Available modules in sys.path:")
                # Log available modules for debugging
                import pkgutil
                for _, modname, ispkg in pkgutil.iter_modules():
                    if modname.startswith(module_name.split('_')[0]):  # Show relevant modules
                        logger.error(f"  - {modname} (package: {ispkg})")
                raise

    def load_extension(self) -> dict:
        """Load the extension and its model (independent, no shared state)."""
        try:
            logger.info(f"[SUBPROCESS] Loading extension: {self.extension_name}")
            logger.info(f"[SUBPROCESS] Before load - is_loaded={self.is_loaded}, is_loading={self.is_loading}, model_loaded={self.model_loaded}")
            self.is_loading = True
            self.model_loaded = False

            # Import the extension module (independent discovery)
            self._ensure_discovered()

            # Get the extension instance via register_extension()
            if not self._extension_module:
                raise RuntimeError(f"Extension module not imported for {self.extension_name}")

            if not hasattr(self._extension_module, 'register_extension'):
                raise RuntimeError(f"Extension {self.extension_name} missing register_extension() function")

            # Create a NEW extension instance (independent state)
            self.extension = self._extension_module.register_extension()

            if not self.extension:
                raise RuntimeError(f"register_extension() returned None for {self.extension_name}")

            # Load the model (call the extension's load method)
            logger.info(f"[SUBPROCESS] Loading model for {self.extension_name}...")
            load_success = self.extension.load()

            # Only set is_loaded if the extension's load() succeeded
            if not load_success:
                raise RuntimeError(f"Extension {self.extension_name}.load() returned False - model initialization failed")

            self.is_loaded = True
            self.is_loading = False
            self.model_loaded = True

            logger.info(f"[SUBPROCESS] After load - is_loaded={self.is_loaded}, is_loading={self.is_loading}, model_loaded={self.model_loaded}")
            logger.info(f"[SUBPROCESS] Extension {self.extension_name} loaded successfully with model")
            return {"status": "ready", "extension": self.extension_name, "model_loaded": True}

        except Exception as e:
            self.is_loading = False
            self.is_loaded = False
            self.model_loaded = False
            logger.error(f"[SUBPROCESS] Error loading extension: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def unload_extension(self) -> dict:
        """Unload the extension and free GPU memory (independent)."""
        try:
            logger.info(f"[SUBPROCESS] Unloading extension: {self.extension_name}")

            # Call unload on the extension instance
            if self.extension and hasattr(self.extension, 'unload'):
                self.extension.unload()

            self.is_loaded = False
            self.model_loaded = False
            self.extension = None

            logger.info(f"[SUBPROCESS] Extension {self.extension_name} unloaded")
            return {"status": "unloaded", "extension": self.extension_name}
        except Exception as e:
            logger.error(f"[SUBPROCESS] Error unloading extension: {e}")
            return {"status": "error", "message": str(e)}

    def get_status(self) -> dict:
        """Get current status of the extension (independent)."""
        if self.is_loading:
            return {
                "status": "loading",
                "extension": self.extension_name,
                "model_loaded": False,
                "voices": 0,
                "service_ready": False
            }

        if self.is_loaded and self.extension:
            voices = len(self.extension.get_voices()) if hasattr(self.extension, 'get_voices') else 0
            return {
                "status": "ready",
                "extension": self.extension_name,
                "model_loaded": self.model_loaded,
                "voices": voices,
                "service_ready": True
            }

        return {
            "status": "unloaded",
            "extension": self.extension_name,
            "model_loaded": False,
            "voices": 0,
            "service_ready": False
        }

    async def generate_audio(self, request: GenerateRequest) -> GenerateResponse:
        """Generate audio from text. Supports both sync and async extensions."""
        logger.debug(f"[SUBPROCESS] generate_audio called - is_loaded={self.is_loaded}, extension_name={self.extension_name}")
        logger.debug(f"[SUBPROCESS] request.extra_params = {request.extra_params}")
        logger.debug(f"[SUBPROCESS] request.text = {request.text[:50] if request.text else None}")
        if not self.is_loaded:
            logger.error(f"[SUBPROCESS] Cannot generate - extension not loaded! is_loaded={self.is_loaded}")
            raise ValueError(f"Extension {self.extension_name} is not loaded")

        # Use self.extension which was set during load_extension()
        # Don't use extension_manager.get_extension() as it might return a different instance
        if not self.extension:
            raise RuntimeError(f"Extension {self.extension_name} not available. Was load_extension() called?")

        # Check if generate_audio is async
        import asyncio
        if asyncio.iscoroutinefunction(self.extension.generate_audio):
            audio_bytes, duration = await self.extension.generate_audio(
                text=request.text,
                voice_id=request.voice_id,
                speed=request.speed,
                pitch=request.pitch,
                **request.extra_params
            )
        else:
            audio_bytes, duration = self.extension.generate_audio(
                text=request.text,
                voice_id=request.voice_id,
                speed=request.speed,
                pitch=request.pitch,
                **request.extra_params
            )

        return GenerateResponse(audio_data=audio_bytes, duration=duration)


def create_app(service: ExtensionService):
    """Create FastAPI app for the extension service."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel

    app = FastAPI(title=f"{service.extension_name} Service")

    class LoadResponse(BaseModel):
        status: str
        extension: str | None = None
        message: str | None = None
        voices: int | None = None

    class StatusResponse(BaseModel):
        status: str  # "loading", "ready", "unloaded", "error"
        extension: str
        model_loaded: bool = False  # Whether the TTS model is loaded
        voices: int | None = None
        service_ready: bool = False  # Whether the service is ready for requests

    @app.post("/load", response_model=LoadResponse)
    async def load():
        """Load the extension model."""
        result = service.load_extension()
        # Use the independent extension instance (not shared extension_manager)
        if result["status"] == "ready" and service.extension:
            result["voices"] = len(service.extension.get_voices()) if hasattr(service.extension, 'get_voices') else 0
        return LoadResponse(**result)

    @app.post("/unload", response_model=LoadResponse)
    async def unload():
        """Unload the extension model."""
        result = service.unload_extension()
        return LoadResponse(**result)

    @app.get("/status", response_model=StatusResponse)
    async def status():
        """Get extension status."""
        result = service.get_status()
        # Use the independent extension instance (not shared extension_manager)
        if service.extension and hasattr(service.extension, 'get_voices'):
            result["voices"] = len(service.extension.get_voices())
        return StatusResponse(**result)

    @app.get("/voices")
    async def voices():
        """Get available voices for this extension."""
        if service.extension and hasattr(service.extension, 'get_voices'):
            return {"voices": service.extension.get_voices()}
        return {"voices": []}

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        """Generate audio from text."""
        # DEBUG: Log the service instance ID and state
        logger.debug(f"[SUBPROCESS] /generate called - service.is_loaded={service.is_loaded}, service.extension_name={service.extension_name}")

        try:
            response = await service.generate_audio(request)
            return Response(
                content=response.audio_data,
                media_type="audio/wav",
                headers={
                    "X-Duration": str(response.duration),
                    "Content-Disposition": f'attachment; filename="output.{request.output_format}"'
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": service.extension_name}

    @app.get("/debug")
    async def debug():
        """Debug endpoint to check internal state."""
        return {
            "extension_name": service.extension_name,
            "is_loaded": service.is_loaded,
            "is_loading": service.is_loading,
            "model_loaded": service.model_loaded,
            "extension_is_none": service.extension is None,
            "extension_has_is_loaded": service.extension is not None and hasattr(service.extension, 'is_loaded'),
            "extension_is_loaded_value": service.extension.is_loaded if service.extension and hasattr(service.extension, 'is_loaded') else None
        }

    @app.post("/config")
    async def set_config(request: dict):
        """Set configuration values before loading the extension."""
        # Use the independent extension instance (not shared extension_manager)
        if service.extension and hasattr(service.extension, 'set_config_value'):
            for key, value in request.items():
                service.extension.set_config_value(key, value)
                logger.info(f"Set config {key}={value} for {service.extension_name}")
        return {"status": "ok", "config": request}

    return app


def main():
    """Run the extension service."""
    parser = argparse.ArgumentParser(description="Run TTS extension service")
    parser.add_argument("--extension", required=True, help="Extension name (e.g., kokoro)")
    parser.add_argument("--port", type=int, default=TTS_EXTENSION_PORT, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # DEBUG: Print all args to see what we're actually receiving
    logger.info(f"=== SUBPROCESS STARTUP DEBUG ===")
    logger.info(f"sys.argv = {sys.argv}")
    logger.info(f"args.extension = {args.extension!r}")
    logger.info(f"args.port = {args.port}")
    logger.info(f"args.host = {args.host}")
    logger.info(f"==================================")
    logger.info(f"Starting {args.extension} service on port {args.port}")

    service = ExtensionService(args.extension, args.port)
    logger.info(f"[DEBUG] service.extension_name = {service.extension_name!r}")

    app = create_app(service)

    # Shutdown handler
    def shutdown_handler(signum, frame):
        logger.info("Shutting down...")
        service.unload_extension()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # START SERVER - don't auto-load, let process manager trigger it
    logger.info("Server started - waiting for /load endpoint to be called")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
