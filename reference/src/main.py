"""Main FastAPI application entry point.

PURE MICROKERNEL ARCHITECTURE:
=============================
Core responsibilities ONLY:
- Middleware (CORS, GZip, Rate Limiting, Request ID)
- Error handling
- Static file serving
- Extension loading

ALL features are injected by extensions via register_routes():
- Chat endpoints → universal_chat extension
- Voice input → voice_input extension
- TTS → tts_engine extension
- Lorebooks → lorebook_engine extension
- Prompts → prompt_engine extension
- Characters → character_engine extension
- Observability → observability_api extension
- Extension management → extensions_management extension
- All mode extensions → register their own routes
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

# Add src to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from src.constants import ROOT_EXTENSIONS_DIR, Assistant_OUTPUT_DIR
from .config import settings
from .shared.logging_config import configure_logging
from .api.core.errors import (
    APIError,
    api_error_handler,
    http_exception_handler,
    validation_error_handler,
    general_exception_handler,
    RequestIDMiddleware,
)
from .api.core.rate_limit import limiter, RateLimitMiddleware, rate_limit_exception_handler

# Configure shared logging format BEFORE importing other modules
configure_logging()

# Now get logger after configuration
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator:
    """Application lifespan manager.

    Pure microkernel: only bootstraps the app and loads extensions.
    """
    logger.info("Starting Unified LLM Interface Backend (Pure Microkernel)")
    logger.info(f"Model: {settings.llm_model}")
    logger.info(f"Base URL: {settings.llm_base_url}")

    # Initialize unified directory structure
    logger.info("Initializing unified directory structure...")
    from src.extensions.libraries.tts_engine.api.utils.storage import ensure_directories_exist
    ensure_directories_exist()
    logger.info("Unified directories ensured: inputs/, outputs/, models/, extensions/")

    # Initialize Extension Manager (Universal extension system)
    logger.info("Initializing Extension Manager...")
    from src.extensions.core import ExtensionManager, ExtensionContext
    from src.shared.factory import get_llm

    extension_manager = ExtensionManager()

    # Create context with LLM factory for extensions
    extension_context = ExtensionContext(
        llm_factory=get_llm,
        settings=settings,
        manager=extension_manager
    )
    extension_manager.set_context(extension_context)

    # Store extension_manager in app.state for router access
    app.state.extension_manager = extension_manager

    # Load all extensions (they will register their own routes via register_routes())
    extensions_dir = ROOT_EXTENSIONS_DIR
    if extensions_dir.exists():
        logger.info(f"Loading extensions from {extensions_dir}...")
        await extension_manager.load_all(extensions_dir, app=app)
        logger.info(f"Extensions loaded: {list(extension_manager.extensions.keys())}")

    else:
        logger.warning(f"Extensions directory not found: {extensions_dir}")

    # Validate prompt configurations on startup
    logger.info("Validating prompt configurations...")
    from .extensions.libraries.prompt_loader import validate_prompts_on_startup
    try:
        errors = validate_prompts_on_startup(raise_on_error=True)
        if errors:
            logger.warning(f"Prompt validation found {len(errors)} issue(s), but continuing...")
    except Exception as e:
        logger.error(f"Prompt validation failed: {e}")
        raise

    # Clean up old research states on startup
    logger.info("Cleaning up old research states...")
    try:
        from src.extensions.content.deep_research.storage import cleanup_old_states
        cleaned = cleanup_old_states(max_age_hours=24 * 7)  # 7 days
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale research states (older than 7 days)")
        else:
            logger.debug("No stale research states to clean up")
    except Exception as e:
        logger.warning(f"Failed to cleanup old research states: {e}")

    # Initialize prompt stack manager
    logger.info("Initializing prompt stack manager...")
    try:
        from src.extensions.libraries.prompt_engine import init_stack_manager
        await init_stack_manager()
        logger.info("Prompt stack manager initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize prompt stack manager: {e}")

    yield

    # Shutdown: Deactivate all service extensions and unload all extensions
    try:
        if extension_manager:
            await extension_manager.shutdown_all()
            logger.info("All extensions deactivated and unloaded")
    except Exception as e:
        logger.warning(f"Error shutting down extensions: {e}")

    # Shutdown prompt stack manager
    try:
        from src.extensions.libraries.prompt_engine import get_stack_manager
        stack_manager = get_stack_manager()
        await stack_manager.shutdown()
        logger.info("Prompt stack manager shut down")
    except Exception:
        pass  # Manager may not be initialized

    logger.info("Shutting down Unified LLM Interface Backend")


# Create FastAPI app with rate limiter state
app = FastAPI(
    title="Unified LLM Interface API",
    description="Multimodal chat API for Qwen3-Omni and other LLMs",
    version="0.1.0",
    lifespan=lifespan,
    state={"limiter": limiter},
)

# Register exception handlers for consistent error responses
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Add Request ID tracking middleware (first in chain)
app.add_middleware(RequestIDMiddleware)

# Add Rate Limiting middleware (after Request ID)
app.add_middleware(RateLimitMiddleware, default_limits=["200/minute"])

# Configure CORS with production-ready settings
import json

cors_origins = json.loads(settings.cors_origins)

# Security: Validate CORS origins are not overly permissive
if "*" in cors_origins or cors_origins == ["*"]:
    logger.warning(
        "CORS configured to allow all origins ('*'). "
        "This is not recommended for production. "
        "Set CORS_ORIGINS environment variable to specific origins."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Request-ID",
        "Accept-Encoding",
    ],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "X-Request-ID",
    ],
    max_age=600,
)

# Add GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
if settings.upload_dir.exists():
    app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")

if Assistant_OUTPUT_DIR.exists():
    app.mount("/static/output", StaticFiles(directory=str(Assistant_OUTPUT_DIR)), name="companion_output")

# Mount observability routes
from src.extensions.content.deep_research.observability.routes import router as observability_router
app.include_router(observability_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Unified LLM Interface API",
        "version": "0.1.0",
        "architecture": "pure-microkernel",
        "core": "middleware, error-handling, static-files, extension-loading",
        "features": "injected-by-extensions",
        "docs": "/docs",
        "health": "/api/v1/system/health",
        "config": "/api/v1/config",
    }


# Compatibility redirects
@app.get("/api/extensions")
async def redirect_extensions():
    """Redirect /api/extensions to /api/v1/extensions for compatibility."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/extensions")


@app.get("/api/extensions/capabilities")
async def redirect_extensions_capabilities():
    """Redirect /api/extensions/capabilities to /api/v1/system/capabilities."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/system/capabilities")


def main():
    """Main entry point for CLI (uvicorn server)."""
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
