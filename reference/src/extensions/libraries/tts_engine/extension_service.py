"""
Base HTTP service wrapper for TTS extensions.

Each extension runs in its own subprocess with its own venv to avoid
dependency conflicts. This service provides a simple HTTP API for
load/unload/generate operations.
"""

import argparse
import signal
import sys
from pathlib import Path

import uvicorn

# Add backend to path
# Use constants for consistent path resolution across the codebase
# IMPORTANT: extension_service.py is at backend/src/extensions/libraries/tts_engine/extension_service.py
# Going up 4 levels gets us to backend/src/, so we need to go up 5 levels to get to backend/
# Then import constants to get correct paths
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Import constants to get correct extension paths
from src.constants import TTS_EXTENSIONS_DIR  # noqa: E402

from src.shared.logging_config import get_logger

logger = get_logger(__name__)

# CRITICAL: Add TTS extensions directory to BEGINNING of sys.path
# IMPORTANT: extension_service.py is at backend/src/extensions/libraries/tts_engine/extension_service.py
