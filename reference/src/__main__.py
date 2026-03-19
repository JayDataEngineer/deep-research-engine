"""Entry point for running the backend server with `python -m src`."""

import os
import sys
from pathlib import Path

# Get directories
src_dir = Path(__file__).parent
backend_dir = src_dir.parent

# Add backend to path (parent of src)
sys.path.insert(0, str(backend_dir))

# Now we can import from src package
from src.main import app
from src.config import settings
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
