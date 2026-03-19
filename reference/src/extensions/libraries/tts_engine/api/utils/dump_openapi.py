#!/usr/bin/env python3
"""
Utility to dump the OpenAPI schema to a JSON file.
"""
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.main import app
from fastapi.openapi.utils import get_openapi

if __name__ == "__main__":
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    with open("openapi.json", "w") as f:
        json.dump(openapi_schema, f, indent=2)
