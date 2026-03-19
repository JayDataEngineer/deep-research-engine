"""Environment variable caches for deep research config.

These are cached at module load time for performance.
"""

import os
from typing import Optional


# =============================================================================
# ENV VAR CACHE (performance optimization)
# =============================================================================

_LLM_MODEL_NAME: Optional[str] = os.getenv("LLM_MODEL_NAME")
_OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
_LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
_OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
_LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL")
