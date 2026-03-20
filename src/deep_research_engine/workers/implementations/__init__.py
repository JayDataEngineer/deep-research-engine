"""Worker implementations.

This package contains concrete worker implementations for various tasks.

Available workers:
- synthesis: Formats research findings into structured outputs
- web_researcher: Autonomous web research using Flash Explorer pattern
- chunk_processor: Classifies and extracts core content from text chunks
- multimodal: Processes audio, images, and video with ASR/VLM/face recognition
- scraper: Scrapes documentation websites and generates LLMs.txt
"""

from .chunk_processor import ChunkProcessorWorker
from .multimodal import MultimodalWorker
from .scraper import ScraperWorker
from .synthesis import SynthesisWorker
from .web_researcher import WebResearcherWorker
from .social_archive import SocialArchiveWorker

__all__ = [
    "ChunkProcessorWorker",
    "MultimodalWorker",
    "ScraperWorker",
    "SynthesisWorker",
    "WebResearcherWorker",
    "SocialArchiveWorker",
]
