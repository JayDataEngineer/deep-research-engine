"""Worker implementations.

This package contains concrete worker implementations for various tasks.

Available workers:
- artifact_maker: Formats research findings into structured artifacts
- web_researcher: Autonomous web research using Flash Explorer pattern
- content_classifier: LLM classification and enrichment of text chunks (replaces chunk_processor)
- multimodal: Processes audio, images, and video with ASR/VLM/face recognition
- scraper: Scrapes documentation websites and generates LLMs.txt
- vibevoice_asr: Advanced ASR with built-in speaker diarization (VibeVoice ASR-7B)
- asr_worker: ASR worker (defaults to VibeVoice with diarization)
"""

from .artifact_maker import ArtifactMakerWorker
from .content_classifier import ContentClassifierWorker
from .multimodal import MultimodalWorker
from .scraper import ScraperWorker
from .social_archive import SocialArchiveWorker
from .vibevoice_asr import ASRWorker, VibeVoiceASRWorker
from .web_researcher import WebResearcherWorker

__all__ = [
    "ASRWorker",
    "ArtifactMakerWorker",
    "ContentClassifierWorker",
    "MultimodalWorker",
    "ScraperWorker",
    "SocialArchiveWorker",
    "VibeVoiceASRWorker",
    "WebResearcherWorker",
]
