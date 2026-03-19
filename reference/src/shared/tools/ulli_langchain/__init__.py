"""LangChain Tools - Organized by purpose.

All tools use the modern @tool decorator pattern.

Categories:
- web: Web search and scraping tools
- image: Image generation and editing tools
- memory: RAG, vector store, and structured storage tools
- agent: Multi-agent subagent tools
- core: Core chat tools (character management, etc.)
- osint: Open-source intelligence tools
- vision: Image and video analysis tools
"""

from .web import web_search, web_get, web_map, web_crawl
from .image import generate_image, edit_character_image
from .memory import vector_search, add_documents, list_collections
from .agent import deep_agent, deep_research_subprocess, index_tts_subprocess, run_cli_script
from .osint import (
    osint_username_search,
    osint_email_investigation,
    osint_domain_whois,
    osint_generate_dorks,
    osint_breach_lookup,
)
from .vision import (
    image_analysis,
    video_analysis,
    ui_to_artifact,
    extract_text_from_screenshot,
    diagnose_error_screenshot,
    understand_technical_diagram,
    analyze_data_visualization,
    ui_diff_check,
)

__all__ = [
    # Web tools
    "web_search",
    "web_get",
    "web_map",
    "web_crawl",
    # Image tools
    "generate_image",
    "edit_character_image",
    # Memory tools
    "vector_search",
    "add_documents",
    "list_collections",
    # Agent tools (subprocess wrappers)
    "deep_agent",
    "deep_research_subprocess",
    "index_tts_subprocess",
    "run_cli_script",
    # OSINT tools
    "osint_username_search",
    "osint_email_investigation",
    "osint_domain_whois",
    "osint_generate_dorks",
    "osint_breach_lookup",
    # Vision tools
    "image_analysis",
    "video_analysis",
    "ui_to_artifact",
    "extract_text_from_screenshot",
    "diagnose_error_screenshot",
    "understand_technical_diagram",
    "analyze_data_visualization",
    "ui_diff_check",
]
