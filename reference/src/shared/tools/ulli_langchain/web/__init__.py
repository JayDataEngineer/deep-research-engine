"""Web scraping and search tools.

Provides tools for:
- Web search (multiple backends) - with semaphore rate limiting (max 10 concurrent)
- Web page fetching and reading - with semaphore rate limiting (max 10 concurrent)
- Sitemap discovery
- Web crawling
- Community sentiment scraping (Reddit, 4chan) - with semaphore rate limiting
- Fact-checking claims against web sources
- Get source content (lazy loading for Information Funnel architecture)
- Smart ingestion (semantic search with LangChain embeddings)
- Semantic knowledge retrieval with tier filtering
- Web research agent (autonomous multi-step research)

LANGGRAPH V1 PATTERN (CURRENT):
- All tools use @tool decorator
- Return Command(update={...}) for state updates
- No in-place state mutations
- State flows naturally through the graph
- State is passed via runtime: ToolRuntime parameter

NOTE: web_search_and_scrape has been removed to enforce Information Funnel architecture.
Workers should use web_search (snippets) + get_source_content (lazy loading) instead.
"""

# =============================================================================
# CORE TOOLS - All using @tool decorator pattern
# =============================================================================

from .web_search import web_search
from .web_get import web_get
from .web_agent import web_research_agent
from .community_scrape import community_scrape
from .get_source_content import get_source_content, generate_source_id

# =============================================================================
# ADDITIONAL TOOLS
# =============================================================================

from .web_map import web_map
from .web_crawl import web_crawl
from .fact_check import fact_check, check_claim

# =============================================================================
# SEMANTIC SEARCH AND INGESTION MODULES
# =============================================================================

from .ingestion import ingest_url, ingest_urls, get_vector_store
from .retrieve import (
    lookup_knowledge,
    lookup_knowledge_full,
    lookup_tier_a,
    lookup_tier_b,
    lookup_tier_c,
)

__all__ = [
    # Core Tools - LangGraph V1 pattern (@tool decorator)
    "web_search",
    "web_get",
    "web_research_agent",
    "community_scrape",
    "get_source_content",
    "generate_source_id",
    # V1 Tools - BaseTool class (legacy, to be migrated)
    "web_map",
    "web_crawl",
    "fact_check",
    "check_claim",
    # Semantic search and ingestion
    "ingest_url",
    "ingest_urls",
    "get_vector_store",
    "lookup_knowledge",
    "lookup_knowledge_full",
    "lookup_tier_a",
    "lookup_tier_b",
    "lookup_tier_c",
]
