"""Web Research Agent tool using Command API pattern (LangGraph V1).

This tool uses the modern LangGraph pattern following LangChain handoffs documentation:
- Uses @tool decorator instead of BaseTool class
- Returns Command(update={...}) to update state
- No in-place state mutations
- State flows naturally through the graph

The agent uses multi-step research with planning, URL selection, critic evaluation,
and automatic persistence to LangGraph Store.

Flow:
    plan → search → select → scrape → extract → critic → write_results
                  ↑                                    ↓
                  └──────── refine (with feedback) ←──┘

    - plan: LLM generates research plan with multiple steps
    - search: Execute web search for each step
    - select: LLM selects best URLs (avoids SEO blogs, paywalls)
    - scrape: Scrape selected URLs
    - extract: LLM extracts relevant findings
    - critic: Evaluates quality, approves or sends feedback
    - write_results: Saves sources to LangGraph Store, synthesizes output

Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""

import asyncio
from .logging_config import get_logger
import re
import time
from typing import Literal, Optional, Dict, Any, List, Set, Type

from pydantic import BaseModel, Field

# Import shared blocklist for URL filtering
from .blocklist import is_url_blocklisted
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage

# Import base class
from ._base_models import BaseWebAgentInput

logger = get_logger(__name__)


# =============================================================================
# TIMING/PROFILING HELPERS
# =============================================================================

def _log_timing(operation: str, timings: Dict[str, float]) -> None:
    """Log timing breakdown for an operation.

    Args:
        operation: Name of the operation (e.g., "web_research", "investigative_research")
        timings: Dict of step name -> time in seconds
    """
    total = sum(timings.values())
    parts = [f"{k}: {v:.2f}s" for k, v in timings.items()]
    logger.info(f"[WEB_AGENT] [{operation}] timing - total: {total:.2f}s ({', '.join(parts)})")


# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

# Check if LangGraph is available
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available - investigative mode disabled")

# Check if LLM is available
try:
    from .factory import get_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM factory not available")


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class WebResearchAgentInput(BaseWebAgentInput):
    """Input for web research agent."""
    # Inherits: query, time_range, category, depth, location from BaseWebAgentInput
    num_results: int = Field(default=10, ge=1, le=50, description="Number of search results per step")
    max_urls: int = Field(default=3, ge=1, le=10, description="Max URLs to scrape per search")
    max_steps: int = Field(default=5, ge=1, le=10, description="Max research steps in plan")
    max_refinements: int = Field(default=2, ge=0, le=5, description="Max query refinement loops")


# =============================================================================
# INTERNAL SEMAPHORE
# =============================================================================

_agent_semaphore: Optional[asyncio.Semaphore] = None
_concurrency_limit: int = 3  # Max 3 concurrent web_research_agent calls (expensive!)


def _get_semaphore():
    """Get or create the shared semaphore."""
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(_concurrency_limit)
    return _agent_semaphore


# =============================================================================
# STATE SCHEMAS (LangGraph V1 - TypedDict)
# =============================================================================

from typing import TypedDict


class InvestigativeState(TypedDict):
    """State for investigative research mode (multi-step with refinement)."""
    # === Core task data ===
    task: str                       # Original research task
    plan: List[str]                 # Research steps (LLM-generated)
    current_step: int               # Index into plan

    # === Research findings ===
    findings: Dict[str, str]        # step -> extracted_info
    search_history: List[Dict]      # For query refinement
    final_synthesis: str            # Final output

    # === Tracking ===
    visited_urls: Set[str]          # Deduplication
    source_ids: List[str]           # Track all source IDs
    raw_sources: Dict[str, Any]     # Collected sources

    # === Flow control ===
    needs_refinement: bool          # Trigger for new search
    refinement_count: int           # Track refinement loops
    total_cycles: int               # TTL counter for soft landing (prevents infinite loops)
    error: Optional[str]            # Error tracking

    # === Critic ===
    critic_approved: bool           # True if critic approves findings
    critic_feedback: str            # Feedback for plan if not approved

    # === Config (immutable per session) ===
    max_steps: int                  # Max research steps
    max_urls_per_step: int          # Max URLs to process per step
    max_refinements: int            # Max query refinements

    # === External state for source collection ===
    worker_id: str                  # Worker ID for tracking
    session_id: Optional[str]       # Session ID for write_file
    store: Optional[Any]            # LangGraph Store for write_file

    # === Tool Model Configuration ===
    tool_model: Optional[str]  # Provider for LLM (bound by research framework)


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS
# =============================================================================

class URLSelection(BaseModel):
    """Structured output for URL selection."""
    indices: List[int] = Field(
        default_factory=list,
        description="List of 0-based indices of the best URLs to scrape"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _llm_call_with_retry(
    llm: Any,
    messages: List,
    max_retries: int = 3,
    operation_name: str = "LLM call",
) -> Any:
    """Call LLM with retry logic.

    Args:
        llm: LangChain LLM instance
        messages: List of messages to send
        max_retries: Maximum retry attempts
        operation_name: Name for logging

    Returns:
        LLM response

    Raises:
        RuntimeError: If all retries fail
    """
    from langchain_core.messages import BaseMessage

    last_error = None
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(messages)
            return response
        except Exception as e:
            last_error = e
            logger.warning(f"[WEB_AGENT] {operation_name} failed attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    raise RuntimeError(
        f"{operation_name} failed after {max_retries} retries. "
        f"Last error: {last_error}"
    )


async def _get_llm(tool_model: Optional[str] = None) -> Optional:
    """Get LLM for agent reasoning.

    Args:
        tool_model: Model name to use (e.g., 'gpt-oss-20b-UD-Q4_K_XL.gguf'). If None, uses default.
    """
    if not LLM_AVAILABLE:
        return None
    try:
        if tool_model:
            return await get_llm(model=tool_model)
        return await get_llm()
    except Exception as e:
        logger.warning(f"[WEB_AGENT] Failed to get LLM: {e}")
        return None


async def _web_search(query: str, num_results: int = 10) -> str:
    """Helper to call web search via MCP server.

    Uses the MCP server at 100.85.22.99:8000/sse which provides
    search_web tool with SearXNG backend.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Formatted search results string
    """
    import time
    import json

    search_start = time.monotonic()
    logger.debug(f"[WEB_AGENT] Running MCP web search: {query[:50]}...")

    try:
        from .tools.logic.web.custom_mcp import mcp_web_search

        # MCP search returns JSON list of content blocks
        result = await mcp_web_search(query, num_results=num_results)

        # Parse the MCP response format
        # Result can be:
        # 1. A list of content blocks: [{'type': 'text', 'text': '{"query":...}'}]
        # 2. A string that's already formatted with **Title** and URL: patterns
        if isinstance(result, list):
            # Extract the text from the first content block
            text_content = None
            for block in result:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_content = block.get('text', '')
                    break

            if text_content:
                # Try to parse the JSON inside
                try:
                    data = json.loads(text_content)
                    results = data.get('results', [])

                    # Format results for web_agent consumption
                    # Expected format: **Title**\nURL: https://...
                    formatted = []
                    for r in results:
                        title = r.get('title', 'No title')
                        url = r.get('url', '')
                        snippet = r.get('snippet', '')
                        formatted.append(f"**{title}**\nURL: {url}\n{snippet}\n")

                    content = "\n".join(formatted)
                except json.JSONDecodeError:
                    # text_content might already be formatted
                    # Check if it has the expected format
                    if '**' in text_content and '\nURL:' in text_content:
                        content = text_content
                    else:
                        # Return as-is for web agent to try parsing
                        content = text_content
            else:
                content = str(result)
        elif isinstance(result, str):
            # Result is already a string - check if it's already formatted
            if '**' in result and '\nURL:' in result:
                content = result
            else:
                # Try to parse as JSON
                try:
                    data = json.loads(result)
                    results = data.get('results', [])
                    formatted = []
                    for r in results:
                        title = r.get('title', 'No title')
                        url = r.get('url', '')
                        snippet = r.get('snippet', '')
                        formatted.append(f"**{title}**\nURL: {url}\n{snippet}\n")
                    content = "\n".join(formatted)
                except (json.JSONDecodeError, AttributeError):
                    # Return as-is for web agent to try parsing
                    content = result
        else:
            content = str(result)

        search_time = time.monotonic() - search_start
        url_count = len(re.findall(r'URL:\s*(https?://[^\s]+)', content, re.MULTILINE))
        logger.info(f"[WEB_AGENT] MCP search completed in {search_time:.2f}s, found {url_count} URLs")

        return content

    except Exception as e:
        logger.error(f"[WEB_AGENT] MCP search failed: {e}")
        raise RuntimeError(f"MCP web search failed: {e}")


async def _web_get(url: str) -> str:
    """Helper to fetch content via MCP server.

    Uses MCP scrape_url tool for full content extraction.
        Uses MCP server via Tailscale Funnel.
    """
    import time
    from .tools.logic.web.custom_mcp import mcp_web_fetch

    fetch_start = time.monotonic()

    logger.debug(f"[WEB_AGENT] MCP fetch: {url}")

    try:
        content = await mcp_web_fetch(url)
        fetch_time = time.monotonic() - fetch_start
        logger.info(f"[WEB_AGENT] MCP fetch completed in {fetch_time:.2f}s")
        return content
    except Exception as e:
        logger.error(f"[WEB_AGENT] MCP fetch failed for {url}: {e}")
        raise


def _extract_urls_from_search(search_results: str) -> List[tuple[str, str]]:
    """Extract (title, url) tuples from web search results.

    Automatically filters out blocklisted URLs to avoid wasting resources
    on known-bad domains (paywalls, PDFs, social media, etc.).
    """
    urls = []
    title_pattern = re.compile(r'^\*\*([^*]+)\*\*$', re.MULTILINE)
    url_pattern = re.compile(r'^URL:\s*(https?://[^\s]+)$', re.MULTILINE)

    titles = title_pattern.findall(search_results)
    urls_found = url_pattern.findall(search_results)

    blocked_count = 0
    for i, url in enumerate(urls_found):
        # Check blocklist BEFORE adding
        is_blocked, reason = is_url_blocklisted(url)
        if is_blocked:
            blocked_count += 1
            continue

        title = titles[i] if i < len(titles) else url
        urls.append((title, url))

    if blocked_count > 0:
        logger.info(f"[WEB_AGENT] Filtered {blocked_count} blocklisted URLs from search results")

    return urls


def _generate_source_id(url: str) -> str:
    """Generate a stable source ID from URL."""
    import hashlib
    return hashlib.sha256(url.encode()).hexdigest()[:12]


async def _ingest_url_to_sources(
    url: str,
    title: str,
    rank: int,
    worker_id: str
) -> Optional[Dict[str, Any]]:
    """Ingest a single URL and return source data.

    This does NOT mutate state - returns source data dict for merging.
    Sources are tracked for Context Engine persistence.

    Args:
        url: The URL to ingest
        title: Source title
        rank: Search result rank (for relevance scoring)
        worker_id: Worker identifier for tracking

    Returns:
        source_data dict if ingested successfully, None otherwise
    """
    try:
        from .tools.logic.web.reader import web_read
        from datetime import datetime

        # Scrape content
        content = await web_read(url=url)

        # Calculate relevance score
        relevance_score = max(0.1, 1.0 - (rank * 0.05))

        # Create snippet
        snippet = content[:300] + "..." if len(content) > 300 else content

        # Generate source ID
        source_id = _generate_source_id(url)

        # Return source data (NOT merging into state yet)
        # Note: Tier system removed - all sources stored via Context Engine
        source_data = {
            "source_id": source_id,
            "url": url,
            "title": title,
            "content": content,
            "snippet": snippet,
            "summary": "",  # Will be populated by async summarizer
            "relevance_score": relevance_score,
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "ingested_by": "web_research_agent",
                "rank": rank
            }
        }

        logger.debug(f"[WEB_AGENT] Ingested {title[:50]}... → {source_id}")
        return source_data

    except Exception as e:
        logger.warning(f"[WEB_AGENT] Failed to ingest {url}: {e}")
        return None


# =============================================================================

# INVESTIGATIVE RESEARCH MODE (LangGraph)
# =============================================================================

async def _plan_research(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Generate a research plan with LLM.

    Has retry logic and fallback to single-step plan if parsing fails.
    """
    tool_model = state.get("tool_model")
    llm = await _get_llm(tool_model=tool_model)
    if not llm:
        raise RuntimeError(
            "LLM required for investigative research planning. "
            "Ensure LLM factory is configured and available."
        )

    task = state["task"]
    max_steps = state.get("max_steps", 5)

    prompt = f"""Generate {max_steps} SEARCH QUERIES for this research task.

Task: {task}

Each query should:
- Be a SHORT search phrase (5-10 words max)
- Be suitable for a search engine like Google
- Focus on finding specific, verifiable information

IMPORTANT: Output actual search queries, NOT research instructions.
BAD: "Step 1: Search academic databases for papers on quantum error correction"
GOOD: "quantum error correction breakthroughs 2025"

Output format (strict JSON list):
["search query 1", "search query 2", "search query 3"]

Only output the JSON list, nothing else."""

    from langchain_core.messages import SystemMessage, HumanMessage
    import json

    messages = [
        SystemMessage(content="You are a research strategist. Output ONLY valid JSON lists."),
        HumanMessage(content=prompt)
    ]

    plan = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = await _llm_call_with_retry(
                llm, messages, max_retries=1, operation_name="Plan generation"
            )

            plan_text = response.content.strip()

            # Extract JSON from code blocks if present
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_text:
                plan_text = plan_text.split("```")[1].split("```")[0].strip()

            # Try to parse JSON
            plan = json.loads(plan_text)

            if isinstance(plan, list) and len(plan) > 0:
                break  # Success!

            # Empty list - retry
            logger.warning(f"[WEB_AGENT] Plan generation attempt {attempt + 1} got empty list")
            plan = None

        except json.JSONDecodeError as e:
            logger.warning(f"[WEB_AGENT] Plan generation attempt {attempt + 1} JSON parse failed: {e}")
            messages.append(HumanMessage(content="Please output ONLY a valid JSON list like: [\"Step 1: ...\", \"Step 2: ...\"]"))
        except Exception as e:
            logger.warning(f"[WEB_AGENT] Plan generation attempt {attempt + 1} failed: {e}")

    # Fallback: Single-step plan
    if not plan:
        logger.warning(f"[WEB_AGENT] Plan generation failed after {max_retries} attempts, using single-step fallback")
        plan = [task]

    logger.info(f"[WEB_AGENT] Generated {len(plan)} step plan: {[p[:50] for p in plan]}")
    state["plan"] = plan[:max_steps]
    state["current_step"] = 0

    return state

    return state


async def _search_step(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Execute search for the current step."""
    # CRITICAL FIX: Add bounds check to prevent IndexError
    current_step = state["current_step"]
    plan = state.get("plan", [])

    if not plan:
        logger.warning("[WEB_AGENT] No plan found in state, marking for refinement")
        state["needs_refinement"] = True
        state["error"] = "No research plan available"
        return state

    if current_step >= len(plan):
        logger.warning(
            f"[WEB_AGENT] current_step ({current_step}) >= plan length ({len(plan)}), "
            f"marking for critic"
        )
        # Don't set needs_refinement - let the natural flow go to critic
        return state

    step = plan[current_step]

    # SOFT LANDING: Increment total cycles for TTL tracking
    state["total_cycles"] = state.get("total_cycles", 0) + 1

    search_start = time.monotonic()
    logger.info(f"[WEB_AGENT] Searching query {state['current_step'] + 1}/{len(plan)}: {step[:50]}...")

    # Search for this step
    search_results = await _web_search(step, num_results=20)
    search_time = time.monotonic() - search_start

    # Extract all URLs
    url_tuples = _extract_urls_from_search(search_results)

    if not url_tuples:
        state["needs_refinement"] = True
        state["error"] = f"No search results found for: {step}"
        return state

    logger.info(f"[WEB_AGENT] Search completed in {search_time:.2f}s, found {len(url_tuples)} URLs")

    # Store search results for selection
    state["search_history"].append({
        "step": step,
        "results": search_results,
        "urls": url_tuples,
        "selected_urls": [],
        "scraped": []
    })

    return state


async def _select_urls(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Select the best URLs from search results using LLM.

    Has retry logic and fallback to first N URLs if parsing fails.
    """
    tool_model = state.get("tool_model")
    llm = await _get_llm(tool_model=tool_model)
    if not llm:
        raise RuntimeError(
            "LLM required for URL selection in investigative mode. "
            "Ensure LLM factory is configured and available."
        )

    # CRITICAL FIX: Add comprehensive bounds checks to prevent IndexError
    current_step = state["current_step"]
    plan = state.get("plan", [])
    search_history = state.get("search_history", [])

    # Check 1: Plan exists and current_step is within bounds
    if not plan:
        logger.warning("[WEB_AGENT] No plan found in state, marking for refinement")
        state["needs_refinement"] = True
        state["error"] = "No research plan available"
        return state

    if current_step >= len(plan):
        logger.warning(
            f"[WEB_AGENT] current_step ({current_step}) >= plan length ({len(plan)}), "
            f"skipping URL selection and marking for critic"
        )
        state["needs_refinement"] = True
        state["error"] = f"Plan step {current_step} does not exist (plan has {len(plan)} steps)"
        return state

    # Check 2: search_history has the current step's data
    if not search_history:
        logger.warning("[WEB_AGENT] Empty search_history, marking for refinement")
        state["needs_refinement"] = True
        state["error"] = "No search results available (search_history is empty)"
        return state

    search_data = search_history[-1]
    url_tuples = search_data.get("urls", [])

    if not url_tuples:
        logger.warning("[WEB_AGENT] No URLs in search data, marking for refinement")
        state["needs_refinement"] = True
        state["error"] = "No URLs found in search results"
        return state

    step = plan[current_step]
    max_urls = state.get("max_urls_per_step", 3)

    # Build snippet summary for LLM
    snippets = []
    for i, (title, url) in enumerate(url_tuples[:15]):
        snippets.append(f"{i}. {title}\n   URL: {url}")

    prompt = f"""Research Step: {step}

I found these search results. Select the {max_urls} BEST URLs that are most likely to contain
verifiable facts, official documentation, or primary sources.

IGNORE:
- Generic landing pages
- SEO blogs with listicles
- Content farms (Medium, Substack personal blogs unless expert-written)
- Paywall sites

PRIORITIZE:
- Official documentation
- Academic sources
- Government sources
- Technical specifications
- Primary sources (court documents, official reports)

Results (use 0-based indices):
{chr(10).join(snippets)}

Select the best URLs by their 0-based index numbers.
"""

    selected_urls = []

    # Use structured output with proper message format
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content="You are a research URL selector. Select the best URLs by returning their indices."),
        HumanMessage(content=prompt)
    ]

    # 3 retries for reliability
    for attempt in range(3):
        try:
            structured_llm = llm.with_structured_output(URLSelection)
            result = await structured_llm.ainvoke(messages)

            # Filter to valid indices
            valid_indices = [i for i in result.indices if 0 <= i < len(url_tuples)]

            if valid_indices:
                # Remove duplicates, preserve order
                seen = set()
                unique_indices = []
                for idx in valid_indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)

                for idx in unique_indices[:max_urls]:
                    selected_urls.append(url_tuples[idx])
                logger.info(f"[WEB_AGENT] Selected {len(selected_urls)} URLs: {unique_indices[:max_urls]}")
                break
            else:
                logger.warning(f"[WEB_AGENT] Attempt {attempt + 1}: no valid indices (got {result.indices})")

        except Exception as e:
            logger.warning(f"[WEB_AGENT] Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(0.5)

    # If LLM rejected all URLs, use first N as fallback
    if not selected_urls:
        logger.warning(f"[WEB_AGENT] LLM rejected all URLs, using first {max_urls} as fallback")
        selected_urls = url_tuples[:max_urls]

    logger.info(f"[WEB_AGENT] Selected {len(selected_urls)} URLs from {len(url_tuples)} results")
    search_data["selected_urls"] = selected_urls

    return state


async def _scrape_urls(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Scrape selected URLs and extract relevant information."""
    # CRITICAL FIX: Add bounds check to prevent IndexError
    current_step = state["current_step"]
    plan = state.get("plan", [])
    search_history = state.get("search_history", [])

    if not plan or current_step >= len(plan):
        logger.warning(
            f"[WEB_AGENT] Invalid state for scraping: plan={len(plan) if plan else 0}, current_step={current_step}"
        )
        state["needs_refinement"] = True
        return state

    if not search_history:
        logger.warning("[WEB_AGENT] No search history available for scraping")
        state["needs_refinement"] = True
        return state

    step = plan[current_step]
    search_data = search_history[-1]
    selected_urls = search_data.get("selected_urls", [])
    worker_id = state.get("worker_id", "web_research_agent")

    if not selected_urls:
        state["needs_refinement"] = True
        return state

    scrape_start = time.monotonic()
    logger.info(f"[WEB_AGENT] Scraping {len(selected_urls)} URLs...")

    # Scrape in parallel and collect sources
    async def scrape_single(title: str, url: str, rank: int) -> tuple:
        try:
            # Get source data
            source_data = await _ingest_url_to_sources(
                url=url,
                title=title,
                rank=rank,
                worker_id=worker_id
            )

            content = await _web_get(url)

            # Check for common failure patterns
            if "Failed to scrape" in content or "temporarily unavailable" in content:
                return (title, url, "USELESS: Failed to fetch", source_data)
            if len(content) < 100:
                return (title, url, "USELESS: Too short/no content", source_data)
            return (title, url, content, source_data)
        except Exception as e:
            return (title, url, f"USELESS: {str(e)}", None)

    scrape_tasks = [
        scrape_single(title, url, i)
        for i, (title, url) in enumerate(selected_urls)
    ]
    results = await asyncio.gather(*scrape_tasks)
    scrape_time = time.monotonic() - scrape_start

    # Store scraped content
    search_data["scraped"] = []

    # Initialize raw_sources if not present
    if "raw_sources" not in state:
        state["raw_sources"] = {}

    successful_scrapes = 0
    for title, url, content, source_data in results:
        if content.startswith("USELESS"):
            logger.warning(f"[WEB_AGENT] URL failed: {url[:50]}... - {content.split(':')[1] if ':' in content else content}")
        else:
            successful_scrapes += 1
            search_data["scraped"].append((title, url, content, source_data))
            if source_data:
                # Add to raw_sources
                state["raw_sources"][source_data["source_id"]] = source_data
                if "source_ids" not in state:
                    state["source_ids"] = []
                state["source_ids"].append(source_data["source_id"])

    logger.info(f"[WEB_AGENT] Scraping completed in {scrape_time:.2f}s, {successful_scrapes}/{len(selected_urls)} successful")

    if not search_data["scraped"]:
        state["needs_refinement"] = True
        return state

    return state


async def _extract_findings(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Extract relevant information from scraped content."""
    tool_model = state.get("tool_model")
    llm = await _get_llm(tool_model=tool_model)
    if not llm:
        raise RuntimeError(
            "LLM required for findings extraction in investigative mode. "
            "Ensure LLM factory is configured and available."
        )

    # CRITICAL FIX: Add bounds checks to prevent IndexError
    current_step = state["current_step"]
    plan = state.get("plan", [])
    search_history = state.get("search_history", [])

    if not plan or current_step >= len(plan):
        logger.warning(f"[WEB_AGENT] Invalid state for extraction, marking for refinement")
        state["needs_refinement"] = True
        state["error"] = f"Plan step {current_step} does not exist (plan has {len(plan)} steps)"
        return state

    if not search_history:
        logger.warning("[WEB_AGENT] Empty search_history for extraction, marking for refinement")
        state["needs_refinement"] = True
        state["error"] = "No search results available (search_history is empty)"
        return state

    step = plan[current_step]
    search_data = search_history[-1]

    # Combine all scraped content
    combined_content = []
    for title, url, content, _ in search_data.get("scraped", []):
        combined_content.append(f"## {title}\nSource: {url}\n\n{content[:15000]}")

    prompt = f"""Extract information relevant to this research step: "{step}"

Source content:
{chr(10).join(combined_content)}

EXTRACTION RULES:
1. Extract ANY facts, figures, or information that could be relevant to the research step
2. Be LENIENT - include partial matches, tangentially related info, or context
3. Include quotes and source references
4. If the content is truly empty or completely unrelated, summarize what was found
5. NEVER respond with "USELESS" - always extract whatever you can find

Provide a structured summary of findings. If nothing relevant was found, briefly describe what the sources actually contained."""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = await llm.ainvoke([
            SystemMessage(content="You are a research analyst. Extract relevant facts with quotes. Be lenient - extract partial matches and context."),
            HumanMessage(content=prompt)
        ])

        extracted = response.content.strip()

        # Record extracted findings - LLM now always returns something
        state["findings"][state["plan"][state["current_step"]]] = extracted
        logger.info(f"[WEB_AGENT] Extracted {len(extracted)} chars for step: {step[:50]}...")

        # Increment step counter for next iteration
        state["current_step"] += 1

    except Exception as e:
        raise RuntimeError(
            f"Findings extraction failed in investigative mode. "
            f"LLM invocation error: {e}"
        ) from e

    return state


# =============================================================================
# CRITIC NODE - Evaluates research quality
# =============================================================================

async def _critic(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Critic evaluates the quality of research findings.

    Decides:
    - APPROVED: Findings are good, proceed to write_results
    - NEEDS_REFINEMENT: Findings are weak, loop back to plan with feedback

    The critic is called AFTER all steps are complete to evaluate overall quality.
    """
    tool_model = state.get("tool_model")
    llm = await _get_llm(tool_model=tool_model)
    if not llm:
        # No LLM available - auto-approve
        logger.warning("[WEB_AGENT] No LLM for critic, auto-approving")
        state["critic_approved"] = True
        state["critic_feedback"] = ""
        return state

    # Build summary of findings for critic - give FULL context
    findings_summary = []
    for step, finding in state.get("findings", {}).items():
        findings_summary.append(f"**Step: {step}**\n{finding}")  # Full finding, no truncation

    # Also include source info
    sources_info = []
    for source_id, source_data in state.get("raw_sources", {}).items():
        title = source_data.get("title", "Unknown")[:80]
        url = source_data.get("url", "")[:100]
        sources_info.append(f"- {title}\n  {url}")

    if not findings_summary:
        # No findings - need to refine
        state["critic_approved"] = False
        state["critic_feedback"] = "No findings were extracted. Try different search queries."
        logger.warning("[WEB_AGENT] Critic: No findings, needs refinement")
        return state

    prompt = f"""You are a research quality critic. Evaluate these research findings.

ORIGINAL TASK: {state['task']}

FINDINGS:
{chr(10).join(findings_summary)}

SOURCES USED:
{chr(10).join(sources_info) if sources_info else 'No sources recorded'}

Evaluate:
1. Are the findings useful for addressing the original task?
2. Are there concrete facts/quotes (not vague summaries)?
3. Is the source quality acceptable (not just SEO blogs)?

Respond in this EXACT format:
DECISION: APPROVED or NEEDS_REFINEMENT
FEEDBACK: [If NEEDS_REFINEMENT, explain what's missing or what to try differently]"""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = await llm.ainvoke([
            SystemMessage(content="You are a strict research quality critic. Be critical but fair."),
            HumanMessage(content=prompt)
        ])

        critic_response = response.content.strip()

        # EDGE CASE: Empty or malformed critic response - default to needs refinement
        if not critic_response or len(critic_response) < 10:
            logger.warning("[WEB_AGENT] Critic response empty or too short, defaulting to NEEDS_REFINEMENT")
            state["critic_approved"] = False
            state["critic_feedback"] = "Critic response was malformed. Please try different search queries."
            return state

        # Parse decision
        if "DECISION: APPROVED" in critic_response.upper():
            state["critic_approved"] = True
            state["critic_feedback"] = ""
            logger.info("[WEB_AGENT] Critic: APPROVED - proceeding to write_results")
        else:
            state["critic_approved"] = False
            # Extract feedback
            feedback_match = re.search(r'FEEDBACK:\s*(.+)', critic_response, re.DOTALL | re.IGNORECASE)
            state["critic_feedback"] = feedback_match.group(1).strip() if feedback_match else "Findings insufficient, try different approach."
            logger.warning(f"[WEB_AGENT] Critic: NEEDS_REFINEMENT - {state['critic_feedback']}")

    except Exception as e:
        logger.warning(f"[WEB_AGENT] Critic failed, auto-approving: {e}")
        state["critic_approved"] = True
        state["critic_feedback"] = ""

    return state


def _critic_decision(state: InvestigativeState) -> str:
    """Route based on critic decision.

    Returns:
        - "approved": Go to write_results
        - "refine": Loop back to plan with feedback
    """
    max_refinements = state.get("max_refinements", 2)
    refinement_count = state.get("refinement_count", 0)

    if state.get("critic_approved", False):
        return "approved"

    if refinement_count < max_refinements:
        return "refine"

    # Max refinements reached - proceed anyway with what we have
    logger.warning(f"[WEB_AGENT] Max refinements ({max_refinements}) reached, proceeding with current findings")
    return "approved"


# =============================================================================
# SAVE SOURCES - Write to LangGraph Store
# =============================================================================

async def _save_sources_to_store(
    state: InvestigativeState,
    store: Optional[Any],
) -> List[str]:
    """Save research sources to LangGraph Store via write_file pattern.

    This is called AFTER critic approves, before write_results.
    Sources are saved deterministically (not LLM-driven).

    Uses hierarchical namespace for GRAPH_BUILDER prefix queries:
    - Save: (session_id, "memories", producer_id)
    - Query all: store.asearch((session_id, "memories"))

    Args:
        state: Current research state
        store: LangGraph Store instance

    Returns:
        List of saved memory keys
    """
    if not store:
        logger.warning("[WEB_AGENT] No store available, skipping source save")
        return []

    session_id = state.get("session_id")
    if not session_id:
        logger.warning("[WEB_AGENT] No session_id, skipping source save")
        return []

    saved_keys = []
    raw_sources = state.get("raw_sources", {})
    producer_id = state.get("worker_id", "web_agent")

    # Use shared persistence layer for consistency
    from .memory.persistence import save_findings_to_memories

    saved_keys = await save_findings_to_memories(
        store=store,
        session_id=session_id,
        findings=raw_sources,
        worker_id=producer_id,
    )

    logger.info(f"[WEB_AGENT] Saved {len(saved_keys)} sources to LangGraph Store")
    return saved_keys


def _should_continue(state: InvestigativeState) -> str:
    """Decide whether to continue to next step or go to critic.

    This is called after extract to check if we should:
    - Continue to next step in plan
    - Go to critic (all steps done)

    Note: current_step is already incremented by _extract_findings.
    Per-step failures are recorded in findings as "[No useful information extracted]".
    The critic evaluates overall quality, not per-step success.
    """
    current_step = state["current_step"]
    plan_length = len(state["plan"])

    if current_step < plan_length:
        # More steps to do - continue
        logger.info(f"[WEB_AGENT] Step {current_step}/{plan_length} next...")
        return "continue"

    # All steps done - go to critic
    logger.info("[WEB_AGENT] All steps complete, going to critic...")
    return "critic"


# =============================================================================
# SOFT LANDING - Graceful exit when hitting cycle limits
# =============================================================================

# Maximum total cycles before soft landing (prevents infinite loops)
MAX_TOTAL_CYCLES = 30


def _check_search_success(state: InvestigativeState) -> str:
    """Check if search was successful before routing to select.

    This is called after search to decide:
    - select: Search returned URLs, proceed to selection
    - refine: Search failed, try refinement
    - soft_exit: Hit cycle limit, exit gracefully with partial results

    This prevents the linear flow bug where empty search_history
    would still route to select, causing cascading failures.
    """
    search_history = state.get("search_history", [])
    total_cycles = state.get("total_cycles", 0)
    max_refinements = state.get("max_refinements", 2)
    refinement_count = state.get("refinement_count", 0)

    # Check if we hit the hard cycle limit
    if total_cycles >= MAX_TOTAL_CYCLES:
        logger.warning(f"[WEB_AGENT] Total cycles ({total_cycles}) >= limit ({MAX_TOTAL_CYCLES}), soft exit")
        return "soft_exit"

    # Check if search returned any URLs
    if not search_history or len(search_history) == 0:
        # Search failed - check if we can refine
        if refinement_count >= max_refinements:
            logger.warning(f"[WEB_AGENT] Search failed and max refinements reached, soft exit")
            return "soft_exit"
        logger.warning("[WEB_AGENT] Search failed, routing to refine")
        return "refine"

    # Search succeeded - proceed to select
    return "select"


async def _soft_exit(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Exit gracefully with partial results when hitting cycle limits.

    This is the "soft landing" pattern - instead of raising GraphRecursionError
    which would crash the parent LLM, we return whatever partial data we found
    with a clear explanation.
    """
    logger.warning(f"[WEB_AGENT] SOFT EXIT: total_cycles={state.get('total_cycles', 0)}, findings={len(state.get('findings', {}))}")

    findings = state.get("findings", {})
    task = state.get("task", "research")

    if findings:
        # We have some partial results
        findings_text = "\n\n".join([
            f"**{step}**\n{finding}"
            for step, finding in findings.items()
        ])
        state["final_synthesis"] = (
            f"## Partial Research Results\n\n"
            f"*Note: Research was terminated early due to cycle limits. "
            f"Found {len(findings)} partial result(s).*\n\n"
            f"### Findings\n\n{findings_text}"
        )
    else:
        # No results at all
        state["final_synthesis"] = (
            f"## Research Unavailable\n\n"
            f"*Note: Could not complete research on '{task[:50]}...' "
            f"after multiple attempts. No relevant information was found.*"
        )

    return state


async def _refine_search(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Regenerate plan with critic feedback and restart research.

    Called when critic rejects findings. Goes back to plan phase
    with the critic's feedback incorporated.
    """
    tool_model = state.get("tool_model")
    llm = await _get_llm(tool_model=tool_model)
    if not llm:
        raise RuntimeError(
            "LLM required for plan refinement in investigative mode. "
            "Ensure LLM factory is configured and available."
        )

    task = state["task"]
    critic_feedback = state.get("critic_feedback", "")
    previous_findings = state.get("findings", {})

    prompt = f"""The previous search queries didn't yield good results.

ORIGINAL TASK: {task}

CRITIC FEEDBACK: {critic_feedback}

PREVIOUS FINDINGS (partial):
{chr(10).join([f'- {k}: {v[:200]}...' for k, v in previous_findings.items()])}

Generate {state.get('max_steps', 5)} NEW SEARCH QUERIES that address the critic's feedback:
- Try different keywords and phrases
- Focus on missing information
- Use more specific or more general terms

IMPORTANT: Output actual search queries (5-10 words max), NOT research instructions.
BAD: "Step 1: Search for papers about quantum computing"
GOOD: "IBM quantum processor 2025 specifications"

Output format (strict JSON list):
["search query 1", "search query 2", "search query 3"]

Only output the JSON list, nothing else."""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = await llm.ainvoke([
            SystemMessage(content="You are a research strategist. Output ONLY valid JSON lists."),
            HumanMessage(content=prompt)
        ])

        import json
        plan_text = response.content.strip()

        if "```json" in plan_text:
            plan_text = plan_text.split("```json")[1].split("```")[0].strip()
        elif "```" in plan_text:
            plan_text = plan_text.split("```")[1].split("```")[0].strip()

        new_plan = json.loads(plan_text)

        if not isinstance(new_plan, list):
            new_plan = [task]

        # EDGE CASE: Empty plan after refinement - use task as fallback
        if not new_plan:
            logger.warning("[WEB_AGENT] Refinement produced empty plan, using task as fallback")
            new_plan = [task]

        # Reset state for new attempt
        state["plan"] = new_plan[:state.get("max_steps", 5)]
        state["current_step"] = 0
        state["findings"] = {}  # Clear old findings
        state["search_history"] = []  # Clear old searches
        state["needs_refinement"] = False
        state["critic_feedback"] = ""  # Clear feedback

        # CRITICAL FIX: Increment refinement count HERE (not in routing function)
        state["refinement_count"] = state.get("refinement_count", 0) + 1

        # SOFT LANDING: Increment total cycles for TTL tracking
        state["total_cycles"] = state.get("total_cycles", 0) + 1

        logger.info(f"[WEB_AGENT] Refined plan (attempt {state['refinement_count']}, cycle {state['total_cycles']}): {[p[:50] for p in state['plan']]}")

    except Exception as e:
        raise RuntimeError(
            f"Plan refinement failed in investigative mode. "
            f"LLM invocation or JSON parsing error: {e}"
        ) from e

    return state


async def _write_results(
    state: InvestigativeState,
    config: RunnableConfig | None = None,
    *,
    store: Optional[Any] = None
) -> InvestigativeState:
    """Write final results after critic approval.

    1. Saves sources to LangGraph Store (deterministic, not LLM-driven)
    2. Synthesizes findings into output for calling agent
    """
    # Step 1: Save sources to store
    store = store or state.get("store")
    saved_keys = await _save_sources_to_store(state, store)
    logger.info(f"[WEB_AGENT] Saved {len(saved_keys)} sources to memories")

    # Step 2: Synthesize findings
    tool_model = state.get("tool_model")
    llm = await _get_llm(tool_model=tool_model)
    if not llm:
        # Fallback: just combine findings
        state["final_synthesis"] = "\n\n".join([
            f"**{step}**\n{finding}"
            for step, finding in state.get("findings", {}).items()
        ])
        return state

    # Build findings summary
    findings_summary = []
    for step, finding in state.get("findings", {}).items():
        findings_summary.append(f"**Step: {step}**\n{finding}")

    if not findings_summary:
        state["final_synthesis"] = f"# Research: {state['task']}\n\nNo significant findings were extracted."
        return state

    prompt = f"""Synthesize the following research findings into a comprehensive report.

Original Task: {state['task']}

Findings by step:
{chr(10).join(findings_summary)}

Sources used: {len(state.get('raw_sources', {}))}

Create a well-structured report that:
1. Directly addresses the original task
2. Synthesizes findings across all steps
3. Highlights key insights and verifiable facts
4. Notes any gaps or limitations

Use markdown formatting with clear headings."""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = await llm.ainvoke([
            SystemMessage(content="You are a research synthesizer. Create comprehensive, well-structured reports."),
            HumanMessage(content=prompt)
        ])

        state["final_synthesis"] = response.content.strip()
        logger.info(f"[WEB_AGENT] Results written: {len(state['final_synthesis'])} chars, {len(saved_keys)} sources saved")

    except Exception as e:
        raise RuntimeError(
            f"Synthesis failed in investigative mode. "
            f"LLM invocation error: {e}"
        ) from e

    return state


def _build_investigative_graph() -> "StateGraph":
    """Build the LangGraph for investigative research.

    Flow (with soft landing):
    plan → search → [select / refine / soft_exit] → scrape → extract → [continue / critic]
                          ↓                                           ↓
                      (if search                                critic → [approved / refine]
                       failed)                                      ↓
                          ↓                               write_results / soft_exit → END
                      refine → search
                          or
                      soft_exit → END

    Key improvements:
    - Conditional edge after search prevents empty search_history bug
    - Soft exit node provides graceful failure instead of GraphRecursionError
    - Structured output for URL selection ensures valid indices
    """
    builder = StateGraph(InvestigativeState)

    # Add nodes
    builder.add_node("plan", _plan_research)
    builder.add_node("search", _search_step)
    builder.add_node("select", _select_urls)
    builder.add_node("scrape", _scrape_urls)
    builder.add_node("extract", _extract_findings)
    builder.add_node("critic", _critic)
    builder.add_node("refine", _refine_search)
    builder.add_node("write_results", _write_results)
    builder.add_node("soft_exit", _soft_exit)  # NEW: Graceful exit node

    # Set entry point
    builder.set_entry_point("plan")

    # Linear flow: plan → search
    builder.add_edge("plan", "search")

    # CONDITIONAL EDGE: search → [select / refine / soft_exit]
    # This fixes the linear flow bug where empty search_history would still go to select
    builder.add_conditional_edges(
        "search",
        _check_search_success,
        {
            "select": "select",     # Search succeeded, proceed to selection
            "refine": "refine",     # Search failed, try refinement
            "soft_exit": "soft_exit",  # Hit cycle limit, exit gracefully
        }
    )

    # Continue linear flow after select
    builder.add_edge("select", "scrape")
    builder.add_edge("scrape", "extract")

    # After extract: continue to next step OR go to critic
    builder.add_conditional_edges(
        "extract",
        _should_continue,
        {
            "continue": "search",  # More steps to do
            "critic": "critic",    # All steps done, evaluate
        }
    )

    # Critic decides: approved → write_results, refine → refine
    builder.add_conditional_edges(
        "critic",
        _critic_decision,
        {
            "approved": "write_results",  # Good to go
            "refine": "refine",           # Loop back with feedback
        }
    )

    # Refine regenerates plan and restarts
    builder.add_edge("refine", "search")

    # Terminal nodes
    builder.add_edge("write_results", END)
    builder.add_edge("soft_exit", END)  # NEW: Soft exit goes to END

    return builder.compile()


async def _investigative_research(
    task: str,
    max_steps: int = 5,
    max_urls_per_step: int = 3,
    max_refinements: int = 2,
    worker_id: str = "web_research_agent",
    tool_model: Optional[str] = None,
    store: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """Execute investigative research with multi-step planning and refinement.

    Args:
        task: Research task
        max_steps: Max research steps
        max_urls_per_step: Max URLs to process per step
        max_refinements: Max query refinements
        worker_id: Worker ID for source tracking
        tool_model: Provider for LLM (bound by research framework)
        store: LangGraph Store for auto-vectorized persistence (Entry Point B)
        session_id: Session ID for namespace isolation

    Returns:
        Tuple of (synthesized_content, raw_sources_dict)
    """
    # Timing tracking
    timings: Dict[str, float] = {}
    total_start = time.monotonic()

    logger.info(f"[WEB_AGENT] Starting INVESTIGATIVE research: '{task[:50]}...'")

    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "LangGraph is required for investigative research mode. "
            "Install langgraph or use simple mode instead."
        )

    # Initial state
    initial_state: InvestigativeState = {
        "task": task,
        "plan": [],
        "current_step": 0,
        "findings": {},
        "visited_urls": set(),
        "search_history": [],
        "needs_refinement": False,
        "refinement_count": 0,
        "total_cycles": 0,  # SOFT LANDING: TTL counter
        "final_synthesis": "",
        "error": None,
        "critic_approved": False,
        "critic_feedback": "",
        "max_steps": max_steps,
        "max_urls_per_step": max_urls_per_step,
        "max_refinements": max_refinements,
        "source_ids": [],
        "raw_sources": {},  # Collect sources in subgraph state
        "worker_id": worker_id,
        "session_id": session_id,  # For auto-vectorized persistence
        "store": store,             # LangGraph Store for Entry Point B
        "tool_model": tool_model,  # Bind tool model provider
    }

    # Build and run graph
    graph_start = time.monotonic()
    graph = _build_investigative_graph()
    result_state = await graph.ainvoke(initial_state)
    timings["graph_execution"] = time.monotonic() - graph_start

    if result_state.get("error"):
        raise RuntimeError(
            f"Investigative research failed with error: {result_state['error']}"
        )

    if not result_state.get("final_synthesis"):
        raise ValueError(
            "Synthesis output missing from investigative research state. "
            "The synthesis node may have failed silently."
        )

    # Log final timing breakdown
    total_time = time.monotonic() - total_start
    timings["total"] = total_time
    _log_timing("investigative_research", timings)

    # Return both content and collected sources
    return result_state["final_synthesis"], result_state.get("raw_sources", {})


# =============================================================================
# MAIN TOOL FUNCTION
# =============================================================================

@tool
async def web_research_agent(
    query: str,
    num_results: int = 10,
    max_urls: int = 3,
    max_steps: int = 5,
    max_refinements: int = 2,
    runtime: Optional[ToolRuntime] = None,
    tool_model: Optional[str] = None,  # Model name (e.g., "gpt-oss-20b-UD-Q4_K_XL.gguf")
) -> Command:
    """Autonomous web research agent for complex multi-step research tasks.

    IMPORTANT: Use this agent INSTEAD of web_search when:
    - You need comprehensive research across multiple angles
    - The topic requires iterating and refining search queries
    - You need strategic URL selection (avoiding SEO blogs, paywalls)
    - Basic web_search results are insufficient or too noisy
    - You need THOROUGH research with synthesized findings (not just links)

    The web is noisy and dynamic. This agent handles SEO blogs, paywalls, and
    unreliable sources through strategic URL selection and refinement loops.

    The agent uses multi-step research with planning, URL selection, critic evaluation,
    and automatic persistence to LangGraph Store. Research findings are saved to
    (session_id, "memories", "web_agent") namespace for GRAPH_BUILDER access.

    Flow: plan → search → select → scrape → extract → critic → write_results (saves)

    Args:
        query: Research question or task
        num_results: Number of search results to retrieve per step (default: 10)
        max_urls: Max URLs to scrape per search step (default: 3)
        max_steps: Max research steps in plan (default: 5)
        max_refinements: Max query refinement loops (default: 2)
        runtime: ToolRuntime context for accessing state

    Returns:
        Command with ToolMessage (content) and state updates (raw_sources)
    """
    semaphore = _get_semaphore()

    logger.info(
        f"[WEB_RESEARCH_AGENT] Acquiring semaphore (concurrency_limit={_concurrency_limit})..."
    )

    # Determine tool_model (priority: parameter > runtime.state > None)
    actual_tool_model = tool_model
    if actual_tool_model is None and runtime is not None:
        actual_tool_model = runtime.state.get("tool_model")

    if actual_tool_model:
        logger.info(f"[WEB_RESEARCH_AGENT] Using tool model: {actual_tool_model}")

    # Acquire semaphore
    async with semaphore:
        logger.info(f"[WEB_RESEARCH_AGENT] Semaphore acquired, executing research: query={query[:50]}...")

        worker_id = "web_research_agent"
        if runtime is not None:
            worker_id = runtime.state.get("current_worker", "web_research_agent")

        try:
            content = ""
            new_sources = {}

            # Get session_id from runtime (for reference)
            session_id = None
            if runtime is not None:
                session_id = runtime.state.get("session_id")

            # Get store from runtime for auto-vectorized persistence (Entry Point B)
            store = None
            if runtime is not None:
                store = runtime.state.get("store")

            # Always use the full investigative research graph
            # This ensures proper planning, URL selection, critic evaluation, and persistence
            logger.info(f"[WEB_RESEARCH_AGENT] Executing full research graph: max_steps={max_steps}, max_urls={max_urls}, max_refinements={max_refinements}")
            if store:
                logger.info(f"[WEB_RESEARCH_AGENT] LangGraph Store available - sources will be auto-vectorized")
            else:
                logger.warning(f"[WEB_RESEARCH_AGENT] No LangGraph Store - sources will NOT be persisted/vectorized")

            content, sources_dict = await _investigative_research(
                query,
                max_steps=max_steps,
                max_urls_per_step=max_urls,
                max_refinements=max_refinements,
                worker_id=worker_id,
                tool_model=actual_tool_model,
                store=store,           # Pass store for auto-vectorization
                session_id=session_id,  # Pass session_id for namespace isolation
            )
            new_sources = sources_dict
            logger.info(f"[WEB_RESEARCH_AGENT] Research complete: {len(content)} chars, {len(new_sources)} sources")

            # Merge with existing sources and artifact_ids
            if runtime is not None:
                current_sources = runtime.state.get("raw_sources", {})
                merged_sources = {**current_sources, **new_sources}

                logger.info(f"[WEB_RESEARCH_AGENT] Adding {len(new_sources)} sources to state")

                state_update = {
                    "messages": [
                        ToolMessage(
                            content=content,
                            tool_call_id=runtime.tool_call_id
                        )
                    ],
                    "raw_sources": merged_sources,
                }

                return Command(update=state_update)

            # Fallback: no runtime available
            logger.debug("[WEB_RESEARCH_AGENT] No runtime context - returning result without state update")
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=content,
                            tool_call_id=""
                        )
                    ]
                }
            )

        except Exception as e:
            logger.error(f"[WEB_RESEARCH_AGENT] Research failed: {e}", exc_info=True)
            error_msg = f"Web research failed: {str(e)}"

            if runtime is not None:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=error_msg,
                                tool_call_id=runtime.tool_call_id or ""
                            )
                        ]
                    }
                )

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=""
                        )
                    ]
                }
            )


# Export the tool function
__all__ = ["web_research_agent"]
