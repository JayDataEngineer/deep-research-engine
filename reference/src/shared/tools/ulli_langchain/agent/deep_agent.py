"""Experimental Deep Agent Tool.

Uses deepagents library to create an autonomous agent with web tools (search, scrape, map, crawl)
and memory system using LangGraph Store for semantic search.

EXPERIMENTAL: This tool is more complex and may take longer to execute.
"""

import asyncio
from .logging_config import get_logger

from langchain_core.tools import tool

logger = get_logger(__name__)


@tool
async def deep_agent(
    query: str,
    max_steps: int = 10,
    enable_web_map: bool = False,
    enable_web_crawl: bool = False,
    enable_memory: bool = True,
) -> str:
    """EXPERIMENTAL: Advanced web research agent for complex queries.

    Can search, scrape, and optionally map/crawl websites.
    Use for multi-step research tasks like: 'Research and summarize topic X',
    'Find all information about Y from website Z', 'Compare multiple sources on topic'.
    Takes longer than simple search but provides comprehensive results.

    Memory System (enable_memory=True):
    - Write files to /memories/ for persistent storage with semantic search
    - Use search_memories to find previously stored information by meaning
    - Files not in /memories/ are temporary (scratchpad)

    Args:
        query: The research query or task for the agent to accomplish. Be specific about what
               you want to find or accomplish. Examples: 'Research the history of quantum computing',
               'Find all information about climate change on nasa.gov', 'Compare 5 sources on AI safety'.
        max_steps: Maximum number of agent steps (tool calls) to perform. Between 1 and 30.
                   More steps means deeper research but takes longer. Default is 10.
                   For simple queries use 5-10, for complex research use 15-20.
        enable_web_map: Enable web mapping (discovers all URLs on a website). Useful for
                       site exploration to discover all pages. Default is False.
        enable_web_crawl: Enable web crawling (scrapes multiple pages). Useful for
                        comprehensive content extraction from a site. Default is False.
        enable_memory: Enable semantic memory system with search_memories tool.
                      Files written to /memories/ are auto-vectorized for semantic search.
                      Default is True.

    Returns:
        The agent's findings and response to the query, synthesized from multiple tool calls.
    """
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
        from .factory import get_llm
        from src.config import settings
        from .tools.ulli_langchain.web import web_search, web_get

        logger.info(f"[DEEP_AGENT] Starting deep agent for query: {query[:80]}...")
        logger.info(f"[DEEP_AGENT] max_steps={max_steps}, web_map={enable_web_map}, web_crawl={enable_web_crawl}, memory={enable_memory}")

        # Get LLM
        llm = await get_llm()

        # Build tools list - start with basic web tools
        tools = [web_search, web_get]

        # Memory system setup
        store = None
        backend_factory = None

        if enable_memory:
            try:
                # Import memory tools from context_engine
                from src.extensions.features.tools.context_engine.tools.search_memories import search_memories

                # Get LangGraph Postgres Store from context_engine
                from src.extensions.features.tools.context_engine.storage.langgraph_store import get_langgraph_store
                store = await get_langgraph_store()

                # Add search_memories tool
                tools.append(search_memories)
                logger.info("[DEEP_AGENT] search_memories tool enabled with LangGraph Store")

            except ImportError as e:
                logger.warning(f"[DEEP_AGENT] Memory system not available: {e}")
                enable_memory = False
            except Exception as e:
                logger.warning(f"[DEEP_AGENT] Failed to initialize memory store: {e}")
                enable_memory = False

        # Optionally add web_map and web_crawl
        if enable_web_map:
            try:
                from .tools.ulli_langchain.web.web_map import web_map
                tools.append(web_map)
                logger.info("[DEEP_AGENT] web_map tool enabled")
            except ImportError:
                logger.warning("[DEEP_AGENT] web_map requested but not available")

        if enable_web_crawl:
            try:
                from .tools.ulli_langchain.web.web_crawl import web_crawl
                tools.append(web_crawl)
                logger.info("[DEEP_AGENT] web_crawl tool enabled")
            except ImportError:
                logger.warning("[DEEP_AGENT] web_crawl requested but not available")

        # Create system prompt
        memory_instructions = """
Memory System (if enabled):
- write_file("/memories/filename.txt", content): Save information for later with semantic search
- read_file("/memories/filename.txt"): Read previously saved content
- search_memories("query"): Find memories by meaning (semantic search)

Files in /memories/ are persisted and searchable by meaning.
Scratchpad files (not in /memories/) are temporary and session-only.
""" if enable_memory else ""

        system_prompt = f"""You are a research assistant with access to web tools and memory.

Your available tools:
- web_search: Search for current information on the web
- web_get: Read and extract content from web pages

Additional tools (if enabled):
- web_map: Map websites to discover all URLs and pages
- web_crawl: Crawl websites to scrape multiple pages
- search_memories: Search your long-term memories by semantic similarity
{memory_instructions}
Instructions:
1. Start by searching for relevant information using web_search
2. Read promising pages using web_get to extract detailed content
3. If web_map is enabled, use it to discover all pages on a website
4. If web_crawl is enabled, use it to scrape multiple pages from a site
5. Save important findings to /memories/ for future reference (if memory enabled)
6. Check existing memories before researching (if memory enabled)
7. Synthesize your findings into a comprehensive answer

Be thorough but efficient. Focus on finding accurate, up-to-date information."""

        # Add tool middleware: retry and rate limiting
        # Note: LLM-level retry is handled by factory middleware (RetryMiddleware in get_llm)
        from langchain.agents.middleware import ToolRetryMiddleware, ToolCallLimitMiddleware

        # Create backend factory for memory routing (if enabled)
        if enable_memory and store is not None:
            # Use unified memory router for all 4 entry points
            try:
                from src.extensions.features.tools.context_engine.storage.unified_memory_router import (
                    create_unified_memory_router
                )

                def backend_factory(runtime):
                    """Create unified memory router with all storage backends.

                    Routes:
                    - /data/raw/dumps/ -> S3 (context overflow dumps)
                    - /data/raw/web/ -> S3 (web scrapes)
                    - /data/insights/ -> Postgres (vectorized insights)
                    - /data/graph/ -> Neo4j (knowledge graph)
                    - /memories/ -> StoreBackend (semantic search)
                    - /* -> StateBackend (ephemeral)
                    """
                    return create_unified_memory_router(
                        runtime=runtime,
                        session_id="global",  # Will be overridden by runtime state
                        include_s3=True,
                        include_postgres=True,
                        include_neo4j=True,
                        include_memories=True,
                        enable_async_vectorization=False,  # Sync for now
                    )

                logger.info("[DEEP_AGENT] Unified Memory Router configured with all backends")

            except ImportError as e:
                # Fallback to simple router if unified router not available
                logger.warning(f"[DEEP_AGENT] Unified router not available, using simple: {e}")

                def backend_factory(runtime):
                    """Create composite backend routing /memories/ to Store.

                    This routes file operations:
                    - /memories/* -> StoreBackend (persistent, searchable)
                    - /* -> StateBackend (temporary, session-only)
                    """
                    return CompositeBackend(
                        default=StateBackend(runtime),
                        routes={
                            "/memories/": StoreBackend(runtime),
                        }
                    )
                logger.info("[DEEP_AGENT] CompositeBackend configured with /memories/ route")
        else:
            backend_factory = None

        # Create the deep agent with middleware
        agent = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            store=store,  # LangGraph Store for semantic memory
            backend=backend_factory,  # CompositeBackend routing /memories/ to Store
            middleware=[
                # RETRY: Retry failed tool calls with exponential backoff
                ToolRetryMiddleware(
                    max_retries=2,  # 3 total attempts (1 initial + 2 retries)
                    backoff_factor=2.0,  # Exponential backoff: 1s, 2s, 4s delays
                    initial_delay=1.0,
                    max_delay=60.0,
                    jitter=True,  # Add random variation to avoid thundering herd
                    on_failure="raise",  # Re-raise exception (no silent tool errors)
                ),
                # RATE LIMITING: Limit concurrent scrape calls to avoid overwhelming the service
                # Queues requests when limit reached (unlimited total calls, max 10 concurrent)
                ToolCallLimitMiddleware(
                    tool_name="web_get",
                    thread_limit=10,  # Max 10 concurrent web_get calls (queues excess)
                ),
            ],
        )

        # Invoke the agent
        logger.info(f"[DEEP_AGENT] Invoking agent with {len(tools)} tools...")
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": max_steps}
        )

        # Extract the response
        if isinstance(result, dict):
            if "messages" in result:
                messages = result["messages"]
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        response = str(last_msg.content)
                    else:
                        response = str(last_msg)

                    logger.info(f"[DEEP_AGENT] Agent completed. Response length: {len(response)} chars")
                    return response

        # Fallback
        response = str(result)
        logger.info(f"[DEEP_AGENT] Agent completed (fallback). Response length: {len(response)} chars")
        return response

    except ImportError as e:
        logger.error(f"[DEEP_AGENT] Import error: {e}")
        return f"Error: Deep Agent import failed: {str(e)}. Run 'uv sync' to ensure dependencies are installed."

    except Exception as e:
        logger.error(f"[DEEP_AGENT] Error: {e}", exc_info=True)
        return f"Error running Deep Agent: {str(e)}"
