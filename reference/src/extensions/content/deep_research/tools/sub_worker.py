"""Specialist Sub-Workers for parallel research.

Parent workers can spawn autonomous specialist sub-workers to handle
parallel research tasks. Each sub-worker can have different capabilities:
- Web Research: web_search, web_get, web_map, web_crawl
- RAG Research: rag_search for local document search
- Hybrid: Both web and RAG for comprehensive coverage
"""

from pathlib import Path
import asyncio
import logging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ..logging_utils import get_logger, log_component

import re
import uuid
import time
from typing import List, Dict, Any, Optional

from ..graph.state import SubWorkerQuery, SubWorkerResult
from ..graph.helpers.llm import create_llm_from_config
from src.shared.tools.registry import get_tool

logger = get_logger(__name__, __name__.split('.')[-1])  # Component-specific level


# =============================================================================
# SPECIALIST AGENT CONFIGURATIONS
# =============================================================================

WEB_SPECIALIST_SYSTEM_PROMPT = """You are a Research Specialist. Your job is to answer queries by conducting thorough research.

Available Capabilities:
- web_search: Search for relevant information on the web
- web_get: Fetch and read specific URLs for detailed information
- web_map: Discover site structure and find linked pages
- web_crawl: Crawl multiple pages from a site for comprehensive coverage

Research Strategy:
1. Start with broad web_search to understand the topic landscape
2. Use web_map to discover relevant sources and site structure
3. Use web_get to read specific pages in detail
4. Use web_crawl when you need comprehensive coverage of multiple pages
5. Synthesize findings into a structured response

Output Format:
Provide your findings in a clear, structured format with:
1. Executive summary of key findings (2-3 sentences)
2. Detailed findings organized by topic/theme
3. Sources consulted (include full URLs)
4. Confidence level in your findings (0-1)

Important Guidelines:
- Always include full URLs in your citations
- Verify information from multiple sources when possible
- Distinguish between facts, opinions, and speculation
- Note any limitations or gaps in available information
"""

RAG_SPECIALIST_SYSTEM_PROMPT = """You are a Document Research Specialist. Your job is to answer queries by searching local documents.

Available Capabilities:
- rag_search: Search within local documents for relevant information

Research Strategy:
1. Use rag_search to find relevant passages in the documents
2. Extract key information directly from document content
3. Synthesize findings with proper source attribution
4. Note which documents provided which information

Output Format:
Provide your findings in a clear, structured format with:
1. Executive summary of key findings (2-3 sentences)
2. Detailed findings organized by topic/theme
3. Document sources consulted (cite document names/paths)
4. Confidence level in your findings (0-1)

Important Guidelines:
- Always cite which documents provided each piece of information
- Extract direct quotes when relevant for accuracy
- Distinguish between explicit content and interpretations
- Note any limitations or gaps in the available documents
"""

HYBRID_SPECIALIST_SYSTEM_PROMPT = """You are a Comprehensive Research Specialist. Your job is to answer queries by combining web and document research.

Available Capabilities:
- web_search: Search for relevant information on the web
- web_get: Fetch and read specific URLs for detailed information
- web_map: Discover site structure and find linked pages
- web_crawl: Crawl multiple pages from a site for comprehensive coverage
- rag_search: Search within local documents for relevant information

Research Strategy:
1. Start with rag_search to find relevant information in local documents
2. Use web_search to supplement and validate document findings
3. Use web_get to read specific sources for detailed information
4. Cross-reference document findings with web sources
5. Synthesize findings from both sources into a coherent response

Output Format:
Provide your findings in a clear, structured format with:
1. Executive summary of key findings (2-3 sentences)
2. Detailed findings organized by topic/theme
3. Sources consulted (include URLs and document names)
4. Confidence level in your findings (0-1)
5. Clear distinction between document-based and web-based findings

Important Guidelines:
- Always cite both document and web sources
- Highlight where documents and web sources agree or disagree
- Use documents as primary source when available
- Use web sources to fill gaps and validate document information
"""

WEB_SPECIALIST_PROMPT_TEMPLATE = """Research Query: {query}

Constraints:
- Maximum depth: {max_depth} levels of research
- Maximum sources: {max_sources}
- Timeout: {timeout} seconds

Please conduct thorough web research and provide structured findings with full source URLs."""


# =============================================================================
# SUB-WORKER CREATION AND EXECUTION
# =============================================================================

async def delegate_to_sub_worker(
    query: SubWorkerQuery,
    yaml_config: dict,
    parent_worker_id: str,
    run_logger=None
) -> SubWorkerResult:
    """Delegate a research query to a web specialist sub-worker.

    Args:
        query: SubWorkerQuery with the research task
        yaml_config: Configuration dict for this run
        parent_worker_id: ID of the parent worker spawning this sub-worker
        run_logger: Optional logger for tracking

    Returns:
        SubWorkerResult with research findings
    """
    start_time = time.time()

    if run_logger:
        log_component(run_logger, parent_worker_id,
            f"Spawning sub-worker {query.query_id} "
            f"for query: {query.query[:100]}..."
        )

    try:
        # Create LLM for sub-worker (same model as parent)
        llm = await create_llm_from_config(yaml_config, role="worker")

        # Get tools from central registry based on tool names
        tools = []
        has_rag = "rag_search" in query.tools
        has_web = any(t in query.tools for t in ["web_search", "web_get", "web_map", "web_crawl"])

        for tool_name in query.tools:
            # Map tool names to actual tools from central registry
            if tool_name == "web_search":
                from src.shared.tools.ulli_langchain.web import web_search
                tools.append(web_search)
            elif tool_name == "web_get":
                from src.shared.tools.ulli_langchain.web import web_get
                tools.append(web_get)
            elif tool_name == "web_map":
                from src.shared.tools.ulli_langchain.web import web_map
                tools.append(web_map)
            elif tool_name == "web_crawl":
                from src.shared.tools.ulli_langchain.web import web_crawl
                tools.append(web_crawl)
            elif tool_name == "rag_search":
                # Get pdf_deep_agent from central registry (the existing RAG deep agent)
                rag_tool = get_tool("pdf_deep_agent")
                if rag_tool:
                    tools.append(rag_tool)
                    if run_logger:
                    log_component(run_logger, parent_worker_id,
                        f"Sub-worker {query.query_id} "
                        f"added pdf_deep_agent tool (from central registry)"
                    )
                else:
                    # Fallback: direct import if not in registry
                    from src.shared.tools.ulli_langchain.memory.pdf_agent import pdf_deep_agent
                    tools.append(pdf_deep_agent)
                    if run_logger:
                        log_component(run_logger, parent_worker_id,
                            f"Sub-worker {query.query_id} "
                            f"added pdf_deep_agent tool (direct import - not in registry)"
                        )

        if run_logger:
            log_component(run_logger, parent_worker_id,
                f"Sub-worker {query.query_id} "
                f"created with {len(tools)} tools: {query.tools}"
            )

        # Choose system prompt based on tool configuration
        if has_rag and has_web:
            system_prompt = HYBRID_SPECIALIST_SYSTEM_PROMPT
            specialist_type = "hybrid"
        elif has_rag:
            system_prompt = RAG_SPECIALIST_SYSTEM_PROMPT
            specialist_type = "rag"
        else:
            system_prompt = WEB_SPECIALIST_SYSTEM_PROMPT
            specialist_type = "web"

        if run_logger:
            log_component(run_logger, parent_worker_id,
                f"Sub-worker {query.query_id} "
                f"using {specialist_type} specialist mode"
            )

        # Create deep_agent for tools
        from deepagents import create_deep_agent

        research_agent = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        # Build the research prompt
        research_prompt = WEB_SPECIALIST_PROMPT_TEMPLATE.format(
            query=query.query,
            max_depth=query.max_depth,
            max_sources=query.max_sources,
            timeout=query.timeout
        )

        # Invoke the agent with timeout
        result = await asyncio.wait_for(
            research_agent.ainvoke(
                {"messages": [{"role": "system", "content": research_prompt}]}
            ),
            timeout=query.timeout
        )

        # Extract findings from agent result
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            findings_list = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    findings_list.append(str(msg.content))
            findings = "\n\n".join(findings_list).strip()
        else:
            findings = str(result).strip()

        # Extract URLs from findings
        url_pattern = r'https?://[^\s\)\]\}>"\',]+(?:\([^\s]*\))?'
        sources = list(set(re.findall(url_pattern, findings)))

        elapsed = time.time() - start_time

        if run_logger:
            log_component(run_logger, parent_worker_id,
                f"Sub-worker {query.query_id} "
                f"completed in {elapsed:.1f}s with {len(sources)} sources"
            )

        return SubWorkerResult(
            query_id=query.query_id,
            query=query.query,
            findings=findings,
            sources=sources,
            confidence=0.8,
            metadata={
                "parent_worker_id": parent_worker_id,
                "elapsed_seconds": elapsed,
                "tools_used": query.tools,
                "depth_reached": min(query.max_depth, 3),  # Approximate
            }
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        error_msg = f"Sub-worker {query.query_id} timed out after {elapsed:.1f}s"

        if run_logger:
            log_component(run_logger, parent_worker_id, error_msg)

        return SubWorkerResult(
            query_id=query.query_id,
            query=query.query,
            findings=f"[TIMEOUT] Research query exceeded {query.timeout}s timeout limit.",
            sources=[],
            confidence=0.0,
            metadata={
                "parent_worker_id": parent_worker_id,
                "error": "timeout",
                "elapsed_seconds": elapsed,
            }
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Sub-worker {query.query_id} failed: {type(e).__name__}: {str(e)[:200]}"

        if run_logger:
            log_component(run_logger, parent_worker_id, error_msg)

        return SubWorkerResult(
            query_id=query.query_id,
            query=query.query,
            findings=f"[ERROR] {error_msg}",
            sources=[],
            confidence=0.0,
            metadata={
                "parent_worker_id": parent_worker_id,
                "error": str(e)[:200],
                "elapsed_seconds": elapsed,
            }
        )


async def delegate_multiple_queries(
    queries: List[str],
    yaml_config: dict,
    parent_worker_id: str,
    run_logger=None,
    max_concurrent: int = 5
) -> List[SubWorkerResult]:
    """Delegate multiple queries to sub-workers in parallel.

    Args:
        queries: List of query strings to research
        yaml_config: Configuration dict for this run
        parent_worker_id: ID of the parent worker
        run_logger: Optional logger for tracking
        max_concurrent: Maximum number of sub-workers to run concurrently

    Returns:
        List of SubWorkerResult with research findings
    """
    if run_logger:
        log_component(run_logger, parent_worker_id,
            f"Spawning {len(queries)} sub-workers "
            f"(max_concurrent={max_concurrent})"
        )

    # Get sub-worker config from yaml_config
    spawner_config = yaml_config.get("spawner_config", {})
    sub_worker_config = spawner_config.get("sub_workers", {})

    # Create SubWorkerQuery objects
    sub_worker_queries = []
    for i, query_text in enumerate(queries):
        sub_worker_queries.append(SubWorkerQuery(
            query_id=f"{parent_worker_id}_sub_{i}_{uuid.uuid4().hex[:8]}",
            query=query_text,
            max_depth=sub_worker_config.get("default_max_depth", 3),
            max_sources=sub_worker_config.get("default_max_sources", 10),
            tools=["web_search", "web_get", "web_map", "web_crawl"],
            timeout=sub_worker_config.get("default_timeout", 300)
        ))

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_delegate(query: SubWorkerQuery) -> SubWorkerResult:
        """Run sub-worker with concurrency limit."""
        async with semaphore:
            return await delegate_to_sub_worker(
                query=query,
                yaml_config=yaml_config,
                parent_worker_id=parent_worker_id,
                run_logger=run_logger
            )

    # Run all queries in parallel
    start_time = time.time()
    results = await asyncio.gather(*[bounded_delegate(q) for q in sub_worker_queries])
    elapsed = time.time() - start_time

    if run_logger:
        # Confidence is 0-100 scale, threshold at 50
        successful = sum(1 for r in results if r.confidence > 50)
        log_component(run_logger, parent_worker_id,
            f"All sub-workers completed in {elapsed:.1f}s "
            f"({successful}/{len(results)} successful)"
        )

    return results


def parse_sub_worker_delegations(text: str) -> List[str]:
    """Parse [SUB_WORKER: query] syntax from worker text.

    Args:
        text: Text that may contain sub-worker delegation markers

    Returns:
        List of extracted query strings
    """

    # Pattern: [SUB_WORKER: query text here]
    pattern = r'\[SUB_WORKER:\s*(.+?)\]'
    matches = re.findall(pattern, text, re.DOTALL)

    # Clean up whitespace
    queries = [m.strip() for m in matches if m.strip()]

    return queries


async def execute_sub_workers_if_needed(
    worker_output: str,
    yaml_config: dict,
    worker_id: str,
    run_logger=None
) -> tuple[str, List[SubWorkerResult]]:
    """Check worker output for sub-worker delegations and execute if found.

    Args:
        worker_output: Text output from a worker that may contain [SUB_WORKER: ...]
        yaml_config: Configuration dict
        worker_id: ID of the worker
        run_logger: Optional logger

    Returns:
        Tuple of (original_output, list_of_sub_worker_results)
    """
    queries = parse_sub_worker_delegations(worker_output)

    if not queries:
        return worker_output, []

    # Get concurrency limit from config
    spawner_config = yaml_config.get("spawner_config", {})
    sub_worker_config = spawner_config.get("sub_workers", {})
    max_concurrent = sub_worker_config.get("max_concurrent_per_worker", 5)

    # Check if sub-workers are enabled
    if not sub_worker_config.get("enable", False):
        if run_logger:
            log_component(run_logger, worker_id,
                f"Found {len(queries)} sub-worker queries but "
                f"sub-workers are disabled in config"
            )
        return worker_output, []

    # Execute sub-workers
    results = await delegate_multiple_queries(
        queries=queries,
        yaml_config=yaml_config,
        parent_worker_id=worker_id,
        run_logger=run_logger,
        max_concurrent=max_concurrent
    )

    return worker_output, results
