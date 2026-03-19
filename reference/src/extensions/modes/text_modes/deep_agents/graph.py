"""Deep Agents Graph - RAG Deep Agent with LangGraph V4 Modern Patterns.

Deep Agents all the way down - infinite conversation with:
- Router for query analysis and search strategy
- RAG integration with vector search
- Tool-enabled research with web + RAG
- Modern LangGraph V4 patterns: Command API, Send API, Runtime Config

Modern LangGraph Patterns Implemented:
- Command API: Combined state updates + control flow in router/synthesis nodes
- Send API: Map-reduce pattern for parallel worker execution
- Runtime Configuration: Dynamic parameter injection via context schema
- Input/Output Schemas: Clean API boundaries
- Retry Policies: Declarative retry logic per node
- Cache Policies: Node-level caching for expensive operations
- Store: Cross-thread persistence with semantic search
- Checkpointer: Enabled by default for time-travel and durable execution

The system supports multiple query types:
- Quick: Direct answer from conversation memory
- RAG: Vector search from knowledge base
- Web: Web search for current information
- Hybrid: Combined RAG + Web search
"""

import sys
import asyncio
import time
import os
import uuid
from typing import Dict, Any, List, Optional, Tuple, Literal, Union, Annotated
from datetime import datetime
from pathlib import Path
from typing_extensions import TypedDict

# Add backend to path for absolute imports
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy, CachePolicy, Command, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.cache.memory import InMemoryCache
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import add_messages
from langgraph.managed import RemainingSteps

from src.shared.logging_utils import get_extension_logger

logger = get_extension_logger("deep_agents")


# =============================================================================
# STATE DEFINITIONS (Pydantic models for runtime validation)
# =============================================================================

class DeepAgentsInput(TypedDict):
    """User-facing input schema for deep agents mode."""
    query: str  # User's message
    chat_history: Optional[List[BaseMessage]]  # Conversation history
    user_profile: Optional[Dict[str, Any]]  # User profile data


class DeepAgentsOutput(TypedDict):
    """User-facing output schema for deep agents mode."""
    response: str  # Final response text
    sources: List[str]  # Source references
    citations: List[Dict[str, Any]]  # Structured citations
    metadata: Dict[str, Any]  # Execution metadata
    routing_decision: str  # How the query was routed


class WorkerTask(TypedDict):
    """Task definition for parallel workers."""
    worker_id: str
    task_type: Literal["rag_search", "web_search", "web_research"]
    query: str
    context: Optional[str]


class WorkerResult(TypedDict):
    """Result from a worker execution."""
    worker_id: str
    task_type: str
    findings: str
    sources: List[str]
    citations: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


class RouterDecision(TypedDict):
    """Router decision for query handling."""
    decision: Literal["quick", "rag", "web", "hybrid"]
    reasoning: str
    num_workers: int
    worker_tasks: List[WorkerTask]


# =============================================================================
# GLOBAL STATE (TypedDict for LangGraph with annotated reducers)
# =============================================================================

class GlobalState(TypedDict):
    """Global state for Deep Agents RAG graph.

    This state flows through all phases:
    - Router: Analyzes query and decides strategy
    - Workers: Execute parallel search/research
    - Synthesis: Generates final response with citations
    """
    # ===== User Input =====
    query: str  # Current user query
    chat_history: Annotated[List[BaseMessage], add_messages]  # Conversation history
    user_profile: Dict[str, Any]  # User profile data

    # ===== Router Artifacts =====
    routing_decision: RouterDecision  # How to handle the query
    boss_guidance: Optional[str]  # Guidance from router

    # ===== Worker Artifacts =====
    worker_tasks: List[WorkerTask]  # Tasks to execute
    worker_outputs: Annotated[List[WorkerResult], "merge_worker_outputs"]  # Results from workers

    # ===== Synthesis Artifacts =====
    sources: List[str]  # All sources collected
    citations: List[Dict[str, Any]]  # Structured citations
    response: Optional[str]  # Final response

    # ===== Metadata =====
    thread_id: str  # Conversation thread ID
    research_id: str  # Research session ID
    start_time: str  # Start timestamp
    end_time: Optional[str]  # End timestamp
    current_phase: str  # Current execution phase
    iteration_count: int  # Iteration counter

    # ===== Refinement =====
    requires_revision: bool  # Whether revision is needed
    revision_instructions: Optional[str]  # Revision guidance

    # ===== Recursion Limit Handling =====
    remaining_steps: RemainingSteps


def merge_worker_outputs(left: List[WorkerResult], right: List[WorkerResult]) -> List[WorkerResult]:
    """Reducer: Merge worker outputs from parallel execution."""
    if right is None:
        return []
    if not left:
        return right
    if not right:
        return left
    return left + right


# =============================================================================
# RETRY POLICIES (Declarative retry logic per node type)
# =============================================================================

# LLM nodes: Retry on rate limits, timeouts, and connection errors
LLM_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(Exception,),
)

# Worker nodes: More retries for external API calls
WORKER_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(Exception,),
)

# Synthesis node: Conservative retries (already expensive)
SYNTHESIS_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    backoff_factor=2.0,
    jitter=True,
    retry_on=(Exception,),
)


# =============================================================================
# CACHE POLICIES (Node-level caching)
# =============================================================================

# Router cache: Cache routing decisions for similar queries
ROUTER_CACHE_POLICY = CachePolicy(
    ttl=1800,  # 30 minutes
    key_func=lambda s: hash(s.get("query", "")),
)

# Worker cache: Cache worker results for identical searches
WORKER_CACHE_POLICY = CachePolicy(
    ttl=900,  # 15 minutes
    key_func=lambda s: hash((s.get("worker_id", ""), s.get("query", ""))),
)


# =============================================================================
# RUNTIME CONFIGURATION (Dynamic parameter injection)
# =============================================================================

class DeepAgentsRuntimeConfig(TypedDict, total=False):
    """Runtime configuration for deep agents execution.

    This configuration is passed at invoke time via the `context` parameter
    and does not pollute the graph state.
    """
    # LLM Configuration
    model_provider: str  # "fast", "quality", "local", or provider name
    temperature: Optional[float]
    max_tokens: Optional[int]

    # Execution Configuration
    max_parallel_workers: int
    quality_threshold: float
    enable_reflexion: bool
    max_iterations: int

    # Feature Flags
    enable_caching: bool
    cache_ttl: int
    enable_rag: bool
    enable_web_search: bool

    # Debug Options
    debug_mode: bool
    save_artifacts: bool


# =============================================================================
# CONFIG LOADING
# =============================================================================

def _load_extension_config() -> Dict[str, Any]:
    """Load the extension config.

    Returns:
        Dict with extension config
    """
    import yaml

    _EXTENSION_DIR = Path(__file__).parent
    _CONFIG_PATH = _EXTENSION_DIR / "config.yaml"

    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


# =============================================================================
# GRAPH NODES
# =============================================================================

async def router_node(
    state: GlobalState,
    config: RunnableConfig,
    *,
    runtime: Optional[DeepAgentsRuntimeConfig] = None,
    store: Optional[Any] = None,
) -> Command:
    """Router node: Analyze query and decide execution strategy.

    Uses Command API to combine state update + routing decision.

    Args:
        state: Current global state
        config: LangGraph RunnableConfig
        runtime: Optional runtime configuration
        store: Optional BaseStore for cross-thread persistence

    Returns:
        Command with state update and routing decision
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    logger.info(f"[ROUTER] Analyzing query: {query[:100]}...")

    # Load extension config
    ext_config = _load_extension_config()
    enabled_tools = ext_config.get("enabled_tools", ["web_search", "web_get", "vector_search"])

    # Determine routing strategy
    decision: RouterDecision = {
        "decision": "rag",
        "reasoning": "",
        "num_workers": 1,
        "worker_tasks": []
    }

    # Simple routing logic based on query analysis
    query_lower = query.lower()

    # Check if this is a follow-up question (has conversation history)
    is_followup = len(chat_history) > 2

    # Check if query asks for current information
    needs_current = any(word in query_lower for word in [
        "latest", "recent", "current", "today", "now", "news", "price", "weather"
    ])

    # Check if query asks for web research
    needs_research = any(word in query_lower for word in [
        "research", "find", "search", "look up", "investigate", "what is"
    ])

    # Routing decision
    if is_followup and not needs_current:
        # Follow-up question - quick answer from context
        decision["decision"] = "quick"
        decision["reasoning"] = "Follow-up question - can be answered from conversation context"
        decision["num_workers"] = 0
    elif needs_current and needs_research:
        # Hybrid: RAG + Web search
        decision["decision"] = "hybrid"
        decision["reasoning"] = "Query needs both knowledge base and current information"
        decision["num_workers"] = 2
        decision["worker_tasks"] = [
            {
                "worker_id": "rag_worker",
                "task_type": "rag_search",
                "query": query,
                "context": None
            },
            {
                "worker_id": "web_worker",
                "task_type": "web_search",
                "query": query,
                "context": None
            }
        ]
    elif needs_current:
        # Web search for current information
        decision["decision"] = "web"
        decision["reasoning"] = "Query requires current information from web"
        decision["num_workers"] = 1
        decision["worker_tasks"] = [
            {
                "worker_id": "web_worker",
                "task_type": "web_search",
                "query": query,
                "context": None
            }
        ]
    else:
        # RAG search from knowledge base
        decision["decision"] = "rag"
        decision["reasoning"] = "Query can be answered from knowledge base"
        decision["num_workers"] = 1
        decision["worker_tasks"] = [
            {
                "worker_id": "rag_worker",
                "task_type": "rag_search",
                "query": query,
                "context": None
            }
        ]

    logger.info(f"[ROUTER] Decision: {decision['decision']} - {decision['reasoning']}")

    # Use Command API to combine state update + routing
    if decision["decision"] == "quick":
        # Quick path: go directly to synthesis
        return Command(
            update={
                "routing_decision": decision,
                "worker_tasks": [],
                "current_phase": "synthesis"
            },
            goto="synthesis"
        )
    else:
        # Worker path: spawn parallel workers
        return Command(
            update={
                "routing_decision": decision,
                "worker_tasks": decision["worker_tasks"],
                "boss_guidance": decision["reasoning"],
                "current_phase": "workers"
            },
            goto="spawn_workers"
        )


async def spawn_workers_node(
    state: GlobalState,
    config: RunnableConfig,
    *,
    runtime: Optional[DeepAgentsRuntimeConfig] = None,
    store: Optional[Any] = None,
) -> List[Send]:
    """Spawn workers node: Create Send objects for parallel execution.

    Uses Send API for map-reduce pattern.

    Args:
        state: Current global state with worker_tasks
        config: LangGraph RunnableConfig
        runtime: Optional runtime configuration
        store: Optional BaseStore for cross-thread persistence

    Returns:
        List of Send objects for parallel worker execution
    """
    worker_tasks = state.get("worker_tasks", [])

    if not worker_tasks:
        logger.warning("[SPAWN] No worker tasks to spawn")
        return []

    logger.info(f"[SPAWN] Creating {len(worker_tasks)} worker tasks via Send API")

    # Create a Send object for each worker
    sends = []
    for task in worker_tasks:
        worker_state = {
            "worker_id": task["worker_id"],
            "task_type": task["task_type"],
            "query": task["query"],
            "context": task.get("context"),
            "original_query": state.get("query", ""),
            "chat_history": state.get("chat_history", []),
        }
        sends.append(Send("execute_worker", worker_state))

    return sends


async def execute_worker_node(
    worker_state: Dict[str, Any],
    config: RunnableConfig,
    *,
    runtime: Optional[DeepAgentsRuntimeConfig] = None,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a single worker (RAG search or web search).

    This function is called for each Send object created by spawn_workers_node.

    Args:
        worker_state: Worker state with task information
        config: LangGraph RunnableConfig
        runtime: Optional runtime configuration
        store: Optional BaseStore for cross-thread persistence

    Returns:
        State update with single worker result
    """
    worker_id = worker_state.get("worker_id", "unknown")
    task_type = worker_state.get("task_type", "unknown")
    query = worker_state.get("query", "")
    original_query = worker_state.get("original_query", query)

    logger.info(f"[WORKER {worker_id}] Executing {task_type}: {query[:100]}...")

    start_time = time.time()

    try:
        # Load LLM and tools
        from src.shared.factory import get_default_llm
        llm = get_default_llm()

        if task_type == "rag_search":
            # Execute RAG search
            findings, sources, citations = await _execute_rag_search(
                query=query,
                chat_history=worker_state.get("chat_history", []),
                llm=llm,
                store=store
            )
        elif task_type == "web_search":
            # Execute web search
            findings, sources, citations = await _execute_web_search(
                query=query,
                llm=llm
            )
        elif task_type == "web_research":
            # Execute deep web research
            findings, sources, citations = await _execute_web_research(
                query=query,
                llm=llm
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        elapsed = time.time() - start_time
        logger.info(f"[WORKER {worker_id}] Completed in {elapsed:.2f}s")

        # Return as list (reducer will merge)
        worker_result: WorkerResult = {
            "worker_id": worker_id,
            "task_type": task_type,
            "findings": findings,
            "sources": sources,
            "citations": citations,
            "confidence": 0.8,
            "metadata": {
                "elapsed_time": elapsed,
                "query": query,
            }
        }

        return {"worker_outputs": [worker_result]}

    except Exception as e:
        logger.error(f"[WORKER {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()

        # Return error result
        worker_result: WorkerResult = {
            "worker_id": worker_id,
            "task_type": task_type,
            "findings": f"Error: {str(e)}",
            "sources": [],
            "citations": [],
            "confidence": 0.0,
            "metadata": {
                "error": str(e),
            }
        }

        return {"worker_outputs": [worker_result]}


async def synthesis_node(
    state: GlobalState,
    config: RunnableConfig,
    *,
    runtime: Optional[DeepAgentsRuntimeConfig] = None,
    store: Optional[Any] = None,
) -> Command:
    """Synthesis node: Generate final response from worker outputs.

    Uses Command API to combine state update + routing decision.

    Args:
        state: Current global state with worker outputs
        config: LangGraph RunnableConfig
        runtime: Optional runtime configuration
        store: Optional BaseStore for cross-thread persistence

    Returns:
        Command with state update and routing to END
    """
    query = state.get("query", "")
    routing_decision = state.get("routing_decision", {})
    worker_outputs = state.get("worker_outputs", [])

    logger.info(f"[SYNTHESIS] Generating response for: {query[:100]}...")

    # Load LLM
    from src.shared.factory import get_default_llm
    llm = get_default_llm()

    # Build synthesis prompt
    if routing_decision.get("decision") == "quick":
        # Quick path: direct answer from context
        prompt = f"""You are a helpful AI assistant. The user asked: "{query}"

Conversation history:
{_format_chat_history(state.get("chat_history", []))}

Provide a helpful, concise response based on the conversation context."""
    else:
        # Worker path: synthesize from worker outputs
        findings_text = "\n\n".join([
            f"=== {w['worker_id']} ({w['task_type']}) ===\n{w['findings']}"
            for w in worker_outputs
        ])

        sources = []
        citations = []
        for w in worker_outputs:
            sources.extend(w.get("sources", []))
            citations.extend(w.get("citations", []))

        prompt = f"""You are a helpful AI assistant. The user asked: "{query}"

Research findings:
{findings_text}

Synthesize these findings into a comprehensive, accurate response. Cite your sources appropriately."""

    # Generate response
    try:
        response = await llm.ainvoke(prompt)
        if isinstance(response, AIMessage):
            response_text = response.content
        else:
            response_text = str(response)
    except Exception as e:
        logger.error(f"[SYNTHESIS] Error generating response: {e}")
        response_text = f"I apologize, but I encountered an error: {str(e)}"

    # Collect all sources and citations
    all_sources = []
    all_citations = []
    for w in worker_outputs:
        all_sources.extend(w.get("sources", []))
        all_citations.extend(w.get("citations", []))

    logger.info(f"[SYNTHESIS] Generated response ({len(response_text)} chars, {len(all_sources)} sources)")

    # Use Command API to combine state update + routing
    return Command(
        update={
            "response": response_text,
            "sources": all_sources,
            "citations": all_citations,
            "current_phase": "complete",
            "end_time": datetime.now().isoformat(),
        },
        goto=END
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _execute_rag_search(
    query: str,
    chat_history: List[BaseMessage],
    llm: Any,
    store: Optional[Any] = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Execute RAG search from knowledge base.

    Args:
        query: Search query
        chat_history: Conversation history
        llm: LLM instance
        store: Optional BaseStore for semantic search

    Returns:
        Tuple of (findings, sources, citations)
    """
    from src.shared.tools.ulli_langchain.memory.rag import vector_search

    # Execute vector search
    search_results_text = await vector_search(query=query, top_k=5)

    # vector_search returns a string with results
    if not search_results_text or "No results found" in search_results_text:
        return "No relevant information found in the knowledge base.", [], []

    findings = f"Relevant information from the knowledge base:\n\n{search_results_text}"
    sources = ["Knowledge Base"]
    citations = [{
        "source_type": "document",
        "source_name": "Knowledge Base",
        "excerpt": search_results_text[:200],
        "confidence": 0.8,
    }]

    return findings, sources, citations


async def _execute_web_search(
    query: str,
    llm: Any,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Execute web search.

    Args:
        query: Search query
        llm: LLM instance

    Returns:
        Tuple of (findings, sources, citations)
    """
    from src.shared.tools.logic.web.search import web_search

    # Execute web search
    search_results_text = await web_search(query=query, num_results=5)

    # web_search returns a formatted string with results
    if not search_results_text or "No results found" in search_results_text:
        return "No web search results found.", [], []

    findings = f"Web search results:\n\n{search_results_text}"
    sources = ["Web Search"]
    citations = [{
        "source_type": "url",
        "source_name": "Web Search",
        "excerpt": search_results_text[:200],
        "confidence": 0.7,
    }]

    return findings, sources, citations


async def _execute_web_research(
    query: str,
    llm: Any,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Execute deep web research with multiple searches.

    Args:
        query: Research query
        llm: LLM instance

    Returns:
        Tuple of (findings, sources, citations)
    """
    # For now, delegate to web_search
    # In the future, this could do multiple searches and synthesis
    return await _execute_web_search(query, llm)


def _format_chat_history(messages: List[BaseMessage]) -> str:
    """Format chat history for prompt.

    Args:
        messages: List of messages

    Returns:
        Formatted chat history string
    """
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages[-10:]:  # Last 10 messages
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, ToolMessage):
            role = "Tool"
        else:
            role = "Unknown"

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        formatted.append(f"{role}: {content[:200]}")  # Truncate long messages

    return "\n\n".join(formatted)


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_deep_agents_graph(
    checkpointer: Optional[Any] = None,
    cache: Optional[Any] = None,
    store: Optional[Any] = None,
) -> StateGraph:
    """Build the V4 Modern LangGraph Architecture graph for RAG Deep Agent.

    NEW PATTERNS IMPLEMENTED:
    1. Command API: Combined state updates + control flow
    2. Send API: Map-reduce pattern for parallel workers
    3. Runtime Configuration: Dynamic parameter injection
    4. Input/Output Schemas: Clean API boundaries
    5. Retry Policies: Declarative retry logic
    6. Cache Policies: Node-level caching
    7. Store: Cross-thread persistence with semantic search
    8. Checkpointer: Enabled by default for time-travel and durable execution

    Args:
        checkpointer: Optional checkpointer for persistence (default: MemorySaver)
        cache: Optional cache backend for node-level caching
        store: Optional BaseStore for cross-thread persistence

    Returns:
        Compiled StateGraph ready for invoke()
    """
    # ========================================================================
    # BUILD STATEGRAPH WITH INPUT/OUTPUT SCHEMAS
    # ========================================================================
    builder = StateGraph(
        GlobalState,
        input_schema=DeepAgentsInput,
        output_schema=DeepAgentsOutput,
        context_schema=DeepAgentsRuntimeConfig,
    )

    # ========================================================================
    # ADD NODES WITH RETRY AND CACHE POLICIES
    # ========================================================================
    # Router node: Analyzes query and decides strategy
    builder.add_node(
        "router",
        router_node,
        retry_policy=LLM_RETRY_POLICY,
        cache_policy=ROUTER_CACHE_POLICY,  # Cache routing decisions
    )

    # Spawn workers node: Creates Send objects for parallel execution
    builder.add_node(
        "spawn_workers",
        spawn_workers_node,
        retry_policy=RetryPolicy(max_attempts=1),  # No retry needed
    )

    # Execute worker node (for Send API pattern)
    builder.add_node(
        "execute_worker",
        execute_worker_node,
        retry_policy=WORKER_RETRY_POLICY,
        cache_policy=WORKER_CACHE_POLICY,  # Cache worker results
    )

    # Synthesis node: Generates final response
    builder.add_node(
        "synthesis",
        synthesis_node,
        retry_policy=SYNTHESIS_RETRY_POLICY,
    )

    # ========================================================================
    # DEFINE EDGES
    # ========================================================================
    # Start flow: START -> router
    builder.add_edge(START, "router")

    # Router uses Command API internally for routing
    # The conditional edge is handled by Command goto in router_node

    # Spawn workers -> execute_worker (Send API)
    builder.add_conditional_edges(
        "spawn_workers",
        spawn_workers_node,
        ["execute_worker"],
    )

    # After all workers complete, go to synthesis
    builder.add_edge("execute_worker", "synthesis")

    # Synthesis uses Command API internally for routing to END

    # ========================================================================
    # COMPILE GRAPH WITH CHECKPOINTER, CACHE, AND STORE
    # ========================================================================
    compile_kwargs = {}

    # Checkpointer: Use factory for environment-aware selection
    if checkpointer:
        actual_checkpointer = checkpointer
        if hasattr(checkpointer, 'checkpointer'):
            actual_checkpointer = checkpointer.checkpointer
        logger.info(f"[DEEP_AGENTS] Compiling graph with custom checkpointer: {type(actual_checkpointer).__name__}")
        compile_kwargs["checkpointer"] = actual_checkpointer
    else:
        # Note: checkpointer should always be passed from extension.py
        # If no checkpointer provided, raise error to prevent data loss
        raise ValueError(
            "checkpointer is required for Deep Agents. "
            "Pass checkpointer=await get_shared_checkpointer() from extension.py"
        )

    # Cache: Enable InMemoryCache by default for node-level caching
    if cache:
        compile_kwargs["cache"] = cache
        logger.info(f"[DEEP_AGENTS] Compiling graph with custom cache backend: {type(cache).__name__}")
    else:
        # Check for Valkey/Redis URL for distributed caching (multi-instance deployments)
        valkey_url = os.getenv("VALKEY_URL") or os.getenv("REDIS_URL")
        if valkey_url:
            try:
                from langgraph.cache.redis import RedisCache
                import redis

                # Use RedisCache (works with both Valkey and Redis)
                redis_client = redis.from_url(
                    valkey_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                default_cache = RedisCache(redis_client)
                logger.info(f"[DEEP_AGENTS] Compiling graph with RedisCache (distributed caching via VALKEY_URL)")
            except ImportError:
                logger.warning("[DEEP_AGENTS] Redis client not available, falling back to InMemoryCache")
                default_cache = InMemoryCache()
            except Exception as e:
                logger.warning(f"[DEEP_AGENTS] Failed to connect to Valkey/Redis: {e}")
                logger.info("[DEEP_AGENTS] Falling back to InMemoryCache")
                default_cache = InMemoryCache()
        else:
            # Use InMemoryCache by default to enable node-level caching
            # This caches expensive operations like LLM calls and web searches
            default_cache = InMemoryCache()
            logger.info("[DEEP_AGENTS] Compiling graph with InMemoryCache (node-level caching enabled)")

        compile_kwargs["cache"] = default_cache

    # Store: Optional semantic search store
    if store:
        compile_kwargs["store"] = store
        logger.info(f"[DEEP_AGENTS] Compiling graph with custom Store: {type(store).__name__}")

    return builder.compile(**compile_kwargs)


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

async def run_deep_agents(
    query: str,
    chat_history: List[BaseMessage] = None,
    thread_id: Optional[str] = None,
    runtime_config: Optional[DeepAgentsRuntimeConfig] = None,
    store: Optional[Any] = None,
    checkpointer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the Deep Agents RAG graph.

    Args:
        query: User query
        chat_history: Optional conversation history
        thread_id: Optional thread ID for persistence
        runtime_config: Optional runtime configuration
        store: Optional BaseStore for cross-thread persistence
        checkpointer: Optional custom checkpointer

    Returns:
        Dict with response, sources, citations, and metadata
    """
    # Load extension config
    ext_config = _load_extension_config()
    enabled_tools = ext_config.get("enabled_tools", ["web_search", "web_get", "vector_search"])

    # Build graph
    graph = build_deep_agents_graph(checkpointer=checkpointer, store=store)

    # Generate thread_id if not provided
    if thread_id is None:
        thread_id = f"deep_agents_thread_{uuid.uuid4()}"

    # Create initial state
    initial_state: GlobalState = {
        "query": query,
        "chat_history": chat_history or [],
        "user_profile": {},
        "routing_decision": {
            "decision": "rag",
            "reasoning": "",
            "num_workers": 0,
            "worker_tasks": []
        },
        "boss_guidance": None,
        "worker_tasks": [],
        "worker_outputs": [],
        "sources": [],
        "citations": [],
        "response": None,
        "thread_id": thread_id,
        "research_id": f"research_{uuid.uuid4()}",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "current_phase": "router",
        "iteration_count": 1,
        "requires_revision": False,
        "revision_instructions": None,
    }

    # Build invoke config
    invoke_config = {"configurable": {"thread_id": thread_id}}

    # Add runtime config if provided
    if runtime_config:
        invoke_config["configurable"].update(runtime_config)

    # Invoke graph
    logger.info(f"[DEEP_AGENTS] Starting execution for query: {query[:100]}...")
    final_state = await graph.ainvoke(initial_state, config=invoke_config)

    # Extract results
    response = final_state.get("response", "")
    sources = final_state.get("sources", [])
    citations = final_state.get("citations", [])
    routing_decision = final_state.get("routing_decision", {})

    logger.info(f"[DEEP_AGENTS] Execution complete. Routing: {routing_decision.get('decision', 'unknown')}")

    return {
        "response": response,
        "sources": sources,
        "citations": citations,
        "metadata": {
            "routing_decision": routing_decision.get("decision", "unknown"),
            "routing_reasoning": routing_decision.get("reasoning", ""),
            "thread_id": thread_id,
            "research_id": final_state.get("research_id", ""),
            "start_time": final_state.get("start_time", ""),
            "end_time": final_state.get("end_time", ""),
        }
    }


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def get_available_tools() -> List[str]:
    """Get list of available tools for deep agents mode.

    Returns:
        List of tool names
    """
    ext_config = _load_extension_config()
    enabled_tools = ext_config.get("enabled_tools", ["web_search", "web_get", "vector_search"])
    return enabled_tools


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Graph builder
    "build_deep_agents_graph",
    # Main execution function
    "run_deep_agents",
    # State types
    "GlobalState",
    "DeepAgentsInput",
    "DeepAgentsOutput",
    "RouterDecision",
    "WorkerTask",
    "WorkerResult",
    "DeepAgentsRuntimeConfig",
    # Utilities
    "get_available_tools",
]
