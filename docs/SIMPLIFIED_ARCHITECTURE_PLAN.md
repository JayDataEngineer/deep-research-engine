# Simplified Context Engine Architecture Plan

**Date**: 2024-03-18
**Status**: Draft v2 - Updated with Clarifications
**Goal**: Strip down to essentials using `deepagents` + LangGraph Store

---

## Executive Summary

Replace the current over-engineered system with a **two-agent architecture** built on `deepagents`:

1. **Main Deep Agent** (Smart Model: Sonnet) - Context builder with flash explorers
2. **Critic Deep Agent** (Smart Model: Sonnet) - Quality gate with read-only tools

Plus **two optional isolated processes**:
3. **NEO4j Graph Builder** - Post-processing (copies from unified-llm-interface)
4. **Synthesis Module** - Separate process, not in pipeline

---

## Key Constraints (Non-Negotiable)

| Constraint | Implementation |
|------------|----------------|
| **LiteLLM Only** | All models via LiteLLM proxy, no parameters, just model names |
| **LangFuse Integration** | Prompt management + tracing via LangFuse |
| **Docker Infrastructure** | Postgres, S3 (MinIO), Neo4j, MongoDB already available |
| **LangGraph Checkpointer** | State persistence via Postgres checkpointer |
| **Auto-Vectorization** | LangGraph Store with `index_config` auto-vectors `write_file` content |
| **Two Model Tiers** | Smart (Sonnet) for high-level, Fast (Haiku) for subagents |
| **Simple Logging** | Loguru + Rich, stage transitions + critic reviews only |
| **Debuggability** | Generalized TypedDict state, clear observability |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTEXT ENGINE PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT (Query + Files)                                              │
│    │                                                                 │
│    ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  1. AUTO-INGEST (Isolated Tool)                               │   │
│  │                                                               │   │
│  │  Detects non-text → Converts to text automatically:          │   │
│  │  • PDFs       → PyMuPDF → text                                │   │
│  │  • Images     → Vision Model → text description               │   │
│  │  • Audio      → Whisper → transcript                          │   │
│  │  • Video      → Frame extraction + Vision → text              │   │
│  │                                                               │   │
│  │  Writes to: store.put((session_id, "input"), key, value)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│    │                                                                 │
│    ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  2. MAIN DEEP AGENT (SMART: Sonnet via LiteLLM)              │   │
│  │                                                               │   │
│  │  Store: LangGraph PostgresStore with auto-vectorization      │   │
│  │  Checkpointer: Postgres for state persistence                │   │
│  │                                                               │   │
│  │  Tools (built-in from deepagents + StoreBackend):            │   │
│  │  • write_todos    - Plan context gathering                    │   │
│  │  • task           - Spawn flash explorers                     │   │
│  │  • read_file      - Read from store (auto-vectorized)         │   │
│  │  • write_file     - Write to store (auto-vectorized!)         │   │
│  │  • ls             - List store contents                       │   │
│  │  • search_vectors - Semantic search via store.asearch()       │   │
│  │                                                               │   │
│  │  Prompts: Loaded from LangFuse                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         │ spawns via task tool                                       │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  FLASH EXPLORERS (FAST: Haiku via LiteLLM)                   │   │
│  │                                                               │   │
│  │  Same store access (auto-vectorization inherited)            │   │
│  │                                                               │   │
│  │  Can delegate to specialized agents:                          │   │
│  │  • Web Search Agent (Tavily)                                  │   │
│  │  • Document Agent (read_file + summarize)                     │   │
│  │                                                               │   │
│  │  Write findings via write_file → AUTO-VECTORIZED             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│    │                                                                 │
│    ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  3. CRITIC DEEP AGENT (SMART: Sonnet via LiteLLM)            │   │
│  │                                                               │   │
│  │  Read-only tools (StoreBackend with no write methods):       │   │
│  │  • read_file      - Read context                              │   │
│  │  • ls             - List files                                │   │
│  │  • search_vectors - Search context semantically               │   │
│  │                                                               │   │
│  │  NO write_file, NO edit_file - Only evaluates                │   │
│  │                                                               │   │
│  │  Outputs JSON:                                                │   │
│  │  {                                                            │   │
│  │    "passed": bool,                                            │   │
│  │    "score": 0.0-1.0,                                          │   │
│  │    "issues": [...],                                           │   │
│  │    "suggestions": [...]                                       │   │
│  │  }                                                            │   │
│  │                                                               │   │
│  │  On FAIL: Feeds CONTEXT + REVIEW back to Main Agent          │   │
│  │  Loop: max 3 iterations                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│    │                                                                 │
│    ▼                                                                 │
│  CONTEXT ENGINE COMPLETE                                             │
│    │                                                                 │
│    ├─────────────────────────────────────────────────────┐          │
│    ▼                                                     ▼          │
│  ┌──────────────────┐                      ┌──────────────────┐     │
│  │  OPTIONAL:       │                      │  OPTIONAL:       │     │
│  │  NEO4j GRAPH     │                      │  SYNTHESIS       │     │
│  │  BUILDER         │                      │  (ISOLATED)      │     │
│  │                  │                      │                  │     │
│  │  Reads from      │                      │  Separate CLI    │     │
│  │  LangGraph Store │                      │  command, NOT    │     │
│  │  via             │                      │  in pipeline     │     │
│  │  get_all_memories│                      │                  │     │
│  │                  │                      │  Generates:      │     │
│  │  Extracts        │                      │  • Twitter thread│     │
│  │  entities →      │                      │  • Podcast script│     │
│  │  Neo4j           │                      │  • Report        │     │
│  │                  │                      │                  │     │
│  │  Copy from       │                      │  Reads from      │     │
│  │  unified-llm-    │                      │  LangGraph Store │     │
│  │  interface       │                      │                  │     │
│  └──────────────────┘                      └──────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Storage Architecture

### LangGraph Store (Auto-Vectorization)

```python
# Store configuration with auto-vectorization
from langgraph.store.postgres import PostgresStore
from langchain_litellm import LiteLLMEmbeddings

store = PostgresStore(
    conn_string="postgresql://...",  # Docker Postgres
    index_config={
        "dims": 384,  # Embedding dimensions
        "embed": LiteLLMEmbeddings(model="text-embedding-3-small"),
        "fields": ["content"],  # Auto-vectorize content field
    }
)

# When agents use write_file via StoreBackend:
# 1. Content is written to store
# 2. Embedding is auto-generated
# 3. Semantic search is available immediately

await store.aput(
    namespace=(session_id, "memories", "insights"),
    key="finding_001",
    value={"content": "Important research finding..."}  # AUTO-VECTORIZED
)

# Semantic search
results = await store.asearch(
    namespace_prefix=(session_id, "memories"),
    query="research about AI",
    limit=10
)
```

### Checkpointer (State Persistence)

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://..."  # Same Docker Postgres
)

# Graph compilation with checkpointer
graph = builder.compile(
    checkpointer=checkpointer,
    store=store  # Auto-vectorized store
)
```

### Namespace Structure

```
(session_id, "input")           # Ingested input files
(session_id, "memories", "dumps")    # Context overflow dumps
(session_id, "memories", "web")      # Web research content
(session_id, "memories", "insights") # Worker findings
(session_id, "summary")              # Final summary
```

---

## State Design (Generalized TypedDict)

```python
"""Minimal state - most data lives in Store, not state."""

from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator


class ContextEngineState(TypedDict, total=False):
    """Minimal state for Context Engine.

    Most context lives in LangGraph Store (auto-vectorized).
    State only tracks orchestration data.
    """

    # Messages (standard LangGraph pattern)
    messages: Annotated[List[BaseMessage], operator.add]

    # Session tracking
    session_id: str
    run_id: str

    # Orchestration
    current_phase: str  # "ingest" | "build" | "critic" | "complete"
    iteration_count: int
    max_iterations: int

    # Critic feedback (for loop)
    critic_score: float
    critic_passed: bool
    critic_issues: List[str]
    critic_suggestions: List[str]

    # Error handling
    errors: List[Dict[str, Any]]
    fatal_error: str | None
```

---

## Model Configuration (LiteLLM Only)

```python
"""Model configuration - LiteLLM proxy only."""

from langchain_litellm import ChatLiteLLM

# Model names only, no parameters
SMART_MODEL = "claude-sonnet-4-6"  # Main agent, Critic
FAST_MODEL = "claude-haiku-4-5"    # Flash explorers, specialists

def get_model(tier: str = "smart") -> ChatLiteLLM:
    """Get model via LiteLLM proxy.

    Args:
        tier: "smart" (Sonnet) or "fast" (Haiku)

    Returns:
        ChatLiteLLM instance
    """
    model = SMART_MODEL if tier == "smart" else FAST_MODEL
    return ChatLiteLLM(
        model=model,
        # All other params handled by LiteLLM proxy config
    )
```

---

## Logging (Loguru + Rich)

```python
"""Simple logging with Loguru + Rich."""

from loguru import logger
from rich.console import Console
from rich.panel import Panel

console = Console()

def log_stage(stage: str, details: dict = None):
    """Log stage transition with Rich panel."""
    console.print(Panel(
        f"[bold]{stage}[/bold]",
        subtitle=details or "",
        border_style="blue"
    ))
    logger.info(f"STAGE: {stage}", **(details or {}))

def log_critic_review(score: float, passed: bool, issues: list):
    """Log critic review on fail."""
    if not passed:
        console.print(Panel(
            f"[red]Critic: FAILED (score: {score:.2f})[/red]\n"
            f"Issues: {issues}",
            border_style="red"
        ))
        logger.warning(f"CRITIC FAILED: score={score}, issues={issues}")
    else:
        console.print(Panel(
            f"[green]Critic: PASSED (score: {score:.2f})[/green]",
            border_style="green"
        ))
        logger.info(f"CRITIC PASSED: score={score}")
```

---

## Project Structure

```
deep-research-engine/
├── src/
│   └── context_engine/
│       ├── __init__.py
│       ├── state.py                 # ContextEngineState TypedDict
│       ├── config.py                # Models, store config
│       │
│       ├── store/
│       │   ├── __init__.py
│       │   ├── setup.py             # PostgresStore + Checkpointer setup
│       │   └── tools.py             # search_vectors tool
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── main.py              # Main Deep Agent (Sonnet)
│       │   ├── critic.py            # Critic Deep Agent (Sonnet, read-only)
│       │   ├── flash_explorer.py    # Flash Explorer (Haiku)
│       │   └── specialists/
│       │       ├── __init__.py
│       │       ├── web_search.py    # Tavily integration
│       │       └── document.py      # Document summarizer
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── auto_detect.py       # Detect non-text input
│       │   └── converters/
│       │       ├── pdf.py           # PyMuPDF
│       │       ├── image.py         # Vision → text
│       │       ├── audio.py         # Whisper
│       │       └── video.py         # Frame extraction
│       │
│       ├── prompts/
│       │   ├── __init__.py
│       │   └── langfuse.py          # LangFuse prompt loading
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── runner.py            # Main orchestration
│       │
│       ├── graph_builder/           # OPTIONAL: NEO4j
│       │   ├── __init__.py
│       │   └── neo4j_builder.py     # Copy from unified-llm-interface
│       │
│       ├── synthesis/               # OPTIONAL: Separate CLI
│       │   ├── __init__.py
│       │   ├── twitter.py
│       │   ├── podcast.py
│       │   └── report.py
│       │
│       ├── logging/
│       │   ├── __init__.py
│       │   └── rich_logging.py      # Loguru + Rich setup
│       │
│       └── cli.py                   # Entry point
│
├── configs/
│   └── context_engine.yaml          # Single config
│
├── tests/
│   ├── test_store.py
│   ├── test_agents.py
│   └── test_pipeline.py
│
└── pyproject.toml
```

---

## What Gets Deleted

```
DELETE:
├── src/deep_research_engine/state/         # Replaced by minimal ContextEngineState
├── src/deep_research_engine/nodes/         # Replaced by agent tools
├── src/deep_research_engine/workers/       # Replaced by flash explorers
├── src/deep_research_engine/storage/       # Replaced by LangGraph Store
├── src/deep_research_engine/engine.py      # Replaced by pipeline/runner.py
├── src/deep_research_engine/prompts/       # Move to LangFuse
└── src/deep_research_engine/config.py      # Simplify
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Day 1-2)

**Goal**: LangGraph Store + Checkpointer + Basic Agents

**Tasks**:
1. [ ] Setup `src/context_engine/` structure
2. [ ] Implement `store/setup.py` with PostgresStore + auto-vectorization
3. [ ] Implement `store/tools.py` with `search_vectors` tool
4. [ ] Implement `state.py` with minimal ContextEngineState
5. [ ] Implement `config.py` with LiteLLM model loading
6. [ ] Implement `logging/rich_logging.py`
7. [ ] Test store + checkpointer with Docker Postgres

**Acceptance**:
```bash
# Store works with auto-vectorization
uv run pytest tests/test_store.py -v
# All tests pass
```

### Phase 2: Core Agents (Day 2-3)

**Goal**: Main + Critic agents working

**Tasks**:
1. [ ] Implement `prompts/langfuse.py` for prompt loading
2. [ ] Implement `agents/main.py` with full StoreBackend access
3. [ ] Implement `agents/critic.py` with read-only StoreBackend
4. [ ] Implement `pipeline/runner.py` with critic loop
5. [ ] Wire up LangFuse tracing
6. [ ] Test end-to-end with simple query

**Acceptance**:
```bash
uv run context-engine build "What is quantum computing?"
# → Main agent runs
# → Critic evaluates
# → Context saved to store with auto-vectorization
```

### Phase 3: Flash Explorers + Specialists (Day 3-4)

**Goal**: Subagent delegation working

**Tasks**:
1. [ ] Implement `agents/flash_explorer.py` with Haiku model
2. [ ] Implement `agents/specialists/web_search.py` (Tavily)
3. [ ] Implement `agents/specialists/document.py`
4. [ ] Wire up subagent delegation via `task` tool
5. [ ] Test parallel context gathering

**Acceptance**:
```bash
uv run context-engine build "Compare React vs Vue"
# → Main spawns flash explorers
# → Flash explorers call web_search specialist
# → All content auto-vectorized in store
```

### Phase 4: Auto-Ingestion (Day 4-5)

**Goal**: Automatic non-text input handling

**Tasks**:
1. [ ] Implement `ingestion/auto_detect.py`
2. [ ] Implement `ingestion/converters/pdf.py`
3. [ ] Implement `ingestion/converters/image.py`
4. [ ] Implement `ingestion/converters/audio.py`
5. [ ] Wire auto-ingestion into pipeline

**Acceptance**:
```bash
uv run context-engine build --input ./mixed_docs/
# → PDFs, images, audio auto-converted to text
# → All stored in (session_id, "input") namespace
```

### Phase 5: Optional Modules (Day 5-7)

**Goal**: NEO4j and Synthesis as separate utilities

**Tasks**:
1. [ ] Copy NEO4j builder from unified-llm-interface
2. [ ] Adapt to read from LangGraph Store
3. [ ] Implement `synthesis/` as separate CLI commands
4. [ ] Add CLI commands for optional modules

**Acceptance**:
```bash
# NEO4j (reads from store)
uv run context-engine build-graph --session-id abc123

# Synthesis (separate CLI, NOT in pipeline)
uv run context-engine synthesize --session-id abc123 --format podcast
```

---

## Core Implementation

### `store/setup.py`

```python
"""LangGraph Store + Checkpointer setup."""

from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_litellm import LiteLLMEmbeddings
import os


async def get_store_and_checkpointer():
    """Create LangGraph Store with auto-vectorization + Checkpointer."""

    conn_string = os.getenv(
        "POSTGRES_URL",
        "postgresql://ulli:ulli_password@localhost:5432/context_engine"
    )

    # Store with auto-vectorization
    store = PostgresStore(
        conn_string,
        index_config={
            "dims": 384,
            "embed": LiteLLMEmbeddings(model="text-embedding-3-small"),
            "fields": ["content"],  # Auto-vectorize 'content' field
        }
    )
    await store.setup()

    # Checkpointer for state persistence
    checkpointer = PostgresSaver.from_conn_string(conn_string)
    await checkpointer.setup()

    return store, checkpointer
```

### `store/tools.py`

```python
"""Custom tools for store access."""

from langchain_core.tools import tool
from langgraph.store.base import BaseStore


def create_search_vectors_tool(store: BaseStore, session_id: str):
    """Create search_vectors tool for semantic search."""

    @tool
    async def search_vectors(query: str, namespace: str = "memories", limit: int = 10) -> str:
        """Search context semantically.

        Args:
            query: Natural language search query
            namespace: Sub-namespace (memories, web, insights)
            limit: Max results

        Returns:
            Formatted search results
        """
        ns = (session_id, namespace)

        results = await store.asearch(
            namespace_prefix=ns,
            query=query,
            limit=limit
        )

        if not results:
            return "No results found."

        output = []
        for r in results:
            output.append(f"[{r.key}] (score: {r.score:.2f})\n{r.value.get('content', '')[:500]}")

        return "\n\n---\n\n".join(output)

    return search_vectors
```

### `agents/main.py`

```python
"""Main Deep Agent - Context Builder."""

from deepagents import create_deep_agent
from langchain_litellm import ChatLiteLLM

from ..config import SMART_MODEL
from ..store.tools import create_search_vectors_tool


def main_agent(store, session_id: str, prompts: dict):
    """Main context builder agent.

    Uses SMART model (Sonnet) via LiteLLM.
    Has full store access (read + write + search).
    """

    # Custom tools
    search_vectors = create_search_vectors_tool(store, session_id)

    return create_deep_agent(
        model=ChatLiteLLM(model=SMART_MODEL),
        tools=[search_vectors],  # Built-in: write_todos, task, read_file, write_file, ls
        store=store,  # StoreBackend will use this
        system_prompt=prompts["main_agent"].format(session_id=session_id),
    )
```

### `agents/critic.py`

```python
"""Critic Deep Agent - Quality Gate."""

from deepagents import create_deep_agent
from langchain_litellm import ChatLiteLLM

from ..config import SMART_MODEL
from ..store.tools import create_search_vectors_tool


def critic_agent(store, session_id: str, prompts: dict):
    """Critic agent with read-only access.

    Uses SMART model (Sonnet) via LiteLLM.
    Can read and search, but NOT write.
    """

    # Read-only tools
    search_vectors = create_search_vectors_tool(store, session_id)

    return create_deep_agent(
        model=ChatLiteLLM(model=SMART_MODEL),
        tools=[search_vectors],
        # NO store= here means no write_file tool
        # Only built-in read_file (read-only) + ls
        system_prompt=prompts["critic_agent"],
    )
```

### `pipeline/runner.py`

```python
"""Main pipeline orchestration."""

import asyncio
import json
from pathlib import Path

from ..store.setup import get_store_and_checkpointer
from ..agents.main import main_agent
from ..agents.critic import critic_agent
from ..ingestion.auto_detect import auto_ingest
from ..logging.rich_logging import log_stage, log_critic_review
from ..prompts.langfuse import load_prompts


async def run_pipeline(
    query: str,
    input_path: Path | None = None,
    session_id: str | None = None,
    max_iterations: int = 3,
    pass_threshold: float = 0.7,
) -> dict:
    """Run the full context engine pipeline.

    Args:
        query: Research query
        input_path: Optional input files
        session_id: Session ID (auto-generated if None)
        max_iterations: Max critic loops
        pass_threshold: Minimum score to pass

    Returns:
        Pipeline result with session_id and score
    """
    import uuid
    session_id = session_id or str(uuid.uuid4())

    # Setup
    store, checkpointer = await get_store_and_checkpointer()
    prompts = await load_prompts()

    # Phase 1: Auto-Ingestion
    if input_path:
        log_stage("INGEST", {"path": str(input_path)})
        await auto_ingest(input_path, store, session_id)

    # Phase 2: Context Building with Critic Loop
    builder = main_agent(store, session_id, prompts)
    critic = critic_agent(store, session_id, prompts)

    log_stage("BUILD", {"session_id": session_id})

    for iteration in range(max_iterations):
        log_stage(f"BUILD (iteration {iteration + 1})")

        # Build context
        result = await builder.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": session_id}}
        )

        # Evaluate
        log_stage("CRITIC")
        critique_result = await critic.ainvoke(
            {"messages": [{"role": "user", "content": "Evaluate context quality."}]},
            config={"configurable": {"thread_id": session_id}}
        )

        # Parse critique
        try:
            last_message = critique_result["messages"][-1].content
            critique = json.loads(last_message)
        except (json.JSONDecodeError, KeyError):
            critique = {"passed": False, "score": 0.5, "issues": ["Parse error"]}

        score = critique.get("score", 0)
        passed = critique.get("passed", False) or score >= pass_threshold

        log_critic_review(score, passed, critique.get("issues", []))

        if passed:
            log_stage("COMPLETE", {"score": score})
            break

        # On fail: feed CONTEXT + REVIEW back to main agent
        log_stage("REFINING", {"reason": "critic_feedback"})
        query = f"""
        The critic reviewed your context and found issues.

        SCORE: {score}
        ISSUES: {critique.get('issues', [])}
        SUGGESTIONS: {critique.get('suggestions', [])}

        Please improve the context based on this feedback.
        The original query was: {query}
        """

    else:
        log_stage("COMPLETE", {"note": f"Max iterations ({max_iterations}) reached"})

    return {
        "session_id": session_id,
        "score": score,
        "passed": passed,
        "iterations": iteration + 1,
    }
```

---

## CLI Design

```bash
# Build context from query
uv run context-engine build "What is quantum computing?"

# Build with input files (auto-ingested)
uv run context-engine build --input ./documents/ "Summarize these docs"

# With options
uv run context-engine build "Query" \
    --max-iterations 5 \
    --pass-threshold 0.8 \
    --session-id my-session

# OPTIONAL: Build NEO4j graph (reads from store)
uv run context-engine build-graph --session-id abc123

# OPTIONAL: Synthesize output (SEPARATE CLI, not in pipeline)
uv run context-engine synthesize \
    --session-id abc123 \
    --format podcast \
    --output ./podcast.md
```

---

## Dependencies

```toml
[project]
dependencies = [
    # Core
    "deepagents>=0.1.0",
    "langgraph>=0.2.0",
    "langgraph-checkpoint-postgres>=0.1.0",
    "langchain-litellm>=0.1.0",

    # Store + Embeddings
    "asyncpg>=0.29.0",
    "pgvector>=0.2.0",

    # LangFuse
    "langfuse>=2.0.0",

    # Logging
    "loguru>=0.7.0",
    "rich>=13.0.0",

    # Ingestion
    "pymupdf>=1.24.0",
    "pillow>=10.0.0",

    # Specialists
    "langchain-community>=0.3.0",  # Tavily
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

# Optional: Whisper for audio
audio = ["openai-whisper>=1.0.0"]

# Optional: NEO4j
graph = ["neo4j>=5.0.0"]
```

---

## Environment Variables

```bash
# LiteLLM (required)
LITELLM_BASE_URL=http://litellm.local:4000

# Postgres (Docker)
POSTGRES_URL=postgresql://ulli:ulli_password@localhost:5432/context_engine

# LangFuse (required)
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: MinIO for large files
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
```

---

## Migration Strategy

1. **Create new structure** in `src/context_engine/` alongside existing
2. **Implement store setup** first (Phase 1)
3. **Test with Docker Postgres** before proceeding
4. **Implement agents** (Phase 2-3)
5. **Delete old code** only after validation
6. **NEO4j + Synthesis** last (optional modules)

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Core lines of code | ~1400 | ~400 |
| State fields | 18 | 6 |
| Config files | 4+ | 1 |
| Manual vectorization | Yes | Auto (Store) |
| Checkpointer | No | Yes (Postgres) |
| Debuggability | Poor | Good (TypedDict + LangFuse traces) |

---

## References

- [LangGraph Store Documentation](https://langchain-ai.github.io/langgraph/reference/store/)
- [LangGraph Postgres Store](https://langchain-ai.github.io/langgraph/reference/checkpoints/)
- [Deep Agents StoreBackend](./reference/deepagents/libs/deepagents/deepagents/backends/store.py)
- [Auto-Vectorized Store Example](./reference/src/shared/memory/auto_vectorized_store.py)
- [LangGraph Postgres Store Tests](./reference/langgraph/libs/checkpoint-postgres/tests/test_store.py)
- Existing unified-llm-interface deep research implementation
