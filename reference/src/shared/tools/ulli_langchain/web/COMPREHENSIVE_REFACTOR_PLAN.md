# Research Framework - Comprehensive Refactor Plan (UPDATED)

## Executive Summary

This document outlines ALL changes required to refactor the entire Research Framework, including:

### Files Affected
- `web_agent.py` - Web research tool (main refactor)
- `deep_research/graph/graph.py` - Graph compilation with Store
- `deep_research/graph/state.py` - State schema updates
- `deep_research/graph/nodes/*.py` - Node signature refactors
- Virtual filesystem backends (unchanged architecture, new integration)

### Changes Required:
1. Remove all fallbacks (9 locations)
2. Remove legacy code and patterns
3. Align with LangGraph V1 best practices
4. Add comprehensive profiling/timing
5. **Integrate LangGraph Store for cross-thread persistence**
6. **Use LangChain ingestion/Embeddings for semantic search**
7. **Preserve our Virtual Filesystem tiering architecture**

---

# Part 1: LangGraph V1 Best Practices

## A. Node Function Signatures

**RULE:** Nodes receive **ONLY the state parameter**. No additional parameters.

```python
# ❌ WRONG - Current pattern
async def _scrape_urls(state, raw_state=None, worker_id="default"):

# ✅ CORRECT - V1 pattern
async def _scrape_urls(state: InvestigativeState) -> InvestigativeState:
```

## B. State Schema - What Belongs in State?

### Include in State (Criteria):
- ✅ Data that needs to persist across steps
- ✅ Data that can't be reconstructed later
- ✅ Data needed by downstream nodes
- ✅ Expensive-to-re-fetch data (search results, API responses)
- ✅ Execution metadata (for debugging and recovery)

### Don't Store in State:
- ❌ Data that can be derived from other data (compute when needed)

### Key Principle: Keep State Raw

**Store raw data, NOT formatted text.** Format prompts inside nodes when needed.

## C. Non-State Data Passing

### Data Classification Table

| Data Type | Storage | Access Pattern | Example |
|-----------|---------|----------------|---------|
| Immutable, per-session | **config** | Passed at graph invocation | `research_id`, session configs |
| Mutable, per-session | **extend state** | Modify state schema | `profiler`, `findings` |
| Per-invocation | **contextvars** | Access via context | `worker_id`, request IDs |
| **Cross-thread, long-term** | **Store** | **`store: BaseStore` parameter** | User preferences, research history |

### NEW: LangGraph Store Integration

LangGraph Store solves **cross-thread persistence** with built-in semantic search.

**When to use Store:**
- User preferences across conversations
- Research history across sessions
- Long-term memory (semantic search)
- Data that needs to survive thread termination

**Store vs VFS:**
| Feature | LangGraph Store | Virtual Filesystem |
|---------|-----------------|-------------------|
| Best for | JSON blobs, metadata, embeddings | Large files, raw content |
| Semantic search | ✅ Built-in | ❌ Manual implementation needed |
| Cross-thread | ✅ Native | ⚠️ Requires custom handling |
| Storage size | Small (KB per item) | Large (MB per file) |

---

# Part 2: HYBRID ARCHITECTURE (NEW)

## The "Body" + "Brain" Pattern

We **combine** our Virtual Filesystem with LangGraph Store rather than choosing one.

### Domain A: Research Archive (Your Virtual Filesystem)

**What goes here:**
- Raw HTML scrapes (50KB+ each)
- PDF content
- Full markdown files
- "Tier A" summaries (large content)

**Why:**
- These are **Artifacts** - large, static, structure-heavy
- LangGraph Store is not optimized for large blobs
- Your SQLite/File-based system already works

**Implementation:** Keep current VFS unchanged.

### Domain B: Semantic Brain (LangGraph Store)

**What goes here:**
- High-level insights
- User preferences
- "Claims" extracted from research
- **Pointers to VFS files** (with embeddings)

**Why:**
- These are **Memories** - retrieved via meaning (Embeddings)
- Semantic search enables "fuzzy" conceptual matching
- Cross-thread persistence for user research history

**Implementation:** NEW - integrate LangGraph Store.

## The Workflow: How They Work Together

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WORKER NODE                                 │
└─────────────────────────────────────────────────────────────────────┘
           │
           ├── Scrape URL (50KB raw HTML)
           │
           ├─► SAVE TO VFS
           │   /research/tier_a/doc_abc123.md
           │   (Full content - "The Body")
           │
           └─► SAVE TO STORE
               namespace=(user_id, "research_index")
               key=doc_abc123
               value={
                   "title": "Nietzsche's Diet",
                   "summary": "He ate only fruit...",
                   "vfs_path": "/research/tier_a/doc_abc123.md",  # ← THE LINK
                   "tier": "A"
               }
               index=["summary", "title"]  # ← Enables semantic search

┌─────────────────────────────────────────────────────────────────────┐
│                      SYNTHESIS NODE                                │
└─────────────────────────────────────────────────────────────────────┘
           │
           ├── Needs info about "fruit"
           │
           ├─► SEARCH STORE (semantic)
           │   query="fruit diet"
           │   Returns: {
           │       vfs_path: "/research/tier_a/doc_abc123.md",
           │       summary: "He ate only fruit..."
           │   }
           │
           └─► READ FROM VFS
               filesystem.read("/research/tier_a/doc_abc123.md")
               (Full content for detailed analysis)
```

---

# Part 3: LangChain Ingestion Integration (UPDATED)

## Hybrid Approach: Keep Our Scraper + Use LangChain Downstream

**IMPORTANT:** We keep our custom `web_get` scraper because it bypasses CAPTCHA blockers.
We only use LangChain for text splitting, embeddings, and vector storage.

### Why This Hybrid Approach?

| Feature | Our Custom | LangChain | Decision |
|---------|------------|-----------|----------|
| **Web Scraping** | ✅ CAPTCHA bypass, markdown conversion | `WebBaseLoader` (basic) | **Keep ours** |
| **Text Chunking** | Manual logic | `RecursiveCharacterTextSplitter` (smart) | **Use LangChain** |
| **Embeddings** | Manual API calls | `OpenAIEmbeddings` (30+ providers) | **Use LangChain** |
| **Vector Storage** | Manual implementation | `InMemoryVectorStore` / Chroma / Pinecone | **Use LangChain** |

### The Smart Ingestion Pattern

```python
# tools/ingest_url.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# Keep our custom scraper
from .scraper import web_get  # Your existing CAPTCHA-bypass scraper

# Global setup (do once)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

async def ingest_url_smart(url: str, tier: str):
    """
    Ingest URL using our scraper + LangChain downstream processing.

    Args:
        url: URL to ingest
        tier: Information Funnel tier ("A", "B", or "C")

    Returns:
        Number of chunks ingested
    """

    # A. Use OUR custom scraper (handles CAPTCHA, converts to markdown)
    content = await web_get(url)

    # B. Use LangChain TextSplitter (smart paragraph/newline handling)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    # Create Document from our scraped content
    doc = Document(page_content=content, metadata={"source_url": url})
    splits = text_splitter.split_documents([doc])

    # C. CRITICAL: Inject our Tiering Logic into Metadata
    source_id = _generate_source_id(url)
    vfs_path = f"/research/tier_{tier.lower()}/{source_id}.md"

    for split in splits:
        split.metadata["tier"] = tier
        split.metadata["source_url"] = url
        split.metadata["vfs_path"] = vfs_path
        split.metadata["source_id"] = source_id

        # For Tier C, we might tag as low_priority or skip entirely
        if tier == "C":
            split.metadata["priority"] = "low"

    # D. Use LangChain VectorStore (handles embedding & indexing)
    vector_store.add_documents(documents=splits)

    # E. Save full content to our VFS (for later retrieval of large files)
    filesystem = get_filesystem_from_context()
    filesystem.write(vfs_path, content)

    return len(splits)
```

### The Hybrid Retrieval Tool

```python
# tools/retrieve.py

from langchain_core.tools import tool

@tool
def lookup_knowledge(query: str, filter_tier: str = None):
    """
    Search research database with semantic search + tier filtering.

    Args:
        query: Semantic concept (e.g. "nietzsche diet")
        filter_tier: Optional tier filter ("A", "B", "C")

    Returns:
        Relevant documents with VFS paths for full reading
    """

    # 1. Prepare metadata filter (LangChain feature)
    filter_dict = {}
    if filter_tier:
        filter_dict["tier"] = filter_tier

    # 2. Semantic search with filter
    results = vector_store.similarity_search(
        query,
        k=5,
        filter=filter_dict  # Only search specified tier
    )

    # 3. Format output with VFS paths
    output = []
    for doc in results:
        output.append(f"""[Source: {doc.metadata['source_url']}]
[VFS Path: {doc.metadata.get('vfs_path', 'N/A')}]
[Tier: {doc.metadata.get('tier', 'unknown')}]

{doc.page_content}""")

    return "\n\n".join(output)
```

## Configuration

### Choose Your Vector Store

**For Development:**
```python
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
```

**For Production:**
```python
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="research_index",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)
```

### Choose Your Embedding Provider

```python
# Option 1: OpenAI (recommended)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Option 2: Local (Ollama)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Option 3: HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

---

# Part 4: Updated State Schema

## New InvestigativeState with Store Integration

```python
from typing import TypedDict, List, Dict, Any, Optional, Set

class InvestigativeState(TypedDict):
    """State for investigative research mode (LangGraph V1 compliant)."""

    # === Core task data ===
    task: str                      # Original research task
    plan: List[str]                # Research steps (LLM-generated)
    current_step: int              # Index into plan

    # === Research findings ===
    findings: Dict[str, str]       # step -> extracted_info
    search_history: List[Dict]     # For query refinement
    final_synthesis: str           # Final output

    # === Tracking ===
    visited_urls: Set[str]         # Deduplication
    source_ids: List[str]          # Track all source IDs (VFS paths)

    # === Flow control ===
    needs_refinement: bool         # Trigger for new search
    refinement_count: int          # Track refinement loops
    error: Optional[str]           # Error tracking

    # === Config (immutable per session) ===
    max_steps: int                 # Max research steps
    max_urls_per_step: int         # Max URLs to process per step
    max_refinements: int           # Max query refinements

    # === NEW: Store integration ===
    store_namespace: Optional[tuple]  # (user_id, "research") for Store
    vector_store_filter: Optional[Dict[str, Any]]  # Tier filter for retrieval

    # === Profiling ===
    profiler: Optional[Dict[str, Any]]  # Timing metrics
```

---

# Part 5: Node Signature with Store

## V1 Compliant Nodes with Store Access

```python
import contextvars
from langgraph.store.base import BaseStore

# Context variables for per-invocation data
_worker_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    'worker_id',
    default="web_research_agent"
)

# Example: Node with Store access
async def _scrape_urls(state: InvestigativeState, config: RunnableConfig, *, store: BaseStore) -> InvestigativeState:
    """
    Scrape selected URLs and save to both VFS and Store.

    Args:
        state: InvestigativeState (only required parameter)
        config: RunnableConfig (injected by LangGraph)
        store: BaseStore (injected by LangGraph)

    Returns:
        Updated InvestigativeState
    """
    # Get worker_id from context
    worker_id = _worker_id_context.get()

    # Get user_id from config for Store namespace
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "research_index")

    # Get profiler from state
    profiler_data = state.get("profiler", {})

    step = state["plan"][state["current_step"]]
    search_data = state["search_history"][-1]
    selected_urls = search_data.get("selected_urls", [])

    logger.info(f"[WEB_AGENT] Scraping {len(selected_urls)} URLs...")

    # For each URL:
    # 1. Scrape content
    # 2. Save full content to VFS (via filesystem context)
    # 3. Save summary + pointer to Store
    # 4. Store summary in state["findings"]

    for i, (title, url) in enumerate(selected_urls):
        # Scrape
        content = await _web_get(url)

        # Generate source ID
        source_id = _generate_source_id(url)

        # Save to VFS (full content)
        vfs_path = f"/research/tier_a/{source_id}.md"
        filesystem = get_filesystem_from_context()
        filesystem.write(vfs_path, content)

        # Save to Store (summary + pointer)
        summary = content[:500] + "..." if len(content) > 500 else content
        store.put(
            namespace,
            source_id,
            {
                "title": title,
                "url": url,
                "vfs_path": vfs_path,
                "summary": summary,
                "tier": "A",
                "timestamp": datetime.now().isoformat(),
            },
            index=["summary", "title"]  # Enable semantic search
        )

        # Track source ID
        state["source_ids"].append(source_id)
        search_data["scraped"].append((title, url, content, source_id))

    return state
```

## Synthesis Node with Semantic Search

```python
async def _synthesize_findings(state: InvestigativeState, config: RunnableConfig, *, store: BaseStore) -> InvestigativeState:
    """
    Synthesize findings using semantic search across Store.
    """
    user_id = config.get("configurable", {}).get("user_id", "default")
    namespace = (user_id, "research_index")

    # Semantic search for relevant findings
    task = state["task"]
    memories = store.search(
        namespace,
        query=task,  # Semantic search!
        limit=10
    )

    # Build context from semantic search results
    context_parts = []
    for memory in memories:
        vfs_path = memory.value.get("vfs_path")
        summary = memory.value.get("summary")
        title = memory.value.get("title")

        context_parts.append(f"""
## {title}
**VFS Path:** {vfs_path}
**Summary:** {summary}
""")

    # Use LLM to synthesize
    llm = _get_llm()
    prompt = f"""Synthesize research findings about: {task}

Relevant sources found via semantic search:
{chr(10).join(context_parts)}

Create a comprehensive report that addresses the original task.
"""

    response = await llm.ainvoke(prompt)
    state["final_synthesis"] = response.content

    return state
```

---

# Part 6: Graph Compilation with Store

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.memory import InMemoryStore

def build_research_graph():
    """
    Build the investigative research graph with Store.
    """
    builder = StateGraph(InvestigativeState)

    # Add nodes
    builder.add_node("plan", _plan_research)
    builder.add_node("search", _search_step)
    builder.add_node("select", _select_urls)
    builder.add_node("scrape", _scrape_urls)
    builder.add_node("extract", _extract_findings)
    builder.add_node("refine", _refine_search)
    builder.add_node("synthesize", _synthesize_findings)

    # Set entry point
    builder.set_entry_point("plan")

    # Add edges
    builder.add_edge("plan", "search")
    builder.add_edge("search", "select")
    builder.add_edge("select", "scrape")
    builder.add_edge("scrape", "extract")

    # Conditional routing
    builder.add_conditional_edges(
        "extract",
        _should_continue,
        {
            "continue": "search",
            "refine": "refine",
            "synthesize": "synthesize"
        }
    )

    builder.add_edge("refine", "select")
    builder.add_edge("synthesize", END)

    return builder.compile()

# Compile with checkpointer AND store
checkpointer = InMemorySaver()
in_memory_store = InMemoryStore()  # For cross-thread persistence

graph = build_research_graph().compile(
    checkpointer=checkpointer,
    store=in_memory_store
)

# Invocation with config
config = {
    "configurable": {
        "thread_id": "research_session_123",
        "user_id": "user_abc"
    }
}

result = await graph.ainvoke(initial_state, config=config)
```

---

# Part 7: Fallbacks to Remove (9 locations)

All fallbacks hide failures and make debugging harder. **REMOVE ENTIRELY.**

| Line | Location | Fallback | Replace With |
|------|----------|----------|--------------|
| 425-430 | `_select_urls` | LLM unavailable → pick top 3 | `RuntimeError("LLM required for URL selection")` |
| 483-485 | `_select_urls` | Parse fails → use top results | `ValueError("Failed to parse LLM response")` |
| 567-573 | `_extract_findings` | LLM unavailable → raw content | `RuntimeError("LLM required for findings extraction")` |
| 609-614 | `_extract_findings` | Exception → raw content | Re-raise with context |
| 692-698 | `_synthesize_findings` | LLM unavailable → concatenate | `RuntimeError("LLM required for synthesis")` |
| 729-735 | `_synthesize_findings` | Exception → concatenate | Re-raise with context |
| 794-796 | `_investigative_research` | LangGraph unavailable → simple | `RuntimeError("LangGraph required for investigative mode")` |
| 829-833 | `_investigative_research` | Synthesis missing → manual | `ValueError("Synthesis output missing from state")` |
| 837-840 | `_investigative_research` | Exception → simple mode | Re-raise with context |

---

# Part 8: Refactoring Implementation Plan (UPDATED)

## Phase 1: Setup LangChain Components (UPDATED)

```bash
# Add dependencies (NO langchain-community needed - we keep our scraper)
uv add langchain-text-splitters
uv add langchain-openai  # or your preferred embedding provider
uv add langchain-chroma  # or your preferred vector store (optional for dev)
```

## Phase 2: Create Ingestion Module (NEW)

Create `src/shared/tools/ulli_langchain/web/ingestion.py`:

```python
"""Smart URL ingestion using our scraper + LangChain downstream components."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from typing import Literal, Dict, Any
import hashlib

# Keep our custom scraper (CAPTCHA bypass + markdown conversion)
from .web_agent import web_get  # Your existing scraper

# Global singleton
embeddings = None
vector_store = None

def get_embeddings():
    """Get or initialize embeddings."""
    global embeddings
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings

def get_vector_store():
    """Get or initialize vector store."""
    global vector_store
    if vector_store is None:
        vector_store = InMemoryVectorStore(get_embeddings())
    return vector_store

def _generate_source_id(url: str) -> str:
    """Generate source ID from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]

async def ingest_url(
    url: str,
    tier: Literal["A", "B", "C"] = "B",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict[str, Any]:
    """
    Ingest URL using our scraper + LangChain downstream processing.

    Args:
        url: URL to ingest
        tier: Information Funnel tier
        chunk_size: Text splitter chunk size
        chunk_overlap: Text splitter overlap

    Returns:
        Dict with chunk counts, VFS path, and source ID
    """
    # A. Use OUR custom scraper (handles CAPTCHA, converts to markdown)
    content = await web_get(url)

    # B. Use LangChain TextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    # Create Document from our scraped content
    doc = Document(page_content=content, metadata={"source_url": url})
    splits = text_splitter.split_documents([doc])

    # C. Generate IDs and paths
    source_id = _generate_source_id(url)
    vfs_path = f"/research/tier_{tier.lower()}/{source_id}.md"

    # D. Inject metadata with tiering logic
    for split in splits:
        split.metadata.update({
            "tier": tier,
            "source_url": url,
            "vfs_path": vfs_path,
            "source_id": source_id,
        })

        # For Tier C, tag as low priority
        if tier == "C":
            split.metadata["priority"] = "low"

    # E. Add to vector store
    get_vector_store().add_documents(splits)

    return {
        "url": url,
        "tier": tier,
        "chunks": len(splits),
        "vfs_path": vfs_path,
        "source_id": source_id
    }
```

## Phase 3: Create Retrieval Tool (NEW)

Create `src/shared/tools/ulli_langchain/web/retrieve.py`:

```python
"""Semantic search tool with tier filtering."""

from langchain_core.tools import tool
from typing import Literal, Optional

@tool
def lookup_knowledge(
    query: str,
    filter_tier: Optional[Literal["A", "B", "C"]] = None,
    k: int = 5
) -> str:
    """
    Search research database with semantic search + tier filtering.

    Args:
        query: Semantic concept to search for
        filter_tier: Only search specified tier (optional)
        k: Number of results to return

    Returns:
        Formatted search results with VFS paths
    """
    from .ingestion import get_vector_store

    vector_store = get_vector_store()

    # Build filter
    filter_dict = {"tier": filter_tier} if filter_tier else None

    # Search
    results = vector_store.similarity_search(
        query,
        k=k,
        filter=filter_dict
    )

    # Format
    output = []
    for doc in results:
        meta = doc.metadata
        output.append(f"""**{meta.get('source_url', 'Unknown')}**
[VFS: {meta.get('vfs_path', 'N/A')}]
[Tier: {meta.get('tier', 'unknown')}]

{doc.page_content}""")

    return "\n\n".join(output)
```

## Phase 4: Update State Schema

See Part 4 above.

## Phase 5: Refactor Nodes for Store Integration

See Part 5 above.

## Phase 6: Add Profiling (Already Done)

The `ResearchProfiler` class has been added to web_agent.py.

## Phase 7: Remove Fallbacks

See Part 7 above.

## Phase 8: Remove Legacy Code

Remove legacy function wrapper (lines 1033-1058).

## Phase 9: Update Graph Compilation

See Part 6 above.

## Phase 10: Testing

```python
# Test ingestion
result = await ingest_url("https://example.com", tier="A")
print(f"Ingested {result['chunks']} chunks")

# Test retrieval
results = lookup_knowledge.invoke({"query": "fruit diet", "filter_tier": "A"})

# Test full flow
config = {"configurable": {"thread_id": "test", "user_id": "test_user"}}
graph_result = await graph.ainvoke(initial_state, config=config)
```

---

# Part 9: Dependencies

## New Dependencies

```bash
# LangChain components (we keep our custom scraper)
uv add langchain-text-splitters
uv add langchain-openai  # or preferred embedding provider

# Vector store (choose one)
uv add langchain-chroma      # For Chroma (optional for dev)
# OR
uv add langchain-postgres    # For PGVector (production)
# OR use InMemoryVectorStore (built-in to langchain-core)

# Already have
# - langgraph (for Store, StateGraph, etc.)
# - langchain-core
# - Our custom scraper with CAPTCHA bypass
```

## Files to Update

1. **web_agent.py** - Main refactor
2. **ingestion.py** - NEW - LangChain-based ingestion
3. **retrieve.py** - NEW - Semantic search tool
4. **graph.py** - Update graph compilation with Store
5. **config.yaml** - Add new tools to enabled_tools
6. **state.py** - Update InvestigativeState TypedDict

---

# Part 10: Migration Checklist

### Code Changes

- [ ] Add LangChain dependencies
- [ ] Create `ingestion.py` with LangChain components
- [ ] Create `retrieve.py` with semantic search tool
- [ ] Add `ResearchProfiler` class ✅ (DONE)
- [ ] Add contextvars for worker_id, raw_state
- [ ] Convert `InvestigativeState` to TypedDict
- [ ] Refactor node signatures (state-only + store)
- [ ] Remove 9 fallback paths
- [ ] Remove legacy function wrapper
- [ ] Update graph compilation with Store
- [ ] Integrate profiling throughout
- [ ] Update return types (include timing)

### Architecture

- [ ] VFS continues to store large files (unchanged)
- [ ] Store handles metadata/pointers/embeddings (NEW)
- [ ] VectorStore handles semantic search (NEW)
- [ ] Tiering logic preserved in metadata
- [ ] Link: Store.vfs_path → VFS file

### Testing

- [ ] Unit tests for ingestion
- [ ] Unit tests for retrieval
- [ ] Integration test for full flow
- [ ] Test Store persistence across threads
- [ ] Test semantic search quality
- [ ] Performance test with profiling

---

# Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| **Scraping** | Our custom scraper (CAPTCHA bypass) | **KEPT** - Our custom scraper |
| **Chunking** | Manual logic | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | Not available | LangChain `OpenAIEmbeddings` (or other) |
| **Search** | Exact match / SQL only | Semantic search via `VectorStore` |
| **Large Files** | VFS (SQLite/Files) | VFS (unchanged) |
| **Metadata** | Not searchable | Store with semantic search |
| **Cross-thread** | Not available | Store (native) |
| **Tiering** | Manual filtering | Metadata filter + VFS paths |
| **Profiling** | None | `ResearchProfiler` throughout |

---

# Quick Reference: Key Patterns

## Saving Research (Worker)

```python
# 1. Scrape
content = await web_get(url)

# 2. Save to VFS (large file)
vfs_path = f"/research/tier_a/{doc_id}.md"
filesystem.write(vfs_path, content)

# 3. Save to Store (metadata + pointer)
store.put(
    (user_id, "research"),
    doc_id,
    {
        "title": title,
        "summary": content[:500],
        "vfs_path": vfs_path,
        "tier": "A"
    },
    index=["summary", "title"]
)
```

## Retrieving Research (Synthesis)

```python
# 1. Semantic search
memories = store.search((user_id, "research"), query="fruit diet", limit=5)

# 2. For each result
for memory in memories:
    vfs_path = memory.value["vfs_path"]
    full_content = filesystem.read(vfs_path)
    # Use full_content in synthesis
```

## Graph Invocation

```python
config = {
    "configurable": {
        "thread_id": "session_123",
        "user_id": "user_abc"
    }
}

result = await graph.ainvoke(initial_state, config=config)
```
