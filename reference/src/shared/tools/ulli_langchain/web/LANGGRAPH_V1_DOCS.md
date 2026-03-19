# LangGraph V1 Best Practices Documentation

## 1. Node Function Signatures

Nodes should receive **only the state parameter**. Additional data is passed through other mechanisms (see section 3).

## 2. State Definition

### What Belongs in State?

**Include in state:**
- Data that needs to persist across steps
- Data that can't be reconstructed later
- Data needed by downstream nodes
- Expensive-to-re-fetch data (search results, API responses)
- Execution metadata (for debugging and recovery)

**Don't store in state:**
- Data that can be derived from other data (compute when needed)

### Key Principle: Keep State Raw

Store raw data, not formatted text. Format prompts inside nodes when needed.

**Benefits:**
- Different nodes can format the same data differently
- Change prompt templates without modifying state schema
- Clearer debugging - see exactly what data each node received
- Agent can evolve without breaking existing state

### Example State for Email Agent
```python
from typing import TypedDict, List, Annotated
from langgraph.graph import MessagesState

class EmailAgentState(MessagesState):
    # Original email and sender info (can't reconstruct later)
    original_email: str
    sender_info: Dict[str, str]

    # Classification results (needed by downstream nodes)
    classification: Dict[str, str]

    # Search results and customer data (expensive to re-fetch)
    search_results: List[Dict]
    customer_data: Dict[str, Any]

    # Draft response (needs to persist through review)
    draft_response: str

    # Execution metadata (for debugging and recovery)
    execution_metadata: Dict[str, Any]
```

## 3. Non-State Data Passing

### Data Classification

| Data Type | Storage | Access Pattern |
|-----------|---------|----------------|
| Immutable, per-session | **config** | Passed at graph invocation |
| Mutable, per-session | **extend state graph** | Modify state schema |
| Per-invocation | **contextvars** | Access via context |
| Long-term | **Store** | ToolRuntime.store |

### ToolRuntime: Accessing Runtime Information

Tools receive `ToolRuntime` parameter providing:

| Component | Description | Use Case |
|-----------|-------------|----------|
| **State** | Short-term mutable data (messages, counters, custom fields) | Conversation history, tool call counts |
| **Context** | Immutable config passed at invocation (user IDs, session info) | User-based personalization |
| **Store** | Long-term persistent memory across conversations | User preferences, knowledge base |
| **Stream Writer** | Emit real-time updates during execution | Progress updates for long operations |
| **Config** | RunnableConfig for execution | Callbacks, tags, metadata |
| **Tool Call ID** | Unique identifier for current invocation | Correlate tool calls for logs |

### Example: Tool Accessing State
```python
from langchain.tools import tool, ToolRuntime

@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """Get the number of messages in the conversation."""
    messages = runtime.state["messages"]
    return f"There are {len(messages)} messages."
```

## 4. Graph Building/Compilation

### Standard Pattern
```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)  # Routes to "tools" or END
builder.add_edge("tools", "llm")

graph = builder.compile()
```

### ToolNode for Tool Execution

**ToolNode** is the prebuilt node for executing tools in LangGraph workflows:
- Handles parallel tool execution
- Error handling
- State injection automatically

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([tool1, tool2])
```

#### ToolNode Error Handling Options

```python
# Default: catch invocation errors, re-raise execution errors
tool_node = ToolNode(tools)

# Catch all errors and return error message to LLM
tool_node = ToolNode(tools, handle_tool_errors=True)

# Custom error message
tool_node = ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

# Custom error handler function
def handle_error(e: ValueError) -> str:
    return f"Invalid input: {e}"

tool_node = ToolNode(tools, handle_tool_errors=handle_error)

# Only catch specific exception types
tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))
```

## 5. Tool Definition Patterns

### Prefer @tool Decorator

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))
```

### Using ToolNode with Custom Workflows

For fine-grained control over tool execution patterns, use ToolNode directly:

```python
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_conditional_edges("llm", tools_condition)
```

## 6. Applying to web_agent.py Refactor

### Current Issues vs V1 Requirements

| Current Issue | V1 Solution |
|---------------|-------------|
| `async def _scrape_urls(state, raw_state, worker_id)` | **Only state parameter**. Use contextvars for worker_id. raw_state should be in state if needed. |
| `class InvestigativeState(Dict)` | **Use TypedDict** for type enforcement |
| Closure to pass raw_state/worker_id | **Extend state** or use **contextvars** |
| Manual tool invocation in nodes | **Use ToolNode** for graph-level tools |
| Class-based `BaseTool` pattern | **Prefer @tool decorator** |
| No error handling strategy | **Define error handling approach** |

### State Schema Refactor

```python
from typing import TypedDict, List, Dict, Any, Optional, Set

class InvestigativeState(TypedDict):
    """State for investigative research mode (multi-step with refinement)."""
    # Core task data
    task: str                      # Original research task
    plan: List[str]                # Research steps (LLM-generated)
    current_step: int              # Index into plan

    # Research findings
    findings: Dict[str, str]       # step -> extracted_info
    search_history: List[Dict]     # For query refinement
    final_synthesis: str           # Final output

    # Tracking
    visited_urls: Set[str]         # Deduplication
    source_ids: List[str]          # Track all source IDs

    # Flow control
    needs_refinement: bool         # Trigger for new search
    refinement_count: int          # Track refinement loops
    error: Optional[str]           # Error tracking

    # Config (immutable per research session)
    max_steps: int                 # Max research steps
    max_urls_per_step: int         # Max URLs to process per step
    max_refinements: int           # Max query refinements

    # Profiling
    profiler: Optional[ResearchProfiler]  # Timing metrics
```

### Non-State Data Handling

| Data | Current Approach | V1 Approach |
|------|------------------|-------------|
| `raw_state` (GlobalState) | Passed via closure | **Put in state** if needed across steps, or use **contextvars** |
| `worker_id` | Function parameter | **contextvar** - per-invocation data |
| `research_id` | In GlobalState | **config** - immutable per session |
| `profiler` | N/A | **state** - persists across steps |

### Node Signature Refactor

```python
# BEFORE (incorrect)
async def _scrape_urls(
    state: InvestigativeState,
    raw_state: Optional[Dict] = None,
    worker_id: str = "web_research_agent"
) -> InvestigativeState:
    ...

# AFTER (V1 compliant)
from contextvars import ContextVar

_worker_id_context: ContextVar[str] = ContextVar('worker_id', default="web_research_agent")

async def _scrape_urls(state: InvestigativeState) -> InvestigativeState:
    """Scrape selected URLs and extract relevant information."""
    # Get worker_id from context
    worker_id = _worker_id_context.get()

    # raw_state should be in state if needed
    # Or access via contextvar if truly external
    ...
```
