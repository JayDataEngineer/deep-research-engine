# Simplified Context Engine Architecture Plan

**Date**: 2024-03-18
**Status**: Draft for Approval
**Goal**: Strip down to essentials using `deepagents` as foundation

---

## Executive Summary

Replace the current over-engineered system (custom state, nodes, workers, routing) with a **two-agent architecture** built on `deepagents`:

1. **Main Deep Agent** - Context builder with flash explorers
2. **Critic Deep Agent** - Quality gate with read-only tools

Plus **two optional isolated processes**:
3. **NEO4j Graph Builder** - Post-processing
4. **Synthesis Module** - Output generation

---

## Current State (Problems)

| Component | Lines | Problem |
|-----------|-------|---------|
| `state/types.py` | 264 | Custom state reinvents deepagents state + filesystem |
| `engine.py` | 221 | Manual phase loop, not real LangGraph |
| `nodes/*.py` | ~400 | Node functions that should be agent tools |
| `workers/` | ~300 | Worker contracts that filesystem handles |
| `storage/` | ~200 | Partially implemented, deepagents has backends |

**Total complexity: ~1400+ lines of abstraction that deepagents provides**

---

## New Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTEXT ENGINE PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT                                                               │
│    │                                                                 │
│    ▼                                                                 │
│  ┌──────────────┐                                                    │
│  │  1. INGEST   │  Photos/Videos/Audio/PDFs → Text                   │
│  │     NODE     │  Saves to context/ filesystem                      │
│  └──────────────┘                                                    │
│    │                                                                 │
│    ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  2. MAIN DEEP AGENT (Context Builder)                         │   │
│  │                                                               │   │
│  │  Built-in tools:                                              │   │
│  │  • write_todos  - Plan context gathering                      │   │
│  │  • task         - Spawn flash explorers                       │   │
│  │  • read_file    - Read input/context                          │   │
│  │  • write_file   - Save context to filesystem                  │   │
│  │  • ls           - List available files                        │   │
│  │                                                               │   │
│  │  System prompt handles orchestration logic                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         │ spawns via task tool                                       │
│         ▼                                                            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  FLASH EXPLORERS (Lightweight haiku agents)                   │   │
│  │                                                               │   │
│  │  Can delegate to specialized agents:                          │   │
│  │  • Web Search Agent (Tavily)                                  │   │
│  │  • Vector DB Agent (Chroma/PGVector)                          │   │
│  │  • Document Agent (PDF/Docx parsing)                          │   │
│  │                                                               │   │
│  │  Write findings to context/ filesystem                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│    │                                                                 │
│    ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  3. CRITIC DEEP AGENT (Quality Gate)                          │   │
│  │                                                               │   │
│  │  Read-only tools:                                             │   │
│  │  • read_file    - Read context                                │   │
│  │  • ls           - List files                                  │   │
│  │                                                               │   │
│  │  NO write tools - Only evaluates                              │   │
│  │                                                               │   │
│  │  Outputs:                                                     │   │
│  │  • passed: bool                                               │   │
│  │  • score: 0-1                                                 │   │
│  │  • issues: []                                                 │   │
│  │  • suggestions: []                                            │   │
│  │                                                               │   │
│  │  If failed → loop back to Main Agent (max 3 iterations)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│    │                                                                 │
│    ▼                                                                 │
│  CONTEXT ENGINE COMPLETE                                             │
│    │                                                                 │
│    ├──────────────────────┬──────────────────────┐                  │
│    ▼                      ▼                      ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  OPTIONAL:   │  │  OPTIONAL:   │  │   DONE       │               │
│  │  NEO4j       │  │  SYNTHESIS   │  │              │               │
│  │  Builder     │  │  Module      │  │  Context     │               │
│  │              │  │              │  │  ready for   │               │
│  │  Reads       │  │  Reads       │  │  RAG queries │               │
│  │  context/    │  │  context/    │  │              │               │
│  │  Builds      │  │  Generates:  │  │              │               │
│  │  graph       │  │  • Twitter   │  │              │               │
│  │  relation-   │  │  • Podcast   │  │              │               │
│  │  ships       │  │  • Report    │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## New Project Structure

```
deep-research-engine/
├── src/
│   └── context_engine/
│       ├── __init__.py
│       ├── core.py                 # Main + Critic agents (~100 lines)
│       ├── pipeline.py             # Orchestration (~80 lines)
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── converter.py        # File → Text conversion
│       │   └── handlers/
│       │       ├── pdf.py
│       │       ├── image.py        # Vision model → text
│       │       ├── audio.py        # Whisper → text
│       │       └── video.py        # Frame extraction + vision
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── flash_explorer.py   # Fast context gatherer
│       │   └── specialists/
│       │       ├── __init__.py
│       │       ├── web_search.py   # Tavily integration
│       │       ├── vector_db.py    # Chroma/PGVector
│       │       └── document.py     # Document retrieval
│       │
│       ├── graph_builder/          # OPTIONAL MODULE
│       │   ├── __init__.py
│       │   └── neo4j_builder.py    # Copy from unified-llm-interface
│       │
│       ├── synthesis/              # OPTIONAL MODULE
│       │   ├── __init__.py
│       │   ├── twitter.py
│       │   ├── podcast.py
│       │   └── report.py
│       │
│       └── cli.py                  # Entry point
│
├── configs/
│   └── context_engine.yaml         # Single config file
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_agents.py
│   └── test_pipeline.py
│
└── pyproject.toml
```

---

## What Gets Deleted

```
DELETE:
├── src/deep_research_engine/state/         # Custom state - gone
├── src/deep_research_engine/nodes/         # Nodes - replaced by agent tools
├── src/deep_research_engine/workers/       # Workers - replaced by subagents
├── src/deep_research_engine/storage/       # Storage - use deepagents backends
├── src/deep_research_engine/engine.py      # Replaced by pipeline.py
├── src/deep_research_engine/prompts/       # Move to LangFuse or inline
└── src/deep_research_engine/config.py      # Simplify

KEEP (with modifications):
├── src/deep_research_engine/cli/           # Simplify
└── configs/                                # Consolidate to one config
```

---

## Implementation Phases

### Phase 1: Core Pipeline (Day 1-2)

**Goal**: Working end-to-end pipeline with minimal features

**Tasks**:
1. [ ] Create `src/context_engine/` structure
2. [ ] Implement `core.py` with Main Agent + Critic Agent
3. [ ] Implement `pipeline.py` with simple orchestration
4. [ ] Create basic `cli.py` entry point
5. [ ] Test with simple text input

**Acceptance**:
```bash
uv run context-engine build "What is quantum computing?"
# → Creates context/ with research findings
```

### Phase 2: Ingestion Node (Day 2-3)

**Goal**: Handle non-text inputs

**Tasks**:
1. [ ] Implement PDF handler (PyMuPDF)
2. [ ] Implement image handler (vision model → text)
3. [ ] Implement audio handler (Whisper)
4. [ ] Implement video handler (frame extraction + vision)
5. [ ] Create `ingestion/converter.py` dispatcher

**Acceptance**:
```bash
uv run context-engine build --input ./documents/
# → Converts all files to text in context/input/
```

### Phase 3: Flash Explorers + Specialists (Day 3-4)

**Goal**: Context gathering from multiple sources

**Tasks**:
1. [ ] Implement `flash_explorer.py`
2. [ ] Implement `specialists/web_search.py` (Tavily)
3. [ ] Implement `specialists/vector_db.py` (Chroma)
4. [ ] Implement `specialists/document.py`
5. [ ] Wire up subagent delegation

**Acceptance**:
```bash
uv run context-engine build "Compare React vs Vue"
# → Main agent spawns flash explorers
# → Flash explorers call web_search specialist
# → Context aggregated in context/research/
```

### Phase 4: Critic Loop (Day 4-5)

**Goal**: Quality gate with retry logic

**Tasks**:
1. [ ] Implement Critic Agent with read-only tools
2. [ ] Add retry logic in `pipeline.py`
3. [ ] Add scoring criteria configuration
4. [ ] Test critic feedback loop

**Acceptance**:
```bash
uv run context-engine build "Complex topic" --max-iterations 3
# → Critic evaluates
# → If score < 0.7, Main Agent refines
# → Loops until pass or max iterations
```

### Phase 5: Optional Modules (Day 5-7)

**Goal**: NEO4j and Synthesis as separate utilities

**Tasks**:
1. [ ] Copy NEO4j builder from unified-llm-interface
2. [ ] Implement synthesis/twitter.py
3. [ ] Implement synthesis/podcast.py
4. [ ] Implement synthesis/report.py
5. [ ] Add CLI commands for optional modules

**Acceptance**:
```bash
uv run context-engine build-graph --context ./context/
uv run context-engine synthesize --format podcast --context ./context/
```

---

## Core Implementation Sketch

### `context_engine/core.py`

```python
"""Core agents for the Context Engine."""

from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from pathlib import Path
from typing import Optional

from .agents.flash_explorer import flash_explorer


def main_agent(context_dir: Path) -> "DeepAgent":
    """Main context builder agent.

    Has full filesystem access and can spawn flash explorers.
    """
    return create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[],  # Built-in: write_todos, task, read_file, write_file, ls
        subagents=[
            ("explorer", flash_explorer(context_dir)),
        ],
        system_prompt=f"""
        You are the Context Engine Builder.

        Your job is to gather comprehensive context for research queries.

        Workflow:
        1. Read input from {context_dir}/input/
        2. Use write_todos to plan your context gathering
        3. Use the task tool to delegate to the explorer subagent
        4. Aggregate findings and write to {context_dir}/research/
        5. Write a summary to {context_dir}/summary.md

        Guidelines:
        - Be thorough but focused on the query
        - Cite sources in your findings
        - Organize context by topic/theme
        """,
    )


def critic_agent(context_dir: Path) -> "DeepAgent":
    """Critic agent with read-only access.

    Evaluates context quality without being able to modify it.
    """
    return create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        tools=[],  # Built-in: read_file, ls (NO write tools)
        system_prompt=f"""
        You are the Context Quality Critic.

        Your job is to evaluate the quality of gathered context.

        Read the context from {context_dir}/research/ and {context_dir}/summary.md

        You MUST respond with a JSON object:
        {{
            "passed": true/false,
            "score": 0.0-1.0,
            "issues": ["list of problems found"],
            "suggestions": ["list of improvements"]
        }}

        Evaluation criteria:
        - Relevance to the original query
        - Coverage of key topics
        - Quality of sources
        - Organization and clarity

        Score thresholds:
        - 0.8+: Excellent, pass immediately
        - 0.6-0.8: Good but could improve
        - Below 0.6: Significant gaps, needs more work
        """,
    )
```

### `context_engine/pipeline.py`

```python
"""Pipeline orchestration."""

import asyncio
from pathlib import Path
from typing import Optional
import json

from .core import main_agent, critic_agent
from .ingestion import convert_input


async def run_pipeline(
    input_path: Path,
    output_dir: Path,
    max_iterations: int = 3,
    pass_threshold: float = 0.7,
) -> dict:
    """Run the full context engine pipeline.

    Args:
        input_path: Path to input (file or directory)
        output_dir: Where to write context
        max_iterations: Max critic retry loops
        pass_threshold: Minimum score to pass

    Returns:
        Pipeline result with context path and score
    """
    # Setup directories
    context_dir = output_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "input").mkdir(exist_ok=True)
    (context_dir / "research").mkdir(exist_ok=True)

    # Phase 1: Ingestion
    print("[1/3] Ingesting input...")
    await convert_input(input_path, context_dir / "input")

    # Phase 2: Context Building with Critic Loop
    builder = main_agent(context_dir)
    critic = critic_agent(context_dir)

    for iteration in range(max_iterations):
        print(f"[2/3] Building context (iteration {iteration + 1})...")

        # Build context
        await builder.ainvoke({
            "messages": [{
                "role": "user",
                "content": "Build comprehensive context from the input."
            }]
        })

        # Evaluate
        print("[3/3] Evaluating context quality...")
        critique_result = await critic.ainvoke({
            "messages": [{
                "role": "user",
                "content": "Evaluate the context quality."
            }]
        })

        # Parse critique
        try:
            last_message = critique_result["messages"][-1].content
            critique = json.loads(last_message)
        except (json.JSONDecodeError, KeyError):
            critique = {"passed": False, "score": 0.5, "issues": ["Failed to parse critique"]}

        # Check if passed
        if critique.get("passed", False) or critique.get("score", 0) >= pass_threshold:
            print(f"Context passed with score {critique.get('score', 0):.2f}")
            break

        print(f"Context needs improvement (score: {critique.get('score', 0):.2f})")
        if iteration < max_iterations - 1:
            # Feed critique back to builder
            await builder.ainvoke({
                "messages": [{
                    "role": "user",
                    "content": f"The critic found these issues: {critique.get('issues', [])}. "
                               f"Please improve the context. Suggestions: {critique.get('suggestions', [])}"
                }]
            })
    else:
        print(f"Max iterations ({max_iterations}) reached")

    return {
        "context_dir": str(context_dir),
        "score": critique.get("score", 0),
        "passed": critique.get("passed", False),
        "iterations": iteration + 1,
    }
```

### `context_engine/agents/flash_explorer.py`

```python
"""Flash Explorer - fast context gathering agent."""

from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from pathlib import Path

from .specialists.web_search import web_search_agent
from .specialists.vector_db import vector_db_agent


def flash_explorer(context_dir: Path) -> "DeepAgent":
    """Fast context gathering agent.

    Lightweight (haiku) agent that can delegate to specialists.
    """
    return create_deep_agent(
        model=ChatAnthropic(model="claude-haiku-4-5"),  # Fast model
        tools=[],
        subagents=[
            ("web", web_search_agent()),
            ("vectors", vector_db_agent()),
        ],
        system_prompt=f"""
        You are a Flash Explorer - a fast context gatherer.

        Your job is to quickly gather relevant information.

        Tools:
        - Use the 'web' subagent for web searches
        - Use the 'vectors' subagent for vector database queries
        - Use write_file to save findings to {context_dir}/research/

        Be fast and focused. Don't overthink.
        """,
    )
```

---

## Dependencies

```toml
# pyproject.toml (dependencies section)

dependencies = [
    "deepagents>=0.1.0",           # Core agent framework
    "langchain-anthropic>=0.3.0",  # Claude models
    "langfuse>=2.0.0",             # Optional: prompt management

    # Ingestion
    "pymupdf>=1.24.0",             # PDF parsing
    "whisper>=1.0.0",              # Audio transcription (optional)
    "pillow>=10.0.0",              # Image handling

    # Specialists
    "langchain-community>=0.3.0",  # Tavily integration
    "chromadb>=0.4.0",             # Vector DB (optional)

    # Optional modules
    "neo4j>=5.0.0",                # Graph builder (optional)
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]
```

---

## CLI Design

```bash
# Build context from query
uv run context-engine build "What is quantum computing?"

# Build from input files
uv run context-engine build --input ./documents/ --output ./my_context/

# With options
uv run context-engine build "Query" \
    --max-iterations 5 \
    --pass-threshold 0.8 \
    --model claude-opus-4-6

# Optional: Build NEO4j graph
uv run context-engine build-graph --context ./my_context/context/

# Optional: Synthesize output
uv run context-engine synthesize \
    --context ./my_context/context/ \
    --format podcast \
    --output ./podcast_script.md

# Optional: Start RAG server
uv run context-engine serve --context ./my_context/context/ --port 8000
```

---

## Migration Strategy

1. **Create new structure** in `src/context_engine/` alongside existing code
2. **Implement core** (Phase 1-2) without deleting old code
3. **Test new pipeline** against existing use cases
4. **Delete old code** only after new pipeline is validated
5. **Move optional modules** (NEO4j, synthesis) as separate packages if needed

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Core lines of code | ~1400 | ~300 |
| Config files | 4+ | 1 |
| State fields | 18 | 0 (use filesystem) |
| Node functions | 4 | 0 (use agent tools) |
| Worker implementations | 4 partial | 3 specialists |
| Time to implement basic pipeline | Days | Hours |

---

## Open Questions

1. **Ingestion**: Should ingestion be a separate CLI step or automatic?
2. **Storage backend**: In-memory (dev) vs filesystem (default) vs LangGraph Store (production)?
3. **Critic model**: Same as main (sonnet) or lighter (haiku)?
4. **Synthesis**: Inline in pipeline or always separate CLI command?

---

## References

- [deepagents documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [deepagents backends](https://docs.langchain.com/oss/python/deepagents/backends)
- [LangGraph Memory Store](https://docs.langchain.com/oss/python/langgraph/persistence#memory-store)
- Existing unified-llm-interface deep research implementation
