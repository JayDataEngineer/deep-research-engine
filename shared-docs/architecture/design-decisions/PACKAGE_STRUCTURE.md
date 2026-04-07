# Deep Research Engine - Package Structure

## Directory Layout

```
deep-research-engine/
├── pyproject.toml                  # Package metadata + dependencies
├── README.md                       # Usage documentation
├── DESIGN.md                       # Architecture documentation
│
├── src/deep_research_engine/
│   ├── __init__.py                 # Package exports
│   ├── version.py                  # Version string
│   │
│   ├── engine.py                   # Main entry point: run_workflow()
│   ├── config.py                   # Config loading and validation
│   │
│   ├── state/
│   │   ├── __init__.py
│   │   ├── types.py                # EngineState, WorkerResult, CriticResult
│   │   ├── factory.py              # create_initial_state()
│   │   └── reducers.py             # State reducers for LangGraph
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py              # build_workflow_graph()
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── boss.py             # BossNode (planning)
│   │   │   ├── workers.py          # WorkerNode (parallel execution)
│   │   │   ├── critic.py           # CriticNode (quality gate)
│   │   │   └── synthesis.py        # SynthesisNode (final output)
│   │   └── edges.py                # Conditional routing
│   │
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseWorker ABC + WorkerRegistry
│   │   ├── config.py               # WorkerConfig dataclass
│   │   ├── errors.py               # Worker exceptions
│   │   └── implementations/        # Built-in worker types
│   │       ├── __init__.py
│   │       ├── web_researcher.py   # Web research (Flash Explorer)
│   │       ├── chunk_processor.py  # PDF chunk processing
│   │       ├── multimodal.py       # ASR/VLM/face processing
│   │       └── scraper.py          # Web scraping (make-llmstxt)
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py                 # StorageBackend ABC
│   │   ├── registry.py             # StorageRegistry (pluggable)
│   │   └── backends/
│   │       ├── __init__.py
│   │       ├── memory.py           # InMemoryStorage (default)
│   │       ├── postgres.py         # PostgreSQL + pgvector
│   │       ├── mongodb.py          # MongoDB (media files)
│   │       ├── neo4j.py            # Neo4j (knowledge graph)
│   │       └── minio.py            # MinIO/S3 (large files)
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── config.py               # LLM config (model slots)
│   │   ├── factory.py              # create_llm() from config
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── litellm.py          # LiteLLM gateway
│   │       ├── openai.py           # Direct OpenAI
│   │       └── anthropic.py        # Direct Anthropic
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py             # ToolRegistry
│   │   └── builtins/
│   │       ├── __init__.py
│   │       ├── web.py              # web_search, web_get, etc.
│   │       ├── rag.py              # vector_search, add_documents
│   │       ├── multimodal.py       # asr_transcribe, vlm_analyze
│   │       └── osint.py            # osint_* tools
│   │
│   ├── critic/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseCritic ABC
│   │   └── builtins/
│   │       ├── __init__.py
│   │       ├── extraction.py       # Extraction quality
│   │       ├── coverage.py         # Coverage validation
│   │       └── multimodal.py       # Multimodal accuracy
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py              # Structured logging
│   │   ├── timing.py               # Performance timing
│   │   └── serialization.py        # JSON/dict serialization
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py                 # CLI entry point
│
├── configs/                        # Example workflow configs
│   ├── rag.yaml
│   ├── pdf_summarizer.yaml
│   ├── telegram_ingest.yaml
│   └── skill_generator.yaml
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/
│   │   ├── test_state.py
│   │   ├── test_worker_registry.py
│   │   ├── test_storage_registry.py
│   │   └── test_config_loader.py
│   ├── integration/
│   │   ├── test_boss_node.py
│   │   ├── test_worker_node.py
│   │   ├── test_critic_node.py
│   │   └── test_synthesis_node.py
│   └── workflows/
│       ├── test_rag_workflow.py
│       ├── test_pdf_workflow.py
│       └── test_telegram_workflow.py
│
└── docs/
    ├── getting_started.md
    ├── workflow_configs.md
    ├── custom_workers.md
    ├── storage_backends.md
    └── api_reference.md
```

---

## pyproject.toml

```toml
[project]
name = "deep-research-engine"
version = "0.1.0"
description = "Config-driven multi-agent workflow engine for AI ingestion tasks"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = [
    "llm",
    "agent",
    "workflow",
    "langgraph",
    "research",
    "ingestion",
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    # Core
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "pydantic>=2.0.0",
    
    # Config loading
    "pyyaml>=6.0",
    "jinja2>=3.1.0",
    
    # Storage backends (optional - users can install only what they need)
    "asyncpg>=0.29.0",              # PostgreSQL
    "pgvector>=0.2.0",              # Vector search
    "pymongo>=4.6.0",               # MongoDB
    "neo4j>=5.0.0",                 # Neo4j
    "boto3>=1.34.0",                # S3/MinIO
    
    # LLM providers
    "litellm>=1.0.0",               # LiteLLM gateway
    "openai>=1.0.0",                # OpenAI SDK
    "anthropic>=0.18.0",            # Anthropic SDK
    
    # Utilities
    "aiofiles>=23.0.0",             # Async file I/O
    "httpx>=0.27.0",                # HTTP client
    "rich>=13.0.0",                 # CLI output
    
    # Observability (optional)
    "langfuse>=2.0.0",              # Langfuse tracing
]

[project.optional-dependencies]
# Install only what you need
postgres = ["asyncpg", "pgvector"]
mongodb = ["pymongo"]
neo4j = ["neo4j"]
minio = ["boto3"]
all-storage = ["asyncpg", "pgvector", "pymongo", "neo4j", "boto3"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.0.0",
    "black>=24.0.0",
    "ruff>=0.4.0",
    "mypy>=1.0.0",
]

[project.scripts]
deep-research = "deep_research_engine.cli.main:main"

[project.urls]
Homepage = "https://github.com/your-org/deep-research-engine"
Documentation = "https://deep-research-engine.readthedocs.io"
Repository = "https://github.com/your-org/deep-research-engine"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/deep_research_engine"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Main Entry Point: engine.py

```python
"""Deep Research Engine - Main entry point.

Usage:
    from deep_research_engine import run_workflow, load_config
    
    # Load config from YAML
    config = load_config("configs/rag.yaml")
    
    # Run workflow
    async for event in run_workflow(
        query="What is quantum computing?",
        config=config,
        stream=True,
    ):
        print(event)
"""

from typing import AsyncIterator, Dict, Any, Optional, Union
from pathlib import Path
import asyncio

from .config import WorkflowConfig, load_config
from .state import EngineState, create_initial_state
from .graph import build_workflow_graph
from .storage import StorageManager
from .utils.logging import setup_logging, get_logger

__all__ = [
    "run_workflow",
    "load_config",
    "WorkflowConfig",
    "EngineState",
]


async def run_workflow(
    input_data: Union[str, Dict[str, Any]],
    config: WorkflowConfig,
    session_id: Optional[str] = None,
    stream: bool = True,
    storage_backends: Optional[list[str]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Run a workflow with the given configuration.
    
    Args:
        input_data: Query string or dict with workflow input
        config: Workflow configuration (from load_config)
        session_id: Optional session ID for tracing
        stream: If True, yield events as they happen
        storage_backends: List of backends to enable (default: from config)
        
    Yields:
        Events (if streaming) or final result (if not streaming)
        
    Event Types:
        - phase_start: {phase: str, timestamp: str}
        - phase_complete: {phase: str, duration: float}
        - worker_start: {worker_id: str, task: str}
        - worker_complete: {worker_id: str, success: bool, duration: float}
        - critic_evaluation: {passed: bool, score: float}
        - synthesis_complete: {output: str, metadata: dict}
        - error: {error: str, phase: str}
        - complete: {success: bool, output: Any, state: EngineState}
    """
    # Setup logging
    setup_logging(level=config.execution.get("logging", "info"))
    logger = get_logger("engine")
    
    # Generate session/run IDs
    import uuid
    run_id = str(uuid.uuid4())
    session_id = session_id or run_id
    
    logger.info(
        f"Starting workflow run: {config.workflow['name']}",
        run_id=run_id,
        session_id=session_id,
    )
    
    # Initialize storage backends
    storage = StorageManager()
    backends_to_enable = storage_backends or config.storage.keys()
    
    for backend_name in backends_to_enable:
        backend_config = config.storage.get(backend_name, {})
        if backend_config.get("enabled", False):
            try:
                await storage.initialize(backend_name, backend_config)
                logger.info(f"Initialized storage backend: {backend_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {backend_name}: {e}")
    
    # Create initial state
    state = create_initial_state(
        workflow_name=config.workflow["name"],
        workflow_type=config.workflow["type"],
        input_data={"query": input_data} if isinstance(input_data, str) else input_data,
        session_id=session_id,
        run_id=run_id,
        max_iterations=config.phases.get("boss", {}).get("config", {}).get("workers", {}).get("max", 3),
    )
    
    # Build workflow graph
    graph = build_workflow_graph(config)
    
    # Run graph
    try:
        if stream:
            async for event in run_graph_streaming(graph, state, config):
                yield event
        else:
            final_state = await run_graph_sync(graph, state, config)
            yield {
                "type": "complete",
                "success": True,
                "output": final_state["synthesis_output"],
                "state": final_state,
            }
            
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        yield {
            "type": "error",
            "error": str(e),
            "phase": "execution",
        }
        
    finally:
        # Cleanup storage
        await storage.cleanup()


async def run_graph_streaming(
    graph,
    state: EngineState,
    config: WorkflowConfig,
) -> AsyncIterator[Dict[str, Any]]:
    """Run graph with streaming events."""
    from langgraph.types import Command
    
    # Stream graph execution
    async for event in graph.astream(
        state,
        stream_mode="updates",
        config={
            "configurable": {
                "config": config,
            }
        },
    ):
        # Convert LangGraph events to our event format
        for node_name, updates in event.items():
            yield {
                "type": "node_update",
                "node": node_name,
                "updates": updates,
                "timestamp": asyncio.get_event_loop().time(),
            }


async def run_graph_sync(
    graph,
    state: EngineState,
    config: WorkflowConfig,
) -> EngineState:
    """Run graph synchronously (no streaming)."""
    result = await graph.ainvoke(
        state,
        config={
            "configurable": {
                "config": config,
            }
        },
    )
    return result


# Convenience function for CLI
async def run_from_cli(
    query: str,
    config_path: str,
    output_path: Optional[str] = None,
    stream: bool = True,
) -> None:
    """Run workflow from CLI.
    
    Args:
        query: User query or input
        config_path: Path to YAML config
        output_path: Optional path to write output
        stream: Stream events to stdout
    """
    config = load_config(config_path)
    
    output = None
    async for event in run_workflow(query, config, stream=stream):
        if stream:
            print_event(event)
        else:
            output = event
    
    if output_path and output:
        write_output(output, output_path)


def print_event(event: Dict[str, Any]) -> None:
    """Print event to stdout with formatting."""
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    event_type = event.get("type", "unknown")
    
    if event_type == "phase_start":
        console.print(Panel(f"Starting phase: {event['phase']}", style="bold blue"))
    elif event_type == "phase_complete":
        console.print(Panel(f"Phase complete: {event['phase']} ({event['duration']:.2f}s)", style="green"))
    elif event_type == "worker_complete":
        status = "✓" if event.get("success") else "✗"
        console.print(f"  {status} Worker {event['worker_id']} complete")
    elif event_type == "critic_evaluation":
        status = "✓ PASS" if event.get("passed") else "✗ FAIL"
        console.print(f"  Critic: {status} (score: {event['score']:.2f})")
    elif event_type == "synthesis_complete":
        console.print(Panel("Synthesis complete", style="bold magenta"))
    elif event_type == "error":
        console.print(f"  ERROR: {event['error']}", style="red")
    elif event_type == "complete":
        console.print(Panel("Workflow complete", style="bold green"))


def write_output(output: Dict[str, Any], path: str) -> None:
    """Write output to file."""
    from pathlib import Path
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".json":
        import json
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
    else:
        with open(output_path, "w") as f:
            f.write(output.get("output", str(output)))
