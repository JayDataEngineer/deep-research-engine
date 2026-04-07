# Examples

Complete examples of using the Deep Research Engine for various workflows.

## Table of Contents

- [RAG Research](#rag-research)
- [PDF Summarization](#pdf-summarization)
- [Skill Generation](#skill-generation)
- [Telegram Ingestion](#telegram-ingestion)
- [Custom Workers](#custom-workers)
- [FastAPI Integration](#fastapi-integration)
- [Batch Processing](#batch-processing)

## RAG Research

### Basic Research

```python
from deep_research_engine import run_workflow, load_config
import asyncio

async def basic_research():
    """Run basic RAG research."""
    # Load configuration
    config = load_config("configs/rag.yaml")
    
    # Run workflow
    async for event in run_workflow(
        "What are the latest developments in quantum computing?",
        config
    ):
        if event["type"] == "phase_start":
            print(f"Starting phase: {event['phase']}")
        elif event["type"] == "phase_complete":
            print(f"Phase complete: {event['phase']} ({event.get('duration', 0):.2f}s)")
        elif event["type"] == "complete":
            print(f"\nResearch Report:\n{event['output']}")

asyncio.run(basic_research())
```

### Research with Custom Storage

```python
from deep_research_engine import run_workflow, load_config
import asyncio

async def research_with_storage():
    """Run research with custom storage backends."""
    config = load_config("configs/rag.yaml")
    
    # Enable specific storage backends
    async for event in run_workflow(
        "Machine learning fundamentals",
        config,
        storage_backends=["postgres", "neo4j", "memory"]
    ):
        if event["type"] == "complete":
            # Save to file
            with open("output/research_report.md", "w") as f:
                f.write(event["output"])
            print("Report saved to output/research_report.md")

asyncio.run(research_with_storage())
```

### Research with Session Tracking

```python
from deep_research_engine import run_workflow, load_config
import asyncio
import uuid

async def tracked_research():
    """Run research with session tracking."""
    config = load_config("configs/rag.yaml")
    
    # Generate session ID for tracing
    session_id = str(uuid.uuid4())
    
    async for event in run_workflow(
        "AI ethics and governance",
        config,
        session_id=session_id,
        stream=True
    ):
        if event["type"] == "complete":
            print(f"Session {session_id} complete")
            print(f"Output length: {len(event['output'])} characters")

asyncio.run(tracked_research())
```

## PDF Summarization

### Single PDF

```python
from deep_research_engine import run_workflow, load_config
import asyncio

async def summarize_pdf():
    """Summarize a single PDF."""
    config = load_config("configs/pdf_summarizer.yaml")
    
    async for event in run_workflow(
        {"pdf_path": "/path/to/document.pdf"},
        config
    ):
        if event["type"] == "complete":
            print(f"Summary:\n{event['output']}")
            
            # Save summary
            with open("output/summary.md", "w") as f:
                f.write(event["output"])

asyncio.run(summarize_pdf())
```

### Multiple PDFs

```python
from deep_research_engine import run_workflow, load_config
import asyncio
from pathlib import Path

async def summarize_multiple_pdfs():
    """Summarize multiple PDFs in parallel."""
    config = load_config("configs/pdf_summarizer.yaml")
    
    pdf_files = list(Path("/path/to/pdfs").glob("*.pdf"))
    
    async def process_pdf(pdf_path):
        async for event in run_workflow(
            {"pdf_path": str(pdf_path)},
            config,
            session_id=f"pdf-{pdf_path.stem}"
        ):
            if event["type"] == "complete":
                # Save individual summary
                output_path = Path("output") / f"{pdf_path.stem}_summary.md"
                output_path.write_text(event["output'])
                print(f"Summarized: {pdf_path.name}")
                return event["output"]
    
    # Process in parallel
    results = await asyncio.gather(*[process_pdf(pdf) for pdf in pdf_files])
    
    # Combine summaries
    combined = "\n\n---\n\n".join(results)
    Path("output/combined_summary.md").write_text(combined)
    print(f"Combined {len(results)} summaries")

asyncio.run(summarize_multiple_pdfs())
```

## Skill Generation

### Documentation Scraping

```python
from deep_research_engine import run_workflow, load_config
import asyncio

async def generate_skill():
    """Generate skill from documentation."""
    config = load_config("configs/skill_generator.yaml")
    
    async for event in run_workflow(
        {"url": "https://docs.python.org/3/tutorial/"},
        config
    ):
        if event["type"] == "complete":
            skill_content = event["output"]
            
            # Save skill
            with open("skills/python_skill.md", "w") as f:
                f.write(skill_content)
            
            print("Skill generated successfully!")
            print(f"Content length: {len(skill_content)}")

asyncio.run(generate_skill())
```

### Multiple URLs

```python
from deep_research_engine import run_workflow, load_config
import asyncio

async def generate_skill_from_multiple_urls():
    """Generate skill from multiple documentation URLs."""
    config_dict = {
        "workflow": {"name": "skill_gen", "type": "skill_generation"},
        "input": {"type": "urls", "source": []},
        "phases": [
            {"name": "boss", "node": "BossNode"},
            {"name": "workers", "node": "WorkerNode", "config": {"worker_type": "scraper"}},
            {"name": "critic", "node": "CriticNode"},
            {"name": "synthesis", "node": "SynthesisNode"}
        ],
        "storage": {"memory": {"enabled": True}},
        "execution": {"max_steps": 100}
    }
    
    from deep_research_engine import WorkflowConfig
    config = WorkflowConfig(**config_dict)
    
    urls = [
        "https://docs.python.org/3/tutorial/",
        "https://realpython.com/tutorials/",
        "https://pythonspeed.com/articles/"
    ]
    
    async for event in run_workflow(
        {"urls": urls},
        config
    ):
        if event["type"] == "complete":
            print(f"Generated skill from {len(urls)} URLs")
            print(event["output"][:500])  # Preview

asyncio.run(generate_skill_from_multiple_urls())
```

## Telegram Ingestion

### Process Export

```python
from deep_research_engine import run_workflow, load_config
import asyncio

async def ingest_telegram_export():
    """Process Telegram export."""
    config = load_config("configs/telegram_ingest.yaml")
    
    async for event in run_workflow(
        {"export_path": "/path/to/telegram_export"},
        config
    ):
        if event["type"] == "complete":
            summary = event["output"]
            
            # Save summary
            with open("output/telegram_summary.md", "w") as f:
                f.write(summary)
            
            print("Telegram export processed!")

asyncio.run(ingest_telegram_export())
```

## Custom Workers

### Database Query Worker

```python
from deep_research_engine.workers import BaseWorker, WorkerConfig, WorkerRegistry
from deep_research_engine.state import EngineState, WorkerResult
import asyncpg

@WorkerRegistry.register("database_query")
class DatabaseQueryWorker(BaseWorker):
    """Worker for database queries."""
    
    worker_type = "database_query"
    description = "Executes SQL queries"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        self._start_timing()
        
        try:
            db_url = config.custom.get("connection_url")
            conn = await asyncpg.connect(db_url)
            
            results = await conn.fetch(config.task)
            await conn.close()
            
            return self._create_result(
                success=True,
                output={"rows": [dict(r) for r in results]},
                output_type="database_results",
                confidence=0.95
            )
        except Exception as e:
            return self._create_result(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )

# Usage
async def query_database():
    """Use custom database worker."""
    config_dict = {
        "workflow": {"name": "db_query", "type": "research"},
        "input": {"type": "query", "source": "SELECT * FROM users"},
        "phases": [
            {"name": "boss", "node": "BossNode"},
            {"name": "workers", "node": "WorkerNode", "config": {"worker_type": "database_query"}},
            {"name": "critic", "node": "CriticNode"},
            {"name": "synthesis", "node": "SynthesisNode"}
        ],
        "storage": {"memory": {"enabled": True}},
        "execution": {"max_steps": 50}
    }
    
    from deep_research_engine import WorkflowConfig, run_workflow
    config = WorkflowConfig(**config_dict)
    
    async for event in run_workflow(
        "SELECT * FROM users LIMIT 10",
        config
    ):
        if event["type"] == "complete":
            print(event["output"])

asyncio.run(query_database())
```

### API Integration Worker

```python
from deep_research_engine.workers import BaseWorker, WorkerConfig, WorkerRegistry
from deep_research_engine.state import EngineState, WorkerResult
import httpx

@WorkerRegistry.register("api_call")
class APICallWorker(BaseWorker):
    """Worker for API calls."""
    
    worker_type = "api_call"
    description = "Makes HTTP requests to APIs"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        self._start_timing()
        
        try:
            url = config.custom.get("url")
            method = config.custom.get("method", "GET")
            headers = config.custom.get("headers", {})
            
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, headers=headers)
                response.raise_for_status()
                
                return self._create_result(
                    success=True,
                    output=response.json(),
                    output_type="api_response",
                    confidence=0.9
                )
        except Exception as e:
            return self._create_result(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )

# Usage
async def call_api():
    """Use custom API worker."""
    config_dict = {
        "workflow": {"name": "api_call", "type": "research"},
        "input": {"type": "query", "source": "Fetch data"},
        "phases": [
            {"name": "boss", "node": "BossNode"},
            {"name": "workers", "node": "WorkerNode", "config": {"worker_type": "api_call"}},
            {"name": "critic", "node": "CriticNode"},
            {"name": "synthesis", "node": "SynthesisNode"}
        ],
        "storage": {"memory": {"enabled": True}}
    }
    
    from deep_research_engine import WorkflowConfig, run_workflow
    config = WorkflowConfig(**config_dict)
    
    async for event in run_workflow(
        "GET https://api.example.com/data",
        config
    ):
        if event["type"] == "complete":
            print(event["output"])

asyncio.run(call_api())
```

## FastAPI Integration

### REST API Wrapper

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import AsyncIterator
import asyncio
from deep_research_engine import run_workflow, load_config

app = FastAPI(title="Deep Research API")

class ResearchRequest(BaseModel):
    query: str
    config_path: str = "configs/rag.yaml"

class ResearchResponse(BaseModel):
    success: bool
    output: str
    session_id: str

@app.post("/research")
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run research workflow."""
    try:
        config = load_config(request.config_path)
        
        result = None
        async for event in run_workflow(
            request.query,
            config,
            stream=False
        ):
            result = event
        
        return ResearchResponse(
            success=result["success"],
            output=result["output"],
            session_id=result["state"].get("session_id", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/stream")
async def research_stream(request: ResearchRequest) -> AsyncIterator[str]:
    """Stream research results."""
    config = load_config(request.config_path)
    
    async for event in run_workflow(request.query, config, stream=True):
        yield f"data: {event}\n\n"

# Run with: uvicorn main:app --reload
```

### WebSocket Support

```python
from fastapi import FastAPI, WebSocket
from deep_research_engine import run_workflow, load_config
import json

app = FastAPI()

@app.websocket("/ws/research")
async def websocket_research(websocket: WebSocket):
    """WebSocket endpoint for research."""
    await websocket.accept()
    
    while True:
        # Receive query
        data = await websocket.receive_text()
        request = json.loads(data)
        
        config = load_config(request.get("config", "configs/rag.yaml"))
        
        # Stream results
        async for event in run_workflow(
            request["query"],
            config,
            stream=True
        ):
            await websocket.send_json(event)
            
            if event["type"] == "complete":
                break
```

## Batch Processing

### Process Multiple Queries

```python
from deep_research_engine import run_workflow, load_config
import asyncio
from pathlib import Path

async def batch_research():
    """Process multiple research queries."""
    config = load_config("configs/rag.yaml")
    
    # Read queries from file
    queries = Path("queries.txt").read_text().splitlines()
    
    async def process_query(query: str, index: int):
        """Process single query."""
        async for event in run_workflow(
            query,
            config,
            session_id=f"query-{index}",
            stream=False
        ):
            if event["type"] == "complete":
                # Save result
                output_path = Path("output") / f"result_{index}.md"
                output_path.write_text(event["output"])
                print(f"Completed query {index}: {query[:50]}...")
                return event["output"]
    
    # Process in batches of 5
    batch_size = 5
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        results = await asyncio.gather(*[
            process_query(query, i + j) for j, query in enumerate(batch)
        ])
        print(f"Batch {i // batch_size + 1} complete")

asyncio.run(batch_research())
```

### Scheduled Research

```python
from deep_research_engine import run_workflow, load_config
import asyncio
from datetime import datetime

async def scheduled_research():
    """Run research on a schedule."""
    config = load_config("configs/rag.yaml")
    
    queries = [
        "Latest AI research",
        "Machine learning breakthroughs",
        "Deep learning applications"
    ]
    
    while True:
        for query in queries:
            print(f"Running scheduled research: {query}")
            
            async for event in run_workflow(
                query,
                config,
                stream=False
            ):
                if event["type"] == "complete":
                    # Save with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(f"output/scheduled_{timestamp}.md")
                    output_path.write_text(event["output"])
            
            await asyncio.sleep(60)  # Wait between queries
        
        # Wait until next scheduled run (e.g., 1 hour)
        await asyncio.sleep(3600)

# asyncio.run(scheduled_research())
```

## Error Handling

### Robust Error Handling

```python
from deep_research_engine import run_workflow, load_config
from deep_research_engine.workers import WorkerError
from deep_research_engine.storage import StorageError
import asyncio

async def robust_research():
    """Research with robust error handling."""
    config = load_config("configs/rag.yaml")
    
    try:
        async for event in run_workflow(
            "Research query",
            config,
            stream=True
        ):
            if event["type"] == "error":
                print(f"Error in phase {event.get('phase')}: {event['error']}")
                
                # Implement retry logic
                if event.get("phase") == "workers":
                    print("Retrying workers...")
                    continue
            
            elif event["type"] == "complete":
                print(f"Success: {event['output'][:200]}...")
                
    except WorkerError as e:
        print(f"Worker error: {e}")
        # Handle worker-specific error
        
    except StorageError as e:
        print(f"Storage error: {e}")
        # Handle storage error
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Handle unexpected error

asyncio.run(robust_research())
```

## Performance Optimization

### Parallel Workflow Execution

```python
from deep_research_engine import run_workflow, load_config
import asyncio

async def parallel_research():
    """Run multiple workflows in parallel."""
    config = load_config("configs/rag.yaml")
    
    queries = [
        "Quantum computing basics",
        "Machine learning fundamentals",
        "Neural network architectures",
        "Deep learning applications",
        "AI ethics and governance"
    ]
    
    async def run_single_query(query: str):
        async for event in run_workflow(query, config, stream=False):
            if event["type"] == "complete":
                return event["output"]
    
    # Run all queries in parallel
    results = await asyncio.gather(*[run_single_query(q) for q in queries])
    
    # Combine results
    combined = "\n\n---\n\n".join(results)
    Path("output/combined_research.md").write_text(combined)
    print(f"Completed {len(queries)} research queries in parallel")

asyncio.run(parallel_research())
```

## Reference

- [Workflow Configs](../workflow-configs/)
- [Workers Guide](../workers/)
- [Storage Backends](../storage/)
- [API Reference](../api-reference/)
