# Examples

Example workflows and integrations.

## RAG Research

```bash
uv run deep-research research "What is quantum computing?" --config rag.yaml
```

**Python API**:
```python
from deep_research_engine import run_workflow, load_config

config = load_config("configs/rag.yaml")

async for event in run_workflow(
    query="What is quantum computing?",
    config=config,
    stream=True,
):
    if event["type"] == "phase_complete":
        print(f"✓ Phase {event['phase']} complete")
    elif event["type"] == "synthesis_complete":
        print(f"\nResult:\n{event['output']}")
```

---

## PDF Summarizer

```bash
uv run deep-research pdf ./book.pdf --config pdf_summarizer.yaml
```

**Python API**:
```python
from deep_research_engine import run_workflow, load_config

config = load_config("configs/pdf_summarizer.yaml")

result = await run_workflow(
    input_data={"pdf_path": "./book.pdf"},
    config=config,
    stream=False,
)

print(result["output"])
```

---

## Telegram Ingestion

```bash
uv run deep-research ingest ./ChatExport_2026-03-13 --config telegram_ingest.yaml
```

**Python API**:
```python
from deep_research_engine import run_workflow, load_config

config = load_config("configs/telegram_ingest.yaml")

async for event in run_workflow(
    input_data={"export_path": "./ChatExport_2026-03-13"},
    config=config,
    stream=True,
):
    print(event)
```

---

## Skill Generator

```bash
uv run deep-research scrape https://docs.example.com --config skill_generator.yaml
```

**Python API**:
```python
from deep_research_engine import run_workflow, load_config

config = load_config("configs/skill_generator.yaml")

result = await run_workflow(
    input_data={"url": "https://docs.example.com"},
    config=config,
    stream=False,
)

# Save output
with open("output/llms.txt", "w") as f:
    f.write(result["output"])
```

---

## FastAPI Integration

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from deep_research_engine import run_workflow, load_config

app = FastAPI()

@app.post("/api/research")
async def research(query: str):
    config = load_config("configs/rag.yaml")
    
    async def generate():
        async for event in run_workflow(query, config, stream=True):
            yield f"data: {event}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/pdf")
async def summarize_pdf(pdf_path: str):
    config = load_config("configs/pdf_summarizer.yaml")
    result = await run_workflow(
        {"pdf_path": pdf_path},
        config,
        stream=False,
    )
    return {"output": result["output"]}
```

**Run**:
```bash
uv run uvicorn app:app --reload
curl -X POST "http://localhost:8000/api/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?"}'
```

---

## Custom Worker Example

```python
# custom_worker.py
from deep_research_engine.workers import BaseWorker, WorkerRegistry

@WorkerRegistry.register("custom_analyzer")
class CustomAnalyzerWorker(BaseWorker):
    worker_type = "custom_analyzer"
    description = "Custom analysis worker"
    
    async def execute(self, state, config) -> WorkerResult:
        self._start_timing()
        
        # Custom analysis logic
        analysis = await self.analyze(state["input_data"])
        
        return self._create_result(
            success=True,
            output=analysis,
            output_type="analysis",
            confidence=0.9,
        )

# Use in config
```yaml
phases:
  - name: workers
    config:
      worker_type: custom_analyzer
```

---

## Custom Storage Backend

```python
# custom_storage.py
from deep_research_engine.storage import StorageBackend, StorageResult, StorageRegistry

@StorageRegistry.register("custom_db")
class CustomStorage(StorageBackend):
    backend_name = "custom_db"
    
    async def initialize(self):
        # Custom initialization
        self.initialized = True
    
    async def store(self, data, namespace="default"):
        # Custom storage logic
        resource_id = generate_id()
        await self.save(resource_id, data)
        return StorageResult(success=True, id=resource_id)
    
    async def retrieve(self, id, namespace="default"):
        # Custom retrieval logic
        return await self.load(id)
```

---

## Batch Processing

```python
import asyncio
from deep_research_engine import run_workflow, load_config

async def process_pdfs(pdf_paths: list):
    config = load_config("configs/pdf_summarizer.yaml")
    
    async def process_one(path):
        result = await run_workflow(
            {"pdf_path": path},
            config,
            stream=False,
        )
        return result["output"]
    
    # Process in parallel
    tasks = [process_one(path) for path in pdf_paths]
    results = await asyncio.gather(*tasks)
    
    return results

# Usage
outputs = await process_pdfs(["book1.pdf", "book2.pdf", "book3.pdf"])
```

---

## Streaming to File

```python
from deep_research_engine import run_workflow, load_config

async def stream_to_file(query, output_path):
    config = load_config("configs/rag.yaml")
    
    with open(output_path, "w") as f:
        async for event in run_workflow(query, config, stream=True):
            if event["type"] == "synthesis_complete":
                f.write(event["output"])
            elif event["type"] == "error":
                f.write(f"ERROR: {event['error']}\n")

# Usage
await stream_to_file("What is quantum computing?", "output/research.md")
```
