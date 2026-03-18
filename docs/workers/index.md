# Workers Reference

Workers are pluggable processing units that execute specific tasks.

## Base Interface

All workers implement:

```python
from deep_research_engine.workers import BaseWorker, WorkerRegistry

@WorkerRegistry.register("worker_type")
class MyWorker(BaseWorker):
    worker_type = "worker_type"
    description = "Description"
    required_tools = ["tool1", "tool2"]
    
    async def execute(self, state, config) -> WorkerResult:
        self._start_timing()
        
        # Your implementation
        output = await self.process(state["input_data"])
        
        return self._create_result(
            success=True,
            output=output,
            output_type="result_type",
            confidence=0.85,
            sources=sources,
        )
```

## Built-in Workers

### WebResearcherWorker

**Purpose**: Autonomous web research using Flash Explorer pattern

**Tools**: `web_search`, `web_get`, `web_research_agent`

**Output**: Research findings with citations

```python
@WorkerRegistry.register("web_researcher")
class WebResearcherWorker(BaseWorker):
    async def execute(self, state, config) -> WorkerResult:
        # Spawns web_deep_agent for parallel research
        findings = await run_flash_explorer(
            query=config.task,
            yaml_config=state.get("config", {}),
        )
        return self._create_result(
            success=True,
            output=findings,
            sources=extract_sources(findings),
        )
```

---

### ChunkProcessorWorker

**Purpose**: Process text chunks (classify, extract, filter)

**Tools**: `llm_classify`, `llm_extract`

**Output**: Classified chunks with core content

```python
@WorkerRegistry.register("chunk_processor")
class ChunkProcessorWorker(BaseWorker):
    async def execute(self, state, config) -> WorkerResult:
        chunk = config.custom.get("chunk")
        
        # Classify if filler
        is_filler = await self.classify_filler(chunk)
        if is_filler:
            return self._create_result(
                success=True,
                output={"is_filler": True},
            )
        
        # Extract core content
        core = await self.extract_core(chunk)
        return self._create_result(
            success=True,
            output={"core_content": core},
        )
```

---

### MultimodalWorker

**Purpose**: Process audio, images, video

**Tools**: `asr_transcribe`, `vlm_analyze`, `face_recognize`

**Output**: Transcriptions, descriptions, entity extractions

```python
@WorkerRegistry.register("multimodal")
class MultimodalWorker(BaseWorker):
    async def execute(self, state, config) -> WorkerResult:
        media_item = config.custom.get("media_item")
        
        if media_item["type"] == "audio":
            result = await self.transcribe(media_item)
        elif media_item["type"] == "image":
            result = await self.analyze_image(media_item)
        elif media_item["type"] == "video":
            result = await self.analyze_video(media_item)
        
        return self._create_result(
            success=True,
            output=result,
        )
```

---

### ScraperWorker

**Purpose**: Scrape documentation websites

**Tools**: `mcp_scrape`, `search_docs`, `get_doc_by_url`

**Output**: Scraped markdown content

```python
@WorkerRegistry.register("scraper")
class ScraperWorker(BaseWorker):
    async def execute(self, state, config) -> WorkerResult:
        url = config.custom.get("url")
        
        content = await scrape_with_mcp(
            url=url,
            max_pages=config.custom.get("max_pages", 50),
        )
        
        return self._create_result(
            success=True,
            output=content,
            sources=[{"url": url}],
        )
```

---

## Custom Workers

Create a custom worker:

```python
# src/my_package/workers/custom.py
from deep_research_engine.workers import BaseWorker, WorkerRegistry

@WorkerRegistry.register("my_custom_worker")
class MyCustomWorker(BaseWorker):
    worker_type = "my_custom_worker"
    description = "My custom worker"
    
    async def execute(self, state, config) -> WorkerResult:
        # Implementation
        pass
```

Register in your package `__init__.py`:

```python
from .workers.custom import MyCustomWorker  # noqa: F401
```

---

## WorkerResult Schema

```python
class WorkerResult(TypedDict, total=False):
    worker_id: str
    worker_type: str
    task: str
    success: bool
    output: Any
    output_type: str
    confidence: float
    quality_score: Optional[float]
    timing: Dict[str, float]  # start, end, duration
    token_usage: Dict[str, int]
    model_info: Dict[str, str]
    sources: List[Dict[str, Any]]
    source_count: int
    error: Optional[str]
    error_type: Optional[str]
    retry_count: int
```

---

## Testing Workers

```python
import pytest
from deep_research_engine.workers import WorkerRegistry

def test_worker_registration():
    klass = WorkerRegistry.get("web_researcher")
    assert klass.worker_type == "web_researcher"

@pytest.mark.asyncio
async def test_worker_execution():
    from deep_research_engine.state import create_initial_state
    
    config = WorkerConfig(
        worker_id="test_0",
        worker_type="web_researcher",
        task="Test task",
    )
    
    state = create_initial_state(
        workflow_name="test",
        workflow_type="research",
        input_data={"query": "test"},
        session_id="test",
        run_id="test",
    )
    
    worker = WorkerRegistry.get("web_researcher")(config)
    result = await worker.execute(state, config)
    
    assert result["success"] is True
    assert "output" in result
```
