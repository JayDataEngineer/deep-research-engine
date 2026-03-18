# BaseWorker ABC Design

## Problem

The Flash Explorer pattern is brilliant but tightly coupled to web research. To support PDF processing, multimodal ingestion, and scraping, we need a **generic worker interface** that all worker types implement.

## Solution: BaseWorker Abstract Base Class

```python
"""BaseWorker - Abstract Base Class for all worker types.

Design Principles:
1. Uniform interface: All workers implement the same methods
2. Standardized output: All workers return WorkerResult
3. Pluggable: Add new worker types without changing core engine
4. Critic integration: Built-in quality validation loop
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
import time

from .state import EngineState, WorkerResult, CriticResult


# =============================================================================
# WORKER CONFIGURATION
# =============================================================================

@dataclass
class WorkerConfig:
    """Configuration for a worker instance.
    
    Passed from YAML config to worker at runtime.
    """
    worker_id: str
    worker_type: str
    task: str
    
    # Tools
    tools: List[str] = field(default_factory=list)
    tool_config: Dict[str, Any] = field(default_factory=dict)
    
    # Execution
    max_retries: int = 3
    timeout: int = 300
    recursion_limit: int = 50
    
    # Critic
    enable_critic: bool = True
    critic_threshold: float = 0.7
    
    # Worker-specific (flexible dict for custom config)
    custom: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE WORKER ABC
# =============================================================================

class BaseWorker(ABC):
    """Abstract base class for all worker types.
    
    All workers (web_researcher, chunk_processor, multimodal, etc.)
    must implement these methods.
    
    Usage:
        class WebResearcherWorker(BaseWorker):
            async def execute(self, state, config) -> WorkerResult:
                # Implementation
                pass
    """
    
    # Class-level metadata (override in subclasses)
    worker_type: str = "base"
    description: str = "Base worker"
    required_tools: List[str] = []
    
    def __init__(self, config: WorkerConfig):
        """Initialize worker with configuration.
        
        Args:
            config: Worker configuration from YAML
        """
        self.config = config
        self.worker_id = config.worker_id
        self.task = config.task
        
        # Observability
        self.start_time: Optional[float] = None
        self.token_usage: Dict[str, int] = {}
        
    # -------------------------------------------------------------------------
    # ABSTRACT METHOD: Execute Worker
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute the worker's task.
        
        This is the MAIN method that all workers must implement.
        
        Args:
            state: Current engine state
            config: Worker configuration
            
        Returns:
            WorkerResult with standardized output
            
        Raises:
            WorkerError: If worker fails (will be caught and logged)
        """
        pass
    
    # -------------------------------------------------------------------------
    # HOOK METHODS (Override as needed)
    # -------------------------------------------------------------------------
    
    async def setup(self, state: EngineState, config: WorkerConfig) -> None:
        """Setup hook called before execute().
        
        Override to:
        - Initialize connections
        - Load models
        - Validate prerequisites
        
        Raises:
            WorkerSetupError: If setup fails
        """
        pass
    
    async def teardown(self, result: WorkerResult) -> None:
        """Teardown hook called after execute() (even on failure).
        
        Override to:
        - Close connections
        - Clean up resources
        - Log final metrics
        """
        pass
    
    async def validate_input(self, state: EngineState, config: WorkerConfig) -> bool:
        """Validate that input is suitable for this worker.
        
        Override to check:
        - Required fields in state
        - Tool availability
        - Resource constraints
        
        Returns:
            True if valid, False if worker should skip
        """
        return True
    
    # -------------------------------------------------------------------------
    # CRITIC INTEGRATION (Optional)
    # -------------------------------------------------------------------------
    
    async def self_evaluate(self, result: WorkerResult) -> Dict[str, float]:
        """Self-evaluation hook for workers that can assess their own output.
        
        Override to provide:
        - Confidence score
        - Quality metrics
        
        Returns:
            Dict of metric_name → score (0-1)
        """
        return {}
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS (Available to all workers)
    # -------------------------------------------------------------------------
    
    def _start_timing(self) -> None:
        """Start timing execution."""
        self.start_time = time.time()
    
    def _end_timing(self) -> float:
        """End timing and return duration."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def _create_result(
        self,
        success: bool,
        output: Any,
        output_type: str = "generic",
        confidence: float = 0.8,
        sources: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> WorkerResult:
        """Create a standardized WorkerResult.
        
        Use this helper to ensure all workers return consistent results.
        
        Args:
            success: Did worker complete successfully?
            output: Worker-specific output
            output_type: Schema hint (e.g., "findings", "chunks", "transcription")
            confidence: 0-1 confidence score
            sources: Citations, URLs, document refs
            error: Error message if failed
            
        Returns:
            WorkerResult with all fields populated
        """
        duration = self._end_timing()
        
        return WorkerResult(
            worker_id=self.worker_id,
            worker_type=self.config.worker_type,
            task=self.task,
            success=success,
            output=output,
            output_type=output_type,
            confidence=confidence,
            quality_score=None,  # Set by critic or self-evaluation
            timing={
                "start": self.start_time or 0.0,
                "end": time.time(),
                "duration": duration,
            },
            token_usage=self.token_usage,
            model_info={},  # Set by LLM wrapper
            sources=sources or [],
            source_count=len(sources) if sources else 0,
            error=error,
            error_type=None,
            retry_count=0,
        )
    
    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message with worker context.
        
        Args:
            message: Log message
            level: Log level (debug, info, warning, error)
            **kwargs: Additional context to log
        """
        # Lazy import to avoid circular dependency
        from .logging import get_logger
        logger = get_logger(f"worker.{self.worker_type}")
        
        log_method = getattr(logger, level, logger.info)
        log_method(
            f"[{self.worker_id}] {message}",
            worker_type=self.worker_type,
            task=self.task[:50],
            **kwargs
        )


# =============================================================================
# WORKER REGISTRY (Pluggable Worker Types)
# =============================================================================

class WorkerRegistry:
    """Registry for worker types.
    
    Workers register themselves, and the engine looks them up by name.
    This enables adding new worker types without changing core engine.
    
    Usage:
        # In your worker module
        from deep_research_engine import WorkerRegistry, BaseWorker
        
        @WorkerRegistry.register("web_researcher")
        class WebResearcherWorker(BaseWorker):
            worker_type = "web_researcher"
            description = "Web research specialist"
            
            async def execute(self, state, config) -> WorkerResult:
                # Implementation
                pass
        
        # In engine
        worker_class = WorkerRegistry.get("web_researcher")
        worker = worker_class(config)
    """
    
    _workers: Dict[str, Type[BaseWorker]] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a worker type.
        
        Usage:
            @WorkerRegistry.register("web_researcher")
            class WebResearcherWorker(BaseWorker):
                ...
        """
        def decorator(worker_class: Type[BaseWorker]) -> Type[BaseWorker]:
            cls._workers[name] = worker_class
            # Store name on class for reference
            worker_class.worker_type = name
            return worker_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseWorker]:
        """Get a worker class by name.
        
        Args:
            name: Worker type name (from YAML config)
            
        Returns:
            Worker class (not instance)
            
        Raises:
            WorkerNotFoundError: If worker type not registered
        """
        if name not in cls._workers:
            available = ", ".join(cls._workers.keys())
            raise WorkerNotFoundError(
                f"Worker type '{name}' not found. Available: {available}"
            )
        return cls._workers[name]
    
    @classmethod
    def list_workers(cls) -> List[str]:
        """List all registered worker types."""
        return list(cls._workers.keys())
    
    @classmethod
    def get_worker_info(cls, name: str) -> Dict[str, Any]:
        """Get metadata about a worker type.
        
        Returns:
            Dict with description, required_tools, etc.
        """
        worker_class = cls.get(name)
        return {
            "name": name,
            "description": worker_class.description,
            "required_tools": worker_class.required_tools,
        }


# =============================================================================
# WORKER ERRORS
# =============================================================================

class WorkerError(Exception):
    """Base exception for worker errors."""
    pass


class WorkerSetupError(WorkerError):
    """Raised when worker setup fails."""
    pass


class WorkerExecutionError(WorkerError):
    """Raised when worker execution fails."""
    pass


class WorkerNotFoundError(WorkerError):
    """Raised when requested worker type is not registered."""
    pass


# =============================================================================
# EXAMPLE WORKER IMPLEMENTATIONS
# =============================================================================

# Example 1: Web Researcher Worker
@WorkerRegistry.register("web_researcher")
class WebResearcherWorker(BaseWorker):
    """Web research worker using Flash Explorer pattern."""
    
    worker_type = "web_researcher"
    description = "Autonomous web research agent"
    required_tools = ["web_search", "web_get"]
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        self._start_timing()
        self._log(f"Starting web research: {config.task[:100]}")
        
        try:
            # Import Flash Explorer (lazy import to avoid circular dependency)
            from .agents.flash_explorer import run_flash_explorer
            
            # Run Flash Explorer for this research angle
            findings = await run_flash_explorer(
                query=config.task,
                yaml_config=state.get("config", {}),
                session_id=state["session_id"],
            )
            
            # Extract sources from findings
            sources = self._extract_sources(findings)
            
            self._log(f"Web research complete: {len(sources)} sources")
            
            return self._create_result(
                success=True,
                output=findings,
                output_type="research_findings",
                confidence=0.8,
                sources=sources,
            )
            
        except Exception as e:
            self._log(f"Web research failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
                confidence=0.0,
            )
    
    def _extract_sources(self, findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from findings."""
        # Implementation depends on findings structure
        return []


# Example 2: Chunk Processor Worker (for PDF)
@WorkerRegistry.register("chunk_processor")
class ChunkProcessorWorker(BaseWorker):
    """Process a chunk of text (classify, extract, validate)."""
    
    worker_type = "chunk_processor"
    description = "Process text chunks for classification and extraction"
    required_tools = ["llm_classify", "llm_extract"]
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        self._start_timing()
        
        # Get chunk from context
        chunk = config.custom.get("chunk")
        if not chunk:
            return self._create_result(
                success=False,
                output=None,
                error="No chunk provided to chunk_processor",
            )
        
        self._log(f"Processing chunk: {len(chunk.get('content', ''))} chars")
        
        try:
            # Classify if filler
            is_filler = await self._classify_filler(chunk)
            
            if is_filler:
                self._log("Chunk classified as filler, skipping")
                return self._create_result(
                    success=True,
                    output={"is_filler": True, "core_content": None},
                    output_type="chunk_classification",
                    confidence=0.9,
                )
            
            # Extract core content
            core_content = await self._extract_core(chunk)
            
            self._log(f"Extracted core content: {len(core_content)} chars")
            
            return self._create_result(
                success=True,
                output={
                    "is_filler": False,
                    "core_content": core_content,
                    "original_chunk": chunk,
                },
                output_type="chunk_extraction",
                confidence=0.85,
            )
            
        except Exception as e:
            self._log(f"Chunk processing failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )
    
    async def _classify_filler(self, chunk: Dict[str, Any]) -> bool:
        """Classify if chunk is filler content."""
        # Implementation using LLM
        pass
    
    async def _extract_core(self, chunk: Dict[str, Any]) -> str:
        """Extract core content from chunk."""
        # Implementation using LLM
        pass


# Example 3: Multimodal Worker (for Telegram)
@WorkerRegistry.register("multimodal")
class MultimodalWorker(BaseWorker):
    """Process multimodal content (ASR, VLM, face recognition)."""
    
    worker_type = "multimodal"
    description = "Process audio, images, and video"
    required_tools = ["asr_transcribe", "vlm_analyze"]
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        self._start_timing()
        
        # Get media item from context
        media_item = config.custom.get("media_item")
        media_type = media_item.get("type") if media_item else None
        
        if not media_item:
            return self._create_result(
                success=False,
                output=None,
                error="No media item provided",
            )
        
        self._log(f"Processing {media_type}: {media_item.get('path')}")
        
        results = {}
        
        try:
            # Route based on media type
            if media_type == "audio":
                results["transcription"] = await self._transcribe_audio(media_item)
            elif media_type == "image":
                results["description"] = await self._analyze_image(media_item)
                results["faces"] = await self._recognize_faces(media_item)
            elif media_type == "video":
                results["frames"] = await self._extract_frames(media_item)
                results["description"] = await self._analyze_video(media_item)
            elif media_type == "text":
                results["entities"] = await self._extract_entities(media_item)
            
            self._log(f"Multimodal processing complete: {len(results)} results")
            
            return self._create_result(
                success=True,
                output=results,
                output_type="multimodal_processing",
                confidence=0.85,
            )
            
        except Exception as e:
            self._log(f"Multimodal processing failed: {e}", level="error")
            return self._create_result(
                success=False,
                output=None,
                error=str(e),
            )
    
    async def _transcribe_audio(self, media_item: Dict[str, Any]) -> str:
        """Transcribe audio using ASR tool."""
        # Implementation
        pass
    
    async def _analyze_image(self, media_item: Dict[str, Any]) -> str:
        """Analyze image using VLM tool."""
        # Implementation
        pass
    
    async def _recognize_faces(self, media_item: Dict[str, Any]) -> List[str]:
        """Recognize faces using CompreFace."""
        # Implementation
        pass
    
    async def _extract_entities(self, media_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        # Implementation
        pass


# =============================================================================
# WORKER EXECUTOR (Engine Side)
# =============================================================================

class WorkerExecutor:
    """Execute workers in parallel with critic loop.
    
    This is called by the engine's WorkerNode.
    
    Usage:
        executor = WorkerExecutor(config)
        results = await executor.execute_parallel(state, worker_configs)
    """
    
    def __init__(self, engine_config: Dict[str, Any]):
        """Initialize executor with engine configuration.
        
        Args:
            engine_config: Full engine config from YAML
        """
        self.config = engine_config
        self.critic_enabled = engine_config.get("phases", {}).get("workers", {}).get("config", {}).get("critic", {}).get("enabled", True)
    
    async def execute_parallel(
        self,
        state: EngineState,
        worker_configs: List[WorkerConfig],
    ) -> List[WorkerResult]:
        """Execute multiple workers in parallel.
        
        Args:
            state: Current engine state
            worker_configs: List of worker configurations
            
        Returns:
            List of WorkerResult from all workers
        """
        import asyncio
        
        # Create worker instances
        tasks = []
        for config in worker_configs:
            worker_class = WorkerRegistry.get(config.worker_type)
            worker = worker_class(config)
            tasks.append(self._execute_with_retry(worker, state, config))
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    WorkerResult(
                        worker_id=worker_configs[i].worker_id,
                        worker_type=worker_configs[i].worker_type,
                        task=worker_configs[i].task,
                        success=False,
                        output=None,
                        output_type="error",
                        confidence=0.0,
                        error=str(result),
                        error_type=type(result).__name__,
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_retry(
        self,
        worker: BaseWorker,
        state: EngineState,
        config: WorkerConfig,
    ) -> WorkerResult:
        """Execute a single worker with retry logic.
        
        Implements:
        - Setup validation
        - Execution with timeout
        - Critic evaluation (if enabled)
        - Retry on failure
        """
        last_error = None
        
        for attempt in range(config.max_retries):
            try:
                # Setup
                await worker.setup(state, config)
                
                # Validate input
                if not await worker.validate_input(state, config):
                    return worker._create_result(
                        success=False,
                        output=None,
                        error="Input validation failed",
                    )
                
                # Execute
                result = await worker.execute(state, config)
                
                # Self-evaluation (optional)
                if config.enable_critic:
                    self_eval = await worker.self_evaluate(result)
                    if self_eval.get("confidence", 1.0) < config.critic_threshold:
                        raise WorkerExecutionError(
                            f"Self-evaluation failed: {self_eval}"
                        )
                
                # Teardown
                await worker.teardown(result)
                
                return result
                
            except Exception as e:
                last_error = e
                worker._log(f"Attempt {attempt + 1} failed: {e}", level="warning")
                
                # Teardown on error
                try:
                    await worker.teardown(
                        worker._create_result(
                            success=False,
                            output=None,
                            error=str(e),
                        )
                    )
                except:
                    pass  # Ignore teardown errors
        
        # All retries failed
        return worker._create_result(
            success=False,
            output=None,
            error=f"All {config.max_retries} retries failed: {last_error}",
        )
