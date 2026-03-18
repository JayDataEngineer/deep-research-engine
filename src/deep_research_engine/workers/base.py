"""BaseWorker ABC and WorkerRegistry."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
import time

from ..state import EngineState, WorkerResult as WorkerResultDict


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
# WORKER RESULT (Re-export from state)
# =============================================================================

WorkerResult = WorkerResultDict


# =============================================================================
# BASE WORKER ABC
# =============================================================================

class BaseWorker(ABC):
    """Abstract base class for all worker types.
    
    All workers (web_researcher, chunk_processor, multimodal, etc.)
    must implement these methods.
    """
    
    # Class-level metadata (override in subclasses)
    worker_type: str = "base"
    description: str = "Base worker"
    required_tools: List[str] = []
    
    def __init__(self, config: WorkerConfig):
        """Initialize worker with configuration."""
        self.config = config
        self.worker_id = config.worker_id
        self.task = config.task
        
        # Observability
        self.start_time: Optional[float] = None
        self.token_usage: Dict[str, int] = {}
    
    @abstractmethod
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute the worker's task.
        
        This is the MAIN method that all workers must implement.
        
        Args:
            state: Current engine state
            config: Worker configuration
            
        Returns:
            WorkerResult with standardized output
        """
        pass
    
    # -------------------------------------------------------------------------
    # HOOK METHODS (Override as needed)
    # -------------------------------------------------------------------------
    
    async def setup(self, state: EngineState, config: WorkerConfig) -> None:
        """Setup hook called before execute()."""
        pass
    
    async def teardown(self, result: WorkerResult) -> None:
        """Teardown hook called after execute() (even on failure)."""
        pass
    
    async def validate_input(self, state: EngineState, config: WorkerConfig) -> bool:
        """Validate that input is suitable for this worker."""
        return True
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
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
        """Create a standardized WorkerResult."""
        duration = self._end_timing()
        
        return WorkerResult(
            worker_id=self.worker_id,
            worker_type=self.config.worker_type,
            task=self.task,
            success=success,
            output=output,
            output_type=output_type,
            confidence=confidence,
            quality_score=None,
            timing={
                "start": self.start_time or 0.0,
                "end": time.time(),
                "duration": duration,
            },
            token_usage=self.token_usage,
            model_info={},
            sources=sources or [],
            source_count=len(sources) if sources else 0,
            error=error,
            error_type=None,
            retry_count=0,
        )
    
    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message with worker context."""
        from ..utils.logging import get_logger
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
    """Registry for worker types."""
    
    _workers: Dict[str, Type[BaseWorker]] = {}
    
    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a worker type.
        
        Usage:
            @WorkerRegistry.register("web_researcher")
            class WebResearcherWorker(BaseWorker):
                worker_type = "web_researcher"
        """
        def decorator(worker_class: Type[BaseWorker]) -> Type[BaseWorker]:
            cls._workers[name] = worker_class
            worker_class.worker_type = name
            return worker_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseWorker]:
        """Get a worker class by name."""
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
