"""BaseWorker ABC and WorkerRegistry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..state import WorkerResult as WorkerResultDict

if TYPE_CHECKING:
    from shared.base import BaseLLM

    from ..state import EngineState


# =============================================================================
# WORKER CONFIGURATION
# =============================================================================


@dataclass
class WorkerConfig:
    """Configuration for a worker instance.

    Passed from YAML config to worker at runtime.

    Attributes:
        worker_id: Unique identifier for this worker instance.
        worker_type: Type of worker (e.g., "web_researcher", "chunk_processor").
        task: Task description for this worker.
        tools: List of tools available to this worker.
        tool_config: Configuration for individual tools.
        max_retries: Maximum number of retry attempts.
        timeout: Timeout in seconds for worker execution.
        recursion_limit: Maximum recursion depth.
        enable_critic: Whether to enable critic evaluation.
        critic_threshold: Minimum confidence threshold for critic.
        custom: Custom configuration for worker-specific needs.
    """

    worker_id: str
    worker_type: str
    task: str

    # Tools
    tools: list[str] = field(default_factory=list)
    tool_config: dict[str, Any] = field(default_factory=dict)

    # Execution
    max_retries: int = 3
    timeout: int = 300
    recursion_limit: int = 50

    # Critic
    enable_critic: bool = True
    critic_threshold: float = 0.7

    # Worker-specific (flexible dict for custom config)
    custom: dict[str, Any] = field(default_factory=dict)


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

    Attributes:
        worker_type: Class-level worker type identifier.
        description: Human-readable description of the worker.
        required_tools: List of tools required by this worker.
        config: Worker configuration instance.
        worker_id: Unique identifier for this worker instance.
        task: Task description for this worker.
        start_time: Timestamp when worker execution started.
        token_usage: Token usage statistics.
    """

    # Class-level metadata (override in subclasses)
    worker_type: str = "base"
    description: str = "Base worker"
    required_tools: list[str] = []

    def __init__(self, config: WorkerConfig) -> None:
        """Initialize worker with configuration.

        Args:
            config: Worker configuration instance.
        """
        self.config = config
        self.worker_id = config.worker_id
        self.task = config.task

        # Observability
        self.start_time: float | None = None
        self.token_usage: dict[str, int] = {}

    @abstractmethod
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute the worker's task.

        This is the MAIN method that all workers must implement.

        Args:
            state: Current engine state.
            config: Worker configuration.

        Returns:
            WorkerResult with standardized output.
        """
        pass

    # -------------------------------------------------------------------------
    # HOOK METHODS (Override as needed)
    # -------------------------------------------------------------------------

    async def setup(self, state: EngineState, config: WorkerConfig) -> None:
        """Setup hook called before execute().

        Args:
            state: Current engine state.
            config: Worker configuration.
        """

    async def teardown(self, result: WorkerResult) -> None:
        """Teardown hook called after execute() (even on failure).

        Args:
            result: Worker result from execution.
        """

    async def validate_input(self, state: EngineState, config: WorkerConfig) -> bool:
        """Validate that input is suitable for this worker.

        Args:
            state: Current engine state.
            config: Worker configuration.

        Returns:
            True if input is valid, False otherwise.
        """
        return True

    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------

    def _start_timing(self) -> None:
        """Start timing execution."""
        self.start_time = time.time()

    def _end_timing(self) -> float:
        """End timing and return duration.

        Returns:
            Duration in seconds since _start_timing was called.
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time  # type: ignore[operator]

    def _create_result(
        self,
        success: bool,
        output: Any,
        output_type: str = "generic",
        confidence: float = 0.8,
        sources: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> WorkerResult:
        """Create a standardized WorkerResult.

        Args:
            success: Whether the worker completed successfully.
            output: Worker output data.
            output_type: Type hint for the output schema.
            confidence: Confidence score (0-1).
            sources: List of source citations.
            error: Error message if failed.

        Returns:
            WorkerResult with standardized format.
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

    async def _get_llm(
        self,
        state: EngineState,
        role: str = "worker",
        **kwargs: Any,
    ) -> BaseLLM:
        """Get an LLM instance for a specific role.

        Resolves model settings from:
        1. worker-specific config (self.config.custom)
        2. phase-specific config (from state['context']['config'])
        3. global defaults

        Args:
            state: Current engine state.
            role: LLM role ("worker", "boss", "critic").
            **kwargs: Overrides for get_chat_llm (temperature, tags, etc.).

        Returns:
            LLM instance.
        """
        try:
            from shared.chat_llm import get_chat_llm
        except ImportError:
            # Fallback for environments where shared is not available
            self._log("shared.chat_llm not found, using direct fallback", level="warning")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", **kwargs)  # type: ignore

        # 1. Get phase/global config from state
        yaml_config = state.get("context", {}).get("config", {})

        # 2. Resolve model name
        # Priority: custom config > phase config > global config > default
        model_name = self.config.custom.get("model")
        
        if not model_name:
            # Try to get from phase config in state
            from ..config import get_phase_config
            phase_cfg = get_phase_config(yaml_config, role)
            model_name = phase_cfg.get("model", {}).get("name")

        if not model_name:
            # Try global llm config
            llm_cfg = yaml_config.get("llm", {})
            model_name = llm_cfg.get("models", {}).get(role) or \
                         llm_cfg.get("models", {}).get("default", "qwen-35-27-fast")

        # 3. Create LLM
        return await get_chat_llm(
            model=model_name,
            session_id=state.get("session_id"),
            tags=[f"worker:{self.worker_type}", role],
            **kwargs,
        )

    def _log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        """Log a message with worker context.

        Args:
            message: Log message.
            level: Logging level (info, warning, error, debug).
            **kwargs: Additional context for logging.
        """
        from ..utils.logging import get_loguru_logger

        logger = get_loguru_logger(f"worker.{self.worker_type}")

        log_method = getattr(logger, level, logger.info)
        
        # Construct log message with context prefix
        ctx_msg = f"[{self.worker_id}] {message}"
        if kwargs:
            ctx_msg += f" | {kwargs}"
            
        log_method(ctx_msg)


# =============================================================================
# WORKER REGISTRY (Pluggable Worker Types)
# =============================================================================


class WorkerRegistry:
    """Registry for worker types."""

    _workers: dict[str, type[BaseWorker]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a worker type.

        Args:
            name: Name to register the worker under.

        Returns:
            Decorator function that registers the worker class.

        Example:
            @WorkerRegistry.register("web_researcher")
            class WebResearcherWorker(BaseWorker):
                worker_type = "web_researcher"
        """

        def decorator(worker_class: type[BaseWorker]) -> type[BaseWorker]:
            cls._workers[name] = worker_class
            worker_class.worker_type = name
            return worker_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseWorker]:
        """Get a worker class by name.

        Args:
            name: Name of the worker type.

        Returns:
            Worker class.

        Raises:
            WorkerNotFoundError: If the worker type is not registered.
        """
        if name not in cls._workers:
            available = ", ".join(cls._workers.keys())
            raise WorkerNotFoundError(f"Worker type '{name}' not found. Available: {available}")
        return cls._workers[name]

    @classmethod
    def list_workers(cls) -> list[str]:
        """List all registered worker types.

        Returns:
            List of registered worker type names.
        """
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
