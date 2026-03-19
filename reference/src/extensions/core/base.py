"""
Extension Base Classes

Defines the abstract base classes for all extension types.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, AsyncGenerator, Any, Dict, List
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

# Import default model constant for consistent default model handling
from src.constants import DEFAULT_LLM_MODEL


class ExtensionType(Enum):
    """Types of extensions."""
    MODE = "mode"        # Shared env, runs in main process (chat modes)
    JOB = "job"          # Isolated env, one-off scripts (CLI tools)
    SERVICE = "service"  # Isolated env, long-running daemons (TTS servers)


class ExtensionStatus(Enum):
    """Status of an extension."""
    LOADED = "loaded"
    MISSING_DEPS = "missing_deps"
    ERROR = "error"
    DISABLED = "disabled"
    REGISTERED = "registered"  # For subprocess-only extensions that are discovered but not loaded


@dataclass
class BaseDataModel:
    """
    Base data model for extension storage.

    Provides common fields and serialization/deserialization methods
    for data models that need to be persisted to disk.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with ISO datetime formatting.

        Returns:
            Dictionary representation of this model
        """
        data = asdict(self)
        if "created_at" in data:
            data["created_at"] = self.created_at.isoformat()
        if "updated_at" in data:
            data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDataModel":
        """
        Create instance from dictionary with datetime parsing.

        Args:
            data: Dictionary representation of this model

        Returns:
            Instance of this model
        """
        data = data.copy()
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)

    def touch(self):
        """Update the updated_at timestamp to now."""
        self.updated_at = datetime.now()


class BaseExtension(ABC):
    """
    Base class for all extensions.

    All extensions must inherit from this and implement the required methods.
    """

    def __init__(self, config: Dict[str, Any], root_path: Path):
        """
        Initialize the extension.

        Args:
            config: The parsed config.yaml content
            root_path: The filesystem path to the extension directory
        """
        self.config = config
        self.name = config['extension']['name']
        self.root_path = root_path
        self.status = ExtensionStatus.LOADED
        self.context: Optional['ExtensionContext'] = None

    @abstractmethod
    async def initialize(self, context: 'ExtensionContext') -> None:
        """
        Called when the extension is loaded.

        Store the context here and perform any initialization.

        Args:
            context: Shared context with LLM factory, settings, etc.
        """
        self.context = context

    @abstractmethod
    async def shutdown(self) -> None:
        """Called when the extension is unloaded. Clean up resources here."""

    def get_capabilities(self) -> Dict[str, Any]:
        """Return the capabilities declared in config."""
        return self.config.get('capabilities', {})

    @property
    def category(self) -> str:
        """Get the extension category from config."""
        return self.config.get('extension', {}).get('category', 'unknown')

    def get_ui_config(self) -> Dict[str, Any]:
        """Get the UI configuration from config.yaml."""
        return self.config.get('ui', {})

    # ==========================================================================
    # API ROUTER REGISTRATION (Micro-Kernel Pattern)
    # ==========================================================================

    def register_routes(self, app) -> None:
        """
        Register API routes with the FastAPI application.

        Extensions can override this method to register their API endpoints.
        This is called during app startup after all extensions are loaded.

        Args:
            app: The FastAPI application instance

        Example:
            ```python
            def register_routes(self, app):
                from fastapi import APIRouter
                router = APIRouter(prefix="/api/v1/my_extension")

                @router.get("/items")
                async def get_items():
                    return {"items": []}

                app.include_router(router)
            ```
        """
        pass

    def get_router_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the router this extension provides.

        Extensions can override this to declare their router metadata
        for frontend discovery.

        Returns:
            Dict with router info, or None if no router:
            {
                "prefix": "/api/v1/my_extension",
                "name": "my_extension",
                "tags": ["my-extension"],
                "description": "My extension API"
            }
        """
        return None


class LogicExtension(BaseExtension):
    """
    Extension that runs in the shared host environment.

    Used for chat modes and other logic that doesn't need isolation.
    These run directly in the main Python process.
    """

    extension_type = ExtensionType.MODE

    async def shutdown(self) -> None:
        """Shutdown the logic extension.

        Default implementation does nothing.
        Subclasses can override for custom cleanup.
        """
        pass

    @abstractmethod
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Main processing method.

        This is called directly without subprocess overhead.

        Args:
            input_data: The input data for processing
            **kwargs: Additional parameters

        Returns:
            The result of processing
        """


class CLIExtension(LogicExtension):
    """
    Base class for CLI-based extensions with storage support.

    Provides common patterns for:
    - Storage directory setup and management
    - CLI command processing
    - Data persistence

    Extensions that provide CLI commands should inherit from this class.
    """

    def __init__(self, config: Dict[str, Any], root_path: Path):
        """
        Initialize the CLI extension.

        Args:
            config: Extension configuration
            root_path: Path to the extension directory
        """
        super().__init__(config, root_path)
        self._storage_dir: Optional[Path] = None
        self._storage_subdir: str = ""  # Subdir for storage (e.g., "variables", "groups")

    async def initialize(self, context: 'ExtensionContext') -> None:
        """
        Initialize the CLI extension with storage setup.

        Args:
            context: Shared extension context
        """
        await super().initialize(context)
        self._setup_storage()
        self._load_data()

    def _setup_storage(self) -> None:
        """
        Set up the storage directory.

        Override _get_storage_subdir() to specify the subdirectory name.
        """
        if not self._storage_subdir:
            return

        try:
            from src.shared.utils.api_helpers import get_inputs_root_dir
            self._storage_dir = get_inputs_root_dir() / self._storage_subdir
        except (ImportError, AttributeError):
            self._storage_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "inputs"
                / self._storage_subdir
            )

        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        """
        Load data from storage.

        Override this method to load extension-specific data.
        """
        pass

    async def shutdown(self) -> None:
        """
        Shutdown the extension and save data.

        Override _save_data() to persist extension-specific data.
        """
        self._save_data()

    def _save_data(self) -> None:
        """
        Save data to storage.

        Override this method to persist extension-specific data.
        """
        pass


# Import the mixin for shared chat functionality
from .mixins import ChatHelperMixin


class ChatModeExtension(LogicExtension, ChatHelperMixin):
    """
    Base class for chat mode extensions.

    Provides chat-specific functionality including state building,
    message conversion, and streaming responses.

    Chat modes should inherit from this class instead of LogicExtension.
    Includes ChatHelperMixin for shared utility methods.
    """

    async def shutdown(self) -> None:
        """Shutdown the chat mode extension.

        Default implementation does nothing.
        Subclasses can override for custom cleanup.
        """
        pass

    @abstractmethod
    def build_state(self, request: 'ChatRequest') -> Dict[str, Any]:
        """
        Build initial state from chat request.

        This is where the extension converts the API request into its
        internal state format. Each extension is responsible for:
        - Message conversion (API format -> LangChain format)
        - Thread ID generation (mode-specific logic)
        - Provider determination
        - Mode-specific state fields
        - Character/user profile objects

        Args:
            request: The chat request from the API

        Returns:
            Initial state dictionary for the LangGraph
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        state: Dict[str, Any],
        config: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response using the extension's graph.

        Args:
            state: The initial state (from build_state())
            config: LangGraph config (thread_id, etc.)

        Yields:
            Stream chunks in Vercel AI SDK format
        """
        pass


class ServiceExtension(BaseExtension):
    """
    Extension that runs as a long-running service daemon.

    Used for TTS servers and other long-running background services.
    These have resource management via lifecycle manager.
    """

    extension_type = ExtensionType.SERVICE

    @abstractmethod
    async def start_service(self) -> None:
        """Start the service daemon."""

    @abstractmethod
    async def stop_service(self) -> None:
        """Stop the service daemon and free resources."""


class JobExtension(BaseExtension):
    """
    Extension that runs as an isolated subprocess.

    Used for CLI tools and batch jobs that need isolation.
    """

    extension_type = ExtensionType.JOB

    @abstractmethod
    async def run_job(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the job in isolated environment.

        Args:
            input_data: Input parameters for the job

        Returns:
            Job execution results
        """


class FailedExtension(BaseExtension):
    """
    Placeholder for extensions that failed to load.

    Used to gracefully handle import/dependency errors without crashing.
    """

    def __init__(self, config: Dict[str, Any], root_path: Path, error: str):
        super().__init__(config, root_path)
        self.status = ExtensionStatus.ERROR
        self.error = error

    async def initialize(self, context: 'ExtensionContext') -> None:
        """No-op for failed extensions."""
        pass

    async def shutdown(self) -> None:
        """No-op for failed extensions."""
        pass

    async def start_service(self) -> None:
        """No-op for failed extensions."""
        pass

    async def stop_service(self) -> None:
        """No-op for failed extensions."""
        pass

    async def run_job(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return error for failed extensions."""
        return {"error": self.error}


class TTSExtensionInterface(ABC):
    """
    Interface for TTS-specific functionality.

    Defines the contract that TTS extensions must implement.
    """

    @abstractmethod
    async def generate_audio(self, text: str, voice: str, **kwargs) -> Any:
        """Generate audio from text.

        Args:
            text: Input text to synthesize
            voice: Voice ID to use
            **kwargs: Additional generation parameters

        Returns:
            Generated audio data
        """
        pass


class TextModeExtension(LogicExtension, ChatHelperMixin):
    """
    Base class for text-based chat mode extensions.

    This provides a structured way for extensions to build chat graphs
    while keeping each extension self-contained.

    Extensions must implement:
    - build_graph(): Create and return the LangGraph for this mode

    Extensions can optionally override:
    - get_personality_enabled(): Whether this mode uses personality
    - get_mode_name(): The internal mode name for routing
    - get_config_overrides(): Configuration overrides for this mode

    Includes ChatHelperMixin for shared utility methods.
    Includes StateBuilderMixin for state building helpers.
    """

    def __init__(self, config: Dict[str, Any], root_path: Path):
        super().__init__(config, root_path)
        self._graph = None

    async def shutdown(self) -> None:
        """Shutdown the text mode extension.

        Default implementation that cleans up the graph.
        Subclasses can override for custom cleanup.
        """
        # Clean up graph if it exists
        if hasattr(self, '_graph') and self._graph is not None:
            self._graph = None

    # Import StateBuilderMixin methods and add them to the class
    # These provide helper methods for building chat state
    from .mixins import StateBuilderMixin

    build_conversation_state = StateBuilderMixin.build_conversation_state
    get_personality_config = StateBuilderMixin.get_personality_config
    get_personality_enabled_from_mixin = StateBuilderMixin.get_personality_enabled
    build_base_state_fields = StateBuilderMixin.build_base_state_fields

    @abstractmethod
    async def build_graph(self) -> Any:
        """
        Build and return the LangGraph for this text mode.

        This method is called during initialization to create the graph
        that will handle chat requests for this mode.

        Extensions can use helper functions from core_chat_engine
        or build completely custom graphs.

        Returns:
            Compiled StateGraph ready for streaming
        """

    async def initialize(self, context: 'ExtensionContext') -> None:
        """
        Initialize the text mode extension.

        Args:
            context: Shared extension context
        """
        self.context = context
        # Build the graph during initialization
        self._graph = await self.build_graph()

    @property
    def graph(self) -> Any:
        """Get the LangGraph for this mode."""
        if self._graph is None:
            raise RuntimeError(f"Graph not initialized for {self.name}")
        return self._graph

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process chat input using this mode's graph.

        Args:
            input_data: Chat request data
            **kwargs: Additional parameters

        Returns:
            Streaming response generator
        """
        # This is called by the chat endpoint
        # The actual streaming is handled by the universal.py router
        raise NotImplementedError("Use graph.astream() directly for streaming")

    def get_mode_name(self) -> str:
        """
        Get the internal mode name for routing.

        By default, uses the extension name. Override if different.

        Returns:
            Mode identifier (e.g., "standard", "companion", "roleplay")
        """
        return self.name

    def get_personality_enabled(self) -> bool:
        """
        Get whether this mode uses personality features.

        Returns:
            True if personality is enabled for this mode
        """
        # Check config first, then use default
        return self.config.get('personality_enabled', False)

    def get_config_overrides(self) -> Optional[Dict[str, Any]]:
        """
        Get any configuration overrides for this mode.

        Extensions can override this to provide custom configuration
        that differs from the preset configs.

        Returns:
            Dict of config field overrides, or None
        """
        return None

    # =============================================================================
    # Chat Request Processing - These replace the business logic in universal.py
    # =============================================================================

    def build_state(self, request: 'ChatRequest') -> Dict[str, Any]:
        """
        Build initial state from chat request.

        Default implementation that converts the API request to state.
        Extensions can override this for custom state building.

        Args:
            request: The chat request from the API (ChatRequest Pydantic model)

        Returns:
            Initial state dictionary for the LangGraph
        """
        import time
        from src.config import settings

        # Convert messages to LangChain format
        langchain_messages = self.convert_messages_to_langchain(request.messages)
        user_input = self.extract_user_input(langchain_messages)

        # Build thread ID (can be overridden by extensions)
        thread_id = self._build_thread_id(request, self.get_mode_name())

        # Determine model and provider
        # Handle frontend sentinel value "select" by using default model
        model = request.model or settings.llm_model
        if model == "select":
            model = settings.llm_model or DEFAULT_LLM_MODEL
        tool_model = getattr(request, 'tool_model', None) or model  # Default to main model if not specified
        if tool_model == "select":
            tool_model = settings.llm_model or DEFAULT_LLM_MODEL
        provider = self.determine_provider(model, settings.llm_base_url or "")

        # Base state
        state = {
            "messages": langchain_messages,
            "user_input": user_input,
            "thread_id": thread_id,
            "model": model,
            "tool_model": tool_model,  # Separate model for tool routing/context gathering
            "provider": provider,
            "start_time": time.time(),
            "processing_time": None,
            "mode": self.get_mode_name(),
            # Required temperature values for LLM nodes
            "main_temperature": getattr(request, 'temperature', None) or 0.7,
            "router_temperature": 0.3,  # Lower temperature for more deterministic tool routing
        }

        return state

    async def stream_chat(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response using the extension's graph.

        This handles the LangGraph streaming and converts to Vercel AI SDK format.

        Args:
            state: The initial state (from build_state())
            config: LangGraph config (thread_id, etc.)

        Yields:
            Stream chunks in Vercel AI SDK format
        """
        if config is None:
            config = {"configurable": {"thread_id": state.get("thread_id")}}

        # Stream from graph - use messages AND custom mode to get OpenAI chunks + custom events
        # Custom events include reasoning chunks, tool status, etc. from writer()
        graph_stream = self.graph.astream(state, config, stream_mode=["messages", "custom"])

        # Yield chunks - streaming now uses simple SSE format
        # No transformation needed - frontend proxy handles Vercel protocol
        async for chunk in graph_stream:
            yield chunk
