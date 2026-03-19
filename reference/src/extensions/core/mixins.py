"""
Extension Mixins

Reusable mixins for extension functionality.
"""

from typing import Optional, List, Any, Dict
from pathlib import Path


class RouterMixin:
    """
    Mixin for extensions that provide FastAPI routers.

    Provides a standardized router property pattern that lazily
    imports and caches the router module.
    """

    def __init__(self, config: Dict[str, Any], root_path: Path):
        """
        Initialize the router mixin.

        Args:
            config: Extension configuration
            root_path: Path to the extension directory
        """
        self._router = None

    @property
    def router(self):
        """
        Get the FastAPI router for this extension.

        This property lazily imports the router module and caches it.
        Subclasses should override the _get_router_module_path() method
        to specify the import path for their router.

        Returns:
            The FastAPI router for this extension
        """
        if self._router is None:
            router_path = self._get_router_module_path()
            if router_path:
                module = self._import_router_module(router_path)
                self._router = getattr(module, "router", None)
            else:
                self._router = self._build_router()
        return self._router

    def _get_router_module_path(self) -> Optional[str]:
        """
        Get the import path for the router module.

        Subclasses can override this to specify the import path.
        Example: "src.extensions.modes.text_modes.companion.routes"

        Returns:
            The import path for the router module, or None
        """
        return None

    def _import_router_module(self, import_path: str):
        """
        Import the router module from the given path.

        Args:
            import_path: The Python import path

        Returns:
            The imported module
        """
        import importlib
        return importlib.import_module(import_path)

    def _build_router(self):
        """
        Build the router inline.

        Subclasses can override this to build the router inline
        instead of importing from a separate module.

        Returns:
            The FastAPI router, or None
        """
        return None


class ChatHelperMixin:
    """
    Shared helper methods for chat-based extensions.

    This mixin contains utility methods that are common across
    ChatModeExtension and TextModeExtension, eliminating code duplication.
    """

    def convert_messages_to_langchain(self, messages: List[Any]) -> List[Any]:
        """
        Convert API messages to LangChain format.

        Args:
            messages: List of message objects (from API request)

        Returns:
            List of LangChain message objects
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        langchain_messages = []
        for msg in messages:
            # Handle Pydantic models with model_dump()
            if hasattr(msg, 'model_dump'):
                msg_dict = msg.model_dump()
            elif isinstance(msg, dict):
                msg_dict = msg
            else:
                continue

            role = msg_dict.get("role", "")
            content = msg_dict.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "tool":
                langchain_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg_dict.get("tool_call_id", "")
                ))
            else:
                # Fallback
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    def extract_user_input(self, messages: List[Any]) -> str:
        """
        Extract the user's input from the messages.

        Args:
            messages: List of LangChain messages

        Returns:
            The last user message content
        """
        from langchain_core.messages import HumanMessage

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return str(msg.content)
        return ""

    def determine_provider(self, model: str, base_url: str) -> Optional[str]:
        """
        Determine the LLM provider from model/base_url.

        Args:
            model: Model name
            base_url: API base URL

        Returns:
            Provider name ("zai", "openai", "local", etc.)
        """
        if base_url:
            if "zai" in base_url.lower() or "bigmodel" in base_url.lower():
                return "zai"
            elif "openai" in base_url.lower():
                return "openai"
            elif "localhost" in base_url.lower() or "127.0.0.1" in base_url:
                return "local"

        # Default to zai for glm models
        if model and "glm" in model.lower():
            return "zai"

        return None

    def _build_thread_id(self, request: Any, mode_name: str) -> str:
        """
        Build thread ID from request.

        Can be overridden by extensions for mode-specific behavior.
        Default: generates unique thread ID with mode prefix.

        Args:
            request: The chat request
            mode_name: The name of the mode for the prefix

        Returns:
            Thread ID string
        """
        import uuid
        import logging

        logger = logging.getLogger(__name__)

        # Log the session_id from request for debugging
        logger.info(f"[THREAD_ID] request.session_id = {request.session_id}")

        if request.session_id:
            logger.info(f"[THREAD_ID] Using session_id from request: {request.session_id}")
            return request.session_id

        # Generate unique thread with mode prefix
        unique_id = str(uuid.uuid4())
        generated_thread_id = f"{mode_name}-{unique_id}"
        logger.info(f"[THREAD_ID] Generated new thread_id: {generated_thread_id}")
        return generated_thread_id


class StateBuilderMixin:
    """
    Mixin for chat mode extensions that build state from requests.

    Provides helper methods for creating conversation state with
    configurable parameters, eliminating duplication across chat modes.
    """

    def build_conversation_state(
        self,
        max_context_turns: int = 5,
        summarize_interval: int = 100,
        context_window_size: int = 4000,
    ) -> Dict[str, Any]:
        """
        Build conversation state with given parameters.

        Args:
            max_context_turns: Maximum number of conversation turns to keep in context
            summarize_interval: Interval for summarization
            context_window_size: Maximum context window size in tokens

        Returns:
            Dictionary with conversation state
        """
        from src.extensions.libraries.core_chat_engine.state.brain_state import (
            ConversationState,
            UserProfile,
        )
        from src.extensions.libraries.core_chat_engine.conversation.manager import (
            create_conversation_manager,
        )

        conversation_manager = create_conversation_manager(
            max_context_turns=max_context_turns,
            summarize_interval=summarize_interval,
            context_window_size=context_window_size,
        )

        return {
            "conversation": conversation_manager.create_state(),
            "user_profile": UserProfile(),
        }

    def get_personality_config(self, request: Any, default: str = "friendly") -> Dict[str, Any]:
        """
        Get personality configuration from request.

        Args:
            request: The chat request
            default: Default personality if none specified

        Returns:
            Dictionary with personality configuration
        """
        personality_enabled = (
            request.personality_enabled
            if hasattr(request, "personality_enabled") and request.personality_enabled is not None
            else self.get_personality_enabled()
        )

        return {
            "personality": getattr(request, "personality", None) or default,
            "personality_enabled": personality_enabled,
            "personality_config": None,
        }

    def get_personality_enabled(self) -> bool:
        """
        Get whether this mode uses personality features.

        Returns:
            True if personality is enabled for this mode
        """
        return False

    def build_base_state_fields(self) -> Dict[str, Any]:
        """
        Build base state fields common to all chat modes.

        Returns:
            Dictionary with common state fields
        """
        return {
            "router_tool_calls": [],
            "router_reasoning": "",
            "executed_tools": [],
            "response": None,
        }
