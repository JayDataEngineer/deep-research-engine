"""Pydantic models for LLM input/output validation.

ALL LLM functions MUST use these models for type safety and validation.
No more passing raw dicts or unvalidated data.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator

# Import shared types and validators
from .types import ExtendedRole, MetadataDict, is_empty_string
from .fields import temperature_field, max_tokens_field
from .validators import create_url_string_validator, create_first_message_role_validator

# Import LangChain message classes for compatibility
try:
    from langchain_core.messages import BaseMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseMessage = BaseModel  # Fallback to BaseModel if LangChain not available
    AIMessage = BaseModel  # Fallback to BaseModel if AIMessage not available


# =============================================================================
# MESSAGE MODELS - All message formats must be validated
# =============================================================================

class ChatMessage(BaseModel):
    """A single chat message with role and content.

    This is the standard format for all LLM communication.
    """
    role: ExtendedRole = Field(
        description="The role of the message sender"
    )
    content: str = Field(
        description="The message content",
        min_length=0
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool call ID (required for tool messages)"
    )

    @field_validator("content")
    @classmethod
    def content_not_empty_for_user_assistant(cls, v: str, info) -> str:
        """Validate that user and assistant messages have non-empty content."""
        role = info.data.get("role")
        if role in ["user", "assistant"] and is_empty_string(v):
            raise ValueError(f"{role} messages must have non-empty content")
        return v


class ToolCall(BaseModel):
    """A tool call from the LLM.

    Validates structure of tool invocations.
    """
    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool to call")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


# =============================================================================
# REQUEST MODELS - All LLM requests must use these
# =============================================================================


# =============================================================================
# REQUEST MODELS - All LLM requests must use these
# =============================================================================

class ToolDefinition(BaseModel):
    """Definition of a tool available to the LLM.

    Matches OpenAI function calling format.
    """
    type: Literal["function"] = "function"
    function: Dict[str, Any] = Field(
        description="Tool function definition with name, description, parameters"
    )


class LLMRequest(BaseModel):
    """Validated request for LLM completion.

    ALL LLM calls must use this model.
    """
    messages: List[ChatMessage] = Field(
        description="Conversation history",
        min_length=1
    )
    tools: Optional[List[ToolDefinition]] = Field(
        default=None,
        description="Tools available for the LLM to call"
    )
    temperature: float = temperature_field()
    max_tokens: Optional[int] = max_tokens_field()

    # Use shared validator for first message role
    validate_messages = create_first_message_role_validator("messages")


# =============================================================================
# RESPONSE MODELS - All LLM responses must use these
# =============================================================================

class LLMChunk(AIMessage):
    """A chunk of streaming LLM response.

    All LLM implementations MUST return chunks in this format.
    Inherits from LangChain's AIMessage for compatibility with deepagents.

    Custom fields:
    - tool_calls: Set via AIMessage's built-in tool_calls attribute (for deepagents compatibility)
    - reasoning, is_complete: Stored in additional_kwargs

    Access via:
    - .tool_calls: Returns AIMessage's built-in tool_calls attribute
    - .reasoning: Returns reasoning from additional_kwargs
    - .is_complete: Returns completion status from additional_kwargs
    """

    def __init__(self, content: str = "", reasoning: str = "", tool_calls: List[ToolCall] = None,
                 is_complete: bool = False, additional_kwargs: Dict[str, Any] = None, **kwargs):
        """Initialize LLMChunk with proper AIMessage setup.

        Args:
            content: The message content
            reasoning: Reasoning/thinking content (stored in additional_kwargs)
            tool_calls: Tool calls in this chunk (set via AIMessage's tool_calls parameter)
            is_complete: Whether this is the final chunk (stored in additional_kwargs)
            additional_kwargs: Additional metadata (merged with our custom fields)
            **kwargs: Other AIMessage arguments
        """
        # Initialize additional_kwargs with our custom fields (except tool_calls)
        if additional_kwargs is None:
            additional_kwargs = {}

        # Store reasoning and is_complete in additional_kwargs
        # NOTE: tool_calls is passed to AIMessage.__init__ via kwargs below
        custom_data = {
            "reasoning": reasoning,
            "is_complete": is_complete
        }

        # Merge with any additional_kwargs provided
        merged_kwargs = {**custom_data, **additional_kwargs}

        # CRITICAL FIX: Set tool_calls as actual AIMessage attribute
        # AIMessage has built-in tool_calls handling that deepagents expects
        if tool_calls:
            # Convert ToolCall objects to dicts for AIMessage compatibility
            kwargs["tool_calls"] = [
                tc.model_dump() if isinstance(tc, ToolCall) else tc
                for tc in tool_calls
            ]

        # Call AIMessage.__init__ with content, merged additional_kwargs, and tool_calls
        super().__init__(content=content, additional_kwargs=merged_kwargs, **kwargs)

    @property
    def reasoning(self) -> str:
        """Get reasoning content from additional_kwargs."""
        return self.additional_kwargs.get("reasoning", "")

    @property
    def is_complete(self) -> bool:
        """Get completion status from additional_kwargs."""
        return self.additional_kwargs.get("is_complete", False)


class LLMResponse(BaseModel):
    """Complete non-streaming LLM response.

    Used for ainvoke() calls.
    """
    content: str = Field(description="Response text content")
    reasoning: str = Field(default="", description="Full reasoning content")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="All tool calls made")
    additional_kwargs: MetadataDict = Field(default_factory=dict, description="Extra metadata")


# =============================================================================
# CONFIG MODELS - LLM configuration validation
# =============================================================================

class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider backend.

    Validates backend settings from config/environment.
    """
    model: str = Field(description="Model name/identifier")
    api_key: str = Field(description="API key for the provider")
    base_url: str = Field(description="Base URL for the API endpoint")
    temperature: float = temperature_field()
    enable_thinking: bool = Field(default=True, description="Enable thinking mode")
    max_tokens: Optional[int] = max_tokens_field()

    # Use shared URL validator
    validate_base_url = create_url_string_validator("base_url")


# =============================================================================
# CONVERSION HELPERS - Convert between formats safely
# =============================================================================

class MessageConverter:
    """Safe conversion between different message formats.

    NO MORE manual dict manipulation!
    """

    @staticmethod
    def from_langchain(message: Any) -> ChatMessage:
        """Convert LangChain message to ChatMessage Pydantic model.

        Args:
            message: LangChain BaseMessage subclass

        Returns:
            Validated ChatMessage
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        if isinstance(message, HumanMessage):
            return ChatMessage(role="user", content=str(message.content))
        elif isinstance(message, AIMessage):
            return ChatMessage(role="assistant", content=str(message.content))
        elif isinstance(message, SystemMessage):
            return ChatMessage(role="system", content=str(message.content))
        elif isinstance(message, ToolMessage):
            return ChatMessage(
                role="tool",
                content=str(message.content),
                tool_call_id=getattr(message, 'tool_call_id', None)
            )
        else:
            # Fallback for unknown types
            return ChatMessage(role="user", content=str(message.content))

    @staticmethod
    def to_langchain(message: ChatMessage) -> Any:
        """Convert ChatMessage to LangChain format.

        Args:
            message: Validated ChatMessage

        Returns:
            LangChain BaseMessage subclass
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        if message.role == "user":
            return HumanMessage(content=message.content)
        elif message.role == "assistant":
            return AIMessage(content=message.content)
        elif message.role == "system":
            return SystemMessage(content=message.content)
        elif message.role == "tool":
            return ToolMessage(
                content=message.content,
                tool_call_id=message.tool_call_id or ""
            )
        else:
            raise ValueError(f"Unknown role: {message.role}")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ChatMessage:
        """Convert dict to ChatMessage with validation.

        Args:
            data: Raw dict with role/content

        Returns:
            Validated ChatMessage
        """
        return ChatMessage(**data)

    @staticmethod
    def to_dict(message: ChatMessage) -> Dict[str, Any]:
        """Convert ChatMessage to dict.

        Args:
            message: Validated ChatMessage

        Returns:
            Dict representation
        """
        return message.model_dump(exclude_none=True)
