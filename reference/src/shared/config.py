"""LLM Configuration - Pydantic models for LLM initialization.

Enforces configuration validation for LLM instances with a unified parameter
system that works across all providers via LiteLLM proxy.

Generation Parameters:
    - temperature: Controls randomness (0.0-2.0)
    - top_p: Nucleus sampling threshold (0.0-1.0)
    - top_k: Top-k sampling (1+)
    - frequency_penalty: Penalize frequent tokens (-2.0 to 2.0)
    - presence_penalty: Penalize repeated tokens (-2.0 to 2.0)
    - repeat_penalty: Repeat penalty (0.0-2.0)
    - min_p: Minimum probability threshold (0.0-1.0)

Note: LiteLLM handles parameter normalization automatically.
Provider-specific methods (get_glm_kwargs, get_llamacpp_kwargs) have been removed.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Unified LLM configuration for all providers via LiteLLM.

    Supports standard generation parameters:
    - temperature, top_p, top_k
    - frequency_penalty, presence_penalty
    - repeat_penalty, min_p
    - enable_thinking (for thinking-capable models)

    All parameters are passed through LiteLLM which handles
    provider-specific normalization automatically.
    """

    # Core connection settings
    model: str = Field(..., description="Model name/identifier")
    api_key: str = Field(..., description="API key for the LLM provider")
    base_url: str = Field(..., description="Base URL for the LLM API")

    # Primary generation parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0.0=deterministic, 2.0=very random)")

    # Thinking/reasoning mode (ZAI/GLM specific)
    enable_thinking: bool = Field(
        default=False,
        description="Enable chain-of-thought reasoning (ZAI/GLM specific)"
    )

    # Output limits
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate"
    )

    # Nucleus / top-p sampling
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (0.0-1.0). Lower = more focused"
    )

    # Top-k sampling
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description="Top-k sampling (1+). Only consider top k tokens"
    )

    # Frequency penalty (OpenAI-compatible)
    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalize frequent tokens (-2.0 to 2.0). Positive = less repetition"
    )

    # Presence penalty (OpenAI-compatible)
    presence_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalize repeated tokens (-2.0 to 2.0). Positive = more variety"
    )

    # Repeat penalty (GLM/llama.cpp specific)
    repeat_penalty: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Repeat penalty for GLM/llama.cpp (0.0-2.0). Higher = less repetition"
    )

    # Min-p sampling (GLM/llama.cpp specific)
    min_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold (0.0-1.0). Filters low-probability tokens"
    )

    # Stop sequences
    stop: Optional[list[str]] = Field(
        default=None,
        description="List of stop sequences (tokens that stop generation)"
    )

    def get_openai_kwargs(self) -> Dict[str, Any]:
        """Get parameters formatted for OpenAI-compatible API (via LiteLLM).

        LiteLLM handles parameter normalization automatically.
        This method is kept for backward compatibility.

        Maps unified config to OpenAI field names:
        - repeat_penalty -> frequency_penalty (if frequency_penalty not set)
        - min_p is not supported by OpenAI (ignored)

        Deprecated: LiteLLM now handles parameter normalization.
        """
        kwargs = {
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        elif self.repeat_penalty is not None:
            # Map repeat_penalty to frequency_penalty for OpenAI
            kwargs["frequency_penalty"] = self.repeat_penalty - 1.0  # Convert scale
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            kwargs["stop"] = self.stop

        return kwargs

    # Deprecated: get_glm_kwargs() and get_llamacpp_kwargs() removed
    # LiteLLM handles parameter normalization automatically.
    # These methods are no longer needed.

    class Config:
        """Pydantic config."""

        frozen = True  # Make config immutable after creation


class ToolRouterConfig(LLMConfig):
    """Configuration specifically for the tool router LLM.

    Tool router should have thinking OFF for faster tool selection.
    """

    enable_thinking: bool = Field(
        default=False,
        description="Tool router has thinking disabled for faster decisions"
    )


class MainChatConfig(LLMConfig):
    """Configuration specifically for the main chat LLM.

    Main chat should have thinking ON to show reasoning process.
    """

    enable_thinking: bool = Field(
        default=True,
        description="Main chat has thinking enabled to show reasoning"
    )
