"""State Types - Pydantic models for input/output validation.

This module contains the Pydantic models that define the API boundaries
for the deep research system. These provide runtime type-safe validation
for user input and system output.

Separated from state.py to reduce file size and improve modularity.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

from langchain_core.messages import BaseMessage


# =============================================================================
# RUNTIME CONFIGURATION (Pydantic model for validation)
# =============================================================================

class ResearchRuntimeConfig(BaseModel):
    """Runtime configuration for deep research execution with validation.

    This configuration is passed at invoke time via the `context` parameter
    and does not pollute the graph state. This allows the same graph to be
    used with different strategies/models without rebuilding.

    Usage:
        config = ResearchRuntimeConfig(model_provider="fast", max_parallel_workers=4)
        graph.invoke(initial_state, context=config)
    """
    # LLM Configuration
    model_provider: str = "default"  # "fast", "quality", "local", or provider name
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=1000000)

    # Execution Configuration
    max_parallel_workers: int = Field(default=8, ge=1, le=100)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_reflexion: bool = True
    max_iterations: int = Field(default=3, ge=1, le=10)

    # Feature Flags
    enable_prompt_adjustment: bool = False
    enable_caching: bool = True
    cache_ttl: int = Field(default=3600, ge=0, le=86400)  # Max 24 hours

    # Debug Options
    debug_mode: bool = False
    save_artifacts: bool = True

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility


# =============================================================================
# INPUT STATE (Pydantic model for runtime validation)
# =============================================================================

class ResearchInput(BaseModel):
    """User-facing input schema for deep research with Pydantic validation.

    This defines what users/clients can provide to the research system.
    All fields are validated at runtime before graph execution.

    NOTE: chat_history is not included here because it's already defined
    in GlobalState with an Annotated reducer. Including it here causes a
    LangGraph type conflict.
    """
    query: str = Field(..., min_length=1, max_length=10000, description="The research query")


# =============================================================================
# OUTPUT STATE (Pydantic model for runtime validation)
# =============================================================================

class ResearchMetadata(BaseModel):
    """Metadata about research execution with Pydantic validation."""
    total_iterations: int = Field(..., ge=0, description="Total iterations performed")
    total_workers: int = Field(..., ge=0, description="Total workers spawned")
    total_sources: int = Field(..., ge=0, description="Total sources collected")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall quality score")
    passed_quality_gate: bool = Field(default=False, description="Whether quality gate passed")
    timestamps: Dict[str, str] = Field(default_factory=dict, description="Phase timestamps")


class CitationOutput(BaseModel):
    """Citation in final output format with Pydantic validation."""
    source_type: str = Field(..., description="Type of source (url, document, etc.)")
    source_name: str = Field(..., description="Name of the source")
    url: Optional[str] = Field(default=None, description="URL if applicable")
    excerpt: str = Field(..., min_length=1, description="Relevant excerpt")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in citation")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number if applicable")


class ResearchOutput(BaseModel):
    """User-facing output schema for deep research with Pydantic validation.

    This defines what the research system returns to users/clients.
    All fields are validated at runtime before returning to the user.
    """
    response: str = Field(..., min_length=1, description="Final research response")
    sources: List[str] = Field(default_factory=list, description="List of source names")
    citations: List[CitationOutput] = Field(default_factory=list, description="Structured citations")
    metadata: ResearchMetadata = Field(..., description="Execution metadata")


class ResearchResult(BaseModel):
    """Complete research result with output and metadata."""
    output: ResearchOutput
    state: Dict[str, Any] = Field(default_factory=dict, description="Final state snapshot")


# =============================================================================
# NODE INPUT/OUTPUT TYPES (Pydantic validation for nodes)
# =============================================================================

class CriticNodeInput(BaseModel):
    """Input to critic node with validation."""
    draft_content: Optional[str] = None
    research_brief_summary: Optional[str] = None
    worker_outputs: List[Any] = Field(default_factory=list)
    all_citations: List[Any] = Field(default_factory=list)
    iteration_count: int = Field(default=1, ge=1)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class CriticNodeOutput(BaseModel):
    """Output from critic node with validation."""
    critique: Optional[Any] = None
    passed_quality_gate: bool = Field(default=False)
    requires_revision: bool = Field(default=False)
    revision_instructions: Optional[str] = None


class SynthesisNodeInput(BaseModel):
    """Input to synthesis node with validation."""
    query: str = Field(..., min_length=1)
    worker_outputs: List[Any] = Field(default_factory=list)
    all_citations: List[Any] = Field(default_factory=list)
    research_brief_summary: Optional[str] = None
    previous_findings: Optional[str] = None
    iteration_count: int = Field(default=1, ge=1)
    tone: str = Field(default="objective")
    format_type: str = Field(default="comprehensive_analysis")


class SynthesisNodeOutput(BaseModel):
    """Output from synthesis node with validation."""
    response: str = Field(..., min_length=1)
    sources: List[str] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)
    word_count: int = Field(default=0, ge=0)
    format_type: str = Field(default="comprehensive_analysis")
