"""Shared base Pydantic models.

This module contains common base models and response schemas
used across different parts of the application.
"""

from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# =============================================================================
# BASE RESPONSE MODELS
# =============================================================================


class BaseHealthResponse(BaseModel):
    """Base health check response with common fields."""

    status: str = Field(..., description="Service status (healthy, degraded, unhealthy)")
    version: str = Field(..., description="Application version")


class SystemHealthResponse(BaseHealthResponse):
    """Extended health response for system-level checks."""

    uptime: Optional[float] = Field(None, description="Server uptime in seconds")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of health check")


class ExtensionHealthResponse(BaseHealthResponse):
    """Health response for extension status."""

    extension_id: str = Field(..., description="Extension identifier")
    extension_type: Optional[str] = Field(None, description="Extension type (tts, chat, etc.)")
    loaded: bool = Field(..., description="Whether the extension is loaded")


# =============================================================================
# ERROR RESPONSE MODELS
# =============================================================================


class BaseErrorResponse(BaseModel):
    """Standard error response format."""

    error: bool = Field(default=True, description="Always true for error responses")
    message: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Machine-readable error code")


class ValidationErrorResponse(BaseErrorResponse):
    """Response for validation errors."""

    details: Dict[str, Any] = Field(default_factory=dict, description="Field-specific validation errors")


class NotFoundResponse(BaseErrorResponse):
    """Response for resource not found errors."""

    resource_type: Optional[str] = Field(None, description="Type of resource that was not found")
    resource_id: Optional[str] = Field(None, description="ID of the resource that was not found")


# =============================================================================
# BASE REQUEST MODELS
# =============================================================================


class BaseRequest(BaseModel):
    """Base request model with common fields."""

    request_id: Optional[str] = Field(None, description="Unique request identifier for tracing")


class PaginationRequest(BaseRequest):
    """Base request with pagination support."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=50, ge=1, le=1000, description="Items per page")


class PaginatedResponse(BaseModel):
    """Base response with pagination metadata."""

    items: List[Any] = Field(default_factory=list, description="List of items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")


# =============================================================================
# MEDIA UPLOAD RESPONSE
# =============================================================================


class MediaUploadResponse(BaseModel):
    """Standard response for media upload."""

    url: str = Field(..., description="URL to access the uploaded media")
    type: str = Field(..., description="Media type (image, audio, video)")
    size: int = Field(..., ge=0, description="File size in bytes")
    filename: str = Field(..., description="Original filename")


# =============================================================================
# VERSION/MODEL INFO
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an AI model."""

    name: str = Field(..., description="Model name/identifier")
    provider: Optional[str] = Field(None, description="Model provider (openai, anthropic, etc.)")
    version: Optional[str] = Field(None, description="Model version if applicable")


class ProviderInfo(BaseModel):
    """Information about an LLM provider."""

    name: str = Field(..., description="Provider name")
    base_url: str = Field(..., description="API base URL")
    models: List[ModelInfo] = Field(default_factory=list, description="Available models")


# =============================================================================
# TIMESTAMPED MODEL
# =============================================================================


class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""

    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# =============================================================================
# METADATA MODEL
# =============================================================================


class MetadataModel(BaseModel):
    """Base model with metadata support."""

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# WEB TOOL INPUT BASE CLASSES
# =============================================================================


class BaseWebToolInput(BaseModel):
    """Base input for web scraping tools with common fields.
    
    All web tools should inherit from this class to ensure consistent
    URL handling and validation.
    """
    
    url: str = Field(..., description="The URL to fetch or scrape")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {"url": "https://example.com/article"}
        }


class BaseWebSearchInput(BaseModel):
    """Base input for web search tools with common fields.
    
    Provides standard search parameters used across all search tools.
    """
    
    query: str = Field(..., description="The search query string")
    time_range: Literal["day", "week", "month", "year", "noLimit"] = Field(
        default="noLimit",
        description="Time filter: day, week, month, year, noLimit"
    )
    category: str = Field(
        default="general",
        description="Search category: general, news, science, it, images, videos"
    )
    num_results: int = Field(default=5, ge=1, le=50, description="Number of top results")
    location: str = Field(default="us", description="Location: cn or us")
    pages: int = Field(default=10, ge=1, le=20, description="Number of pages to fetch")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "query": "latest AI news",
                "time_range": "week",
                "category": "news",
                "num_results": 10,
                "location": "us",
                "pages": 5
            }
        }


class BaseWebAgentInput(BaseModel):
    """Base input for web research agent tools.
    
    Extends search with additional research parameters.
    """
    
    query: str = Field(..., description="The research query or topic")
    time_range: Literal["day", "week", "month", "year", "noLimit"] = Field(
        default="month",
        description="Time filter for search"
    )
    category: str = Field(
        default="general",
        description="Search category"
    )
    depth: Literal["quick", "standard", "deep"] = Field(
        default="standard",
        description="Research depth: quick (5 results), standard (10), deep (20)"
    )
    location: str = Field(default="us", description="Location: cn or us")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "query": "quantum computing breakthroughs",
                "time_range": "year",
                "category": "science",
                "depth": "deep",
                "location": "us"
            }
        }
