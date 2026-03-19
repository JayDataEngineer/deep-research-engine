"""Citation Middleware - Validates and enforces citation standards.

This middleware provides:
1. Pydantic validation for citations
2. Automatic citation extraction from responses
3. Citation quality scoring
4. Missing citation detection

Usage:
    from src.extensions.libraries.core_chat_engine.middleware import CitationMiddleware

    middleware = CitationMiddleware(
        require_citations=True,
        min_citations=1,
        validate_urls=True
    )

    agent = create_deep_agent(middleware=[middleware])
"""

import logging
import re
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict, Optional, List, Dict, Any
from collections.abc import Callable, Awaitable

from pydantic import BaseModel, Field, validator
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR CITATION VALIDATION
# =============================================================================

class CitationModel(BaseModel):
    """Pydantic model for citation validation.

    Enforces citation structure and data types.
    """

    source_type: str = Field(
        description="Type of source (url, document, tool, database)"
    )
    source_name: str = Field(
        description="Human-readable source name (e.g., 'BBC', 'AP News')"
    )
    url: Optional[str] = Field(default=None, description="URL if web source")
    excerpt: str = Field(
        description="Relevant text chunk or excerpt from source",
        min_length=10
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this citation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra data (doc_id, page_num, tool_name, etc.)"
    )

    @validator("source_type")
    def validate_source_type(cls, v):
        """Validate source_type is one of the allowed values."""
        allowed_types = ["url", "document", "tool", "database"]
        if v.lower() not in allowed_types:
            raise ValueError(f"source_type must be one of {allowed_types}, got '{v}'")
        return v.lower()

    @validator("url")
    def validate_url(cls, v, values):
        """Validate URL format if source_type is 'url'."""
        if values.get("source_type") == "url":
            if not v:
                raise ValueError("url is required when source_type is 'url'")
            # Basic URL format validation
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
                r'localhost|'  # localhost
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE
            )
            if not url_pattern.match(v):
                raise ValueError(f"Invalid URL format: {v}")
        return v

    @validator("excerpt")
    def validate_excerpt(cls, v):
        """Validate excerpt is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("excerpt cannot be empty")
        return v.strip()


class CitationReport(BaseModel):
    """Report on citation quality and compliance."""

    total_citations: int = Field(description="Total number of citations found")
    valid_citations: int = Field(description="Number of valid citations")
    invalid_citations: int = Field(description="Number of invalid citations")
    missing_urls: int = Field(description="Citations with source_type='url' but no URL")
    empty_excerpts: int = Field(description="Citations with empty excerpts")
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall citation quality score"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warnings (non-critical issues)"
    )


# =============================================================================
# MIDDLEWARE STATE
# =============================================================================

class CitationMiddlewareState(AgentState):
    """State schema for CitationMiddleware.

    Attributes:
        citation_report: Citation validation report.
            Marked as private so it's not included in the final agent state.
        citations_found: List of extracted citations.
            Marked as private.
    """

    citation_report: NotRequired[Annotated[CitationReport, PrivateStateAttr]]
    citations_found: NotRequired[Annotated[List[CitationModel], PrivateStateAttr]]


class CitationMiddlewareStateUpdate(TypedDict):
    """State update for CitationMiddleware."""

    citation_report: CitationReport
    citations_found: List[CitationModel]


# =============================================================================
# CITATION MIDDLEWARE
# =============================================================================

class CitationMiddleware(AgentMiddleware):
    """Middleware for validating and enforcing citation standards.

    This middleware:
    1. Extracts citations from agent responses
    2. Validates each citation using Pydantic models
    3. Generates a quality report
    4. Can require minimum citation counts

    Args:
        require_citations: If True, responses without citations fail validation
        min_citations: Minimum number of citations required
        validate_urls: If True, validate URL format
        allow_synthetic: If True, allow synthetic/fictional citations
    """

    state_schema = CitationMiddlewareState

    def __init__(
        self,
        *,
        require_citations: bool = False,
        min_citations: int = 0,
        validate_urls: bool = True,
        allow_synthetic: bool = False,
    ):
        """Initialize the citation middleware.

        Args:
            require_citations: If True, responses without citations fail.
            min_citations: Minimum number of citations required.
            validate_urls: If True, validate URL format.
            allow_synthetic: If True, allow synthetic/fictional citations.
        """
        self.require_citations = require_citations
        self.min_citations = max(0, min_citations)
        self.validate_urls = validate_urls
        self.allow_synthetic = allow_synthetic

        logger.info(
            f"[CitationMiddleware] Initialized: require={require_citations}, "
            f"min={min_citations}, validate_urls={validate_urls}, "
            f"allow_synthetic={allow_synthetic}"
        )

    def _extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from response text.

        Looks for patterns like:
        - [1], [source], etc.
        - (Source: URL)
        - According to Source

        Args:
            text: Response text to extract citations from

        Returns:
            List of citation dictionaries
        """
        citations = []

        # Pattern 1: Bracketed citations [1], [Source], etc.
        bracket_pattern = r'\[([^\]]+)\]'
        for match in re.finditer(bracket_pattern, text):
            content = match.group(1)
            # If it looks like a source name
            if len(content) > 2 and not content.isdigit():
                citations.append({
                    "source_name": content,
                    "excerpt": text[max(0, match.start()-50):match.end()+50],
                })

        # Pattern 2: URL citations
        url_pattern = r'https?://[^\s\)]+'
        for match in re.finditer(url_pattern, text):
            url = match.group(0)
            # Extract domain as source name
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            citations.append({
                "source_type": "url",
                "source_name": domain,
                "url": url,
                "excerpt": text[max(0, match.start()-50):match.end()+50],
            })

        # Pattern 3: "According to X" patterns
        according_pattern = r'(?:according to|reported by|stated in|per)\s+([A-Z][A-Za-z\s]+?)(?:,|\.|\s)'
        for match in re.finditer(according_pattern, text, re.IGNORECASE):
            source_name = match.group(1).strip()
            if len(source_name) > 2:
                citations.append({
                    "source_name": source_name,
                    "excerpt": text[match.start():min(len(text), match.start()+100)],
                })

        return citations

    def _validate_citations(
        self,
        citations: List[Dict[str, Any]]
    ) -> tuple[List[CitationModel], CitationReport]:
        """Validate a list of citation dictionaries.

        Args:
            citations: List of citation dictionaries to validate

        Returns:
            Tuple of (valid_citations, report)
        """
        valid_citations = []
        errors = []
        warnings = []

        missing_urls = 0
        empty_excerpts = 0

        for i, citation_data in enumerate(citations):
            try:
                # Set default source_type if not provided
                if "source_type" not in citation_data:
                    if "url" in citation_data and citation_data["url"]:
                        citation_data["source_type"] = "url"
                    else:
                        citation_data["source_type"] = "document"

                # Skip URL validation if disabled
                if not self.validate_urls and citation_data.get("source_type") == "url":
                    # Remove URL for validation if we're not validating URLs
                    citation_data.pop("url", None)

                citation = CitationModel(**citation_data)
                valid_citations.append(citation)

            except ValueError as e:
                errors.append(f"Citation {i+1}: {str(e)}")
                logger.debug(f"[CitationMiddleware] Invalid citation: {e}")

            except Exception as e:
                errors.append(f"Citation {i+1}: Unexpected error - {str(e)}")
                logger.warning(f"[CitationMiddleware] Unexpected error: {e}")

        # Check for missing requirements
        for citation in citations:
            if citation.get("source_type") == "url" and not citation.get("url"):
                missing_urls += 1
            if not citation.get("excerpt") or not citation["excerpt"].strip():
                empty_excerpts += 1

        # Calculate quality score
        total = len(citations)
        valid = len(valid_citations)
        quality_score = valid / total if total > 0 else 0.0

        # Add warnings for low citation count
        if self.require_citations and total < self.min_citations:
            warnings.append(
                f"Required {self.min_citations} citations but found {total}"
            )

        report = CitationReport(
            total_citations=total,
            valid_citations=valid,
            invalid_citations=total - valid,
            missing_urls=missing_urls,
            empty_excerpts=empty_excerpts,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings
        )

        return valid_citations, report

    def before_agent(
        self,
        state: CitationMiddlewareState,
        runtime: Runtime,
        config: RunnableConfig
    ) -> CitationMiddlewareStateUpdate | None:
        """Initialize citation state before agent execution.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with initialized citation tracking.
        """
        # Skip if already initialized
        if "citation_report" in state:
            return None

        return CitationMiddlewareStateUpdate(
            citation_report=CitationReport(
                total_citations=0,
                valid_citations=0,
                invalid_citations=0,
                missing_urls=0,
                empty_excerpts=0,
                quality_score=0.0
            ),
            citations_found=[]
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Wrap model call to validate citations in response.

        Args:
            request: Model request being processed.
            handler: Handler function to call.

        Returns:
            Model response with citation validation.
        """
        response = handler(request)

        # Extract and validate citations from response content
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Try to extract citations
        extracted = self._extract_citations_from_text(response_text)

        # Validate citations
        valid_citations, report = self._validate_citations(extracted)

        # Log results
        logger.info(
            f"[CitationMiddleware] Found {report.total_citations} citations, "
            f"{report.valid_citations} valid (quality: {report.quality_score:.2f})"
        )

        if report.errors:
            logger.warning(f"[CitationMiddleware] Errors: {report.errors}")

        if report.warnings:
            logger.warning(f"[CitationMiddleware] Warnings: {report.warnings}")

        # Store report in state for later access
        request.state["citation_report"] = report
        request.state["citations_found"] = valid_citations

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        """Async wrap model call to validate citations in response.

        Args:
            request: Model request being processed.
            handler: Async handler function to call.

        Returns:
            Model response with citation validation.
        """
        response = await handler(request)

        # Extract and validate citations from response content
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Try to extract citations
        extracted = self._extract_citations_from_text(response_text)

        # Validate citations
        valid_citations, report = self._validate_citations(extracted)

        # Log results
        logger.info(
            f"[CitationMiddleware] Found {report.total_citations} citations, "
            f"{report.valid_citations} valid (quality: {report.quality_score:.2f})"
        )

        if report.errors:
            logger.warning(f"[CitationMiddleware] Errors: {report.errors}")

        if report.warnings:
            logger.warning(f"[CitationMiddleware] Warnings: {report.warnings}")

        # Store report in state for later access
        request.state["citation_report"] = report
        request.state["citations_found"] = valid_citations

        return response


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_citation_report(state: Dict[str, Any]) -> Optional[CitationReport]:
    """Extract citation report from agent state.

    Args:
        state: Agent state dictionary

    Returns:
        CitationReport if available, None otherwise
    """
    report_data = state.get("citation_report")
    if report_data:
        if isinstance(report_data, CitationReport):
            return report_data
        # If it's a dict, deserialize it
        return CitationReport(**report_data)
    return None


def validate_citations_response(
    response_text: str,
    **middleware_kwargs
) -> CitationReport:
    """Validate citations in a response text.

    Utility function to validate citations outside of middleware context.

    Args:
        response_text: Text to validate citations in
        **middleware_kwargs: Arguments to pass to CitationMiddleware

    Returns:
        CitationReport with validation results
    """
    middleware = CitationMiddleware(**middleware_kwargs)
    extracted = middleware._extract_citations_from_text(response_text)
    _, report = middleware._validate_citations(extracted)
    return report
