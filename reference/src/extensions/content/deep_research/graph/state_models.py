"""State Models - Pydantic models for state data structures.

This module contains the Pydantic models that define the data structures
used within the graph state. These models provide type-safe validation
for state data.

Separated from state.py to reduce file size and improve modularity.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from langchain_core.messages import BaseMessage

# Use centralized logging
from ..logging_utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# CITATION MODELS
# =============================================================================

class Citation(BaseModel):
    """Citation model with Pydantic validation.

    Represents a single citation with source information and confidence.
    """
    source_type: Literal["url", "document", "database", "api", "tool"] = Field(
        default="document",
        description="Type of source"
    )
    source_name: str = Field(..., min_length=1, description="Name of the source")
    url: Optional[str] = Field(default=None, description="URL if applicable")
    excerpt: str = Field(..., min_length=1, description="Relevant excerpt from source")
    confidence: int = Field(default=50, ge=0, le=100, description="Confidence in citation accuracy (0-100)")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number if applicable")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Citation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("source_name")
    @classmethod
    def validate_source_name(cls, v: str) -> str:
        """Validate source name is not empty."""
        if not v or not v.strip():
            raise ValueError("source_name cannot be empty")
        return v.strip()


def validate_citation(data: Dict[str, Any]) -> Optional[Citation]:
    """Validate citation data and return Citation model if valid.

    Args:
        data: Dictionary containing citation data

    Returns:
        Citation model if valid

    Raises:
        ValueError: If citation data is invalid
    """
    try:
        return Citation(**data)
    except Exception as e:
        # CRITICAL: Citation validation failure - always visible
        logger.error(f"[STATE] Invalid citation data: {str(data)[:200]}... Error: {e}")
        raise ValueError(f"Invalid citation data: {e}") from e


# =============================================================================
# RESEARCH BRIEF MODELS
# =============================================================================

class ResearchBrief(BaseModel):
    """Research brief model with Pydantic validation.

    Created by the Boss node during initial planning to guide the research process.
    """
    query_analysis: str = Field(..., min_length=1, description="Analysis of the research query")
    key_entities: List[str] = Field(default_factory=list, description="Key entities identified")
    research_vectors: List[str] = Field(default_factory=list, description="Research direction vectors")
    proposed_workers: List[str] = Field(default_factory=list, description="Suggested worker types")
    complexity_assessment: Literal["simple", "moderate", "complex"] = Field(
        default="moderate",
        description="Query complexity assessment"
    )
    estimated_iterations: int = Field(default=1, ge=1, le=5, description="Estimated iterations needed")
    special_instructions: Optional[str] = Field(default=None, description="Any special instructions")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Brief creation timestamp")

    # =============================================================================
    # BACKWARD COMPATIBILITY FIELDS
    # =============================================================================
    # These fields maintain compatibility with existing code that uses the old API
    # They are computed from the new fields for backward compatibility
    # =============================================================================

    @property
    def original_query(self) -> str:
        """Extract original query from query_analysis for backward compatibility."""
        # Parse the query_analysis to extract the original query
        lines = self.query_analysis.split('\n')
        for line in lines:
            if line.strip().startswith('Query:') or line.strip().startswith('Original Query:'):
                return line.split(':', 1)[1].strip() if ':' in line else self.query_analysis.split('\n')[0]
        # Fallback: return first line or whole analysis if it's short
        return self.query_analysis.split('\n')[0] if '\n' in self.query_analysis else self.query_analysis

    @property
    def refined_query(self) -> str:
        """Extract refined query from query_analysis for backward compatibility."""
        lines = self.query_analysis.split('\n')
        for line in lines:
            if 'Refined:' in line or 'refined' in line.lower():
                return line.split(':', 1)[1].strip() if ':' in line else self.original_query
        return self.original_query

    @property
    def entities_found(self) -> dict:
        """Convert key_entities list to dict format for backward compatibility."""
        return {e: "entity" for e in self.key_entities}

    @property
    def ambiguity_detected(self) -> bool:
        """Determine ambiguity from query_analysis for backward compatibility."""
        return "ambiguity" in self.query_analysis.lower() and "true" in self.query_analysis.lower()

    @property
    def complexity_estimate(self) -> float:
        """Convert complexity_assessment to numeric estimate for backward compatibility."""
        mapping = {"simple": 0.2, "moderate": 0.5, "complex": 0.8}
        return mapping.get(self.complexity_assessment, 0.5)

    @property
    def recommended_approach(self) -> str:
        """Derive recommended approach from complexity for backward compatibility."""
        mapping = {"simple": "quick", "moderate": "standard", "complex": "deep"}
        return mapping.get(self.complexity_assessment, "standard")


class CitationClaim(BaseModel):
    """Individual claim within a citation."""
    claim: str = Field(..., min_length=1, description="The claim being made")
    confidence: int = Field(default=50, ge=0, le=100, description="Confidence in the claim (0-100)")
    source_id: str = Field(..., min_length=1, description="ID of the source supporting this claim")


class CitationManifest(BaseModel):
    """Manifest of all citations from a worker."""
    worker_id: str = Field(..., min_length=1, description="Worker that produced these citations")
    claims: List[CitationClaim] = Field(default_factory=list, description="List of claims made")
    total_claims: int = Field(default=0, ge=0, description="Total number of claims")
    high_confidence_claims: int = Field(default=0, ge=0, description="Claims with confidence > 70")
    abstract: str = Field(default="", description="Abstract summary of findings")
    key_themes: List[str] = Field(default_factory=list, description="Key themes extracted")
    artifact_path: Optional[str] = Field(default=None, description="Path to full artifact")


class ResearchAngle(BaseModel):
    """A single research angle for Flash Explorer to investigate."""
    angle: str = Field(..., min_length=1, description="The research angle to investigate")
    rationale: str = Field(default="", description="Why this angle is important")
    keywords: List[str] = Field(default_factory=list, description="Keywords for this angle")


class ResearchPlan(BaseModel):
    """Research plan for Flash Explorer.

    Contains multiple research angles to investigate.
    """
    angles: List[ResearchAngle] = Field(
        default_factory=list,
        description="List of research angles to investigate"
    )
    rationale: str = Field(default="", description="Overall rationale for the research plan")
    estimated_queries: int = Field(default=3, ge=1, description="Estimated number of queries needed")


# =============================================================================
# WORKER MODELS
# =============================================================================

class WorkerResult(BaseModel):
    """Result from a worker execution with Pydantic validation.

    Extended model for Iron Forge and Information Funnel architectures.
    Includes both standard fields and extended fields for citation management.
    """
    # ===== Standard Fields =====
    worker_id: str = Field(..., min_length=1, description="Worker identifier")
    worker_type: str = Field(default="generic", description="Worker type/classification")
    task_type: str = Field(default="research", description="Type of task performed")
    task_description: str = Field(default="", description="Description of the task")

    # ===== Findings =====
    findings: str = Field(..., min_length=1, description="Research findings (summary)")
    findings_full: Optional[str] = Field(default=None, description="Full findings with detailed analysis")

    # ===== Sources and Citations =====
    sources: List[str] = Field(default_factory=list, description="List of source names/URLs")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Citations from worker")

    # ===== Source References =====
    source_refs: List[str] = Field(default_factory=list, description="References to source IDs for citation linking")

    # ===== Manifest Flow: Citation Manifest =====
    manifest: Optional["CitationManifest"] = Field(default=None, description="Citation manifest with claims")

    # ===== Metrics =====
    confidence: int = Field(default=50, ge=0, le=100, description="Confidence in findings (0-100)")
    summary_tokens: int = Field(default=0, ge=0, description="Tokens in summary findings")
    full_tokens: int = Field(default=0, ge=0, description="Tokens in full findings")

    # ===== Metadata =====
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Result timestamp")
    execution_time: float = Field(default=0.0, ge=0.0, description="Execution time in seconds")
    artifact_path: Optional[str] = Field(default=None, description="Path to artifact")

    # ===== Computed Properties (for backward compatibility) =====
    @property
    def citation_count(self) -> int:
        """Number of citations."""
        return len(self.citations)

    @model_validator(mode="after")
    def validate_execution_time(self) -> "WorkerResult":
        """Ensure execution_time is reasonable."""
        if self.execution_time > 3600:  # More than 1 hour seems wrong
            self.execution_time = 0.0
        return self


class SubWorkerQuery(BaseModel):
    """Query for a sub-worker (used within a worker for multi-step research)."""
    query: str = Field(..., min_length=1, description="The sub-query")
    context: str = Field(..., min_length=1, description="Context for the sub-query")
    parent_worker_id: str = Field(..., min_length=1, description="Parent worker that spawned this")
    worker_type: str = Field(default="research", description="Type of worker to spawn")


class SubWorkerResult(BaseModel):
    """Result from a sub-worker execution."""
    sub_worker_id: str = Field(..., min_length=1, description="Sub-worker identifier")
    parent_worker_id: str = Field(..., min_length=1, description="Parent worker that spawned this")
    findings: str = Field(..., min_length=1, description="Research findings")
    sources: List[str] = Field(default_factory=list, description="List of sources")


# =============================================================================
# ARTIFACT MODELS
# =============================================================================

class ArtifactPointer(BaseModel):
    """Pointer to an artifact stored in the virtual filesystem."""
    artifact_type: Literal["research_brief", "worker_output", "critique", "synthesis_draft"] = Field(
        ...,
        description="Type of artifact"
    )
    path: str = Field(..., min_length=1, description="Path to artifact in virtual filesystem")
    size: int = Field(default=0, ge=0, description="Size in bytes")
    checksum: Optional[str] = Field(default=None, description="Artifact checksum for verification")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class SourceContent(BaseModel):
    """Content from a source with metadata."""
    source_id: str = Field(..., min_length=1, description="Unique source identifier")
    source_type: Literal["url", "document", "database", "api"] = Field(..., description="Source type")
    content: str = Field(..., min_length=1, description="Source content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Retrieval timestamp")
    tier: Literal["A", "B", "C"] = Field(default="B", description="Source quality tier")


def merge_source_contents(left: Dict[str, SourceContent], right: Dict[str, SourceContent]) -> Dict[str, SourceContent]:
    """Merge two source content dictionaries.

    Args:
        left: First dictionary
        right: Second dictionary (values override left on key conflict)

    Returns:
        Merged dictionary
    """
    return {**left, **right}


# =============================================================================
# PROMPT ADJUSTMENT MODELS
# =============================================================================

class PromptAdjustment(BaseModel):
    """Prompt adjustment for workers."""
    worker_id: str = Field(..., min_length=1, description="Worker to apply adjustment to")
    adjustment_type: Literal["emphasis", "deemphasis", "refine", "expand"] = Field(
        ...,
        description="Type of adjustment"
    )
    instructions: str = Field(..., min_length=1, description="Adjustment instructions")
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="Adjustment weight")


# =============================================================================
# CRITIC MODELS
# =============================================================================

class CriticLLMResponse(BaseModel):
    """LLM response from critic node."""
    quality_score: int = Field(default=50, ge=0, le=100, description="Quality score (0-100)")
    completeness_score: int = Field(default=50, ge=0, le=100, description="Completeness score (0-100)")
    issues: List[str] = Field(default_factory=list, description="Issues identified")
    missing_info: List[str] = Field(default_factory=list, description="Missing information")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions found")
    strengths: List[str] = Field(default_factory=list, description="Strengths identified")
    feedback: str = Field(default="", description="Overall feedback")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    passed_quality_gate: bool = Field(default=False, description="Whether quality gate passed")


class CritiqueResult(BaseModel):
    """Result from critic node with Pydantic validation."""
    quality_score: int = Field(default=0, ge=0, le=100, description="Overall quality score (0-100)")
    issues: List[str] = Field(default_factory=list, description="Issues identified")
    strengths: List[str] = Field(default_factory=list, description="Strengths identified")
    feedback: str = Field(default="", description="Overall feedback")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    passed_quality_gate: bool = Field(default=False, description="Whether quality gate passed")
    requires_revision: bool = Field(default=False, description="Whether revision is required")
    revision_instructions: Optional[str] = Field(default=None, description="Instructions for revision")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Critic timestamp")


# =============================================================================
# FLASH EXPLORER MODELS
# =============================================================================

class FlashExplorerFindings(BaseModel):
    """Findings from a single Flash Explorer run.

    Flash Explorer explores one research avenue using deep agents.
    Returns structured findings with citation tracking via raw_sources.
    """
    avenue: str = Field(..., min_length=1, description="Research avenue explored")
    summary: str = Field(..., min_length=1, description="Summary of findings")
    key_findings: List[str] = Field(default_factory=list, description="Key findings from research")
    sources: List[str] = Field(default_factory=list, description="Source URLs")
    confidence: int = Field(default=50, ge=0, le=100, description="Confidence in findings (0-100)")
    gaps: List[str] = Field(default_factory=list, description="Identified gaps in research")
    deep_agents_run: int = Field(default=0, ge=0, description="Number of deep agents spawned")
    successful_agents: int = Field(default=0, ge=0, description="Number of successful agents")
    loops_completed: int = Field(default=1, ge=1, description="Research loops completed")
    rag_enabled: bool = Field(default=False, description="Whether RAG mode was enabled")
    documents_analyzed: int = Field(default=0, ge=0, description="Number of documents analyzed")


class LabArchitectFindings(BaseModel):
    """Findings from Lab Architect node (Pro-Flash Hybrid synthesis).

    Synthesized output from Lab Architect after Flash Explorers complete research.
    Used for structured output validation with Pydantic.
    """
    executive_summary: str = Field(..., min_length=1, description="Executive summary of findings")
    key_findings: List[str] = Field(default_factory=list, description="Key findings from research")
    cross_cutting_themes: List[str] = Field(default_factory=list, description="Cross-cutting themes identified")
    avenues_summary: Dict[str, str] = Field(default_factory=dict, description="Summary by research avenue")
    overall_confidence: int = Field(default=50, ge=0, le=100, description="Overall confidence in findings (0-100)")
    gaps: List[str] = Field(default_factory=list, description="Identified gaps in research")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations based on findings")
    artifacts_created: List[str] = Field(default_factory=list, description="All artifact IDs from Flash Explorers")


class ResearchEvaluation(BaseModel):
    """Evaluation result from Flash Explorer internal critic.

    Used by Flash Explorer to determine if research is sufficient or needs more iteration.
    """
    sufficient: bool = Field(..., description="Whether current findings are sufficient")
    confidence: int = Field(default=50, ge=0, le=100, description="Confidence in findings (0-100)")
    quality_score: int = Field(default=50, ge=0, le=100, description="Quality of research (0-100)")
    gaps: List[str] = Field(default_factory=list, description="Identified gaps in research")
