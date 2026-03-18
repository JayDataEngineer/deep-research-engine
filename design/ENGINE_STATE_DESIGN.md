# Engine State Design

## Problem: State Explosion

The original `GlobalState` had 50+ fields. With 4 workflows unified, this could explode to 150+ fields. This design prevents that.

## Solution: Generic State with Extensible Context

```python
"""Engine State - Generic TypedDict for all workflows.

Design Principles:
1. Minimal core fields (under 20)
2. Extensible context dict for workflow-specific data
3. Strong typing where it matters, flexibility where needed
4. No hardcoded workflow logic in core state
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated, Union
from datetime import datetime
import operator


# =============================================================================
# CORE STATE (Minimal - under 20 fields)
# =============================================================================

class EngineState(TypedDict, total=False):
    """Generic state for ALL workflows.
    
    Core fields are workflow-agnostic. Workflow-specific data lives in `context`.
    
    Total fields: 18 (not 50+)
    """
    
    # ===== Workflow Identity =====
    workflow_id: str                          # Unique workflow run ID
    workflow_name: str                        # Config name (e.g., "rag", "pdf_summarizer")
    workflow_type: str                        # "research" | "ingestion" | "generation"
    
    # ===== Current Phase =====
    current_phase: str                        # "boss" | "workers" | "critic" | "synthesis"
    phase_history: List[str]                  # Track phase transitions
    
    # ===== Input (Generic) =====
    input_data: Dict[str, Any]                # Workflow input (query, pdf_path, etc.)
    
    # ===== Planning (Boss Output) =====
    plan: List[str]                           # Research angles / tasks / chunks
    plan_metadata: Dict[str, Any]             # Planning metadata (complexity, strategy)
    
    # ===== Worker Results (Accumulator) =====
    # Using operator.add for parallel worker accumulation
    worker_results: Annotated[List[Dict[str, Any]], operator.add]
    worker_errors: Annotated[List[Dict[str, Any]], operator.add]
    
    # ===== Critic =====
    critique: Optional[Dict[str, Any]]        # Last critique result
    critique_history: List[Dict[str, Any]]    # Full critique history
    critic_passed: bool                       # Simple pass/fail flag
    
    # ===== Synthesis =====
    synthesis_output: Optional[str]           # Final output (markdown, json, etc.)
    synthesis_metadata: Dict[str, Any]        # Token usage, timing, etc.
    
    # ===== Execution Control =====
    remaining_steps: int                      # LangGraph RemainingSteps
    iteration_count: int                      # How many boss→workers→critic loops
    max_iterations: int                       # Config max iterations
    
    # ===== Context (Extensible - Workflow-Specific) =====
    # This is where workflow-specific data lives WITHOUT bloating core state
    context: Dict[str, Any]                   # Workflow-specific context
    
    # ===== Observability =====
    start_time: str
    end_time: Optional[str]
    session_id: str                           # For tracing (Langfuse, etc.)
    run_id: str                               # Unique run identifier
    
    # ===== Error Handling =====
    errors: List[Dict[str, Any]]              # Accumulated errors
    fatal_error: Optional[str]                # If workflow failed


# =============================================================================
# WORKER RESULT CONTRACT (Standardized)
# =============================================================================

class WorkerResult(TypedDict):
    """Standardized output from ANY worker.
    
    All workers (web_researcher, chunk_processor, multimodal, etc.)
    must return results in this format.
    
    This enables:
    - Uniform accumulation in state
    - Generic critic evaluation
    - Consistent synthesis input
    """
    
    # ===== Identity =====
    worker_id: str                            # e.g., "researcher_0", "chunk_processor_3"
    worker_type: str                          # e.g., "web_researcher", "chunk_processor"
    task: str                                 # The task this worker was assigned
    
    # ===== Result =====
    success: bool                             # Did worker complete successfully?
    output: Any                               # Worker-specific output (findings, chunks, etc.)
    output_type: str                          # Schema hint for synthesis
    
    # ===== Quality =====
    confidence: float                         # 0-1 confidence score
    quality_score: Optional[float]            # 0-1 quality (if self-evaluated)
    
    # ===== Metadata =====
    timing: Dict[str, float]                  # {start, end, duration}
    token_usage: Dict[str, int]               # {prompt, completion, total}
    model_info: Dict[str, str]                # {model, provider, version}
    
    # ===== Sources (Optional) =====
    sources: List[Dict[str, Any]]             # Citations, URLs, document refs
    source_count: int                         # Quick count for evaluation
    
    # ===== Error (If Failed) =====
    error: Optional[str]                      # Error message if failed
    error_type: Optional[str]                 # Exception type if failed
    retry_count: int                          # How many retries attempted


# =============================================================================
# CRITIC RESULT CONTRACT (Standardized)
# =============================================================================

class CriticResult(TypedDict):
    """Standardized output from critic evaluation.
    
    All critics use this schema for consistent evaluation.
    """
    
    # ===== Decision =====
    passed: bool                              # Pass/fail decision
    score: float                              # 0-1 overall quality score
    
    # ===== Evaluation Details =====
    criteria_scores: Dict[str, float]         # Per-criteria scores
    issues: List[str]                         # Specific problems found
    suggestions: List[str]                    # Actionable improvements
    
    # ===== Routing =====
    recommendation: str                       # "pass" | "revise" | "terminate"
    feedback_to: str                          # "boss" | "workers" | "synthesis"
    
    # ===== Metadata =====
    evaluation_time: float                    # Seconds spent evaluating
    model_info: Dict[str, str]                # Which model evaluated


# =============================================================================
# CONTEXT SCHEMAS (Workflow-Specific)
# =============================================================================
# These define what goes in state["context"] for each workflow.
# The core engine doesn't care about these - they're for workers/synthesis.

class RAGContext(TypedDict, total=False):
    """Context for RAG research workflow."""
    # Boss planning
    research_brief: Dict[str, Any]
    complexity_score: float
    
    # Worker outputs (indexed by worker_id)
    findings_by_worker: Dict[str, Dict[str, Any]]
    
    # Citations
    citations: List[Dict[str, Any]]
    citation_count: int
    
    # Storage references
    artifact_ids: List[str]                   # IDs in Context Engine
    vector_store_results: List[Dict[str, Any]]


class PDFContext(TypedDict, total=False):
    """Context for PDF summarizer workflow."""
    # Document info
    pdf_path: str
    total_pages: int
    total_chunks: int
    
    # Chunk processing
    original_chunks: List[Dict[str, Any]]
    filtered_chunks: List[Dict[str, Any]]
    deduplicated_chunks: List[Dict[str, Any]]
    
    # Hierarchy
    chapters: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    
    # Summaries
    section_summaries: Dict[str, str]
    chapter_summaries: Dict[str, str]
    
    # Statistics
    filler_removed_percent: float
    deduplication_reduction: float


class TelegramContext(TypedDict, total=False):
    """Context for Telegram ingestion workflow."""
    # Export info
    export_path: str
    total_messages: int
    date_range: Dict[str, str]
    
    # Processing results
    transcriptions: Dict[str, str]            # item_id → transcription
    image_descriptions: Dict[str, str]        # item_id → VLM description
    faces_detected: Dict[str, List[str]]      # item_id → [person_names]
    entities_extracted: List[Dict[str, Any]]  # People, topics
    
    # Storage references
    postgres_ids: List[str]
    mongo_ids: List[str]
    neo4j_nodes: List[str]
    
    # Clustering
    person_clusters: List[Dict[str, Any]]
    topic_clusters: List[Dict[str, Any]]


class SkillGeneratorContext(TypedDict, total=False):
    """Context for skill generation workflow."""
    # Scraping
    base_url: str
    pages_scraped: int
    total_content_length: int
    
    # Content
    page_content: Dict[str, str]              # url → markdown
    semantic_index: Dict[str, Any]            # InMemoryStore state
    
    # Generation
    llmstxt_content: str
    llmstxt_full_content: str
    skill_package_path: Optional[str]
    
    # Validation
    coverage_score: float
    validation_issues: List[str]


# =============================================================================
# STATE FACTORY (Create Initial State)
# =============================================================================

def create_initial_state(
    workflow_name: str,
    workflow_type: str,
    input_data: Dict[str, Any],
    session_id: str,
    run_id: str,
    max_iterations: int = 3,
    context: Optional[Dict[str, Any]] = None,
) -> EngineState:
    """Create initial state for a workflow run.
    
    Args:
        workflow_name: Config name (e.g., "rag", "pdf_summarizer")
        workflow_type: "research" | "ingestion" | "generation"
        input_data: Workflow-specific input
        session_id: Session ID for tracing
        run_id: Unique run identifier
        max_iterations: Max boss→workers→critic loops
        context: Optional workflow-specific context
        
    Returns:
        Initial EngineState
    """
    return {
        "workflow_id": run_id,
        "workflow_name": workflow_name,
        "workflow_type": workflow_type,
        "current_phase": "boss",
        "phase_history": ["boss"],
        "input_data": input_data,
        "plan": [],
        "plan_metadata": {},
        "worker_results": [],
        "worker_errors": [],
        "critique": None,
        "critique_history": [],
        "critic_passed": False,
        "synthesis_output": None,
        "synthesis_metadata": {},
        "remaining_steps": 100,  # Will be set by LangGraph
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "context": context or {},
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "session_id": session_id,
        "run_id": run_id,
        "errors": [],
        "fatal_error": None,
    }


# =============================================================================
# STATE REDUCERS (For LangGraph Annotated Fields)
# =============================================================================

def merge_worker_results(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Merge worker results from parallel execution."""
    return left + right


def merge_errors(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Merge errors from parallel execution."""
    return left + right


def overwrite_reducer(left: Any, right: Any) -> Any:
    """Reducer that keeps the last value."""
    return right if right is not None else left


# =============================================================================
# TYPE ALIASES (For Convenience)
# =============================================================================

# Phase names
PhaseName = Union[
    Literal["boss"],
    Literal["workers"],
    Literal["critic"],
    Literal["synthesis"],
]

# Workflow types
WorkflowType = Union[
    Literal["research"],
    Literal["ingestion"],
    Literal["generation"],
]

# Worker types
WorkerType = Union[
    Literal["web_researcher"],
    Literal["chunk_processor"],
    Literal["multimodal"],
    Literal["scraper"],
    Literal["asr_worker"],
    Literal["vlm_worker"],
    Literal["face_worker"],
    Literal["text_worker"],
]

# Output formats
OutputFormat = Union[
    Literal["markdown"],
    Literal["json"],
    Literal["html"],
    Literal["graph"],
    Literal["podcast_script"],
]
