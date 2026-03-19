"""
Constants for Deep Research Graph

Centralizes all magic numbers, strings, and configuration values
used throughout the deep research system.
"""

from enum import Enum


# ============================================================================
# Phase Constants
# ============================================================================

class Phase(str, Enum):
    """Research phases"""
    WORKER = "worker"  # Default worker for research tasks
    BOSS = "boss"
    LAB = "lab"
    WORKERS = "workers"
    GRAPH_BUILDER = "graph_builder"  # Knowledge graph construction
    CRITIC = "critic"
    SYNTHESIS = "synthesis"


# ============================================================================
# Log Prefix Patterns
# ============================================================================

class LogPrefix(str, Enum):
    """Standard log prefixes for consistent logging"""
    PHASE_START = "[{phase}] ===== PHASE {phase} START ====="
    PHASE_END = "[{phase}] ===== PHASE {phase} END ====="
    ERROR = "[{phase}] ERROR: {message}"
    INFO = "[{phase}] {message}"
    DEBUG = "[{phase}] [DEBUG] {message}"
    DECISION = "[{phase}] Decision: {decision_type} = {decision}"
    TASK_ASSIGNMENT = "[{phase}] Task assigned: {worker_id} -> {task}"
    TOOL_CALL = "[{phase}] Tool: {tool_name} | {args}"
    TOOL_RESULT = "[{phase}] Tool result: {tool_name} | {result_length} chars"
    PARSING = "[{phase}] Parsing: {what}"
    LLM_START = "[{phase}] Calling LLM: {prompt_length} chars prompt"
    LLM_END = "[{phase}] LLM response: {response_length} chars, {duration:.2f}s"


# ============================================================================
# Special Log Prefixes
# ============================================================================

WORKER_START = "[WORKER START] {worker_id} (type={worker_type}): {task}"
WORKER_END = "[WORKER END] {worker_id}: {status}, {llm_calls} calls, {tokens} tokens, {tool_calls} tools, {output_length} chars output"
WORKER_ERROR = "[WORKER ERROR] {worker_id}: {error_message}"

QUERY_LOG = "[QUERY] {query}"
LLM_CALL_LOG = "[LLM_CALL] {worker}{phase} | Model: {model} ({provider}) | Duration: {duration}s | Response: {preview}"

BOSS_SPAWN = "[BOSS] Spawned {count} workers: {worker_ids}"

CRITIC_INPUT_START = "\n[CRITIC INPUT] ========== BEGIN CRITIC INPUT =========="
CRITIC_INPUT_END = "[CRITIC INPUT] ========== END CRITIC INPUT ==========\n"
CRITIC_RESULT = "[CRITIC] Iteration {iteration}, Quality: {quality}, Passed: {passed}"

SYNTHESIS_RESULT = "[SYNTHESIS] {response_length} chars, {iterations} iterations, {source_count} sources"
SOURCES_RESULT = "[SOURCES] {source_count} sources, {citation_count} total citations"

# ============================================================================
# Research Complexity Levels
# ============================================================================

class Complexity(str, Enum):
    """Research complexity classifications"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    DEEP = "deep"


# ============================================================================
# Worker Types
# ============================================================================

class WorkerType(str, Enum):
    """Worker specialization types"""
    GENERIC = "generic"
    RESEARCHER = "researcher"
    SYNTHESIZER = "synthesizer"
    ANALYZER = "analyzer"
    VERIFIER = "verifier"


# ============================================================================
# Quality Thresholds
# ============================================================================

class QualityThresholds:
    """Quality gate thresholds"""
    EXCELLENT = 0.9
    GOOD = 0.7
    FAIR = 0.5
    POOR = 0.0

    PASS = 0.7
    FAIL = 0.7

    @classmethod
    def get_quality_label(cls, score: float) -> str:
        """Get human-readable quality label"""
        if score >= cls.EXCELLENT:
            return "excellent"
        elif score >= cls.GOOD:
            return "good"
        elif score >= cls.FAIR:
            return "fair"
        else:
            return "poor"


# ============================================================================
# Default Values
# ============================================================================

class Defaults:
    """Default configuration values"""
    MAX_ITERATIONS = 3
    MAX_WORKERS = 8
    MIN_WORKERS = 1
    DEFAULT_WORKERS = 4
    MAX_TOKENS = 128000
    TIMEOUT_SECONDS = 300
    POLL_INTERVAL_MS = 1000

    # Token limits for various outputs
    QUERY_PREVIEW_LENGTH = 200
    RESPONSE_PREVIEW_LENGTH = 1000
    FINDINGS_PREVIEW_LENGTH = 500
    LOG_TRUNCATE_LENGTH = 10000


# ============================================================================
# File Paths
# ============================================================================

class Paths:
    """Standard file paths for outputs"""
    APP_LOG = "app.log"
    LLM_CALLS_LOG = "llm_calls.md"
    METRICS_JSON = "metrics.json"
    OUTPUT_MD = "output.md"
    WORKER_OUTPUTS_JSON = "worker_outputs.json"
    RAW_SOURCES_JSON = "raw_sources.json"
    PROFILING_JSON = "profiling_{worker_id}_{run_id}.json"
    LAB_PROFILING_JSON = "lab_profiling_{run_id}.json"


# ============================================================================
# Database Paths
# ============================================================================

class DatabasePaths:
    """Standard database paths"""
    RUNS_DB = "data/outputs/deep_research/runs.db"
    CHECKPOINTER_DB = "data/outputs/checkpoints/conversations.db"


# ============================================================================
# Timing Constants
# ============================================================================

class Timing:
    """Timing-related constants"""
    FAST_POLL_MS = 2000
    NORMAL_POLL_MS = 5000
    SLOW_POLL_MS = 10000

    SSE_HEARTBEAT_SECONDS = 30
    SSE_RETRY_MS = 1000


# ============================================================================
# Metric Increments
# ============================================================================

class MetricIncrement:
    """Standard metric increment values"""
    DEFAULT = 1
    ZERO = 0


# ============================================================================
# Error Messages
# ============================================================================

class ErrorMessages:
    """Standardized error messages"""
    NO_RESPONSE = "ERROR: No response generated"
    RUN_FAILED = "Run failed: {error_type}: {error}"
    WORKER_FAILED = "Worker {worker_id} failed: {error}"
    PHASE_FAILED = "Phase {phase} failed: {error}"
    TOOL_CALL_FAILED = "Tool call {tool_name} failed: {error}"
    PARSE_FAILED = "Failed to parse {what}: {error}"


# ============================================================================
# Status Values
# ============================================================================

class RunStatus(str, Enum):
    """Run status values"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerStatus(str, Enum):
    """Worker status values"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# Context Engine Tool Names
# ============================================================================

class ContextEngineTools(str, Enum):
    """Context Engine READ tool names

    These are the actual tool names used in LangChain tool calls.
    CRITICAL: Single source of truth - all code MUST reference these constants.
    """
    QUERY_POSTGRES = "query_postgres"
    GET_ARTIFACT = "get_artifact"
    READ_VFS_RAW = "read_vfs_raw"
    QUERY_NEO4J = "query_neo4j"


# List of all Context Engine tools for easy iteration
CONTEXT_ENGINE_READ_TOOLS = [
    ContextEngineTools.QUERY_POSTGRES,
    ContextEngineTools.GET_ARTIFACT,
    ContextEngineTools.READ_VFS_RAW,
    ContextEngineTools.QUERY_NEO4J,
]


# ============================================================================
# Tool-to-Skill Mapping
# ============================================================================

# Tool-to-skill mapping for Boss node
# Maps tool names to the skill name that Boss should advertise
# CRITICAL: Single source of truth - all code MUST reference this mapping.
TOOL_TO_SKILL_MAPPING = {
    # Vector/RAG tools
    "vector_search": "vector_lookup",
    "add_documents": "vector_lookup",
    "list_collections": "vector_lookup",
    "pdf_deep_agent": "vector_lookup",
    "vector_store": "vector_lookup",
    "rag": "vector_lookup",
    # Web tools
    "web_search": "web_search",
    "web_get": "web_reader",
    "web_research_agent": "web_search",
    # Vision tools
    "image_analysis": "vision_analysis",
    "video_analysis": "vision_analysis",
    "vision": "vision_analysis",
    # Code reading
    "zread": "zread",
    "github": "zread",
    # OSINT tools
    "osint_username_search": "osint_tools",
    "osint_email_investigation": "osint_tools",
    "osint_domain_whois": "osint_tools",
    "osint": "osint_tools",
    # Market data
    "market_data": "market_data",
    # News tools
    "news_search": "news_search",
    "news": "news_search",
    # Content generation
    "content_gen": "content_generation",
    "content_generation": "content_generation",
}


# ============================================================================
# String Formatting Helpers
# ============================================================================

def format_log_prefix(prefix: str, **kwargs) -> str:
    """Format a log prefix with provided variables"""
    return prefix.format(**kwargs)


def format_phase_log(phase: Phase, message: str) -> str:
    """Format a phase-specific log message"""
    return f"[{phase.value.upper()}] {message}"


def format_worker_log(worker_id: str, message: str) -> str:
    """Format a worker-specific log message"""
    return f"[{worker_id}] {message}"


def format_error_log(phase: Phase, error: Exception, context: str = "") -> str:
    """Format an error log message"""
    error_type = type(error).__name__
    error_msg = str(error)
    if context:
        return f"[{phase.value.upper()}] ERROR: {error_type}: {error_msg} | {context}"
    return f"[{phase.value.upper()}] ERROR: {error_type}: {error_msg}"
