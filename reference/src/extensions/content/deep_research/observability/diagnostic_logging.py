"""Diagnostic logging for detecting and tracking issues in deep research.

This module extends the logging system to catch the issues identified in code review:
1. LLM returning content unrelated to the original query
2. Flash Explorers failing silently (0% success rate)
3. Context Engine tool call validation errors
4. Critic returning empty responses after tool calls

LangFuse Integration:
    This module optionally emits LangFuse scores when a session_id is provided.
    This enables real-time observability and evaluation in the LangFuse dashboard.

    Scores emitted:
    - diagnostic_llm_relevance: 1.0 if relevant, 0.0 if off-topic
    - diagnostic_flash_explorer: 1.0 if success, 0.0 if failed
    - diagnostic_tool_validation: 1.0 if valid, 0.0 if error

Usage:
    from .observability.diagnostic_logging import (
        DiagnosticLogger,
        validate_llm_response_relevance,
        track_flash_explorer_result,
        log_tool_call_validation_error,
        track_critic_response,
    )

    # In your code:
    diagnostic = DiagnosticLogger(run_logger)

    # Validate LLM response
    is_relevant, reason = validate_llm_response_relevance(
        original_query="Create 5 tweets about gym discipline",
        llm_response="The whole nine yards means...",
        context="worker_planning"
    )
    if not is_relevant:
        diagnostic.log_llm_hallucination(...)

    # Track Flash Explorer
    track_flash_explorer_result(
        avenue="Research gym discipline",
        success=False,
        error=exception,
        elapsed=0.21
    )
"""

import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ..logging_utils import log_component, safe_get_logger

logger = safe_get_logger(__name__, "diagnostic")


# =============================================================================
# DATA MODELS FOR DIAGNOSTICS
# =============================================================================

@dataclass
class LLMResponseDiagnostic:
    """Diagnostic info for an LLM response."""
    context: str  # Where this response came from (worker_planning, lab_synth, etc.)
    original_query: str
    llm_response_preview: str
    is_relevant: bool
    relevance_reason: str
    detected_topics: List[str] = field(default_factory=list)
    expected_topics: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FlashExplorerDiagnostic:
    """Diagnostic info for a Flash Explorer run."""
    avenue: str
    success: bool
    elapsed_seconds: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    findings_length: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ToolCallDiagnostic:
    """Diagnostic info for a tool call validation error."""
    tool_name: str
    validation_errors: List[str]
    runtime_provided: Dict[str, Any] = field(default_factory=dict)
    runtime_expected: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CriticResponseDiagnostic:
    """Diagnostic info for critic response extraction."""
    num_llm_calls: int
    num_tool_calls: int
    messages_received: int
    response_length: int
    last_message_type: str
    last_message_preview: str
    is_empty: bool
    extraction_method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# LLM RESPONSE RELEVANCE VALIDATION
# =============================================================================

# Default off-topic indicators (can be overridden via config)
DEFAULT_OFF_TOPIC_INDICATORS = [
    # Idioms and phrases (like "the whole nine yards")
    r'\b(the whole nine yards|break a leg|bite the bullet|piece of cake)\b.*\b(meaning|origin|etymology)\b',
    # Programming problems (like "climbing stairs", "fibonacci")
    r'\b(climbing stairs|fibonacci|dynamic programming|leetcode|hackerrank)\b',
    r'\bdef\s+\w+\s*\([^)]*\)\s*:',  # Python function definitions
    r'```python\s*def\s+',  # Python code blocks
    # Wikipedia-style explanations
    r'^#{1,3}\s+(history|etymology|origin|usage)\b',
    # Generic explanations that don't reference the query
    r'\b(here is|this is|the following)\s+(the|a)\s+(solution|answer|explanation)\b',
]

# Default domain keywords (can be overridden via config)
DEFAULT_DOMAIN_KEYWORDS = {
    'default': [
        'gym', 'fitness', 'workout', 'exercise', 'discipline', 'training',
        'muscle', 'strength', 'health', 'motivation', 'athletic', 'lift',
        'weights', 'cardio', 'bodybuilding', 'crossfit', 'physical'
    ],
    'research': [
        'research', 'study', 'analysis', 'finding', 'evidence', 'data',
        'methodology', 'conclusion', 'citation', 'reference'
    ],
    'news': [
        'news', 'report', 'article', 'breaking', 'update', 'journalism',
        'media', 'press'
    ],
}


class DiagnosticConfig:
    """Configuration for diagnostic validation.

    Can be loaded from observability config YAML.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize diagnostic configuration.

        Args:
            config: Configuration dictionary (from observability config)
        """
        config = config or {}

        # Get diagnostics config section
        diag_config = config.get("diagnostics", {})

        # Off-topic indicators
        self.off_topic_indicators = diag_config.get(
            "off_topic_indicators",
            DEFAULT_OFF_TOPIC_INDICATORS
        )

        # Domain keywords
        raw_domain_keywords = diag_config.get("domain_keywords", {})
        self.domain_keywords: Dict[str, List[str]] = {}
        for key, keywords in raw_domain_keywords.items():
            if isinstance(keywords, list):
                self.domain_keywords[key] = [k.lower() for k in keywords]

        # Add defaults if not provided
        for key, keywords in DEFAULT_DOMAIN_KEYWORDS.items():
            if key not in self.domain_keywords:
                self.domain_keywords[key] = keywords

        # Minimum relevance threshold
        self.min_relevance_threshold = diag_config.get("min_relevance_threshold", 0.3)

    def get_keywords_for_use_case(self, use_case: str) -> List[str]:
        """Get domain keywords for a specific use case.

        Args:
            use_case: The use case (rag, soc_media_gen, etc.)

        Returns:
            List of domain keywords
        """
        return self.domain_keywords.get(use_case, self.domain_keywords.get("default", []))


# Global config instance (fallback when no context is set)
_global_config: DiagnosticConfig = None


def get_diagnostic_config() -> DiagnosticConfig:
    """Get the diagnostic configuration.

    First checks the observability context (thread-safe), then falls back
    to the global config for backward compatibility.

    Returns:
        DiagnosticConfig instance
    """
    # First, try to get from context
    from .context import get_context_or_none
    ctx = get_context_or_none()
    if ctx and ctx.diagnostic_config:
        return ctx.diagnostic_config

    # Fall back to global config
    global _global_config
    if _global_config is None:
        _global_config = DiagnosticConfig()
    return _global_config


def set_diagnostic_config(config: Dict[str, Any]) -> None:
    """Set the diagnostic configuration.

    Sets in the observability context if available, otherwise sets globally
    for backward compatibility.

    Args:
        config: Configuration dictionary
    """
    new_config = DiagnosticConfig(config)

    # Try to set in context first
    from .context import get_context_or_none, set_diagnostic_config as set_ctx_config
    ctx = get_context_or_none()
    if ctx:
        set_ctx_config(new_config)
        return

    # Fall back to global config
    global _global_config
    _global_config = new_config


def get_off_topic_indicators() -> List[str]:
    """Get the configured off-topic indicators.

    Returns:
        List of regex patterns
    """
    return get_diagnostic_config().off_topic_indicators


def get_domain_keywords(use_case: str = "default") -> List[str]:
    """Get domain keywords for a use case.

    Args:
        use_case: The use case

    Returns:
        List of domain keywords
    """
    return get_diagnostic_config().get_keywords_for_use_case(use_case)


def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from a text for relevance checking.

    Args:
        text: Input text to extract terms from

    Returns:
        List of lowercase key terms
    """
    if not text:
        return []

    # Convert to lowercase
    text_lower = text.lower()

    # Remove common words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below',
                  'between', 'under', 'again', 'further', 'then', 'once',
                  'here', 'there', 'when', 'where', 'why', 'how', 'all',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such',
                  'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                  'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                  'until', 'while', 'about', 'against', 'this', 'that', 'these',
                  'those', 'what', 'which', 'who', 'whom', 'i', 'me', 'my',
                  'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                  'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                  'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                  'itself', 'they', 'them', 'their', 'theirs', 'themselves'}

    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)

    # Filter out stop words and return unique terms
    return list(set(w for w in words if w not in stop_words))


def detect_off_topic_patterns(response: str) -> Tuple[bool, Optional[str]]:
    """Detect if response contains off-topic patterns.

    Uses configurable patterns from diagnostic config.

    Args:
        response: LLM response text

    Returns:
        Tuple of (is_off_topic, pattern_matched)
    """
    response_lower = response.lower()

    # Get configured off-topic indicators
    indicators = get_off_topic_indicators()

    for pattern in indicators:
        if re.search(pattern, response_lower, re.IGNORECASE | re.MULTILINE):
            return True, pattern

    return False, None


def calculate_term_overlap(query_terms: List[str], response_terms: List[str]) -> float:
    """Calculate overlap between query terms and response terms.

    Args:
        query_terms: Key terms from the query
        response_terms: Key terms from the response

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if not query_terms:
        return 0.0

    query_set = set(query_terms)
    response_set = set(response_terms)

    overlap = query_set & response_set
    return len(overlap) / len(query_set)


def validate_llm_response_relevance(
    original_query: str,
    llm_response: str,
    context: str,
    run_logger: Any = None,
    min_overlap: float = 0.1,
) -> Tuple[bool, str, LLMResponseDiagnostic]:
    """Validate that an LLM response is relevant to the original query.

    This catches the bug where the LLM returns completely unrelated content
    like "the whole nine yards" explanation when asked about "gym discipline".

    Args:
        original_query: The user's original query
        llm_response: The LLM's response text
        context: Where this response came from (e.g., "worker_planning")
        run_logger: Optional RunLogger for logging
        min_overlap: Minimum term overlap ratio (default 0.1 = 10%)

    Returns:
        Tuple of (is_relevant, reason, diagnostic)
    """
    # Extract terms
    query_terms = extract_key_terms(original_query)
    response_terms = extract_key_terms(llm_response)

    # Check for off-topic patterns
    is_off_topic, pattern = detect_off_topic_patterns(llm_response)

    if is_off_topic:
        diagnostic = LLMResponseDiagnostic(
            context=context,
            original_query=original_query,
            llm_response_preview=llm_response[:500],
            is_relevant=False,
            relevance_reason=f"Off-topic pattern detected: {pattern}",
            detected_topics=response_terms[:10],
            expected_topics=query_terms,
        )

        if run_logger:
            _log_llm_hallucination(run_logger, context, diagnostic, query_terms, response_terms, pattern, llm_response)

        return False, f"Off-topic pattern: {pattern}", diagnostic

    # Calculate term overlap
    overlap = calculate_term_overlap(query_terms, response_terms)

    # Check if response is too short to be meaningful
    if len(llm_response.strip()) < 50:
        diagnostic = LLMResponseDiagnostic(
            context=context,
            original_query=original_query,
            llm_response_preview=llm_response,
            is_relevant=False,
            relevance_reason="Response too short",
            detected_topics=response_terms[:10],
            expected_topics=query_terms,
        )
        return False, "Response too short", diagnostic

    # Check for sufficient overlap
    if overlap < min_overlap:
        diagnostic = LLMResponseDiagnostic(
            context=context,
            original_query=original_query,
            llm_response_preview=llm_response[:500],
            is_relevant=False,
            relevance_reason=f"Low term overlap: {overlap:.2%} < {min_overlap:.2%}",
            detected_topics=response_terms[:10],
            expected_topics=query_terms,
        )

        if run_logger:
            _log_low_overlap(run_logger, context, diagnostic, query_terms, response_terms, overlap)

        return False, f"Low overlap: {overlap:.2%}", diagnostic

    # Response appears relevant
    diagnostic = LLMResponseDiagnostic(
        context=context,
        original_query=original_query,
        llm_response_preview=llm_response[:500],
        is_relevant=True,
        relevance_reason=f"Term overlap: {overlap:.2%}",
        detected_topics=response_terms[:10],
        expected_topics=query_terms,
    )

    return True, f"Overlap: {overlap:.2%}", diagnostic


def _log_llm_hallucination(
    run_logger: Any,
    context: str,
    diagnostic: LLMResponseDiagnostic,
    query_terms: List[str],
    response_terms: List[str],
    pattern: str,
    llm_response: str,
) -> None:
    """Log LLM hallucination diagnostic with consistent formatting."""
    log_component(run_logger, "DIAGNOSTIC", f"LLM_HALLUCINATION in {context}!")
    log_component(run_logger, "DIAGNOSTIC", f"  Expected topics: {query_terms[:10]}")
    log_component(run_logger, "DIAGNOSTIC", f"  Got topics: {response_terms[:10]}")
    log_component(run_logger, "DIAGNOSTIC", f"  Pattern: {pattern}")
    log_component(run_logger, "DIAGNOSTIC", f"  Response preview: {llm_response[:200]}...")


def _log_low_overlap(
    run_logger: Any,
    context: str,
    diagnostic: LLMResponseDiagnostic,
    query_terms: List[str],
    response_terms: List[str],
    overlap: float,
) -> None:
    """Log low term overlap diagnostic with consistent formatting."""
    log_component(run_logger, "DIAGNOSTIC", f"LOW_OVERLAP in {context}: {overlap:.2%}")
    log_component(run_logger, "DIAGNOSTIC", f"  Expected: {query_terms[:10]}")
    log_component(run_logger, "DIAGNOSTIC", f"  Got: {response_terms[:10]}")


# =============================================================================
# FLASH EXPLORER TRACKING
# =============================================================================


def track_flash_explorer_result(
    avenue: str,
    success: bool,
    elapsed_seconds: float,
    run_logger: Any = None,
    error: Optional[Exception] = None,
    tools_called: Optional[List[str]] = None,
    findings_length: int = 0,
    session_id: Optional[str] = None,
) -> FlashExplorerDiagnostic:
    """Track a Flash Explorer result for diagnostics.

    This catches the 0% success rate issue by logging detailed error info.

    Args:
        avenue: The research avenue being explored
        success: Whether the explorer succeeded
        elapsed_seconds: Time elapsed
        run_logger: Optional RunLogger for logging
        error: Optional exception if failed
        tools_called: List of tools that were called
        findings_length: Length of findings in characters
        session_id: Optional session ID for LangFuse score emission

    Returns:
        FlashExplorerDiagnostic with the tracking info
    """
    diagnostic = FlashExplorerDiagnostic(
        avenue=avenue,
        success=success,
        elapsed_seconds=elapsed_seconds,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
        error_traceback=traceback.format_exc() if error else None,
        tools_called=tools_called or [],
        findings_length=findings_length,
    )

    if run_logger:
        _log_flash_explorer_result(run_logger, diagnostic)

    # Emit LangFuse score if session_id is provided
    if session_id:
        _emit_langfuse_diagnostic_score(
            session_id=session_id,
            diagnostic_type="flash_explorer",
            value=1.0 if success else 0.0,
            comment=f"{'Success' if success else 'Failed'}: {avenue[:50]}... ({elapsed_seconds:.2f}s)",
        )

    return diagnostic


def _log_flash_explorer_result(run_logger: Any, diagnostic: FlashExplorerDiagnostic) -> None:
    """Log Flash Explorer result with consistent formatting."""
    if diagnostic.success:
        log_component(
            run_logger, "FLASH_EXPLORER",
            f"SUCCESS: {diagnostic.avenue[:50]}... "
            f"({diagnostic.elapsed_seconds:.2f}s, {diagnostic.findings_length} chars)"
        )
    else:
        log_component(run_logger, "DIAGNOSTIC", f"FLASH_EXPLORER_FAILED: {diagnostic.avenue[:50]}...")
        log_component(run_logger, "DIAGNOSTIC", f"  Elapsed: {diagnostic.elapsed_seconds:.2f}s (very fast = immediate crash)")
        if diagnostic.error_type:
            log_component(run_logger, "DIAGNOSTIC", f"  Error: {diagnostic.error_type}: {diagnostic.error_message[:200] if diagnostic.error_message else 'N/A'}")
            if diagnostic.error_traceback:
                tb_lines = diagnostic.error_traceback.split('\n')[:5]
                for line in tb_lines:
                    log_component(run_logger, "DIAGNOSTIC", f"    {line}")
        if diagnostic.tools_called:
            log_component(run_logger, "DIAGNOSTIC", f"  Tools called: {diagnostic.tools_called}")


def diagnose_flash_explorer_batch(
    results: List[Any],
    elapsed_total: float,
    run_logger: Any = None,
) -> Dict[str, Any]:
    """Diagnose a batch of Flash Explorer results.

    Detects the 0% success rate pattern and logs detailed diagnostics.

    Args:
        results: List of results (may be exceptions or successful results)
        elapsed_total: Total elapsed time for all explorers
        run_logger: Optional RunLogger for logging

    Returns:
        Dict with diagnostic summary
    """
    total = len(results)
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    success_count = len(successful)
    fail_count = len(failed)

    # Calculate success rate
    success_rate = success_count / total if total > 0 else 0

    # Detect immediate crash pattern (very fast execution = instant failure)
    avg_time_per_explorer = elapsed_total / total if total > 0 else 0
    is_immediate_crash = avg_time_per_explorer < 1.0 and fail_count > 0

    summary = {
        "total": total,
        "successful": success_count,
        "failed": fail_count,
        "success_rate": success_rate,
        "avg_time_per_explorer": avg_time_per_explorer,
        "is_immediate_crash": is_immediate_crash,
        "error_types": [type(e).__name__ for e in failed],
        "error_messages": [str(e)[:200] for e in failed],
    }

    if run_logger:
        _log_flash_explorer_batch_summary(run_logger, summary, failed)

    return summary


def _log_flash_explorer_batch_summary(
    run_logger: Any,
    summary: Dict[str, Any],
    failed: List[Exception],
) -> None:
    """Log Flash Explorer batch summary with consistent formatting."""
    log_component(
        run_logger, "DIAGNOSTIC",
        f"FLASH_EXPLORER_BATCH: {summary['successful']}/{summary['total']} successful"
    )
    log_component(run_logger, "DIAGNOSTIC", f"  Success rate: {summary['success_rate']:.1%}")
    log_component(run_logger, "DIAGNOSTIC", f"  Avg time: {summary['avg_time_per_explorer']:.2f}s per explorer")

    if summary['success_rate'] == 0:
        log_component(run_logger, "DIAGNOSTIC", "CRITICAL: 0% success rate - all Flash Explorers failed!")
        for i, e in enumerate(failed[:3]):
            log_component(run_logger, "DIAGNOSTIC", f"  Failure {i+1}: {type(e).__name__}: {str(e)[:200]}")

    if summary['is_immediate_crash']:
        log_component(run_logger, "DIAGNOSTIC", "IMMEDIATE_CRASH_PATTERN: Explorers failing instantly")
        log_component(run_logger, "DIAGNOSTIC", "  This suggests initialization error, not runtime error")


# =============================================================================
# TOOL CALL VALIDATION ERROR LOGGING
# =============================================================================


def log_tool_call_validation_error(
    tool_name: str,
    validation_errors: List[str],
    runtime_provided: Dict[str, Any],
    run_logger: Any = None,
    expected_keys: Optional[List[str]] = None,
) -> ToolCallDiagnostic:
    """Log a tool call validation error with detailed diagnostics.

    This catches the Context Engine mock runtime validation errors.

    Args:
        tool_name: Name of the tool that failed validation
        validation_errors: List of Pydantic validation error messages
        runtime_provided: The runtime dict that was provided
        run_logger: Optional RunLogger for logging
        expected_keys: Expected keys for the runtime

    Returns:
        ToolCallDiagnostic with the error info
    """
    expected_keys = expected_keys or ["state", "context", "config", "stream_writer", "tool_call_id", "store"]

    diagnostic = ToolCallDiagnostic(
        tool_name=tool_name,
        validation_errors=validation_errors,
        runtime_provided=runtime_provided,
        runtime_expected={k: "required" for k in expected_keys},
    )

    if run_logger:
        _log_tool_validation_error(run_logger, diagnostic, expected_keys)

    return diagnostic


def _log_tool_validation_error(
    run_logger: Any,
    diagnostic: ToolCallDiagnostic,
    expected_keys: List[str],
) -> None:
    """Log tool validation error with consistent formatting."""
    log_component(run_logger, "DIAGNOSTIC", f"TOOL_VALIDATION_ERROR: {diagnostic.tool_name}")
    for error in diagnostic.validation_errors:
        log_component(run_logger, "DIAGNOSTIC", f"  Error: {error}")
    log_component(run_logger, "DIAGNOSTIC", f"  Provided keys: {list(diagnostic.runtime_provided.keys())}")
    log_component(run_logger, "DIAGNOSTIC", f"  Expected keys: {expected_keys}")

    # Check for specific issues
    if "config" in diagnostic.validation_errors and diagnostic.runtime_provided.get("config") is None:
        log_component(run_logger, "DIAGNOSTIC", "  ISSUE: config is None - need to pass a dict, not None")
    if "stream_writer" in str(diagnostic.validation_errors) and diagnostic.runtime_provided.get("stream_writer") is None:
        log_component(run_logger, "DIAGNOSTIC", "  ISSUE: stream_writer is None - need to pass a callable, not None")


def create_valid_mock_runtime(
    context_engine: Any,
    tool_call_id: str = "mock_call",
) -> Dict[str, Any]:
    """Create a valid mock runtime dict for Context Engine tools.

    This provides a solution for the validation error by creating a properly
    structured runtime dict.

    Args:
        context_engine: The context engine instance
        tool_call_id: Optional tool call ID

    Returns:
        Dict with all required fields properly typed
    """
    return {
        "state": {"_context_engine": context_engine},
        "context": None,  # Must be None per Pydantic validation (none_required)
        "config": {},   # Empty dict instead of None
        "stream_writer": lambda x: None,  # No-op callable instead of None
        "tool_call_id": tool_call_id,
        "store": None,  # store can be None
    }


# =============================================================================
# CRITIC RESPONSE TRACKING
# =============================================================================


def track_critic_response(
    messages: List[Any],
    response_text: str,
    run_logger: Any = None,
) -> CriticResponseDiagnostic:
    """Track critic response for diagnostics.

    This catches the "empty response after 30 messages" issue.

    Args:
        messages: List of messages from the agent
        response_text: The extracted response text
        run_logger: Optional RunLogger for logging

    Returns:
        CriticResponseDiagnostic with the tracking info
    """
    # Count LLM calls and tool calls
    llm_calls = 0
    tool_calls = 0
    last_message_type = "unknown"
    last_message_preview = ""

    for msg in messages:
        msg_type = type(msg).__name__
        if msg_type in ("AIMessage", "LLMChunk", "ChatMessage"):
            llm_calls += 1
            last_message_type = msg_type
            if hasattr(msg, 'content'):
                last_message_preview = str(msg.content)[:200] if msg.content else ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls += len(msg.tool_calls)
        elif msg_type == "ToolMessage":
            tool_calls += 1
            last_message_type = msg_type

    response_length = len(response_text) if response_text else 0
    is_empty = response_length == 0

    diagnostic = CriticResponseDiagnostic(
        num_llm_calls=llm_calls,
        num_tool_calls=tool_calls,
        messages_received=len(messages),
        response_length=response_length,
        last_message_type=last_message_type,
        last_message_preview=last_message_preview,
        is_empty=is_empty,
        extraction_method="last_message",
    )

    if run_logger:
        _log_critic_response(run_logger, diagnostic, messages)

    return diagnostic


def _log_critic_response(
    run_logger: Any,
    diagnostic: CriticResponseDiagnostic,
    messages: List[Any],
) -> None:
    """Log critic response with consistent formatting."""
    if diagnostic.is_empty:
        log_component(run_logger, "DIAGNOSTIC", "CRITIC_EMPTY_RESPONSE!")
        log_component(run_logger, "DIAGNOSTIC", f"  Messages: {len(messages)}, LLM calls: {diagnostic.num_llm_calls}, Tool calls: {diagnostic.num_tool_calls}")
        log_component(run_logger, "DIAGNOSTIC", f"  Last message type: {diagnostic.last_message_type}")
        log_component(run_logger, "DIAGNOSTIC", f"  Last message preview: {diagnostic.last_message_preview}")
        log_component(run_logger, "DIAGNOSTIC", "  ISSUE: Response extraction failing - check message content structure")
    else:
        log_component(run_logger, "CRITIC", f"Response OK: {diagnostic.response_length} chars from {diagnostic.num_llm_calls} LLM calls")


# =============================================================================
# DIAGNOSTIC LOGGER CLASS
# =============================================================================


class DiagnosticLogger:
    """Central diagnostic logging for deep research.

    Collects all diagnostic information during a run and provides
    summary reports for debugging.

    Usage:
        diagnostic = DiagnosticLogger(run_logger)

        # Track issues
        diagnostic.track_llm_response(...)
        diagnostic.track_flash_explorer(...)
        diagnostic.track_tool_error(...)
        diagnostic.track_critic_response(...)

        # Get summary
        summary = diagnostic.get_summary()
    """

    def __init__(self, run_logger: Any = None):
        self.run_logger = run_logger
        self.llm_diagnostics: List[LLMResponseDiagnostic] = []
        self.flash_explorer_diagnostics: List[FlashExplorerDiagnostic] = []
        self.tool_diagnostics: List[ToolCallDiagnostic] = []
        self.critic_diagnostics: List[CriticResponseDiagnostic] = []

    def track_llm_response(
        self,
        original_query: str,
        llm_response: str,
        context: str,
    ) -> Tuple[bool, str]:
        """Track an LLM response and validate relevance."""
        is_relevant, reason, diagnostic = validate_llm_response_relevance(
            original_query=original_query,
            llm_response=llm_response,
            context=context,
            run_logger=self.run_logger,
        )
        self.llm_diagnostics.append(diagnostic)
        return is_relevant, reason

    def track_flash_explorer(
        self,
        avenue: str,
        success: bool,
        elapsed_seconds: float,
        error: Optional[Exception] = None,
        tools_called: Optional[List[str]] = None,
        findings_length: int = 0,
    ) -> FlashExplorerDiagnostic:
        """Track a Flash Explorer result."""
        diagnostic = track_flash_explorer_result(
            avenue=avenue,
            success=success,
            elapsed_seconds=elapsed_seconds,
            run_logger=self.run_logger,
            error=error,
            tools_called=tools_called,
            findings_length=findings_length,
        )
        self.flash_explorer_diagnostics.append(diagnostic)
        return diagnostic

    def track_tool_error(
        self,
        tool_name: str,
        validation_errors: List[str],
        runtime_provided: Dict[str, Any],
    ) -> ToolCallDiagnostic:
        """Track a tool call validation error."""
        diagnostic = log_tool_call_validation_error(
            tool_name=tool_name,
            validation_errors=validation_errors,
            runtime_provided=runtime_provided,
            run_logger=self.run_logger,
        )
        self.tool_diagnostics.append(diagnostic)
        return diagnostic

    def track_critic(
        self,
        messages: List[Any],
        response_text: str,
    ) -> CriticResponseDiagnostic:
        """Track a critic response."""
        diagnostic = track_critic_response(
            messages=messages,
            response_text=response_text,
            run_logger=self.run_logger,
        )
        self.critic_diagnostics.append(diagnostic)
        return diagnostic

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all diagnostics."""
        return {
            "llm_issues": {
                "total": len(self.llm_diagnostics),
                "off_topic": sum(1 for d in self.llm_diagnostics if not d.is_relevant),
                "contexts": list(set(d.context for d in self.llm_diagnostics)),
            },
            "flash_explorer_issues": {
                "total": len(self.flash_explorer_diagnostics),
                "failed": sum(1 for d in self.flash_explorer_diagnostics if not d.success),
                "success_rate": (
                    sum(1 for d in self.flash_explorer_diagnostics if d.success) /
                    len(self.flash_explorer_diagnostics)
                ) if self.flash_explorer_diagnostics else 1.0,
                "error_types": list(set(
                    d.error_type for d in self.flash_explorer_diagnostics
                    if d.error_type
                )),
            },
            "tool_issues": {
                "total": len(self.tool_diagnostics),
                "tools_affected": list(set(d.tool_name for d in self.tool_diagnostics)),
            },
            "critic_issues": {
                "total": len(self.critic_diagnostics),
                "empty_responses": sum(1 for d in self.critic_diagnostics if d.is_empty),
            },
        }

    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        # 0% success rate on Flash Explorers
        if self.flash_explorer_diagnostics:
            success_rate = (
                sum(1 for d in self.flash_explorer_diagnostics if d.success) /
                len(self.flash_explorer_diagnostics)
            )
            if success_rate == 0:
                return True

        # Empty critic responses
        if any(d.is_empty for d in self.critic_diagnostics):
            return True

        # Off-topic LLM responses
        if any(not d.is_relevant for d in self.llm_diagnostics):
            return True

        return False

    def log_summary(self) -> None:
        """Log a summary of all diagnostics."""
        if not self.run_logger:
            return

        summary = self.get_summary()

        log_component(self.run_logger, "DIAGNOSTIC", "=" * 60)
        log_component(self.run_logger, "DIAGNOSTIC", "DIAGNOSTIC SUMMARY")
        log_component(self.run_logger, "DIAGNOSTIC", "=" * 60)

        # LLM issues
        llm = summary["llm_issues"]
        if llm["off_topic"] > 0:
            log_component(self.run_logger, "DIAGNOSTIC", f"LLM OFF-TOPIC RESPONSES: {llm['off_topic']}/{llm['total']}")
            log_component(self.run_logger, "DIAGNOSTIC", f"  Contexts affected: {llm['contexts']}")

        # Flash Explorer issues
        fe = summary["flash_explorer_issues"]
        if fe["failed"] > 0:
            log_component(
                self.run_logger, "DIAGNOSTIC",
                f"FLASH EXPLORER FAILURES: {fe['failed']}/{fe['total']} "
                f"(success rate: {fe['success_rate']:.1%})"
            )
            if fe["error_types"]:
                log_component(self.run_logger, "DIAGNOSTIC", f"  Error types: {fe['error_types']}")

        # Tool issues
        tool = summary["tool_issues"]
        if tool["total"] > 0:
            log_component(self.run_logger, "DIAGNOSTIC", f"TOOL VALIDATION ERRORS: {tool['total']}")
            log_component(self.run_logger, "DIAGNOSTIC", f"  Tools affected: {tool['tools_affected']}")

        # Critic issues
        critic = summary["critic_issues"]
        if critic["empty_responses"] > 0:
            log_component(self.run_logger, "DIAGNOSTIC", f"CRITIC EMPTY RESPONSES: {critic['empty_responses']}/{critic['total']}")

        # Critical issues warning
        if self.has_critical_issues():
            log_component(self.run_logger, "DIAGNOSTIC", "")
            log_component(self.run_logger, "DIAGNOSTIC", "CRITICAL ISSUES DETECTED - Research may have failed!")

        log_component(self.run_logger, "DIAGNOSTIC", "=" * 60)


# =============================================================================
# LANGFUSE INTEGRATION
# =============================================================================


def _emit_langfuse_diagnostic_score(
    session_id: str,
    diagnostic_type: str,
    value: float,
    comment: Optional[str] = None,
) -> bool:
    """Emit a diagnostic score to LangFuse.

    This integrates diagnostic logging with LangFuse observability,
    enabling real-time monitoring of research quality issues.

    Args:
        session_id: Research session ID (maps to LangFuse session_id)
        diagnostic_type: Type of diagnostic (flash_explorer, llm_relevance, tool_validation)
        value: Score value (typically 0.0 for failure, 1.0 for success)
        comment: Optional comment explaining the score

    Returns:
        True if score was emitted successfully, False otherwise
    """
    try:
        from src.extensions.integrations.langfuse.client import get_langfuse_client
        from src.extensions.integrations.langfuse.tracing import get_trace_id_for_session

        client = get_langfuse_client()
        if client is None:
            return False

        # Get trace ID for the session
        trace_id = get_trace_id_for_session(session_id)
        if trace_id is None:
            # Session doesn't have an active trace, skip scoring
            return False

        client.score(
            trace_id=trace_id,
            name=f"diagnostic_{diagnostic_type}",
            value=value,
            comment=comment,
            data_type="NUMERIC",
        )

        logger.debug(
            "langfuse_diagnostic_score_emitted",
            session_id=session_id,
            diagnostic_type=diagnostic_type,
            value=value,
        )

        return True

    except Exception as e:
        logger.warning(
            "langfuse_diagnostic_score_failed",
            session_id=session_id,
            diagnostic_type=diagnostic_type,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


def emit_langfuse_research_summary(
    session_id: str,
    diagnostic_summary: Dict[str, Any],
) -> None:
    """Emit research diagnostic summary scores to LangFuse.

    This should be called at the end of a research run to emit
    aggregate diagnostic scores for the entire run.

    Args:
        session_id: Research session ID
        diagnostic_summary: Summary dict from DiagnosticLogger.get_summary()
    """
    try:
        from src.extensions.integrations.langfuse.client import get_langfuse_client
        from src.extensions.integrations.langfuse.tracing import get_trace_id_for_session

        client = get_langfuse_client()
        if client is None:
            return

        trace_id = get_trace_id_for_session(session_id)
        if trace_id is None:
            return

        # Emit Flash Explorer success rate score
        fe = diagnostic_summary.get("flash_explorer_issues", {})
        if fe.get("total", 0) > 0:
            client.score(
                trace_id=trace_id,
                name="diagnostic_flash_explorer_success_rate",
                value=fe.get("success_rate", 1.0),
                comment=f"{fe.get('failed', 0)}/{fe.get('total', 0)} failed",
                data_type="NUMERIC",
            )

        # Emit LLM relevance score (1.0 if all relevant, 0.0 if any off-topic)
        llm = diagnostic_summary.get("llm_issues", {})
        if llm.get("total", 0) > 0:
            relevance_score = 1.0 if llm.get("off_topic", 0) == 0 else 0.0
            client.score(
                trace_id=trace_id,
                name="diagnostic_llm_relevance",
                value=relevance_score,
                comment=f"{llm.get('off_topic', 0)}/{llm.get('total', 0)} off-topic",
                data_type="NUMERIC",
            )

        # Emit critic response score
        critic = diagnostic_summary.get("critic_issues", {})
        if critic.get("total", 0) > 0:
            critic_score = 1.0 if critic.get("empty_responses", 0) == 0 else 0.0
            client.score(
                trace_id=trace_id,
                name="diagnostic_critic_response",
                value=critic_score,
                comment=f"{critic.get('empty_responses', 0)}/{critic.get('total', 0)} empty",
                data_type="NUMERIC",
            )

        # Emit overall health score
        has_critical = (
            fe.get("success_rate", 1.0) == 0 or
            llm.get("off_topic", 0) > 0 or
            critic.get("empty_responses", 0) > 0
        )
        client.score(
            trace_id=trace_id,
            name="diagnostic_research_health",
            value=0.0 if has_critical else 1.0,
            comment="Critical issues detected" if has_critical else "All checks passed",
            data_type="NUMERIC",
        )

        logger.debug(
            "langfuse_research_summary_emitted",
            session_id=session_id,
            has_critical_issues=has_critical,
        )

    except Exception as e:
        logger.warning(
            "langfuse_research_summary_failed",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )
