"""Centralized exception hierarchy for deep research.

This module provides a SINGLE SOURCE OF TRUTH for all exceptions
in the deep research system. All exceptions inherit from
DeepResearchError to enable consistent error handling.

Usage:
    from deep_research.exceptions import (
        DeepResearchError,
        StorageError,
        NodeExecutionError,
        ConfigValidationError,
    )
    
    try:
        await save_worker_finding(...)
    except StorageError as e:
        logger.error(f"Storage failed: {e}")
        raise
"""


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class DeepResearchError(Exception):
    """Base exception for all deep research errors.
    
    All deep research exceptions inherit from this class to enable
    consistent error handling and catching.
    
    Attributes:
        message: Human-readable error message
        session_id: Optional session ID for context
        details: Optional additional error details
    """
    
    def __init__(
        self,
        message: str,
        session_id: str = None,
        details: dict = None,
    ):
        self.message = message
        self.session_id = session_id
        self.details = details or {}
        
        # Build informative error message
        error_msg = message
        if session_id:
            error_msg = f"[Session: {session_id}] {error_msg}"
        if details:
            error_msg += f" | Details: {details}"
        
        super().__init__(error_msg)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "session_id": self.session_id,
            "details": self.details,
        }


# =============================================================================
# STORAGE EXCEPTIONS
# =============================================================================

class StorageError(DeepResearchError):
    """Raised when storage operations fail.
    
    Used for:
    - Failed save/load operations
    - Database connection errors
    - Namespace construction errors
    
    Example:
        try:
            await save_worker_finding(store, session_id, worker_id, content)
        except StorageError as e:
            logger.error(f"Failed to save finding: {e}")
            raise NodeExecutionError(f"Storage failure: {e}") from e
    """
    pass


class DuplicateContentError(StorageError):
    """Raised when duplicate content is detected and rejection is configured.
    
    Used for:
    - Strict deduplication mode
    - Content integrity checks
    
    Example:
        if await is_duplicate(store, namespace, content):
            raise DuplicateContentError(
                f"Duplicate content detected in {namespace}",
                details={"content_hash": compute_content_hash(content)}
            )
    """
    pass


class NamespaceError(StorageError):
    """Raised when namespace construction fails.
    
    Used for:
    - Invalid namespace format
    - Missing required namespace components
    
    Example:
        if not session_id:
            raise NamespaceError(
                "session_id is required for namespace construction"
            )
    """
    pass


class ArtifactNotFoundError(StorageError):
    """Raised when requested artifact is not found.
    
    Used for:
    - Missing research briefs
    - Missing synthesis results
    - Missing worker findings
    
    Example:
        brief = await load_research_brief(store, session_id)
        if brief is None:
            raise ArtifactNotFoundError(
                f"Research brief not found for session {session_id}"
            )
    """
    pass


# =============================================================================
# NODE EXECUTION EXCEPTIONS
# =============================================================================

class NodeExecutionError(DeepResearchError):
    """Raised when node execution fails.
    
    Used for:
    - Boss node failures
    - Critic node failures
    - Synthesis node failures
    - Architect node failures
    
    Example:
        try:
            result = await boss_node(state)
        except Exception as e:
            raise NodeExecutionError(
                f"Boss node failed: {e}",
                session_id=state.get("session_id"),
                details={"node": "boss", "phase": state.get("current_phase")}
            ) from e
    """
    pass


class BossDecisionError(NodeExecutionError):
    """Raised when boss decision-making fails.
    
    Used for:
    - Failed research strategy decisions
    - Worker spawning failures
    - Synthesis handoff failures
    
    Example:
        if not decision.get("action"):
            raise BossDecisionError(
                "Boss failed to make a decision",
                session_id=session_id,
                details={"state": state}
            )
    """
    pass


class CriticEvaluationError(NodeExecutionError):
    """Raised when critic evaluation fails.
    
    Used for:
    - Failed quality assessments
    - Feedback generation failures
    
    Example:
        if not critique.get("passed"):
            # This is OK - critic can reject
            pass
        else:
            raise CriticEvaluationError(
                "Critic node failed to produce evaluation"
            )
    """
    pass


class SynthesisError(NodeExecutionError):
    """Raised when synthesis generation fails.
    
    Used for:
    - Content generation failures
    - Formatting failures
    - Citation resolution failures
    
    Example:
        try:
            synthesis = await generate_synthesis(findings)
        except Exception as e:
            raise SynthesisError(
                f"Synthesis generation failed: {e}",
                session_id=session_id
            ) from e
    """
    pass


class ArchitectError(NodeExecutionError):
    """Raised when architect node fails.
    
    Used for:
    - Research planning failures
    - Worker spawning failures
    
    Example:
        if not architect_result.get("findings"):
            raise ArchitectError(
                "Architect produced no findings",
                session_id=session_id
            )
    """
    pass


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigValidationError(DeepResearchError):
    """Raised when configuration validation fails.
    
    Used for:
    - Invalid YAML config
    - Missing required fields
    - Invalid field values
    
    Example:
        if "workers" not in config:
            raise ConfigValidationError(
                "Config missing required 'workers' field",
                details={"config_keys": list(config.keys())}
            )
    """
    pass


class ConfigValueError(ConfigValidationError):
    """Raised when configuration value is invalid.
    
    Used for:
    - Invalid model names
    - Invalid temperature values
    - Invalid worker configurations
    
    Example:
        if temperature < 0 or temperature > 2:
            raise ConfigValueError(
                f"Invalid temperature: {temperature} (must be 0-2)",
                details={"field": "temperature", "value": temperature}
            )
    """
    pass


class ConfigMissingError(ConfigValidationError):
    """Raised when required configuration is missing.
    
    Used for:
    - Missing environment variables
    - Missing required config fields
    
    Example:
        if not os.getenv("POSTGRES_URL"):
            raise ConfigMissingError(
                "POSTGRES_URL environment variable not set",
                details={"variable": "POSTGRES_URL"}
            )
    """
    pass


# =============================================================================
# INFRASTRUCTURE EXCEPTIONS
# =============================================================================

class InfrastructureError(DeepResearchError):
    """Raised when infrastructure is unavailable.
    
    Used for:
    - PostgreSQL connection failures
    - Neo4j connection failures
    - MinIO/S3 connection failures
    - MCP service failures
    
    Example:
        try:
            await conn.execute("SELECT 1")
        except Exception as e:
            raise InfrastructureError(
                f"PostgreSQL unavailable: {e}",
                details={"service": "postgres"}
            ) from e
    """
    pass


class PostgresError(InfrastructureError):
    """Raised when PostgreSQL operations fail.
    
    Used for:
    - Connection failures
    - Query errors
    - Transaction failures
    
    Example:
        try:
            await store.aput(namespace, key, value)
        except Exception as e:
            raise PostgresError(
                f"Postgres operation failed: {e}",
                session_id=session_id
            ) from e
    """
    pass


class Neo4jError(InfrastructureError):
    """Raised when Neo4j operations fail.
    
    Used for:
    - Connection failures
    - Query errors
    - Graph operation failures
    """
    pass


class MinIOError(InfrastructureError):
    """Raised when MinIO/S3 operations fail.
    
    Used for:
    - Connection failures
    - Upload/download errors
    - Permission errors
    """
    pass


class MCPError(InfrastructureError):
    """Raised when MCP service operations fail.
    
    Used for:
    - Docker service failures
    - Tool execution failures
    
    Example:
        if not docker_result["success"]:
            raise MCPError(
                f"MCP service failed: {docker_result.get('error')}",
                details={"service": "lang-tools"}
            )
    """
    pass


# =============================================================================
# TOOL EXCEPTIONS
# =============================================================================

class ToolError(DeepResearchError):
    """Raised when tool execution fails.
    
    Used for:
    - Web search failures
    - API call failures
    - Tool timeout errors
    
    Example:
        try:
            result = await tavily_search.ainvoke(query)
        except Exception as e:
            raise ToolError(
                f"Tool execution failed: {e}",
                details={"tool": "tavily_search", "query": query}
            ) from e
    """
    pass


class ToolTimeoutError(ToolError):
    """Raised when tool execution times out.
    
    Used for:
    - Long-running web searches
    - Slow API calls
    
    Example:
        try:
            result = await asyncio.wait_for(
                tool.ainvoke(input),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise ToolTimeoutError(
                f"Tool timed out after 30s",
                details={"tool": tool.name}
            )
    """
    pass


class ToolValidationError(ToolError):
    """Raised when tool input validation fails.
    
    Used for:
    - Invalid tool arguments
    - Missing required parameters
    
    Example:
        if not query:
            raise ToolValidationError(
                "Query is required for search",
                details={"tool": "web_search"}
            )
    """
    pass


# =============================================================================
# OBSERVABILITY EXCEPTIONS
# =============================================================================

class ObservabilityError(DeepResearchError):
    """Raised when observability operations fail.
    
    Used for:
    - Logging failures
    - Metrics collection failures
    - Tracing failures
    
    Example:
        try:
            await run_logger.log_event("research_started")
        except Exception as e:
            raise ObservabilityError(
                f"Failed to log event: {e}"
            ) from e
    """
    pass


class CheckpointerError(ObservabilityError):
    """Raised when checkpointer operations fail.
    
    Used for:
    - Checkpoint save failures
    - Checkpoint recovery failures
    
    Example:
        if not checkpointer:
            raise CheckpointerError(
                "Checkpointer not initialized"
            )
    """
    pass


class MetricsError(ObservabilityError):
    """Raised when metrics collection fails.
    
    Used for:
    - Metric recording failures
    - Performance tracking failures
    """
    pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base exception
    "DeepResearchError",
    # Storage exceptions
    "StorageError",
    "DuplicateContentError",
    "NamespaceError",
    "ArtifactNotFoundError",
    # Node execution exceptions
    "NodeExecutionError",
    "BossDecisionError",
    "CriticEvaluationError",
    "SynthesisError",
    "ArchitectError",
    # Configuration exceptions
    "ConfigValidationError",
    "ConfigValueError",
    "ConfigMissingError",
    # Infrastructure exceptions
    "InfrastructureError",
    "PostgresError",
    "Neo4jError",
    "MinIOError",
    "MCPError",
    # Tool exceptions
    "ToolError",
    "ToolTimeoutError",
    "ToolValidationError",
    # Observability exceptions
    "ObservabilityError",
    "CheckpointerError",
    "MetricsError",
]
