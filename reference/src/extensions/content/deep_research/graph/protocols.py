"""Protocol-based interfaces for Deep Research components.

This module defines Protocol-based interfaces for key components in the deep research
system. Using protocols provides:
- Type safety without requiring inheritance
- Duck typing validation with mypy
- Clear contracts between components
- Easier testing with mock implementations

CRASH-ONLY: Protocol violations should be caught at development/testing time.
"""

from typing import Protocol, Any, Optional, List, Dict
from typing_extensions import Self
from abc import abstractmethod


# =============================================================================
# CONTEXT ENGINE PROTOCOLS
# =============================================================================

class PostgresVectorStoreProtocol(Protocol):
    """Protocol for Postgres vector store operations.

    Any implementation must provide these methods for semantic search.
    """

    async def search(
        self,
        query: str,
        session_id: str,
        k: int = 10,
        artifact_type: Optional[str] = None,
    ) -> List[Any]:
        """Search for semantically similar artifacts.

        Args:
            query: Search query
            session_id: Session for isolation
            k: Number of results
            artifact_type: Optional type filter

        Returns:
            List of artifact objects

        Raises:
            RuntimeError: If search fails (CRASH-ONLY)
        """
        ...

    async def add(
        self,
        artifacts: List[Any],
        session_id: str,
    ) -> None:
        """Add artifacts to the vector store.

        Args:
            artifacts: Artifacts to add
            session_id: Session for isolation

        Raises:
            RuntimeError: If add fails (CRASH-ONLY)
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if the store is available."""
        ...


class Neo4jStoreProtocol(Protocol):
    """Protocol for Neo4j graph store operations.

    Any implementation must provide these methods for knowledge graph operations.
    """

    async def query(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            List of result records

        Raises:
            RuntimeError: If query fails (CRASH-ONLY)
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if the store is available."""
        ...


class Neo4jVectorStoreProtocol(Protocol):
    """Protocol for Neo4j vector search with entity awareness.

    Combines vector search with graph traversal for entity-aware results.
    """

    async def search_entities(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for entities similar to query with source tracing.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of entity results with:
            - entity_name: str
            - entity_type: str
            - score: float
            - chunk_text: str
            - doc_title: str
            - doc_url: str

        Raises:
            RuntimeError: If search fails (CRASH-ONLY)
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if the store is available."""
        ...


class S3BackendProtocol(Protocol):
    """Protocol for S3/MinIO large file storage.

    Any implementation must provide these methods for artifact storage.
    """

    async def store(
        self,
        data: bytes,
        path: str,
        bucket: str,
    ) -> str:
        """Store data in S3.

        Args:
            data: Data to store
            path: Object path
            bucket: Bucket name

        Returns:
            Full URL to stored object

        Raises:
            RuntimeError: If storage fails (CRASH-ONLY)
        """
        ...

    async def retrieve(
        self,
        path: str,
        bucket: str,
    ) -> bytes:
        """Retrieve data from S3.

        Args:
            path: Object path
            bucket: Bucket name

        Returns:
            Retrieved data

        Raises:
            RuntimeError: If retrieval fails (CRASH-ONLY)
        """
        ...

    @property
    def is_available(self) -> bool:
        """Check if the backend is available."""
        ...


class ContextEngineProtocol(Protocol):
    """Protocol for the complete Context Engine.

    Combines all storage backends into a single interface.
    """

    postgres_store: Optional[PostgresVectorStoreProtocol]
    neo4j_store: Optional[Neo4jStoreProtocol]
    neo4j_vector_store: Optional[Neo4jVectorStoreProtocol]
    s3_backend: Optional[S3BackendProtocol]

    @property
    def is_available(self) -> bool:
        """Check if context engine is properly initialized."""
        return (
            self.postgres_store is not None and
            self.postgres_store.is_available
        )


# =============================================================================
# LLM PROTOCOLS
# =============================================================================

class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers.

    Any LLM implementation must provide this interface.
    """

    async def ainvoke(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Invoke the LLM asynchronously.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
            **kwargs: Additional provider-specific params

        Returns:
            Generated text

        Raises:
            RuntimeError: If invocation fails (CRASH-ONLY)
        """
        ...

    async def ainvoke_structured(
        self,
        prompt: str,
        schema: type,
        **kwargs,
    ) -> Any:
        """Invoke the LLM with structured output.

        Args:
            prompt: The prompt to send
            schema: Pydantic model for output
            **kwargs: Additional params

        Returns:
            Parsed structured output

        Raises:
            RuntimeError: If invocation fails (CRASH-ONLY)
        """
        ...

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...


class LLMFactoryProtocol(Protocol):
    """Protocol for LLM factory.

    Creates LLM instances based on configuration.
    """

    async def create_llm(
        self,
        role: str = "default",
        **kwargs,
    ) -> LLMProviderProtocol:
        """Create an LLM instance for the given role.

        Args:
            role: Role name (maps to config section)
            **kwargs: Override parameters

        Returns:
            LLM provider instance

        Raises:
            RuntimeError: If creation fails (CRASH-ONLY)
        """
        ...


# =============================================================================
# VALIDATOR PROTOCOLS
# =============================================================================

class ValidatorProtocol(Protocol):
    """Protocol for validation components.

    Validators check data against rules and schemas.
    """

    def validate(self, data: Any) -> bool:
        """Validate data against rules.

        Args:
            data: Data to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If invalid (CRASH-ONLY)
        """
        ...

    def validate_and_raise(self, data: Any) -> Any:
        """Validate data, raising detailed error if invalid.

        Args:
            data: Data to validate

        Returns:
            The validated data

        Raises:
            ValidationError: If invalid with details (CRASH-ONLY)
        """
        ...


class ConfigValidatorProtocol(ValidatorProtocol):
    """Protocol for configuration validators.

    Validates YAML configuration before use.
    """

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dict.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If invalid with specific field errors
        """
        ...


# =============================================================================
# NODE PROTOCOLS
# =============================================================================

class ResearchNodeProtocol(Protocol):
    """Protocol for research graph nodes.

    All nodes in the research graph must implement this protocol.
    """

    async def __call__(
        self,
        state: Dict[str, Any],
        config: Any,
    ) -> Dict[str, Any]:
        """Execute the node logic.

        Args:
            state: Current graph state
            config: Runtime configuration

        Returns:
            State updates

        Raises:
            RuntimeError: If node execution fails (CRASH-ONLY)
        """
        ...


# =============================================================================
# CHECKPOINTER PROTOCOLS
# =============================================================================

class CheckpointerProtocol(Protocol):
    """Protocol for checkpoint savers.

    Checkpointers enable graph persistence and resumption.
    """

    async def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
    ) -> None:
        """Store a checkpoint.

        Args:
            config: Thread/config identifier
            checkpoint: Checkpoint data

        Raises:
            RuntimeError: If storage fails (CRASH-ONLY)
        """
        ...

    async def get(
        self,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a checkpoint.

        Args:
            config: Thread/config identifier

        Returns:
            Checkpoint data or None if not found

        Raises:
            RuntimeError: If retrieval fails (CRASH-ONLY)
        """
        ...

    @property
    def is_persistent(self) -> bool:
        """Check if this checkpointer persists to disk."""
        ...


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_context_engine(context_engine: Any) -> ContextEngineProtocol:
    """Validate that an object implements ContextEngineProtocol.

    Args:
        context_engine: Object to validate

    Returns:
        The validated object

    Raises:
        TypeError: If object doesn't implement the protocol
    """
    # Check for required attributes
    if not hasattr(context_engine, 'postgres_store'):
        raise TypeError(
            f"Object does not implement ContextEngineProtocol: "
            f"missing 'postgres_store' attribute"
        )

    if not hasattr(context_engine, 'neo4j_store'):
        raise TypeError(
            f"Object does not implement ContextEngineProtocol: "
            f"missing 'neo4j_store' attribute"
        )

    if not hasattr(context_engine, 's3_backend'):
        raise TypeError(
            f"Object does not implement ContextEngineProtocol: "
            f"missing 's3_backend' attribute"
        )

    return context_engine


def validate_llm_provider(llm: Any) -> LLMProviderProtocol:
    """Validate that an object implements LLMProviderProtocol.

    Args:
        llm: Object to validate

    Returns:
        The validated object

    Raises:
        TypeError: If object doesn't implement the protocol
    """
    # Check for required methods
    if not hasattr(llm, 'ainvoke'):
        raise TypeError(
            f"Object does not implement LLMProviderProtocol: "
            f"missing 'ainvoke' method"
        )

    return llm


def validate_checkpointer(checkpointer: Any) -> CheckpointerProtocol:
    """Validate that an object implements CheckpointerProtocol.

    Args:
        checkpointer: Object to validate

    Returns:
        The validated object

    Raises:
        TypeError: If object doesn't implement the protocol
    """
    # Check for required methods
    if not hasattr(checkpointer, 'put'):
        raise TypeError(
            f"Object does not implement CheckpointerProtocol: "
            f"missing 'put' method"
        )

    if not hasattr(checkpointer, 'get'):
        raise TypeError(
            f"Object does not implement CheckpointerProtocol: "
            f"missing 'get' method"
        )

    return checkpointer
