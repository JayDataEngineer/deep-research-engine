"""
Storage Backend base classes and mixins for Context Engine.

Provides abstract interfaces and common functionality for all storage backends.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import logging

from ..health_check import HealthCheckResult, HealthStatus
from ..artifacts.base import BaseArtifact

logger = logging.getLogger(__name__)


class StorageCapability(str, Enum):
    """Capabilities that storage backends may support."""
    ADD = "add"
    GET = "get"
    SEARCH = "search"
    DELETE = "delete"
    VECTOR_SEARCH = "vector_search"
    GRAPH_TRAVERSAL = "graph"
    RAW_STORAGE = "raw"


class StorageBackend(ABC):
    """Base class for all Context Engine storage backends.

    Provides:
    - Common interface for CRUD operations
    - Consistent error handling
    - Logging hooks
    - Capability reporting
    - Health check integration
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"storage.{name}")

    @property
    @abstractmethod
    def capabilities(self) -> set[StorageCapability]:
        """Return the set of capabilities this backend supports."""
        pass

    def supports(self, capability: StorageCapability) -> bool:
        """Check if backend supports a capability."""
        return capability in self.capabilities

    async def add(self, artifact: BaseArtifact) -> Tuple[str, bool]:
        """Add an artifact to storage.

        Returns:
            Tuple of (artifact_id, is_new)
        """
        self.logger.debug(f"Adding artifact: {artifact.id}")

        try:
            result = await self._add_internal(artifact)
            self.logger.info(f"Added artifact: {artifact.id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to add artifact {artifact.id}: {e}")
            raise

    async def get(self, artifact_id: str) -> Optional[BaseArtifact]:
        """Get an artifact by ID.

        Returns:
            Artifact if found, None otherwise
        """
        self.logger.debug(f"Getting artifact: {artifact_id}")

        try:
            result = await self._get_internal(artifact_id)
            if result:
                self.logger.debug(f"Retrieved artifact: {artifact_id}")
            else:
                self.logger.warning(f"Artifact not found: {artifact_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get artifact {artifact_id}: {e}")
            raise

    async def search(
        self,
        query: str,
        *,
        k: int = 10,
        session_id: Optional[str] = None,
        artifact_type: Optional[str] = None,
        **filters
    ) -> List[BaseArtifact]:
        """Search for artifacts.

        Args:
            query: Search query (semantic for vector stores, text for others)
            k: Max results to return
            session_id: Filter by session
            artifact_type: Filter by artifact type
            **filters: Additional backend-specific filters

        Returns:
            List of matching artifacts
        """
        self.logger.debug(
            f"Searching: query='{query[:50]}...', k={k}, "
            f"session={session_id}, type={artifact_type}"
        )

        try:
            results = await self._search_internal(
                query=query,
                k=k,
                session_id=session_id,
                artifact_type=artifact_type,
                **filters
            )
            self.logger.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise

    async def delete(self, artifact_id: str) -> bool:
        """Delete an artifact.

        Returns:
            True if deleted, False if not found
        """
        if not self.supports(StorageCapability.DELETE):
            raise NotImplementedError(f"{self.name} does not support deletion")

        self.logger.debug(f"Deleting artifact: {artifact_id}")

        try:
            result = await self._delete_internal(artifact_id)
            if result:
                self.logger.info(f"Deleted artifact: {artifact_id}")
            else:
                self.logger.warning(f"Artifact not found for deletion: {artifact_id}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            raise

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on this backend.

        Returns:
            HealthCheckResult with status and details
        """
        try:
            # Default: try a simple search
            await self._search_internal(query="", k=1, session_id="__health_check__")
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"{self.name} is operational"
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.name} health check failed: {str(e)}"
            )

    # Abstract methods
    @abstractmethod
    async def _add_internal(self, artifact: BaseArtifact) -> Tuple[str, bool]:
        """Internal add implementation."""
        pass

    @abstractmethod
    async def _get_internal(self, artifact_id: str) -> Optional[BaseArtifact]:
        """Internal get implementation."""
        pass

    async def _search_internal(
        self,
        query: str,
        k: int,
        session_id: Optional[str],
        artifact_type: Optional[str],
        **filters
    ) -> List[BaseArtifact]:
        """Internal search implementation."""
        if not self.supports(StorageCapability.SEARCH):
            raise NotImplementedError(f"{self.name} does not support search")
        return []

    async def _delete_internal(self, artifact_id: str) -> bool:
        """Internal delete implementation."""
        raise NotImplementedError(f"{self.name} does not support deletion")


class VectorStorageMixin:
    """Mixin for backends that support vector similarity search."""

    async def vector_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        session_id: Optional[str] = None,
        artifact_type: Optional[str] = None
    ) -> List[Tuple[BaseArtifact, float]]:
        """Search by vector similarity.

        Returns:
            List of (artifact, score) tuples
        """
        raise NotImplementedError("Subclass must implement vector_search")


class SessionScopedMixin:
    """Mixin for backends that support session isolation."""

    def _validate_session_id(
        self,
        session_id: Optional[str],
        mode: str = "research"
    ) -> str:
        """Validate and return session_id.

        Args:
            session_id: Session ID to validate
            mode: 'research' (requires session_id) or 'companion' (optional)

        Returns:
            Validated session_id

        Raises:
            ValueError: If session_id is required but not provided
        """
        if mode == "research" and session_id is None:
            raise ValueError(
                "session_id is REQUIRED for research mode to prevent "
                "cross-session contamination."
            )
        return session_id or ""

    def _build_session_filter(
        self,
        session_id: Optional[str],
        param_index: int = 1
    ) -> Tuple[str, list]:
        """Build SQL WHERE clause for session filtering.

        Returns:
            Tuple of (where_clause, parameters)
        """
        if session_id:
            return f"session_id = ${param_index}", [session_id]
        return "1=1", []


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for deduplication."""
    import hashlib
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
