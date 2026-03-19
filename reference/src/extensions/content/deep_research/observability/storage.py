"""Unified storage interface for observability data.

This module provides a clean abstraction over storage backends:
- Context Engine (MinIO/S3) as primary storage
- Local filesystem as fallback for debugging/offline mode

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    StorageManager                            │
    │  - Primary: Context Engine (MinIO + PostgreSQL)             │
    │  - Fallback: Local files (optional, for debugging)          │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from .storage import StorageManager, StorageConfig

    config = StorageConfig(primary_backend="context_engine")
    storage = StorageManager(session_id="abc-123", config=config)

    # Save to primary (with optional fallback)
    storage.save("logs/app.log", "Log content")

    # Load from primary (falls back to local if primary fails)
    content = storage.load("logs/app.log")

    # Check health
    health = storage.health_check()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import os
import json

from ..logging_utils import safe_get_logger

logger = safe_get_logger(__name__, "storage")


# =============================================================================
# PROTOCOLS
# =============================================================================

@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends.

    Any storage backend must implement these methods to be used
    with the StorageManager.
    """

    def write(self, path: str, content: str) -> bool:
        """Write content to path.

        Args:
            path: Storage path (e.g., "logs/app.log")
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        ...

    def read(self, path: str) -> Optional[str]:
        """Read content from path.

        Args:
            path: Storage path

        Returns:
            Content string or None if not found
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Storage path

        Returns:
            True if exists, False otherwise
        """
        ...

    def delete(self, path: str) -> bool:
        """Delete content at path.

        Args:
            path: Storage path

        Returns:
            True if deleted, False if not found or error
        """
        ...

    def list_files(self, prefix: str = "") -> List[str]:
        """List files with given prefix.

        Args:
            prefix: Path prefix to filter

        Returns:
            List of file paths
        """
        ...


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StorageConfig:
    """Configuration for storage behavior.

    Attributes:
        primary_backend: Which backend to use as primary ("context_engine" or "local")
        fallback_to_local: Whether to use local files as fallback when primary fails
        local_base_dir: Base directory for local storage (defaults to data/outputs/logs)
        context_engine_bucket: MinIO bucket name for Context Engine
        fail_on_primary_error: If True, raise errors from primary (no fallback)
    """
    primary_backend: str = "context_engine"
    fallback_to_local: bool = True
    local_base_dir: Optional[Path] = None
    context_engine_bucket: str = "context-engine"
    fail_on_primary_error: bool = False

    def __post_init__(self):
        if self.local_base_dir is None:
            self.local_base_dir = Path("data/outputs/logs/deep_research")


# =============================================================================
# BACKEND IMPLEMENTATIONS
# =============================================================================

class ContextEngineBackend:
    """Storage backend using Context Engine's S3Backend (MinIO).

    This backend stores all data in MinIO via the S3Backend class
    from the context_engine extension. All paths are prefixed with
    /logs/{session_id}/ for isolation.

    Paths are translated to S3 keys:
        "logs/app.log" -> "/logs/{session_id}/app.log"
        "metrics.json" -> "/logs/{session_id}/metrics.json"
    """

    def __init__(self, session_id: str, bucket: str = "context-engine"):
        """Initialize Context Engine backend.

        Args:
            session_id: Research session ID for path isolation
            bucket: MinIO bucket name
        """
        self.session_id = session_id
        self.bucket = bucket
        self._backend = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize S3Backend."""
        try:
            from src.extensions.features.tools.context_engine.storage.s3_backend import S3Backend

            self._backend = S3Backend(
                bucket=self.bucket,
                endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:8899/minio"),
                aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin")),
                aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")),
            )
            logger.debug(f"[STORAGE] Context Engine backend initialized for session: {self.session_id}")
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to initialize Context Engine backend: {e}")
            self._backend = None

    def _get_s3_path(self, path: str) -> str:
        """Convert relative path to S3 path with session isolation."""
        # Ensure path starts with /logs/{session_id}/
        path = path.lstrip("/")
        if not path.startswith("logs/"):
            path = f"logs/{path}"
        return f"/{path}".replace("logs/", f"logs/{self.session_id}/", 1)

    @property
    def available(self) -> bool:
        """Check if backend is available."""
        return self._backend is not None

    def write(self, path: str, content: str) -> bool:
        """Write content to S3."""
        if not self._backend:
            return False

        try:
            s3_path = self._get_s3_path(path)
            self._backend.write(s3_path, content)
            logger.debug(f"[STORAGE] Wrote to Context Engine: {s3_path}")
            return True
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to write to Context Engine: {e}")
            return False

    def read(self, path: str) -> Optional[str]:
        """Read content from S3."""
        if not self._backend:
            return None

        try:
            s3_path = self._get_s3_path(path)
            content = self._backend.read(s3_path)
            return content
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to read from Context Engine: {e}")
            return None

    def exists(self, path: str) -> bool:
        """Check if path exists in S3."""
        if not self._backend:
            return False

        try:
            s3_path = self._get_s3_path(path)
            return self._backend.exists(s3_path)
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to check existence in Context Engine: {e}")
            return False

    def delete(self, path: str) -> bool:
        """Delete content from S3."""
        if not self._backend:
            return False

        try:
            s3_path = self._get_s3_path(path)
            self._backend.delete(s3_path)
            return True
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to delete from Context Engine: {e}")
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        """List files with prefix in S3."""
        if not self._backend:
            return []

        try:
            s3_prefix = self._get_s3_path(prefix)
            return self._backend.list(s3_prefix)
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to list files in Context Engine: {e}")
            return []


class LocalFileBackend:
    """Storage backend using local filesystem.

    This is primarily used as a fallback when Context Engine is unavailable,
    or for debugging purposes. Files are stored in a local directory.

    Paths are translated to filesystem paths:
        "logs/app.log" -> {base_dir}/{session_id}/logs/app.log
    """

    def __init__(self, base_dir: Path, session_id: str):
        """Initialize local file backend.

        Args:
            base_dir: Base directory for all logs
            session_id: Research session ID for path isolation
        """
        self.base_dir = Path(base_dir) / session_id
        self.session_id = session_id
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the session directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_fs_path(self, path: str) -> Path:
        """Convert relative path to filesystem path."""
        path = path.lstrip("/")
        return self.base_dir / path

    def write(self, path: str, content: str) -> bool:
        """Write content to local file."""
        try:
            fs_path = self._get_fs_path(path)
            fs_path.parent.mkdir(parents=True, exist_ok=True)
            fs_path.write_text(content, encoding="utf-8")
            logger.debug(f"[STORAGE] Wrote to local file: {fs_path}")
            return True
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to write to local file: {e}")
            return False

    def read(self, path: str) -> Optional[str]:
        """Read content from local file."""
        try:
            fs_path = self._get_fs_path(path)
            if fs_path.exists():
                return fs_path.read_text(encoding="utf-8")
            return None
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to read from local file: {e}")
            return None

    def exists(self, path: str) -> bool:
        """Check if path exists in filesystem."""
        fs_path = self._get_fs_path(path)
        return fs_path.exists()

    def delete(self, path: str) -> bool:
        """Delete content from filesystem."""
        try:
            fs_path = self._get_fs_path(path)
            if fs_path.exists():
                fs_path.unlink()
                return True
            return False
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to delete from local file: {e}")
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        """List files with prefix in filesystem."""
        try:
            fs_prefix = self._get_fs_path(prefix)
            if not fs_prefix.parent.exists():
                return []

            files = []
            for f in fs_prefix.parent.glob(f"{fs_prefix.name}*" if prefix else "*"):
                if f.is_file():
                    files.append(str(f.relative_to(self.base_dir)))
            return files
        except Exception as e:
            logger.warning(f"[STORAGE] Failed to list local files: {e}")
            return []


# =============================================================================
# STORAGE MANAGER
# =============================================================================

class StorageManager:
    """Unified storage interface with primary/fallback backends.

    This class provides a single interface for all storage operations,
    automatically handling fallback from primary to secondary storage
    when needed.

    Usage:
        config = StorageConfig(primary_backend="context_engine", fallback_to_local=True)
        storage = StorageManager(session_id="abc-123", config=config)

        # Save (tries primary, falls back to local)
        storage.save("metrics.json", json_content)

        # Load (tries primary first, then local)
        content = storage.load("metrics.json")

        # Check health
        health = storage.health_check()
    """

    def __init__(self, session_id: str, config: Optional[StorageConfig] = None):
        """Initialize storage manager.

        Args:
            session_id: Research session ID for path isolation
            config: Storage configuration (uses defaults if not provided)
        """
        self.session_id = session_id
        self.config = config or StorageConfig()

        # Initialize backends
        self._primary: Optional[StorageBackend] = None
        self._fallback: Optional[StorageBackend] = None
        self._fallback_count = 0

        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize primary and fallback backends."""
        # Initialize primary backend
        if self.config.primary_backend == "context_engine":
            self._primary = ContextEngineBackend(
                session_id=self.session_id,
                bucket=self.config.context_engine_bucket,
            )
            # Check if it's actually available
            if not self._primary.available:
                logger.warning("[STORAGE] Context Engine unavailable, will use local fallback")
                self._primary = None
        else:
            # Use local as primary
            self._primary = LocalFileBackend(
                base_dir=self.config.local_base_dir,
                session_id=self.session_id,
            )

        # Initialize fallback backend if enabled
        if self.config.fallback_to_local and self.config.primary_backend == "context_engine":
            self._fallback = LocalFileBackend(
                base_dir=self.config.local_base_dir,
                session_id=self.session_id,
            )

    @property
    def primary(self) -> Optional[StorageBackend]:
        """Get primary backend."""
        return self._primary

    @property
    def fallback(self) -> Optional[StorageBackend]:
        """Get fallback backend."""
        return self._fallback

    @property
    def fallback_count(self) -> int:
        """Get number of times fallback was used."""
        return self._fallback_count

    def save(self, path: str, content: str, also_to_fallback: bool = False) -> bool:
        """Save content to storage.

        Args:
            path: Storage path (e.g., "logs/app.log", "metrics.json")
            content: Content to save
            also_to_fallback: If True, also save to fallback (for redundancy)

        Returns:
            True if saved successfully to at least one backend
        """
        success = False

        # Try primary first
        if self._primary:
            try:
                if self._primary.write(path, content):
                    success = True
                    logger.debug(f"[STORAGE] Saved to primary: {path}")
            except Exception as e:
                logger.warning(f"[STORAGE] Primary save failed for {path}: {e}")

        # If primary failed or we want redundancy, try fallback
        if (not success or also_to_fallback) and self._fallback:
            try:
                if self._fallback.write(path, content):
                    if not success:
                        self._fallback_count += 1
                        logger.info(f"[STORAGE] Saved to fallback (primary unavailable): {path}")
                    success = True
            except Exception as e:
                logger.warning(f"[STORAGE] Fallback save failed for {path}: {e}")

        # If configured to fail on primary error and primary exists but failed
        if not success and self.config.fail_on_primary_error and self._primary:
            raise RuntimeError(
                f"╔═══════════════════════════════════════════════════════════════════╗\n"
                f"║              STORAGE SAVE FAILED - PRIMARY ERROR                    ║\n"
                f"╚═══════════════════════════════════════════════════════════════════╝\n"
                f"\n"
                f"Path: {path}\n"
                f"Session: {self.session_id}\n"
                f"Primary backend: {self.config.primary_backend}\n"
            )

        return success

    def load(self, path: str) -> Optional[str]:
        """Load content from storage.

        Tries primary first, then fallback if available.

        Args:
            path: Storage path

        Returns:
            Content string or None if not found in any backend
        """
        # Try primary first
        if self._primary:
            content = self._primary.read(path)
            if content is not None:
                return content

        # Try fallback
        if self._fallback:
            content = self._fallback.read(path)
            if content is not None:
                self._fallback_count += 1
                logger.debug(f"[STORAGE] Loaded from fallback: {path}")
                return content

        return None

    def exists(self, path: str) -> bool:
        """Check if content exists in any backend.

        Args:
            path: Storage path

        Returns:
            True if exists in primary or fallback
        """
        if self._primary and self._primary.exists(path):
            return True
        if self._fallback and self._fallback.exists(path):
            return True
        return False

    def delete(self, path: str) -> bool:
        """Delete content from all backends.

        Args:
            path: Storage path

        Returns:
            True if deleted from at least one backend
        """
        success = False
        if self._primary and self._primary.delete(path):
            success = True
        if self._fallback and self._fallback.delete(path):
            success = True
        return success

    def list_files(self, prefix: str = "") -> List[str]:
        """List files with prefix from all backends.

        Args:
            prefix: Path prefix to filter

        Returns:
            Combined list of files from all backends
        """
        files = set()
        if self._primary:
            files.update(self._primary.list_files(prefix))
        if self._fallback:
            files.update(self._fallback.list_files(prefix))
        return sorted(list(files))

    def save_json(self, path: str, data: Dict[str, Any]) -> bool:
        """Save JSON data to storage.

        Args:
            path: Storage path (should end with .json)
            data: Dictionary to save as JSON

        Returns:
            True if saved successfully
        """
        content = json.dumps(data, indent=2, default=str)
        return self.save(path, content)

    def load_json(self, path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data from storage.

        Args:
            path: Storage path

        Returns:
            Parsed JSON dict or None if not found/invalid
        """
        content = self.load(path)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"[STORAGE] Failed to parse JSON from {path}: {e}")
        return None

    def health_check(self) -> Dict[str, Any]:
        """Check health of storage backends.

        Returns:
            Dict with health status of each backend
        """
        health = {
            "session_id": self.session_id,
            "primary_backend": self.config.primary_backend,
            "primary_available": self._primary is not None,
            "fallback_available": self._fallback is not None,
            "fallback_count": self._fallback_count,
            "status": "healthy",
        }

        # Determine overall status
        if not self._primary and not self._fallback:
            health["status"] = "unhealthy"
            health["message"] = "No storage backends available"
        elif not self._primary:
            health["status"] = "degraded"
            health["message"] = "Primary backend unavailable, using fallback"
        elif self._fallback_count > 10:
            health["status"] = "degraded"
            health["message"] = f"High fallback usage ({self._fallback_count} operations)"

        return health


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_storage_manager(
    session_id: str,
    primary_backend: str = "context_engine",
    fallback_to_local: bool = True,
    local_base_dir: Optional[Path] = None,
) -> StorageManager:
    """Create a StorageManager with common configuration.

    Args:
        session_id: Research session ID
        primary_backend: Primary backend type ("context_engine" or "local")
        fallback_to_local: Whether to use local files as fallback
        local_base_dir: Base directory for local storage

    Returns:
        Configured StorageManager instance
    """
    config = StorageConfig(
        primary_backend=primary_backend,
        fallback_to_local=fallback_to_local,
        local_base_dir=local_base_dir,
    )
    return StorageManager(session_id=session_id, config=config)
