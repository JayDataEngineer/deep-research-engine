"""MinIO Backend for DeepAgents filesystem tools.

This module implements the BackendProtocol from deepagents, allowing
all filesystem tools (read_file, write_file, edit_file, ls, glob, grep)
to work with MinIO storage.

This enables:
- Deep agents context dumping to MinIO
- Web session data storage with unified interface
- Context dumps for state management
- All filesystem tools work across both ephemeral state and MinIO

Usage:
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, StateBackend
    from src.extensions.features.tools.context_engine.storage.minio_backend import MinioBackend

    def make_backend(runtime):
        return CompositeBackend(
            default=StateBackend(runtime),  # Ephemeral state
            routes={
                "/web/": MinioBackend(prefix="web/session", runtime=runtime),
                "/dumps/": MinioBackend(prefix="dumps/session", runtime=runtime),
            }
        )

    agent = create_deep_agent(
        model=llm,
        backend=make_backend,
        tools=[web_scrape_tool]
    )
"""

from __future__ import annotations

import abc
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES (matching deepagents BackendProtocol)
# =============================================================================

@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    name: str
    size: int
    modified_at: str
    is_dir: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "modified_at": self.modified_at,
            "is_dir": self.is_dir,
        }


@dataclass
class WriteResult:
    """Result of a write operation."""
    path: str
    error: Optional[str] = None
    files_update: Optional[dict] = None  # For state backends

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "error": self.error,
            "files_update": self.files_update,
        }


@dataclass
class EditResult:
    """Result of an edit operation."""
    path: str
    error: Optional[str] = None
    files_update: Optional[dict] = None
    occurrences: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "error": self.error,
            "files_update": self.files_update,
            "occurrences": self.occurrences,
        }


@dataclass
class GrepMatch:
    """A single grep match."""
    path: str
    line_number: int
    line: str
    match_start: int
    match_end: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "line_number": self.line_number,
            "line": self.line,
            "match_start": self.match_start,
            "match_end": self.match_end,
        }


@dataclass
class FileUploadResponse:
    """Response for file upload."""
    path: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"path": self.path, "error": self.error}


@dataclass
class FileDownloadResponse:
    """Response for file download."""
    path: str
    content: bytes
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "content_size": len(self.content),
            "error": self.error,
        }


# =============================================================================
# MINIO BACKEND IMPLEMENTATION
# =============================================================================

class MinioBackend:
    """MinIO backend implementing deepagents BackendProtocol.

    This backend routes all filesystem operations to MinIO, enabling:
    - read_file, write_file, edit_file for MinIO storage
    - ls, glob, grep for MinIO objects
    - Session-based folder isolation

    Path Structure:
        /{prefix}/{session_id}/{file_path}

    Example:
        prefix="web/session", session_id="abc123"
        write_file("/example.html", content) → /web/session/abc123/example.html
    """

    def __init__(
        self,
        prefix: str = "",
        runtime: Optional["ToolRuntime"] = None,
        session_id: Optional[str] = None,
        bucket: Optional[str] = None,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """Initialize MinIO backend.

        Args:
            prefix: Path prefix (e.g., "web/session", "dumps/session")
            runtime: ToolRuntime for accessing session_id
            session_id: Explicit session_id (overrides runtime)
            bucket: MinIO bucket name (default: context-engine)
            endpoint: MinIO endpoint URL
            access_key: MinIO access key
            secret_key: MinIO secret key
        """
        self.prefix = prefix.strip("/")
        self.runtime = runtime
        self._session_id = session_id

        # MinIO configuration
        self.bucket = bucket or os.getenv("MINIO_BUCKET", "context-engine")
        self.endpoint = endpoint or os.getenv(
            "MINIO_ENDPOINT",
            os.getenv("MINIO_ENDPOINT_URL", "http://localhost:8899/minio")
        )
        self.access_key = access_key or os.getenv(
            "MINIO_ACCESS_KEY",
            os.getenv("MINIO_ROOT_USER", "minioadmin")
        )
        self.secret_key = secret_key or os.getenv(
            "MINIO_SECRET_KEY",
            os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
        )

        # Lazy-load S3Backend
        self._s3_backend = None

    @property
    def session_id(self) -> str:
        """Get session_id from runtime or explicit setting."""
        if self._session_id:
            return self._session_id

        if self.runtime:
            # Try to get from runtime config (LangGraph thread_id)
            config = getattr(self.runtime, "config", {})
            if isinstance(config, dict):
                configurable = config.get("configurable", {})
                thread_id = configurable.get("thread_id")
                if thread_id:
                    return thread_id

            # Try to get from runtime state
            state = getattr(self.runtime, "state", {})
            if isinstance(state, dict):
                sid = state.get("session_id")
                if sid:
                    return sid

        # Fallback to global
        return "global"

    @property
    def s3_backend(self):
        """Lazy-load S3Backend."""
        if self._s3_backend is None:
            from .s3_backend import S3Backend
            self._s3_backend = S3Backend(
                bucket=self.bucket,
                endpoint_url=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
            )
        return self._s3_backend

    def _make_path(self, file_path: str) -> str:
        """Construct full MinIO path with prefix and session_id.

        Args:
            file_path: Relative file path from tool

        Returns:
            Full MinIO path: /{prefix}/{session_id}/{file_path}
        """
        # Ensure path starts with /
        path = file_path.lstrip("/")

        # Build full path
        if self.prefix:
            if self.session_id and self.session_id != "global":
                return f"/{self.prefix}/{self.session_id}/{path}"
            return f"/{self.prefix}/{path}"

        return f"/{path}"

    def _strip_prefix(self, full_path: str) -> str:
        """Remove prefix from path for display."""
        if self.prefix:
            prefix_path = f"/{self.prefix}/"
            if full_path.startswith(prefix_path):
                return full_path[len(prefix_path):]
        return full_path

    # =========================================================================
    # SYNC OPERATIONS (required by BackendProtocol)
    # =========================================================================

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        """List files with metadata.

        Args:
            path: Directory path to list

        Returns:
            List of FileInfo objects
        """
        full_path = self._make_path(path)
        files = self.s3_backend.ls(full_path)

        results = []
        for f in files:
            file_path = f.get("name", f.get("path", ""))
            display_path = self._strip_prefix(file_path)

            # Check if it's a "directory" (ends with /)
            is_dir = file_path.endswith("/")

            results.append(FileInfo(
                path=display_path,
                name=file_path.split("/")[-1] or file_path.split("/")[-2] + "/",
                size=f.get("size", 0),
                modified_at=str(f.get("last_modified", f.get("modified_at", ""))),
                is_dir=is_dir,
            ))

        return results

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Path to file
            offset: Starting line number (0-indexed)
            limit: Maximum lines to return

        Returns:
            Formatted string with line numbers (like cat -n)
        """
        full_path = self._make_path(file_path)

        try:
            content = self.s3_backend.read(full_path)
            lines = content.split("\n")

            # Apply offset and limit
            selected_lines = lines[offset:offset + limit]

            # Format with line numbers (like cat -n)
            result_lines = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                result_lines.append(f"{i:6d}\t{line}")

            return "\n".join(result_lines)

        except FileNotFoundError:
            return f"Error: File not found: {file_path}"

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content to file.

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            WriteResult with path and optional error
        """
        full_path = self._make_path(file_path)

        try:
            result = self.s3_backend.write(full_path, content)

            if "error" in result:
                return WriteResult(path=file_path, error=result["error"])

            logger.info(f"[MinioBackend] Wrote {len(content)} bytes to {full_path}")
            return WriteResult(path=file_path)

        except Exception as e:
            logger.error(f"[MinioBackend] Write failed: {e}")
            return WriteResult(path=file_path, error=str(e))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> EditResult:
        """Edit file by replacing string.

        Args:
            file_path: Path to file
            old_string: String to replace
            new_string: Replacement string
            replace_all: Replace all occurrences

        Returns:
            EditResult with occurrences count
        """
        full_path = self._make_path(file_path)

        try:
            content = self.s3_backend.read(full_path)

            if old_string not in content:
                return EditResult(path=file_path, error="String not found")

            if replace_all:
                new_content = content.replace(old_string, new_string)
                occurrences = content.count(old_string)
            else:
                if content.count(old_string) > 1:
                    return EditResult(
                        path=file_path,
                        error="Multiple occurrences found. Use replace_all=True"
                    )
                new_content = content.replace(old_string, new_string, 1)
                occurrences = 1

            result = self.s3_backend.write(full_path, new_content)

            if "error" in result:
                return EditResult(path=file_path, error=result["error"])

            return EditResult(path=file_path, occurrences=occurrences)

        except FileNotFoundError:
            return EditResult(path=file_path, error="File not found")
        except Exception as e:
            return EditResult(path=file_path, error=str(e))

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None
    ) -> Union[list[GrepMatch], str]:
        """Search for pattern in files.

        Args:
            pattern: Regex pattern to search
            path: Directory to search (default: /)
            glob: File pattern to match (e.g., "*.html")

        Returns:
            List of GrepMatch objects or error string
        """
        search_path = self._make_path(path or "/")

        try:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = []

            # Get files to search
            files = self.s3_backend.ls(search_path)

            for f in files:
                file_path = f.get("name", f.get("path", ""))

                # Apply glob filter
                if glob and not fnmatch(file_path, glob):
                    continue

                # Skip directories
                if file_path.endswith("/"):
                    continue

                try:
                    content = self.s3_backend.read(file_path)
                    lines = content.split("\n")

                    for i, line in enumerate(lines, 1):
                        for match in regex.finditer(line):
                            matches.append(GrepMatch(
                                path=self._strip_prefix(file_path),
                                line_number=i,
                                line=line,
                                match_start=match.start(),
                                match_end=match.end(),
                            ))
                except Exception as e:
                    logger.debug(f"[MinioBackend] grep skipped {file_path}: {e}")
                    continue

            return matches

        except Exception as e:
            return f"Error: {str(e)}"

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching pattern.

        Args:
            pattern: Glob pattern (e.g., "*.html")
            path: Base path to search

        Returns:
            List of matching FileInfo objects
        """
        search_path = self._make_path(path)
        files = self.s3_backend.ls(search_path)

        results = []
        for f in files:
            file_path = f.get("name", f.get("path", ""))

            if fnmatch(file_path, pattern) or fnmatch(file_path.split("/")[-1], pattern):
                results.append(FileInfo(
                    path=self._strip_prefix(file_path),
                    name=file_path.split("/")[-1],
                    size=f.get("size", 0),
                    modified_at=str(f.get("last_modified", f.get("modified_at", ""))),
                    is_dir=file_path.endswith("/"),
                ))

        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files.

        Args:
            files: List of (path, content) tuples

        Returns:
            List of FileUploadResponse
        """
        results = []

        for file_path, content in files:
            full_path = self._make_path(file_path)

            try:
                # Decode bytes to string for S3Backend
                if isinstance(content, bytes):
                    content_str = content.decode("utf-8")
                else:
                    content_str = content

                result = self.s3_backend.write(full_path, content_str)

                if "error" in result:
                    results.append(FileUploadResponse(path=file_path, error=result["error"]))
                else:
                    results.append(FileUploadResponse(path=file_path))

            except Exception as e:
                results.append(FileUploadResponse(path=file_path, error=str(e)))

        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files.

        Args:
            paths: List of file paths

        Returns:
            List of FileDownloadResponse
        """
        results = []

        for file_path in paths:
            full_path = self._make_path(file_path)

            try:
                content = self.s3_backend.read(full_path)
                results.append(FileDownloadResponse(
                    path=file_path,
                    content=content.encode("utf-8")
                ))

            except FileNotFoundError:
                results.append(FileDownloadResponse(
                    path=file_path,
                    content=b"",
                    error="File not found"
                ))
            except Exception as e:
                results.append(FileDownloadResponse(
                    path=file_path,
                    content=b"",
                    error=str(e)
                ))

        return results

    # =========================================================================
    # ASYNC OPERATIONS (required by BackendProtocol)
    # =========================================================================

    async def als_info(self, path: str = "/") -> list[FileInfo]:
        """Async list files."""
        return self.ls_info(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Async read file."""
        return self.read(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async write file."""
        return self.write(file_path, content)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False
    ) -> EditResult:
        """Async edit file."""
        return self.edit(file_path, old_string, new_string, replace_all)

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None
    ) -> Union[list[GrepMatch], str]:
        """Async grep."""
        return self.grep_raw(pattern, path, glob)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Async glob."""
        return self.glob_info(pattern, path)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async upload files."""
        return self.upload_files(files)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async download files."""
        return self.download_files(paths)


# =============================================================================
# COMPOSITE BACKEND FACTORY
# =============================================================================

def make_minio_composite_backend(
    runtime: "ToolRuntime",
    session_id: Optional[str] = None,
    include_web: bool = True,
    include_dumps: bool = True,
    include_artifacts: bool = True,
) -> "CompositeBackend":
    """Create a CompositeBackend with MinIO routes.

    This creates a unified filesystem that routes:
    - Default "/" → StateBackend (ephemeral agent state)
    - "/web/" → MinIO for web session data
    - "/dumps/" → MinIO for context dumps
    - "/artifacts/" → MinIO for research artifacts

    Args:
        runtime: ToolRuntime from deepagents
        session_id: Explicit session_id (overrides runtime)
        include_web: Include /web/ route
        include_dumps: Include /dumps/ route
        include_artifacts: Include /artifacts/ route

    Returns:
        CompositeBackend instance
    """
    try:
        from deepagents.backends import CompositeBackend, StateBackend
    except ImportError:
        raise ImportError(
            "deepagents package required. Install with: uv add deepagents"
        )

    # Default state backend for ephemeral agent state
    default_backend = StateBackend(runtime)

    # Build routes
    routes = {}

    if include_web:
        routes["/web/"] = MinioBackend(
            prefix="web/session",
            runtime=runtime,
            session_id=session_id,
        )

    if include_dumps:
        routes["/dumps/"] = MinioBackend(
            prefix="dumps/session",
            runtime=runtime,
            session_id=session_id,
        )

    if include_artifacts:
        routes["/artifacts/"] = MinioBackend(
            prefix="raw",  # Use existing raw folder for artifacts
            runtime=runtime,
            session_id=session_id,
        )

    return CompositeBackend(default=default_backend, routes=routes)


def minio_backend_factory(
    prefix: str = "",
    session_id: Optional[str] = None,
) -> Callable[["ToolRuntime"], MinioBackend]:
    """Factory function for creating MinioBackend instances.

    Use with create_deep_agent:

        agent = create_deep_agent(
            model=llm,
            backend=minio_backend_factory("web/session"),
        )

    Args:
        prefix: Path prefix for MinIO
        session_id: Explicit session_id

    Returns:
        Factory function
    """
    def factory(runtime: "ToolRuntime") -> MinioBackend:
        return MinioBackend(
            prefix=prefix,
            runtime=runtime,
            session_id=session_id,
        )

    return factory


# =============================================================================
# HELPER FUNCTIONS FOR EXISTING CODE
# =============================================================================

def get_minio_backend_for_session(
    prefix: str,
    session_id: str,
) -> MinioBackend:
    """Get a MinioBackend for a specific session.

    This is useful for non-deep-agents code that wants to use
    the unified filesystem interface.

    Args:
        prefix: Path prefix (e.g., "web/session", "dumps/session")
        session_id: Session ID

    Returns:
        MinioBackend instance

    Example:
        backend = get_minio_backend_for_session("web/session", "abc123")
        backend.write("/page.html", html_content)
        matches = backend.grep_raw("keyword", glob="*.html")
    """
    return MinioBackend(prefix=prefix, session_id=session_id)


__all__ = [
    # Types
    "FileInfo",
    "WriteResult",
    "EditResult",
    "GrepMatch",
    "FileUploadResponse",
    "FileDownloadResponse",
    # Backend
    "MinioBackend",
    # Factories
    "make_minio_composite_backend",
    "minio_backend_factory",
    "get_minio_backend_for_session",
]
