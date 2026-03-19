"""File handler for run logging with context_engine storage.

ARCHITECTURE:
- Primary storage: Context Engine S3Backend (MinIO)
- Local fallback: Filesystem for dev/offline mode
- Session-based isolation: All logs go to /logs/{session_id}/

This module provides file I/O for run logs with proper resource management:
- Context manager support for automatic cleanup
- Exception-safe file operations
- Thread-safe writes
- Context Engine integration for centralized logging

Usage:
    from .file_handler import FileHandler

    # Using context manager with session_id (context_engine mode)
    async with FileHandler(run_dir, session_id="session_123") as handler:
        handler.write_to_log("Message")
        handler.write_to_llm_calls_log("# LLM Call")

    # Legacy mode (local files only)
    handler = FileHandler(run_dir)
    handler.open_log_files()
    handler.write_to_log("Message")
    handler.close_log_files()
"""

import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

# Get logger - safe_get_logger always succeeds
from ..logging_utils import safe_get_logger
logger = safe_get_logger(__name__, "file_handler")


class LogFormat(str, Enum):
    """Log format options for structured logging."""

    TEXT = "text"  # Human-readable text format
    JSON = "json"  # Structured JSON format for machine parsing


class StructuredLogWriter:
    """Writer for structured log entries.

    Supports both text and JSON formats for flexible log consumption:
    - TEXT: Human-readable format with timestamps and levels
    - JSON: Machine-parseable format for log aggregation systems

    Usage:
        writer = StructuredLogWriter(LogFormat.JSON)
        entry = writer.format_entry("Processing started", level="INFO", phase="worker")
    """

    def __init__(self, log_format: LogFormat = LogFormat.TEXT):
        """Initialize the structured log writer.

        Args:
            log_format: Output format (TEXT or JSON)
        """
        self.log_format = log_format

    def format_entry(
        self,
        message: str,
        level: str = "INFO",
        phase: Optional[str] = None,
        **extra: Any
    ) -> str:
        """Format a log entry according to the configured format.

        Args:
            message: Log message content
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            phase: Optional phase name (worker, boss, lab, critic, synthesis)
            **extra: Additional metadata to include

        Returns:
            Formatted log entry string (with newline for TEXT, without for JSON)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        iso_timestamp = datetime.now().isoformat()

        if self.log_format == LogFormat.JSON:
            entry: Dict[str, Any] = {
                "timestamp": iso_timestamp,
                "level": level.upper(),
                "message": message,
            }
            if phase:
                entry["phase"] = phase
            if extra:
                entry["extra"] = extra
            return json.dumps(entry, ensure_ascii=False)
        else:
            # Text format: [timestamp] [LEVEL] [PHASE] message
            prefix = f"[{phase}] " if phase else ""
            return f"[{timestamp}] [{level.upper()}] {prefix}{message}"

    def format_batch(self, entries: List[Dict[str, Any]]) -> str:
        """Format multiple log entries at once.

        Args:
            entries: List of entry dicts with 'message', 'level', 'phase' keys

        Returns:
            Formatted string with all entries
        """
        formatted = []
        for entry in entries:
            formatted.append(self.format_entry(
                message=entry.get("message", ""),
                level=entry.get("level", "INFO"),
                phase=entry.get("phase"),
                **entry.get("extra", {})
            ))
        return "\n".join(formatted) + "\n" if formatted else ""


class FileHandler:
    """File handler for run logging with context_engine storage.

    Features:
    - Primary storage via Context Engine S3Backend
    - Local file fallback for dev/offline mode
    - Async context manager support
    - Automatic directory creation
    - Proper file cleanup on exit
    - Exception-safe operations
    - Thread-safe writes

    Storage Paths (Context Engine):
    - /logs/{session_id}/app.log
    - /logs/{session_id}/llm_calls.md
    - /logs/{session_id}/metrics.json
    """

    def __init__(
        self,
        run_dir: Path,
        session_id: Optional[str] = None,
        encoding: str = "utf-8",
        max_file_size: int = 0,
        backup_count: int = 5,
        use_context_engine: bool = True,
        log_format: LogFormat = LogFormat.TEXT,
    ):
        """Initialize the file handler.

        Args:
            run_dir: Directory for local log files (fallback)
            session_id: Research session ID for context_engine storage
            encoding: File encoding (default: utf-8)
            max_file_size: Maximum file size before rotation (0 = no rotation)
            backup_count: Number of backup files to keep
            use_context_engine: Try to use context_engine for storage (default: True)
            log_format: Log output format (TEXT or JSON)
        """
        self.run_dir = Path(run_dir) if isinstance(run_dir, str) else run_dir
        self.session_id = session_id
        self.encoding = encoding
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.use_context_engine = use_context_engine and session_id is not None
        self.log_format = log_format
        self._log_file = None
        self._llm_calls_file = None
        self._write_lock = threading.Lock()
        self._s3_backend = None
        self._buffer: Dict[str, list] = {"app": [], "llm_calls": []}
        self._structured_writer = StructuredLogWriter(log_format)

        # Initialize S3Backend if using context_engine
        if self.use_context_engine:
            self._init_s3_backend()

    def _init_s3_backend(self) -> None:
        """Initialize S3Backend for context_engine storage."""
        try:
            from src.extensions.features.tools.context_engine.storage.s3_backend import S3Backend

            self._s3_backend = S3Backend(
                bucket=os.getenv("MINIO_BUCKET", "context-engine"),
                endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:8899/minio"),
                aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minioadmin")),
                aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")),
            )
            logger.info(f"[FILE_HANDLER] Context Engine S3Backend initialized for session: {self.session_id}")
        except Exception as e:
            logger.warning(f"[FILE_HANDLER] S3Backend initialization failed, using local files: {e}")
            self._s3_backend = None
            self.use_context_engine = False

    def ensure_directory(self) -> None:
        """Ensure run directory exists (for local fallback)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def open_log_files(self) -> None:
        """Open log files for writing (local fallback)."""
        self.ensure_directory()
        log_path = self.run_dir / "app.log"
        llm_path = self.run_dir / "llm_calls.md"

        self._log_file = open(log_path, "a", encoding=self.encoding)
        self._llm_calls_file = open(llm_path, "a", encoding=self.encoding)

    def close_log_files(self) -> None:
        """Close log files safely and flush to context_engine."""
        # Flush any buffered content to context_engine
        if self.use_context_engine and self._s3_backend:
            self._flush_to_context_engine()

        if self._log_file:
            try:
                self._log_file.close()
            except Exception as e:
                logger.warning(f"Error closing log file: {e}")
            finally:
                self._log_file = None

        if self._llm_calls_file:
            try:
                self._llm_calls_file.close()
            except Exception as e:
                logger.warning(f"Error closing LLM calls file: {e}")
            finally:
                self._llm_calls_file = None

    def _flush_to_context_engine(self) -> None:
        """Flush buffered content to context_engine S3Backend."""
        if not self._s3_backend or not self.session_id:
            return

        try:
            # Flush app log buffer
            if self._buffer["app"]:
                path = f"/logs/{self.session_id}/app.log"
                content = "".join(self._buffer["app"])
                self._s3_backend.append(path, content)
                self._buffer["app"] = []

            # Flush LLM calls buffer
            if self._buffer["llm_calls"]:
                path = f"/logs/{self.session_id}/llm_calls.md"
                content = "".join(self._buffer["llm_calls"])
                self._s3_backend.append(path, content)
                self._buffer["llm_calls"] = []

        except Exception as e:
            logger.warning(f"[FILE_HANDLER] Failed to flush to context_engine: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.open_log_files()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.close_log_files()
        return False  # Don't suppress exceptions

    async def __aenter__(self):
        """Async context manager entry."""
        self.open_log_files()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        self.close_log_files()
        return False  # Don't suppress exceptions

    def _check_rotation(self, file) -> None:
        """Check if file needs rotation and rotate if necessary.

        Args:
            file: File object to check
        """
        if self.max_file_size <= 0:
            return

        try:
            current_pos = file.tell()
            if current_pos > self.max_file_size:
                # Close current file
                file.close()

                # Rotate files
                for i in range(self.backup_count - 1, 0, -1):
                    old_backup = self.run_dir / f"app.log.{i}"
                    new_backup = self.run_dir / f"app.log.{i + 1}"
                    if old_backup.exists():
                        old_backup.rename(new_backup)

                # Move current to .1
                log_path = self.run_dir / "app.log"
                backup_path = self.run_dir / "app.log.1"
                if log_path.exists():
                    log_path.rename(backup_path)

                # Reopen file
                self._log_file = open(log_path, "a", encoding=self.encoding)
        except Exception as e:
            logger.error(f"Error during log rotation: {e}")

    def write_to_log(
        self,
        message: str,
        level: str = "INFO",
        phase: Optional[str] = None,
        **extra: Any
    ) -> None:
        """Write to app.log (thread-safe).

        Writes to both:
        1. Context Engine S3Backend (if available)
        2. Local file (as fallback/cache)

        Supports structured logging when log_format is JSON.

        Args:
            message: Message to write
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            phase: Optional phase name (worker, boss, lab, critic, synthesis)
            **extra: Additional metadata for structured logging
        """
        # Use structured writer for formatting
        formatted_message = self._structured_writer.format_entry(
            message=message,
            level=level,
            phase=phase,
            **extra
        )
        # Add newline for text format (JSON already has proper formatting)
        if self.log_format == LogFormat.TEXT:
            formatted_message += "\n"
        else:
            formatted_message += "\n"  # JSON also needs newline for readability

        # Write to context_engine buffer
        if self.use_context_engine:
            self._buffer["app"].append(formatted_message)

            # Flush periodically (every 100 messages)
            if len(self._buffer["app"]) >= 100:
                self._flush_to_context_engine()

        # Write to local file (fallback)
        if self._log_file:
            with self._write_lock:
                self._check_rotation(self._log_file)
                self._log_file.write(formatted_message)
                self._log_file.flush()
        elif not self.use_context_engine:
            logger.warning("Attempted to write to log file but file is not open")

    def write_structured(
        self,
        entries: List[Dict[str, Any]],
        flush: bool = True
    ) -> None:
        """Write multiple structured log entries at once (thread-safe).

        Efficient batch writing for structured logs. Useful for:
        - Dumping accumulated phase logs
        - Writing diagnostic summaries
        - Bulk log export

        Args:
            entries: List of dicts with 'message', 'level', 'phase', 'extra' keys
            flush: Whether to flush to context_engine immediately
        """
        if not entries:
            return

        formatted_batch = self._structured_writer.format_batch(entries)

        # Write to context_engine buffer
        if self.use_context_engine:
            self._buffer["app"].append(formatted_batch)
            if flush or len(self._buffer["app"]) >= 100:
                self._flush_to_context_engine()

        # Write to local file
        if self._log_file:
            with self._write_lock:
                self._check_rotation(self._log_file)
                self._log_file.write(formatted_batch)
                self._log_file.flush()

    def write_to_llm_calls_log(self, message: str) -> None:
        """Write to llm_calls.md (thread-safe).

        Writes to both:
        1. Context Engine S3Backend (if available)
        2. Local file (as fallback/cache)

        Args:
            message: Message to write
        """
        # Write to context_engine buffer
        if self.use_context_engine:
            self._buffer["llm_calls"].append(message)

            # Flush periodically
            if len(self._buffer["llm_calls"]) >= 50:
                self._flush_to_context_engine()

        # Write to local file
        if self._llm_calls_file:
            with self._write_lock:
                self._llm_calls_file.write(message)
                self._llm_calls_file.flush()
        elif not self.use_context_engine:
            logger.warning("Attempted to write to LLM calls file but file is not open")

    def write_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Write JSON data to storage.

        Writes to both:
        1. Context Engine S3Backend (if available)
        2. Local file (as fallback)

        Args:
            filename: Name of the file to write
            data: Data to write as JSON
        """
        content = json.dumps(data, indent=2, ensure_ascii=False, default=str)

        # Write to context_engine
        if self.use_context_engine and self._s3_backend:
            try:
                path = f"/logs/{self.session_id}/{filename}"
                self._s3_backend.write(path, content)
                logger.debug(f"[FILE_HANDLER] Wrote {filename} to context_engine")
            except Exception as e:
                logger.warning(f"[FILE_HANDLER] Failed to write {filename} to context_engine: {e}")

        # Write to local file
        self.ensure_directory()
        output_path = self.run_dir / filename

        try:
            with open(output_path, 'w', encoding=self.encoding) as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write JSON to {filename}: {e}")
            raise

    def read_file(self, filename: str) -> str:
        """Read file contents.

        Tries context_engine first, then local file.

        Args:
            filename: Name of the file to read

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        # Try context_engine first
        if self.use_context_engine and self._s3_backend:
            try:
                path = f"/logs/{self.session_id}/{filename}"
                content = self._s3_backend.read(path)
                return content
            except Exception as e:
                logger.debug(f"[FILE_HANDLER] Context engine read failed, trying local: {e}")

        # Fallback to local file
        output_path = self.run_dir / filename

        if not output_path.exists():
            raise FileNotFoundError(f"File not found: {output_path}")

        try:
            with open(output_path, 'r', encoding=self.encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            raise

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists.

        Args:
            filename: Name of the file to check

        Returns:
            True if file exists, False otherwise
        """
        # Check local file
        return (self.run_dir / filename).exists()


class FileHandlerFactory:
    """Factory for creating FileHandler instances."""

    @staticmethod
    def create_file_handler(
        base_dir: Path,
        run_name: str,
        timestamp: str,
        session_id: Optional[str] = None,
        encoding: str = "utf-8",
        max_file_size: int = 0,
        backup_count: int = 5,
        use_context_engine: bool = True,
        log_format: LogFormat = LogFormat.TEXT,
    ) -> FileHandler:
        """Create a FileHandler for a new run.

        Args:
            base_dir: Base directory for all runs (local fallback)
            run_name: Name of the run
            timestamp: Timestamp string for unique directory
            session_id: Research session ID for context_engine storage
            encoding: File encoding
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            use_context_engine: Try to use context_engine for storage
            log_format: Log output format (TEXT or JSON)

        Returns:
            Configured FileHandler instance
        """
        run_dir = base_dir / f"{run_name}_{timestamp}"
        return FileHandler(
            run_dir=run_dir,
            session_id=session_id,
            encoding=encoding,
            max_file_size=max_file_size,
            backup_count=backup_count,
            use_context_engine=use_context_engine,
            log_format=log_format,
        )


@asynccontextmanager
async def create_file_handler_async(
    base_dir: Path,
    run_name: str,
    timestamp: str,
    session_id: Optional[str] = None,
    encoding: str = "utf-8",
    max_file_size: int = 0,
    backup_count: int = 5,
    use_context_engine: bool = True,
    log_format: LogFormat = LogFormat.TEXT,
):
    """Async context manager for creating a FileHandler.

    Usage:
        async with create_file_handler_async(base_dir, run_name, timestamp, session_id) as handler:
            handler.write_to_log("Message")

    Args:
        base_dir: Base directory for all runs
        run_name: Name of the run
        timestamp: Timestamp string for unique directory
        session_id: Research session ID for context_engine storage
        encoding: File encoding
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
        use_context_engine: Try to use context_engine for storage
        log_format: Log output format (TEXT or JSON)

    Yields:
        FileHandler instance with files opened
    """
    handler = FileHandlerFactory.create_file_handler(
        base_dir=base_dir,
        run_name=run_name,
        timestamp=timestamp,
        session_id=session_id,
        encoding=encoding,
        max_file_size=max_file_size,
        backup_count=backup_count,
        use_context_engine=use_context_engine,
        log_format=log_format,
    )
    async with handler:
        yield handler


__all__ = [
    "FileHandler",
    "FileHandlerFactory",
    "create_file_handler_async",
    "LogFormat",
    "StructuredLogWriter",
]
