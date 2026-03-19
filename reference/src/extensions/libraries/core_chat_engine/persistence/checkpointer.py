"""
Conversation Persistence with LangGraph Checkpointer

Provides durable conversation state management using LangGraph's checkpointer system.

Note: In langgraph 1.0+, only InMemorySaver is built-in. For persistent storage,
we provide custom SQLite checkpointer implementation.

Key concepts:
- thread_id: Unique identifier for a conversation thread (per user/session)
- checkpoint_id: Specific version of state (can rewind/branch)
- State is automatically persisted after each turn
"""

import os
import sqlite3
import json
import uuid
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Literal, TypedDict, Annotated, Iterator, AsyncIterator, Tuple, Sequence, Union
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import MutableMapping

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import RunnableConfig

# Import WRITES_IDX_MAP for put_writes compatibility
try:
    from langgraph.pregel._executor import WRITES_IDX_MAP
except ImportError:
    # Fallback if the internal API changes
    WRITES_IDX_MAP = {}


def _generate_default_title() -> str:
    """Generate a default title based on current timestamp.

    Returns:
        Formatted timestamp like "Jan 10, 2:30 PM"
    """
    now = datetime.now()
    return now.strftime("%b %d, %-I:%M %p").lstrip("0").replace(" 0", " ")

from src.extensions.libraries.core_chat_engine.state.brain_state import (
    BrainState,
    ConversationState,
    UserProfile,
    UserMood,
    ConversationTurn,
)
from src.extensions.libraries.core_chat_engine.state.intent_types import IntentType, IntentResult

logger = logging.getLogger(__name__)

class CompanionState(TypedDict):
    """State for LangGraph-based companion conversation."""

    # Messages (LangChain format)
    messages: Annotated[List[BaseMessage], add_messages]

    # Core brain state
    brain_state: Dict[str, Any]

    # Conversation metadata
    thread_id: str
    user_id: str
    mode: str  # "companion" or "rp"

    # Timestamps
    last_activity: str

    # Optional config overrides
    config_overrides: Dict[str, Any]


class SQLiteSaver(BaseCheckpointSaver):
    """
    SQLite-based checkpoint saver for persistent conversation storage.

    Custom implementation since langgraph 1.0+ only includes InMemorySaver.
    """

    def __init__(self, db_path: str):
        """Initialize SQLite checkpointer."""
        # Add timeout to prevent hanging on locked database
        self.db_path = db_path
        logger.info(f"[SQLiteSaver] Initializing with db_path={db_path}")
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.serde = JsonPlusSerializer()
        # Enable WAL mode for better concurrency (readers don't block writers)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
        # Initialize writes storage (in-memory, like InMemorySaver)
        from collections import defaultdict
        self.writes = defaultdict(dict)
        logger.info(f"[SQLiteSaver] Creating tables...")
        self._create_tables()
        logger.info(f"[SQLiteSaver] Initialization complete")

    def _create_tables(self):
        """Create checkpoint tables."""
        logger.info("[SQLiteSaver] _create_tables: Starting table creation")
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_id TEXT,
                    checkpoint BLOB NOT NULL,
                    metadata BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (thread_id, checkpoint_id)
                )
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_thread_created
                ON checkpoints(thread_id, created_at DESC)
            """)

            # Create thread_titles table for storing custom and auto-generated titles
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_titles (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT,
                    is_auto_generated BOOLEAN DEFAULT 1,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES checkpoints(thread_id) ON DELETE CASCADE
                )
            """)

            # Migrate existing threads - add default titles for threads without titles
            # Use timestamp-based default titles
            self.conn.execute("""
                INSERT OR IGNORE INTO thread_titles (thread_id, title, is_auto_generated)
                SELECT DISTINCT thread_id, NULL, 1
                FROM checkpoints
                WHERE thread_id IS NOT NULL
            """)

            # Create chain_of_thought table for storing reasoning steps
            # Only stores non-tool-router CoT (actual reasoning, not tool execution events)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chain_of_thought (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    content TEXT,
                    status TEXT DEFAULT 'active',
                    icon TEXT DEFAULT 'sparkle',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES checkpoints(thread_id) ON DELETE CASCADE
                )
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cot_thread_checkpoint
                ON chain_of_thought(thread_id, checkpoint_id)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cot_created
                ON chain_of_thought(created_at DESC)
            """)

            # Create message_versions table for storing multiple message generations/branches
            # Supports SillyTavern-like swipe/regeneration functionality
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS message_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    version_id TEXT NOT NULL UNIQUE,
                    version_number INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    parts TEXT,
                    reasoning_content TEXT,
                    model TEXT,
                    temperature REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 0,
                    UNIQUE (message_id, version_number),
                    FOREIGN KEY (thread_id) REFERENCES checkpoints(thread_id) ON DELETE CASCADE
                )
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_versions_message
                ON message_versions(message_id, version_number)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_versions_thread
                ON message_versions(thread_id, created_at DESC)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_versions_active
                ON message_versions(message_id, is_active)
            """)

            self.conn.commit()
            logger.info("[SQLiteSaver] _create_tables: All tables created and committed")
            # Verify tables were created
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"[SQLiteSaver] _create_tables: Tables in database: {tables}")
        except Exception as e:
            logger.error(f"[SQLiteSaver] _create_tables: Error creating tables: {e}", exc_info=True)
            raise

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id", str(uuid.uuid4()))

        # Ensure checkpoint has 'v' key for LangGraph 1.0+ compatibility
        # If checkpoint doesn't have 'v', set it to the current version format
        checkpoint_copy = dict(checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(checkpoint_copy, dict) and "v" not in checkpoint_copy:
            checkpoint_copy["v"] = 1

        # Serialize checkpoint - dumps_typed returns (type_name, bytes)
        checkpoint_type, checkpoint_data = self.serde.dumps_typed(checkpoint_copy)
        # Store as single BLOB with type prefix
        checkpoint_bytes = json.dumps({"type": checkpoint_type, "data": checkpoint_data.hex()}).encode()

        if metadata:
            metadata_type, metadata_data = self.serde.dumps_typed(metadata)
            metadata_bytes = json.dumps({"type": metadata_type, "data": metadata_data.hex()}).encode()
        else:
            metadata_bytes = None

        # Store
        self.conn.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (thread_id, checkpoint_id, checkpoint, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (thread_id, checkpoint_id, checkpoint_bytes, metadata_bytes)
        )
        self.conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async version of put - wraps sync method for SQLite compatibility."""
        return self.put(config, checkpoint, metadata, new_versions)

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get checkpoint tuple."""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        if checkpoint_id:
            row = self.conn.execute(
                """
                SELECT checkpoint, metadata, checkpoint_id FROM checkpoints
                WHERE thread_id = ? AND checkpoint_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (thread_id, checkpoint_id)
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT checkpoint, metadata, checkpoint_id FROM checkpoints
                WHERE thread_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (thread_id,)
            ).fetchone()

        if not row:
            return None

        # Deserialize checkpoint
        checkpoint_wrapper = json.loads(row["checkpoint"])
        checkpoint_data = bytes.fromhex(checkpoint_wrapper["data"])
        checkpoint = self.serde.loads_typed((checkpoint_wrapper["type"], checkpoint_data))

        # Migrate old checkpoints to LangGraph 1.0+ format
        # LangGraph expects checkpoints to have: {id, channel_values, channel_versions, versions_seen, v}
        # Old format checkpoints may have state directly at top level
        if isinstance(checkpoint, dict):
            # Add 'v' key if missing (version number)
            if "v" not in checkpoint:
                checkpoint["v"] = 1
                logger.info(f"[SQLiteSaver] Added v=1 to checkpoint for thread {thread_id}")

            # Wrap in channel_values if missing (LangGraph 1.0+ expects this structure)
            # The state data should be nested under channel_values for proper migration
            if "channel_values" not in checkpoint:
                # Create a new checkpoint structure with channel_values
                # Keep id and versions_seen if they exist, move everything else to channel_values
                old_checkpoint = checkpoint
                checkpoint = {
                    "id": old_checkpoint.pop("id", str(uuid.uuid4())),
                    "channel_values": old_checkpoint,
                    "channel_versions": old_checkpoint.pop("channel_versions", {}),
                    "versions_seen": old_checkpoint.pop("versions_seen", {}),
                    "v": old_checkpoint.pop("v", 1),
                }
                logger.info(f"[SQLiteSaver] Migrated checkpoint to channel_values format for thread {thread_id}")
        else:
            # If checkpoint is not a dict (e.g., a custom type), log a warning
            logger.warning(f"[SQLiteSaver] Checkpoint is not a dict: type={type(checkpoint)}")
            # Try to add 'v' key if the object supports attribute assignment
            try:
                checkpoint.v = 1
            except (AttributeError, TypeError):
                logger.error(f"[SQLiteSaver] Cannot add 'v' key to checkpoint of type {type(checkpoint)}")

        # Deserialize metadata
        if row["metadata"]:
            metadata_wrapper = json.loads(row["metadata"])
            metadata_data = bytes.fromhex(metadata_wrapper["data"])
            metadata = self.serde.loads_typed((metadata_wrapper["type"], metadata_data))
        else:
            metadata = {}

        # Ensure metadata has required keys for LangGraph 1.0+ compatibility
        # LangGraph expects at minimum: step, source, writes, parents
        if isinstance(metadata, dict):
            if "step" not in metadata:
                # Default to step 1 if missing (new checkpoint or old format)
                metadata["step"] = 1
                logger.info(f"[SQLiteSaver] Added step=1 to metadata for thread {thread_id}")
            if "source" not in metadata:
                metadata["source"] = "input"
            if "writes" not in metadata:
                metadata["writes"] = {}
            if "parents" not in metadata:
                metadata["parents"] = {}

        # Build config with checkpoint_ns if needed
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        # Use the checkpoint_id from the database row if not provided in config
        result_checkpoint_id = checkpoint_id or row["checkpoint_id"]

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": result_checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=None,
            pending_writes=[],
        )

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Async version of get_tuple - wraps sync method for SQLite compatibility."""
        # SQLite operations are fast enough to run synchronously
        # If needed, can use asyncio.to_thread for true async
        return self.get_tuple(config)

    def list(
        self,
        config: Dict[str, Any],
        limit: int = 10,
        before: Optional[str] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")

        query = """
            SELECT checkpoint, metadata, checkpoint_id
            FROM checkpoints
            WHERE thread_id = ?
        """
        params = [thread_id]

        if before:
            query += " AND created_at < ?"
            params.append(before)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        for row in rows:
            # Deserialize checkpoint
            checkpoint_wrapper = json.loads(row["checkpoint"])
            checkpoint_data = bytes.fromhex(checkpoint_wrapper["data"])
            checkpoint = self.serde.loads_typed((checkpoint_wrapper["type"], checkpoint_data))

            # Migrate old checkpoints to LangGraph 1.0+ format (same logic as get_tuple)
            if isinstance(checkpoint, dict):
                if "v" not in checkpoint:
                    checkpoint["v"] = 1

                # Wrap in channel_values if missing
                if "channel_values" not in checkpoint:
                    old_checkpoint = checkpoint
                    checkpoint = {
                        "id": old_checkpoint.pop("id", str(uuid.uuid4())),
                        "channel_values": old_checkpoint,
                        "channel_versions": old_checkpoint.pop("channel_versions", {}),
                        "versions_seen": old_checkpoint.pop("versions_seen", {}),
                        "v": old_checkpoint.pop("v", 1),
                    }

            # Deserialize metadata
            if row["metadata"]:
                metadata_wrapper = json.loads(row["metadata"])
                metadata_data = bytes.fromhex(metadata_wrapper["data"])
                metadata = self.serde.loads_typed((metadata_wrapper["type"], metadata_data))
            else:
                metadata = {}

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["checkpoint_id"],
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
                pending_writes=[],
            )

    async def alist(
        self,
        config: Dict[str, Any],
        limit: int = 10,
        before: Optional[str] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list - wraps sync method for SQLite compatibility."""
        for item in self.list(config, limit, before):
            yield item

    def delete_thread(self, config: Union[str, Dict[str, Any], None]) -> None:
        """Delete a thread.

        Args:
            config: Either a thread_id string or a config dict with configurable.thread_id
        """
        # Handle None, empty string, and invalid inputs gracefully
        if config is None or config == "":
            return

        # Handle both string thread_id and dict config for compatibility
        if isinstance(config, str):
            thread_id = config
        else:
            thread_id = config.get("configurable", {}).get("thread_id", "")

        # Delete from checkpoints table
        self.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        # Delete from thread_titles table to remove orphaned entries
        self.conn.execute(
            "DELETE FROM thread_titles WHERE thread_id = ?",
            (thread_id,)
        )
        self.conn.commit()

    async def adelete_thread(self, config: Union[str, Dict[str, Any], None]) -> None:
        """Async version of delete_thread - wraps sync method for SQLite compatibility."""
        self.delete_thread(config)

    def list_all_threads(self, limit: Optional[int] = None, before: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all unique thread_ids with their creation timestamps and titles.

        PERFORMANCE OPTIMIZED:
        - Simplified query - moved complex title formatting to Python
        - Uses subquery to filter by created_at before aggregation
        - Avoids expensive CASE statements in SQL

        Args:
            limit: Maximum number of threads to return
            before: Return threads created before this timestamp (for pagination)

        Returns:
            List of dicts with thread_id, created_at, and title
        """
        # Build query with optional before filter
        # Use a CTE to filter early, then group by thread_id
        params = []

        if before:
            # Filter threads before pagination, then group
            query = """
                WITH filtered_checkpoints AS (
                    SELECT thread_id, created_at, title
                    FROM checkpoints c
                    LEFT JOIN thread_titles tt ON c.thread_id = tt.thread_id
                    WHERE c.thread_id IS NOT NULL
                      AND datetime(c.created_at) < datetime(?, '-1 second')
                    ORDER BY c.created_at DESC
                    LIMIT ?
                )
                SELECT
                    thread_id,
                    replace(datetime(created_at), ' ', 'T') || 'Z' as created_at,
                    title
                FROM filtered_checkpoints
                GROUP BY thread_id
                ORDER BY MIN(created_at) DESC
            """
            params = [before, limit if limit else 1000]
        else:
            # Simple query without pagination filter
            query = """
                SELECT
                    c.thread_id,
                    replace(datetime(MIN(c.created_at)), ' ', 'T') || 'Z' as created_at,
                    tt.title
                FROM checkpoints c
                LEFT JOIN thread_titles tt ON c.thread_id = tt.thread_id
                WHERE c.thread_id IS NOT NULL
                GROUP BY c.thread_id
                ORDER BY MIN(c.created_at) DESC
            """
            if limit:
                query += " LIMIT ?"
                params.append(limit)

        rows = self.conn.execute(query, params).fetchall()

        result = []
        for row in rows:
            # Use stored title or generate default (faster in Python than complex SQL CASE)
            title = row["title"]
            if not title:
                # Generate timestamp-based title in Python (simpler than SQL CASE)
                try:
                    dt = datetime.fromisoformat(row["created_at"].replace('Z', '+00:00'))
                    title = dt.strftime("%b %d, %-I:%M %p").lstrip("0").replace(" 0", " ")
                except Exception as e:
                    logger.debug(f"Could not parse timestamp for title: {e}, using default")
                    title = _generate_default_title()

            result.append({
                "thread_id": row["thread_id"],
                "created_at": row["created_at"],
                "title": title,
            })

        return result

    def get_thread_title(self, thread_id: str) -> str:
        """
        Get the title for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Title string, or timestamp-based default if not found
        """
        # Try to get stored title first
        row = self.conn.execute(
            "SELECT title FROM thread_titles WHERE thread_id = ?",
            (thread_id,)
        ).fetchone()

        if row and row["title"]:
            return row["title"]

        # No stored title - generate timestamp-based from thread creation time
        created_row = self.conn.execute(
            "SELECT MIN(created_at) as created_at FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        ).fetchone()

        if created_row and created_row["created_at"]:
            # Parse the created_at timestamp and format it
            try:
                dt = datetime.fromisoformat(created_row["created_at"].replace('Z', '+00:00'))
                return dt.strftime("%b %d, %-I:%M %p").lstrip("0").replace(" 0", " ")
            except Exception as e:
                logger.debug(f"Could not parse created_at timestamp: {e}, using default title")

        # Ultimate fallback - current time
        return _generate_default_title()

    def has_thread_title(self, thread_id: str) -> bool:
        """
        Check if a thread has a title stored in the database.

        This is used to prevent redundant title generation.

        Args:
            thread_id: Thread identifier

        Returns:
            True if a title exists in thread_titles table, False otherwise
        """
        row = self.conn.execute(
            "SELECT 1 FROM thread_titles WHERE thread_id = ?",
            (thread_id,)
        ).fetchone()
        return row is not None

    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """
        Update the title for a thread (manual edit by user).

        Args:
            thread_id: Thread identifier
            title: New title

        Returns:
            True if updated, False otherwise
        """
        try:
            self.conn.execute(
                """INSERT INTO thread_titles (thread_id, title, is_auto_generated, updated_at)
                VALUES (?, ?, 0, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                title = excluded.title,
                is_auto_generated = 0,
                updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, title)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating thread title: {e}")
            return False

    def set_thread_title_auto_generated(self, thread_id: str, title: str) -> bool:
        """
        Set an auto-generated title for a thread.

        Args:
            thread_id: Thread identifier
            title: Auto-generated title

        Returns:
            True if set, False otherwise
        """
        try:
            # Only set if title doesn't exist or if current title is also auto-generated
            self.conn.execute(
                """INSERT INTO thread_titles (thread_id, title, is_auto_generated, updated_at)
                VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                title = excluded.title,
                is_auto_generated = excluded.is_auto_generated,
                updated_at = excluded.updated_at
                WHERE is_auto_generated = 1
                """,
                (thread_id, title)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error setting auto-generated title: {e}")
            return False

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save intermediate writes to in-memory storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        outer_key = (thread_id, checkpoint_ns, checkpoint_id)

        for idx, (c, v) in enumerate(writes):
            inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
            # Skip if already written
            if inner_key[1] >= 0 and inner_key in self.writes.get(outer_key, {}):
                continue

            self.writes[outer_key][inner_key] = (
                task_id,
                c,
                self.serde.dumps_typed(v),
                task_path,
            )

    async def aput_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes - delegates to sync version."""
        # SQLite writes are fast, just call sync version
        self.put_writes(config, writes, task_id, task_path)

    def get_next_version(
        self,
        current: Optional[str],
        channel: None,
    ) -> str:
        """Get next version number for a channel."""
        import random
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def save_chain_of_thought(
        self,
        thread_id: str,
        checkpoint_id: str,
        steps: List[Dict[str, Any]],
    ) -> bool:
        """
        Save chain of thought steps for a checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Checkpoint identifier
            steps: List of CoT steps with keys: id, step_type, label, content, status, icon

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            for step in steps:
                # Filter out tool router events - only save actual reasoning
                step_type = step.get("type", "status")
                label = step.get("label", "")

                # Skip tool execution events (only save reasoning)
                if step_type == "tool_start" or step_type == "tool_end":
                    continue

                # Skip empty reasoning steps
                if not label or label == "Thinking...":
                    continue

                # Get or generate step ID (use the UUID from the step if available)
                step_id = step.get("id") or str(uuid.uuid4())

                self.conn.execute(
                    """
                    INSERT INTO chain_of_thought
                    (thread_id, checkpoint_id, step_type, label, content, status, icon)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_id,
                        step_type,
                        label,
                        step.get("content", ""),
                        step.get("status", "complete"),
                        step.get("icon", "sparkle"),
                    )
                )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving chain of thought: {e}")
            return False

    def get_chain_of_thought(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get chain of thought steps for a thread or checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Optional checkpoint identifier (gets latest if not provided)

        Returns:
            List of CoT steps with keys: id, step_type, label, content, status, icon, created_at
        """
        try:
            if checkpoint_id:
                rows = self.conn.execute(
                    """
                    SELECT id, step_type, label, content, status, icon, created_at
                    FROM chain_of_thought
                    WHERE thread_id = ? AND checkpoint_id = ?
                    ORDER BY created_at ASC
                    """,
                    (thread_id, checkpoint_id)
                ).fetchall()
            else:
                # Get all CoT for the thread, ordered by creation time
                rows = self.conn.execute(
                    """
                    SELECT id, step_type, label, content, status, icon, created_at, checkpoint_id
                    FROM chain_of_thought
                    WHERE thread_id = ?
                    ORDER BY created_at ASC
                    """,
                    (thread_id,)
                ).fetchall()

            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "step_type": row["step_type"],
                    "label": row["label"],
                    "content": row["content"],
                    "status": row["status"],
                    "icon": row["icon"],
                    "created_at": row["created_at"],
                    "checkpoint_id": row.get("checkpoint_id"),
                })
            return result
        except Exception as e:
            logger.error(f"Error getting chain of thought: {e}")
            return []

    # =============================================================================
    # MESSAGE VERSIONING / BRANCHING SYSTEM
    # =============================================================================

    def save_message_version(
        self,
        message_id: str,
        thread_id: str,
        version_id: str,
        version_number: int,
        role: str,
        content: str,
        parts: Optional[List[Dict[str, Any]]] = None,
        reasoning_content: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        is_active: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save a message version to the database.

        Args:
            message_id: Unique identifier for this message position
            thread_id: Thread/conversation ID
            version_id: Unique identifier for this version
            version_number: Version number (1=original, 2=first regeneration, etc.)
            role: Message role ("user" or "assistant")
            content: Message content text
            parts: Vercel SDK parts (reasoning, tools, etc.)
            reasoning_content: Chain of thought reasoning
            model: Model that generated this version
            temperature: Temperature used
            is_active: Whether this is the currently active version
            metadata: Additional metadata (tool_calls, etc.)

        Returns:
            True if saved successfully
        """
        try:
            # Ensure metadata column exists (for backward compatibility)
            try:
                self.conn.execute(
                    "ALTER TABLE message_versions ADD COLUMN metadata TEXT"
                )
                self.conn.commit()
                logger.info("[MESSAGE VERSION] Added metadata column to message_versions table")
            except Exception:
                # Column already exists or error, ignore
                pass

            # If this is the new active version, deactivate all other versions for this message
            if is_active:
                self.conn.execute(
                    "UPDATE message_versions SET is_active = 0 WHERE message_id = ?",
                    (message_id,)
                )

            # Insert the new version
            self.conn.execute(
                """
                INSERT INTO message_versions (
                    message_id, thread_id, version_id, version_number, role,
                    content, parts, reasoning_content, model, temperature, is_active, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    thread_id,
                    version_id,
                    version_number,
                    role,
                    content,
                    json.dumps(parts) if parts else None,
                    reasoning_content,
                    model,
                    temperature,
                    1 if is_active else 0,
                    json.dumps(metadata) if metadata else None,
                )
            )
            self.conn.commit()
            logger.info(f"[MESSAGE VERSION] Saved version {version_number} for message {message_id} with metadata={bool(metadata)}")
            return True
        except Exception as e:
            logger.error(f"Error saving message version: {e}")
            self.conn.rollback()
            return False

    def get_message_versions(self, message_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a message.

        Args:
            message_id: Message position identifier

        Returns:
            List of message version dicts
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT id, message_id, thread_id, version_id, version_number, role,
                       content, parts, reasoning_content, model, temperature, created_at, is_active, metadata
                FROM message_versions
                WHERE message_id = ?
                ORDER BY version_number ASC
                """,
                (message_id,)
            )
            rows = cursor.fetchall()

            result = []
            for row in rows:
                version_data = {
                    "id": row["id"],
                    "message_id": row["message_id"],
                    "thread_id": row["thread_id"],
                    "version_id": row["version_id"],
                    "version_number": row["version_number"],
                    "role": row["role"],
                    "content": row["content"],
                    "parts": json.loads(row["parts"]) if row["parts"] else None,
                    "reasoning_content": row["reasoning_content"],
                    "model": row["model"],
                    "temperature": row["temperature"],
                    "created_at": row["created_at"],
                    "is_active": bool(row["is_active"]),
                }
                # Parse metadata if exists
                if row["metadata"]:
                    try:
                        version_data["metadata"] = json.loads(row["metadata"])
                    except Exception:
                        version_data["metadata"] = None
                else:
                    version_data["metadata"] = None

                result.append(version_data)
            return result
        except Exception as e:
            logger.error(f"Error getting message versions: {e}")
            return []

    def switch_active_version(self, message_id: str, version_id: str) -> bool:
        """Switch to a different active version of a message.

        Args:
            message_id: Message position identifier
            version_id: Version ID to make active

        Returns:
            True if switched successfully
        """
        try:
            # Verify the version exists
            cursor = self.conn.execute(
                "SELECT version_id FROM message_versions WHERE message_id = ? AND version_id = ?",
                (message_id, version_id)
            )
            if not cursor.fetchone():
                logger.error(f"Version {version_id} not found for message {message_id}")
                return False

            # Deactivate all versions for this message
            self.conn.execute(
                "UPDATE message_versions SET is_active = 0 WHERE message_id = ?",
                (message_id,)
            )

            # Activate the selected version
            self.conn.execute(
                "UPDATE message_versions SET is_active = 1 WHERE message_id = ? AND version_id = ?",
                (message_id, version_id)
            )
            self.conn.commit()
            logger.info(f"[MESSAGE VERSION] Switched to version {version_id} for message {message_id}")
            return True
        except Exception as e:
            logger.error(f"Error switching active version: {e}")
            self.conn.rollback()
            return False

    def get_active_version(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get the currently active version of a message.

        Args:
            message_id: Message position identifier

        Returns:
            Active version dict or None
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT id, message_id, thread_id, version_id, version_number, role,
                       content, parts, reasoning_content, model, temperature, created_at, is_active, metadata
                FROM message_versions
                WHERE message_id = ? AND is_active = 1
                LIMIT 1
                """,
                (message_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            version_data = {
                "id": row["id"],
                "message_id": row["message_id"],
                "thread_id": row["thread_id"],
                "version_id": row["version_id"],
                "version_number": row["version_number"],
                "role": row["role"],
                "content": row["content"],
                "parts": json.loads(row["parts"]) if row["parts"] else None,
                "reasoning_content": row["reasoning_content"],
                "model": row["model"],
                "temperature": row["temperature"],
                "created_at": row["created_at"],
                "is_active": bool(row["is_active"]),
            }
            # Parse metadata if exists
            if row["metadata"]:
                try:
                    version_data["metadata"] = json.loads(row["metadata"])
                except Exception:
                    version_data["metadata"] = None
            else:
                version_data["metadata"] = None

            return version_data
        except Exception as e:
            logger.error(f"Error getting active version: {e}")
            return None

    def get_thread_messages_with_versions(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a thread with versioning information.

        Args:
            thread_id: Thread/conversation ID

        Returns:
            List of message dicts with version info
        """
        try:
            # First get the earliest version_number for each message_id to sort by creation order
            cursor = self.conn.execute(
                """
                SELECT message_id, MIN(version_number) as first_version
                FROM message_versions
                WHERE thread_id = ?
                GROUP BY message_id
                ORDER BY first_version ASC
                """,
                (thread_id,)
            )
            message_ids = [row["message_id"] for row in cursor.fetchall()]

            result = []
            for message_id in message_ids:
                versions = self.get_message_versions(message_id)
                active_version = self.get_active_version(message_id)

                if active_version:
                    result.append({
                        "message_id": message_id,
                        "role": active_version["role"],
                        "version_count": len(versions),
                        "active_version_id": active_version["version_id"],
                        "active_version_number": active_version["version_number"],
                        "versions": versions,
                    })
            return result
        except Exception as e:
            logger.error(f"Error getting thread messages with versions: {e}")
            return []


class CompanionCheckpointManager:
    """
    Manages conversation persistence using LangGraph checkpointers.

    Supports:
    - InMemorySaver: In-memory (for development)
    - SQLiteSaver: SQLite file (for production)
    """

    def __init__(
        self,
        backend: Literal["memory", "sqlite"] = "sqlite",
        connection_string: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            backend: Checkpointer backend to use
            connection_string: Database connection string (currently unused, for future expansion)
            checkpoint_dir: Directory for SQLite checkpoints (default: data/outputs/checkpoints)
        """
        self.backend = backend
        if checkpoint_dir is None:
            from src.constants import CHECKPOINTS_DIR
            checkpoint_dir = CHECKPOINTS_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpointer = self._create_checkpointer(connection_string)
        # Don't create graph on init - it's expensive and not needed for thread listing
        self._graph = None

    @property
    def graph(self):
        """Lazy-load the graph only when needed."""
        if self._graph is None:
            self._graph = self._create_graph()
        return self._graph

    def _create_checkpointer(self, connection_string: Optional[str]):
        """Create appropriate checkpointer based on backend."""
        if self.backend == "memory":
            return InMemorySaver()

        elif self.backend == "sqlite":
            db_path = self.checkpoint_dir / "conversations.db"
            return SQLiteSaver(str(db_path))

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _create_graph(self) -> StateGraph:
        """Create LangGraph StateGraph for companion conversation."""
        builder = StateGraph(CompanionState)

        # Nodes (can be expanded with actual processing logic)
        builder.add_node("process_input", self._process_input_node)
        builder.add_node("generate_response", self._generate_response_node)
        builder.add_node("update_state", self._update_state_node)

        # Edges
        builder.add_edge(START, "process_input")
        builder.add_edge("process_input", "generate_response")
        builder.add_edge("generate_response", "update_state")
        builder.add_edge("update_state", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _process_input_node(self, state: CompanionState, config: RunnableConfig) -> CompanionState:
        """Process user input (placeholder for actual logic)."""
        # This would integrate with InputProcessor
        return state

    def _generate_response_node(self, state: CompanionState, config: RunnableConfig) -> CompanionState:
        """Generate response (placeholder for actual logic)."""
        # This would integrate with DirectResponseGenerator
        return state

    def _update_state_node(self, state: CompanionState, config: RunnableConfig) -> CompanionState:
        """Update conversation state (placeholder for actual logic)."""
        state["last_activity"] = datetime.now(timezone.utc).isoformat()
        return state

    # Delegate checkpointer methods to the underlying checkpointer
    def put(self, config, checkpoint, metadata, new_versions=None):
        """Store a checkpoint."""
        return self.checkpointer.put(config, checkpoint, metadata, new_versions)

    def get_tuple(self, config):
        """Get checkpoint tuple."""
        return self.checkpointer.get_tuple(config)

    def list(self, config, limit=10, before=None):
        """List checkpoints for a thread."""
        return self.checkpointer.list(config, limit, before)

    def delete_thread(self, config: Union[str, Dict[str, Any], None]) -> None:
        """Delete a thread.

        Args:
            config: Either a thread_id string or a config dict with configurable.thread_id
        """
        return self.checkpointer.delete_thread(config)

    def list_all_threads(self, limit=None, before=None):
        """List all conversation threads.

        PERFORMANCE NOTE: Message count is NOT fetched here because it requires
        deserializing the entire checkpoint state for each thread (N+1 query problem).
        The frontend doesn't currently use message_count, so we skip it entirely.
        """
        try:
            if hasattr(self.checkpointer, 'list_all_threads'):
                # SQLiteSaver returns thread info with titles (optimized)
                thread_info_list = self.checkpointer.list_all_threads(limit=limit, before=before)
            else:
                # For InMemorySaver or other checkpointers without pagination
                return []

            # Pass through the thread info with titles from SQLiteSaver
            threads = []
            for thread_info in thread_info_list:
                thread_id = thread_info["thread_id"]
                threads.append({
                    "thread_id": thread_id,
                    "title": thread_info.get("title", "New Chat"),
                    "message_count": 0,  # Set to 0 - not used by frontend, avoids expensive get_state call
                    "created_at": thread_info["created_at"],
                })

            # Sort by created_at descending (most recent first)
            threads.sort(key=lambda x: x["created_at"], reverse=True)
            return threads
        except Exception as e:
            logger.error(f"Error listing threads: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_thread_title(self, thread_id: str) -> str:
        """Get the title for a thread."""
        if hasattr(self.checkpointer, 'get_thread_title'):
            return self.checkpointer.get_thread_title(thread_id)
        return "New Chat"

    def update_thread_title(self, thread_id: str, title: str) -> bool:
        """Update the title for a thread (manual edit by user)."""
        if hasattr(self.checkpointer, 'update_thread_title'):
            return self.checkpointer.update_thread_title(thread_id, title)
        return False

    def set_thread_title_auto_generated(self, thread_id: str, title: str) -> bool:
        """Set an auto-generated title for a thread."""
        if hasattr(self.checkpointer, 'set_thread_title_auto_generated'):
            return self.checkpointer.set_thread_title_auto_generated(thread_id, title)
        return False

    def has_thread_title(self, thread_id: str) -> bool:
        """Check if a thread has a title stored in the database."""
        if hasattr(self.checkpointer, 'has_thread_title'):
            return self.checkpointer.has_thread_title(thread_id)
        return False

    def get_next_version(self, config, channel):
        """Get next version number for a channel."""
        if hasattr(self.checkpointer, 'get_next_version'):
            return self.checkpointer.get_next_version(config, channel)
        import random
        import uuid
        # Fallback implementation
        return f"{random.randint(0, 999999):032}.{random.random():016}"

    def put_writes(self, config, writes, task_id, task_path=""):
        """Save intermediate writes."""
        if hasattr(self.checkpointer, 'put_writes'):
            return self.checkpointer.put_writes(config, writes, task_id, task_path)
        # No-op if not supported - return empty list for writes
        return []

    # LangGraph looks for aput_writes (async put_writes)
    async def aput_writes(self, config, writes, task_id, task_path=""):
        """Async version of put_writes."""
        if hasattr(self.checkpointer, 'aput_writes'):
            return await self.checkpointer.aput_writes(config, writes, task_id, task_path)
        elif hasattr(self.checkpointer, 'put_writes'):
            return self.checkpointer.put_writes(config, writes, task_id, task_path)
        # No-op if not supported - return empty list for writes
        return []

    # LangGraph also needs async versions of other methods
    async def aget_tuple(self, config):
        """Async version of get_tuple."""
        return self.get_tuple(config)

    async def alist(self, config, limit=10, before=None):
        """Async version of list."""
        return self.list(config, limit, before)

    async def aput(self, config, checkpoint, metadata, new_versions=None):
        """Async version of put."""
        return self.put(config, checkpoint, metadata, new_versions)

    def save_chain_of_thought(
        self,
        thread_id: str,
        checkpoint_id: str,
        steps: List[Dict[str, Any]],
    ) -> bool:
        """Save chain of thought steps for a checkpoint."""
        if hasattr(self.checkpointer, 'save_chain_of_thought'):
            return self.checkpointer.save_chain_of_thought(thread_id, checkpoint_id, steps)
        return False

    async def asave_chain_of_thought(
        self,
        thread_id: str,
        checkpoint_id: str,
        steps: List[Dict[str, Any]],
    ) -> bool:
        """Async version of save_chain_of_thought."""
        return self.save_chain_of_thought(thread_id, checkpoint_id, steps)

    def get_chain_of_thought(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get chain of thought steps for a thread or checkpoint."""
        if hasattr(self.checkpointer, 'get_chain_of_thought'):
            return self.checkpointer.get_chain_of_thought(thread_id, checkpoint_id)
        return []

    async def aget_chain_of_thought(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of get_chain_of_thought."""
        return self.get_chain_of_thought(thread_id, checkpoint_id)

    # =============================================================================
    # MESSAGE VERSIONING / BRANCHING SYSTEM (Wrappers)
    # =============================================================================

    def save_message_version(
        self,
        message_id: str,
        thread_id: str,
        version_id: str,
        version_number: int,
        role: str,
        content: str,
        parts: Optional[List[Dict[str, Any]]] = None,
        reasoning_content: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        is_active: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save a message version to the database."""
        if hasattr(self.checkpointer, 'save_message_version'):
            return self.checkpointer.save_message_version(
                message_id, thread_id, version_id, version_number, role,
                content, parts, reasoning_content, model, temperature, is_active, metadata
            )
        return False

    def get_message_versions(self, message_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a message."""
        if hasattr(self.checkpointer, 'get_message_versions'):
            return self.checkpointer.get_message_versions(message_id)
        return []

    def switch_active_version(self, message_id: str, version_id: str) -> bool:
        """Switch to a different active version of a message."""
        if hasattr(self.checkpointer, 'switch_active_version'):
            return self.checkpointer.switch_active_version(message_id, version_id)
        return False

    def get_active_version(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get the currently active version of a message."""
        if hasattr(self.checkpointer, 'get_active_version'):
            return self.checkpointer.get_active_version(message_id)
        return None

    def get_thread_messages_with_versions(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a thread with versioning information."""
        if hasattr(self.checkpointer, 'get_thread_messages_with_versions'):
            return self.checkpointer.get_thread_messages_with_versions(thread_id)
        return []

    async def asave_message_version(
        self,
        message_id: str,
        thread_id: str,
        version_id: str,
        version_number: int,
        role: str,
        content: str,
        parts: Optional[List[Dict[str, Any]]] = None,
        reasoning_content: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        is_active: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Async version of save_message_version."""
        return self.save_message_version(
            message_id, thread_id, version_id, version_number, role,
            content, parts, reasoning_content, model, temperature, is_active, metadata
        )

    async def aget_message_versions(self, message_id: str) -> List[Dict[str, Any]]:
        """Async version of get_message_versions."""
        return self.get_message_versions(message_id)

    async def aswitch_active_version(self, message_id: str, version_id: str) -> bool:
        """Async version of switch_active_version."""
        return self.switch_active_version(message_id, version_id)

    def invoke(
        self,
        messages: List[BaseMessage],
        thread_id: str,
        user_id: str = "default",
        mode: str = "companion",
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> CompanionState:
        """
        Sync invoke with conversation persistence.

        Args:
            messages: List of messages
            thread_id: Unique conversation thread ID
            user_id: User identifier
            mode: "companion" or "rp"
            config_overrides: Optional config overrides

        Returns:
            Updated conversation state
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        initial_state: CompanionState = {
            "messages": messages,
            "brain_state": {},
            "thread_id": thread_id,
            "user_id": user_id,
            "mode": mode,
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "config_overrides": config_overrides or {},
        }

        result = self.graph.invoke(initial_state, config)
        return result

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        thread_id: str,
        user_id: str = "default",
        mode: str = "companion",
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> CompanionState:
        """Async invoke (uses sync for now)."""
        return self.invoke(messages, thread_id, user_id, mode, config_overrides)

    def get_state(self, thread_id: str) -> Optional[CompanionState]:
        """
        Get current state for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Current state or None if thread doesn't exist
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = self.graph.get_state(config)
            if result and result.values:
                state = result.values
                # Check if state has meaningful content (has messages or thread_id)
                if state.get("messages") or state.get("thread_id") == thread_id:
                    return state
            return None
        except Exception:
            return None

    def get_history(
        self,
        thread_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of checkpoints to return

        Returns:
            List of historical states
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            history = list(self.graph.get_state_history(config, limit=limit))
            result = []
            for state in history:
                # Access checkpoint_id from the config's configurable dict
                checkpoint_id = None
                if hasattr(state, 'config') and state.config:
                    checkpoint_id = state.config.get("configurable", {}).get("checkpoint_id")

                # Get step from metadata
                step = None
                if hasattr(state, 'metadata') and state.metadata:
                    step = state.metadata.get("step")

                result.append({
                    "checkpoint_id": checkpoint_id,
                    "step": step,
                    "messages": state.values.get("messages", []),
                    "timestamp": state.values.get("last_activity"),
                })
            return result
        except Exception:
            return []

    def rewind_to_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
        steps_back: Optional[int] = None,
    ) -> Optional[CompanionState]:
        """
        Rewind conversation to a previous checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID to restore (None for latest)
            steps_back: Number of input checkpoints back to go (alternative to checkpoint_id)

        Returns:
            State at checkpoint, or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            if steps_back is not None:
                # Get history and find the Nth input checkpoint
                # LangGraph creates multiple checkpoints per invoke (input + loop)
                # We only count "input" checkpoints for steps_back
                history = list(self.graph.get_state_history(config))
                input_checkpoints = [
                    state for state in history
                    if state.metadata.get("source") == "input"
                ]

                if len(input_checkpoints) > steps_back:
                    target_config = input_checkpoints[steps_back].config
                    result = self.graph.get_state(target_config)
                    if result and result.values and result.values.get("messages"):
                        return result.values
                return None

            if checkpoint_id is not None:
                config["configurable"]["checkpoint_id"] = checkpoint_id

            result = self.graph.get_state(config)
            if result and result.values and result.values.get("messages"):
                return result.values
            return None
        except Exception:
            return None

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a conversation thread.

        Args:
            thread_id: Thread identifier

        Returns:
            True if deleted, False otherwise
        """
        try:
            # First check if thread exists
            state = self.get_state(thread_id)
            if state is None:
                return False

            # Delete using the checkpointer's delete_thread method
            # InMemorySaver and SQLiteSaver both have this method with signature (thread_id: str)
            self.checkpointer.delete_thread(thread_id)

            return True
        except Exception:
            return False

    def get_thread_stats(self, thread_id: str) -> Dict[str, Any]:
        """
        Get statistics for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Dictionary with thread statistics
        """
        state = self.get_state(thread_id)
        if not state:
            return {}

        messages = state.get("messages", [])
        return {
            "thread_id": thread_id,
            "user_id": state.get("user_id"),
            "mode": state.get("mode"),
            "message_count": len(messages),
            "last_activity": state.get("last_activity"),
            "human_messages": sum(1 for m in messages if isinstance(m, dict) or hasattr(m, 'type') and m.type == "human"),
            "ai_messages": sum(1 for m in messages if isinstance(m, dict) or hasattr(m, 'type') and m.type == "ai"),
            "system_messages": sum(1 for m in messages if isinstance(m, dict) or hasattr(m, 'type') and m.type == "system"),
        }


# Factory functions for easy creation

def create_memory_checkpointer() -> CompanionCheckpointManager:
    """Create in-memory checkpointer (for development/testing)."""
    return CompanionCheckpointManager(backend="memory")


# Singleton cache for checkpointer instances
_global_checkpointer: Optional[CompanionCheckpointManager] = None
_checkpointer_lock = threading.Lock()


def create_sqlite_checkpointer(checkpoint_dir: Optional[Path] = None) -> CompanionCheckpointManager:
    """Create SQLite checkpointer (default for production).

    Uses singleton pattern to avoid creating multiple database connections.
    """
    global _global_checkpointer

    with _checkpointer_lock:
        if _global_checkpointer is None:
            logger.info("[CHECKPOINTER] Creating singleton SQLite checkpointer")
            _global_checkpointer = CompanionCheckpointManager(backend="sqlite", checkpoint_dir=checkpoint_dir)
        return _global_checkpointer
