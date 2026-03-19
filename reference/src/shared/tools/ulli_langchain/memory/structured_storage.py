"""
Structured Storage Tool

A tool for the companion brain to store structured information about users.
Can be called during conversation when the user shares important facts.
"""

import sqlite3
import json
import re
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
from datetime import datetime, timezone

from langchain_core.tools import tool


class StructuredStorage:
    """
    Structured storage for user information.

    Stores facts in a searchable SQL database with:
    - User facts (key-value pairs)
    - Events (time-stamped occurrences)
    - Relationships (entity triples)
    - Documents (full text searchable)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        backend: Literal["sqlite", "memory"] = "sqlite",
    ):
        """
        Initialize structured storage.

        Args:
            db_path: Path to SQLite database
            backend: Storage backend
        """
        self.backend = backend
        if db_path is None:
            from src.constants import ROOT_DATA_DIR
            db_path = ROOT_DATA_DIR / "user_data.db"
        self.db_path = db_path

        if backend == "sqlite":
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
            self.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA busy_timeout=30000")  # 30 seconds
            self._create_tables()
        else:
            self._data: Dict[str, Dict[str, Any]] = {}

    def _create_tables(self):
        """Create storage tables."""
        # User facts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                value_type TEXT DEFAULT 'string',
                metadata TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, key)
            )
        """)

        # Events table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                event_date TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Relationships table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, subject, predicate, object)
            )
        """)

        # Documents table (for full text search)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_type TEXT DEFAULT 'note',
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON user_facts(user_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(user_id, event_date)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_user ON relationships(user_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id)")

        self.conn.commit()

    def store_fact(
        self,
        user_id: str,
        key: str,
        value: Any,
        value_type: str = "string",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a fact about a user."""
        if self.backend == "sqlite":
            try:
                value_str = str(value)
                metadata_json = json.dumps(metadata) if metadata else None

                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO user_facts
                    (user_id, key, value, value_type, metadata, confidence, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, key, value_str, value_type, metadata_json, confidence, datetime.now(timezone.utc).isoformat())
                )
                self.conn.commit()
                return True
            except Exception:
                return False
        else:
            if user_id not in self._data:
                self._data[user_id] = {"facts": {}, "events": [], "relationships": {}, "documents": []}
            self._data[user_id]["facts"][key] = {
                "value": value,
                "value_type": value_type,
                "confidence": confidence,
                "metadata": metadata,
            }
            return True

    def get_fact(
        self,
        user_id: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a fact about a user."""
        if self.backend == "sqlite":
            row = self.conn.execute(
                "SELECT * FROM user_facts WHERE user_id = ? AND key = ?",
                (user_id, key)
            ).fetchone()

            if row:
                return {
                    "key": row["key"],
                    "value": row["value"],
                    "value_type": row["value_type"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    "confidence": row["confidence"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            return None
        else:
            return self._data.get(user_id, {}).get("facts", {}).get(key)

    def store_event(
        self,
        user_id: str,
        event_type: str,
        description: str,
        event_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store an event."""
        if self.backend == "sqlite":
            try:
                metadata_json = json.dumps(metadata) if metadata else None
                self.conn.execute(
                    """
                    INSERT INTO events
                    (user_id, event_type, description, event_date, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, event_type, description, event_date, metadata_json)
                )
                self.conn.commit()
                return True
            except Exception:
                return False
        else:
            if user_id not in self._data:
                self._data[user_id] = {"facts": {}, "events": [], "relationships": {}, "documents": []}
            self._data[user_id]["events"].append({
                "event_type": event_type,
                "description": description,
                "event_date": event_date,
                "metadata": metadata,
            })
            return True

    def store_relationship(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a relationship."""
        if self.backend == "sqlite":
            try:
                metadata_json = json.dumps(metadata) if metadata else None
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO relationships
                    (user_id, subject, predicate, object, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, subject, predicate, obj, confidence, metadata_json)
                )
                self.conn.commit()
                return True
            except Exception:
                return False
        else:
            if user_id not in self._data:
                self._data[user_id] = {"facts": {}, "events": [], "relationships": {}, "documents": []}
            key = f"{subject}_{predicate}_{obj}"
            self._data[user_id]["relationships"][key] = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence,
                "metadata": metadata,
            }
            return True

    def search(
        self,
        user_id: str,
        query: str,
        search_in: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search across all stored data.

        Args:
            user_id: User identifier
            query: Search query
            search_in: List of areas to search ("facts", "events", "relationships", "documents")

        Returns:
            List of matching results
        """
        results = []
        search_in = search_in or ["facts", "events", "relationships", "documents"]
        pattern = f"%{query}%"

        if "facts" in search_in and self.backend == "sqlite":
            rows = self.conn.execute(
                "SELECT 'fact' as type, key, value, created_at FROM user_facts WHERE user_id = ? AND (key LIKE ? OR value LIKE ?)",
                (user_id, pattern, pattern)
            ).fetchall()
            for row in rows:
                results.append({
                    "type": "fact",
                    "key": row["key"],
                    "value": row["value"],
                    "created_at": row["created_at"],
                })

        if "events" in search_in and self.backend == "sqlite":
            rows = self.conn.execute(
                "SELECT 'event' as type, event_type, description, event_date FROM events WHERE user_id = ? AND description LIKE ?",
                (user_id, pattern)
            ).fetchall()
            for row in rows:
                results.append({
                    "type": "event",
                    "event_type": row["event_type"],
                    "description": row["description"],
                    "event_date": row["event_date"],
                })

        if "relationships" in search_in and self.backend == "sqlite":
            rows = self.conn.execute(
                "SELECT 'relationship' as type, subject, predicate, object FROM relationships WHERE user_id = ? AND (subject LIKE ? OR object LIKE ?)",
                (user_id, pattern, pattern)
            ).fetchall()
            for row in rows:
                results.append({
                    "type": "relationship",
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                })

        return results

    def get_all_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all facts for a user."""
        if self.backend == "sqlite":
            rows = self.conn.execute(
                "SELECT * FROM user_facts WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            ).fetchall()

            return [
                {
                    "key": row["key"],
                    "value": row["value"],
                    "value_type": row["value_type"],
                    "confidence": row["confidence"],
                }
                for row in rows
            ]
        else:
            facts = self._data.get(user_id, {}).get("facts", {})
            return [{"key": k, **v} for k, v in facts.items()]

    def get_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of all stored data for a user."""
        facts = self.get_all_facts(user_id)

        summary = {
            "fact_count": len(facts),
            "facts": {f["key"]: f["value"] for f in facts},
        }

        return summary


# Extract patterns for identifying structured information

PATTERNS = {
    "work": [
        r"I work at (.+)",
        r"I'm employed at (.+)",
        r"I work for (.+)",
        r"My job is at (.+)",
    ],
    "live": [
        r"I live in (.+)",
        r"I'm from (.+)",
        r"I reside in (.+)",
        r"My home is in (.+)",
    ],
    "relationship": [
        r"(.+) is my (.+)",  # "Alice is my sister"
        r"I am (.+)'s (.+)",  # "I am John's friend"
    ],
    "preference": [
        r"I like (.+)",
        r"I love (.+)",
        r"I enjoy (.+)",
        r"I don't like (.+)",
        r"I hate (.+)",
    ],
}


def extract_structured_info(text: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Extract structured information from text using patterns.

    Args:
        text: Text to analyze
        user_id: User identifier

    Returns:
        List of extracted facts with metadata
    """
    extracted = []

    for category, patterns in PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                # Remove trailing punctuation
                value = re.sub(r'[.!?]+$', '', value)

                extracted.append({
                    "category": category,
                    "key": category,  # Could be more specific
                    "value": value,
                    "confidence": 0.9,  # High confidence for explicit patterns
                    "source": "extracted",
                })

    return extracted


# LangChain tool interface

def create_structured_storage_tool(
    storage: StructuredStorage,
    user_id: str,
):
    """
    Create a LangChain tool for structured storage.

    This tool can be called by the LLM during conversation
    to store important information.
    """

    @tool
    def store_user_info(
        key: str,
        value: str,
        category: str = "general",
    ) -> str:
        """
        Store important information about the user.

        Use this when the user shares:
        - Personal details (name, location, job)
        - Preferences (likes, dislikes)
        - Important events
        - Relationships

        Args:
            key: What type of information (e.g., "location", "job", "favorite_color")
            value: The information to store
            category: Category for grouping (personal, preference, event, etc.)

        Returns:
            Confirmation message
        """
        success = storage.store_fact(user_id, key, value, value_type="string", confidence=0.9)
        if success:
            return f"I've noted that {key}: {value}. I'll remember this for future conversations."
        else:
            return "I had trouble saving that information, but I'll keep it in mind for now."

    @tool
    def recall_user_info(
        query: str,
    ) -> str:
        """
        Recall stored information about the user.

        Args:
            query: What to search for

        Returns:
            Retrieved information
        """
        results = storage.search(user_id, query)

        if results:
            response_parts = []
            for result in results[:5]:  # Limit to 5 results
                if result["type"] == "fact":
                    response_parts.append(f"- {result['key']}: {result['value']}")
                elif result["type"] == "event":
                    response_parts.append(f"- Event: {result['description']}")
                elif result["type"] == "relationship":
                    response_parts.append(f"- {result['subject']} {result['predicate']} {result['object']}")

            return "I recall the following:\n" + "\n".join(response_parts)
        else:
            return f"I don't have any stored information matching '{query}'."

    return [store_user_info, recall_user_info]


# Factory function

def create_structured_storage(
    db_path: Optional[Path] = None,
    backend: Literal["sqlite", "memory"] = "sqlite",
) -> StructuredStorage:
    """Create a structured storage instance."""
    return StructuredStorage(db_path=db_path, backend=backend)
