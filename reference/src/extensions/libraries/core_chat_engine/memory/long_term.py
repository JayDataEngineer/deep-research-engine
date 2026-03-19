"""
Long-Term Memory Module

Stores and retrieves important facts about users across conversations.
Provides persistent memory of user preferences, important events, and personal details.
"""

import json
import sqlite3
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class MemoryFact:
    """A single fact stored in long-term memory."""
    key: str
    value: str
    category: str  # "preference", "personal", "event", "fact", "relationship"
    confidence: float  # 0.0 to 1.0
    source: str  # "user_stated", "inferred", "conversation"
    timestamp: str
    access_count: int = 0
    last_accessed: Optional[str] = None


class LongTermMemory:
    """
    Long-term memory store for user facts and preferences.

    Stores:
    - Personal information (name, location, job, etc.)
    - Preferences (likes, dislikes, interests)
    - Important events or milestones
    - Facts mentioned in conversation
    - Relationships between concepts

    Memory can be:
    - Explicit: Directly stated by user ("I live in Tokyo")
    - Inferred: Derived from conversation
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        backend: Literal["sqlite", "memory"] = "sqlite",
    ):
        """
        Initialize long-term memory.

        Args:
            storage_path: Path to SQLite database (for sqlite backend)
            backend: Storage backend to use
        """
        self.backend = backend
        if storage_path is None:
            from src.constants import ROOT_DATA_DIR
            storage_path = ROOT_DATA_DIR / "memory.db"
        self.storage_path = storage_path

        if backend == "sqlite":
            self.conn = sqlite3.connect(str(self.storage_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
        else:
            self._memory: Dict[str, List[MemoryFact]] = {}

    def _create_tables(self):
        """Create memory tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'user_stated',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                UNIQUE(user_id, key)
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_facts_user
            ON user_facts(user_id)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_facts_category
            ON user_facts(user_id, category)
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'user_stated',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, subject, predicate, object)
            )
        """)

        self.conn.commit()

    def store_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        category: str = "fact",
        confidence: float = 1.0,
        source: str = "user_stated",
    ) -> bool:
        """
        Store a fact about a user.

        Args:
            user_id: User identifier
            key: Fact key (e.g., "location", "job", "favorite_color")
            value: Fact value
            category: Category for grouping facts
            confidence: Confidence score (0.0 to 1.0)
            source: How we learned this fact

        Returns:
            True if stored successfully
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.backend == "sqlite":
            try:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO user_facts
                    (user_id, key, value, category, confidence, source, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, key, value, category, confidence, source, timestamp)
                )
                self.conn.commit()
                return True
            except Exception:
                return False
        else:
            if user_id not in self._memory:
                self._memory[user_id] = []
            fact = MemoryFact(
                key=key, value=value, category=category,
                confidence=confidence, source=source, timestamp=timestamp
            )
            # Remove existing fact with same key
            self._memory[user_id] = [f for f in self._memory[user_id] if f.key != key]
            self._memory[user_id].append(fact)
            return True

    def get_fact(
        self,
        user_id: str,
        key: str,
    ) -> Optional[MemoryFact]:
        """
        Get a specific fact about a user.

        Args:
            user_id: User identifier
            key: Fact key

        Returns:
            MemoryFact if found, None otherwise
        """
        if self.backend == "sqlite":
            row = self.conn.execute(
                "SELECT * FROM user_facts WHERE user_id = ? AND key = ?",
                (user_id, key)
            ).fetchone()

            if row:
                # Update access count
                self.conn.execute(
                    "UPDATE user_facts SET access_count = access_count + 1, last_accessed = ? WHERE user_id = ? AND key = ?",
                    (datetime.now(timezone.utc).isoformat(), user_id, key)
                )
                self.conn.commit()

                return MemoryFact(
                    key=row["key"],
                    value=row["value"],
                    category=row["category"],
                    confidence=row["confidence"],
                    source=row["source"],
                    timestamp=row["timestamp"],
                    access_count=row["access_count"],
                    last_accessed=row["last_accessed"],
                )
            return None
        else:
            facts = self._memory.get(user_id, [])
            for fact in facts:
                if fact.key == key:
                    fact.access_count += 1
                    fact.last_accessed = datetime.now(timezone.utc).isoformat()
                    return fact
            return None

    def get_all_facts(
        self,
        user_id: str,
        category: Optional[str] = None,
    ) -> List[MemoryFact]:
        """
        Get all facts for a user, optionally filtered by category.

        Args:
            user_id: User identifier
            category: Optional category filter

        Returns:
            List of MemoryFact objects
        """
        if self.backend == "sqlite":
            if category:
                rows = self.conn.execute(
                    "SELECT * FROM user_facts WHERE user_id = ? AND category = ? ORDER BY timestamp DESC",
                    (user_id, category)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM user_facts WHERE user_id = ? ORDER BY timestamp DESC",
                    (user_id,)
                ).fetchall()

            return [
                MemoryFact(
                    key=row["key"],
                    value=row["value"],
                    category=row["category"],
                    confidence=row["confidence"],
                    source=row["source"],
                    timestamp=row["timestamp"],
                    access_count=row["access_count"],
                    last_accessed=row["last_accessed"],
                )
                for row in rows
            ]
        else:
            facts = self._memory.get(user_id, [])
            if category:
                facts = [f for f in facts if f.category == category]
            return facts

    def search_facts(
        self,
        user_id: str,
        query: str,
    ) -> List[MemoryFact]:
        """
        Search for facts matching a query string.

        Args:
            user_id: User identifier
            query: Search query (matches against key and value)

        Returns:
            List of matching MemoryFact objects
        """
        if self.backend == "sqlite":
            pattern = f"%{query}%"
            rows = self.conn.execute(
                "SELECT * FROM user_facts WHERE user_id = ? AND (key LIKE ? OR value LIKE ?) ORDER BY timestamp DESC",
                (user_id, pattern, pattern)
            ).fetchall()

            return [
                MemoryFact(
                    key=row["key"],
                    value=row["value"],
                    category=row["category"],
                    confidence=row["confidence"],
                    source=row["source"],
                    timestamp=row["timestamp"],
                    access_count=row["access_count"],
                    last_accessed=row["last_accessed"],
                )
                for row in rows
            ]
        else:
            facts = self._memory.get(user_id, [])
            query_lower = query.lower()
            return [
                f for f in facts
                if query_lower in f.key.lower() or query_lower in f.value.lower()
            ]

    def delete_fact(
        self,
        user_id: str,
        key: str,
    ) -> bool:
        """Delete a fact about a user."""
        if self.backend == "sqlite":
            self.conn.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND key = ?",
                (user_id, key)
            )
            self.conn.commit()
            return True
        else:
            if user_id in self._memory:
                self._memory[user_id] = [f for f in self._memory[user_id] if f.key != key]
                return True
            return False

    def store_relationship(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "user_stated",
    ) -> bool:
        """
        Store a relationship triple (subject, predicate, object).

        Examples:
            - ("user", "lives_in", "Tokyo")
            - ("user", "works_at", "Google")
            - ("Alice", "is_friend_of", "user")

        Args:
            user_id: User identifier
            subject: Subject entity
            predicate: Relationship type
            obj: Object entity
            confidence: Confidence score
            source: How we learned this

        Returns:
            True if stored successfully
        """
        if self.backend == "sqlite":
            try:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO relationships
                    (user_id, subject, predicate, object, confidence, source, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, subject, predicate, obj, confidence, source, datetime.now(timezone.utc).isoformat())
                )
                self.conn.commit()
                return True
            except Exception:
                return False
        else:
            # For in-memory, we could implement this too
            # For now, just store as a fact
            return self.store_fact(
                user_id,
                f"{subject}_{predicate}",
                obj,
                "relationship",
                confidence,
                source,
            )

    def get_relationships(
        self,
        user_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get relationships, optionally filtered by subject or predicate.

        Args:
            user_id: User identifier
            subject: Optional subject filter
            predicate: Optional predicate filter

        Returns:
            List of relationship dicts
        """
        if self.backend == "sqlite":
            query = "SELECT * FROM relationships WHERE user_id = ?"
            params = [user_id]

            if subject:
                query += " AND subject = ?"
                params.append(subject)
            if predicate:
                query += " AND predicate = ?"
                params.append(predicate)

            rows = self.conn.execute(query + " ORDER BY timestamp DESC", params).fetchall()

            return [
                {
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]
        else:
            # For in-memory, not fully implemented
            return []

    def get_context_string(
        self,
        user_id: str,
        include_categories: Optional[List[str]] = None,
    ) -> str:
        """
        Get a formatted string of all facts for prompt injection.

        Args:
            user_id: User identifier
            include_categories: Optional list of categories to include

        Returns:
            Formatted string of facts
        """
        facts = self.get_all_facts(user_id)

        if include_categories:
            facts = [f for f in facts if f.category in include_categories]

        if not facts:
            return ""

        lines = ["User Information:"]
        for fact in facts:
            lines.append(f"- {fact.key}: {fact.value}")

        # Also add relationships
        relationships = self.get_relationships(user_id)
        if relationships:
            lines.append("\nRelationships:")
            for rel in relationships[:10]:  # Limit to 10 relationships
                lines.append(f"- {rel['subject']} {rel['predicate'].replace('_', ' ')} {rel['object']}")

        return "\n".join(lines)

    def get_summary(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of all stored information about a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with fact counts and key facts
        """
        facts = self.get_all_facts(user_id)
        relationships = self.get_relationships(user_id)

        summary = {
            "total_facts": len(facts),
            "total_relationships": len(relationships),
            "categories": {},
            "key_facts": {},
        }

        # Count by category
        for fact in facts:
            cat = fact.category
            summary["categories"][cat] = summary["categories"].get(cat, 0) + 1

        # Get important facts
        important_categories = ["personal", "preference"]
        for cat in important_categories:
            cat_facts = [f for f in facts if f.category == cat]
            for fact in cat_facts[:5]:  # Top 5 per category
                summary["key_facts"][fact.key] = fact.value

        return summary


# Factory functions

def create_long_term_memory(
    storage_path: Optional[Path] = None,
    backend: Literal["sqlite", "memory"] = "sqlite",
) -> LongTermMemory:
    """Create a long-term memory instance."""
    return LongTermMemory(storage_path=storage_path, backend=backend)


def create_memory_long_term_memory() -> LongTermMemory:
    """Create an in-memory long-term memory (for testing)."""
    return LongTermMemory(backend="memory")
