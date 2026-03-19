"""API Key Management Service.

Provides secure storage and retrieval of user API keys.
Keys are encrypted at rest using Fernet symmetric encryption.

Supports two backends:
- SQLite (legacy, file-based)
- PostgreSQL (production-ready, shared storage)
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union
from pydantic import BaseModel, Field

from src.utils.database import get_async_pool, get_pool
from src.utils.encryption import encrypt_value, decrypt_value
from src.constants import API_KEYS_DB_PATH

logger = logging.getLogger(__name__)


def get_postgres_url() -> Optional[str]:
    """Get PostgreSQL connection string from environment."""
    return os.getenv("POSTGRES_URL") or os.getenv("CONTEXT_ENGINE_POSTGRES_CONNECTION_STRING")


# API key types supported
class KeyType:
    """Supported API key types.

    With LiteLLM proxy handling all provider routing, we only need:
    - LITELLM: Primary key for LiteLLM proxy
    - CUSTOM: For custom OpenAI-compatible endpoints

    Legacy provider-specific keys (OPENAI, ANTHROPIC, ZAI, etc.) are
    deprecated but kept for backward compatibility during migration.
    """
    LITELLM = "litellm"  # Primary: LiteLLM proxy key
    CUSTOM = "custom"     # Custom OpenAI-compatible endpoint

    # Deprecated: Provider-specific keys managed by LiteLLM now
    OPENAI = "openai"      # Deprecated
    ANTHROPIC = "anthropic"  # Deprecated
    OPENROUTER = "openrouter"  # Deprecated
    ZAI = "zai"            # Deprecated
    GLM = "glm"            # Deprecated
    GOOGLE = "google"      # Deprecated
    COHERE = "cohere"      # Deprecated
    OLLAMA = "ollama"      # Deprecated
    LLAMACPP = "llamacpp"  # Deprecated


class APIKey(BaseModel):
    """API key model."""
    id: int = Field(..., description="Database ID")
    key_type: str = Field(..., description="Type of API key (e.g., 'openai', 'anthropic')")
    name: str = Field(..., description="User-provided name for this key")
    encrypted_key: str = Field(..., description="Encrypted API key")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    last_used: Optional[str] = Field(None, description="Last used timestamp (ISO format)")
    is_active: bool = Field(True, description="Whether the key is active")
    base_url: Optional[str] = Field(None, description="Custom base URL for this key")


class APIKeyCreate(BaseModel):
    """Model for creating a new API key."""
    key_type: str = Field(..., description="Type of API key")
    name: str = Field(..., min_length=1, max_length=100, description="User-provided name")
    api_key: str = Field(..., min_length=1, description="The API key to store")
    base_url: Optional[str] = Field(None, description="Custom base URL (overrides default)")


class APIKeyUpdate(BaseModel):
    """Model for updating an API key."""
    api_key: str = Field(..., min_length=1, description="The new API key")


class APIKeyResponse(BaseModel):
    """Response model for API key (without the actual key)."""
    id: int
    key_type: str
    name: str
    created_at: str
    last_used: Optional[str]
    is_active: bool
    base_url: Optional[str] = None
    # Partial key for identification (first 8 chars)
    key_preview: str


class APIKeyService:
    """Service for managing encrypted API keys.

    Supports SQLite and PostgreSQL backends via environment configuration.
    - PostgreSQL: Used when POSTGRES_URL is set (production)
    - SQLite: Used when POSTGRES_URL is not set (legacy/dev)
    """

    def __init__(self, db_path: Optional[Path] = None, postgres_url: Optional[str] = None):
        """Initialize the API key service.

        Args:
            db_path: Path to the SQLite database file (only used if POSTGRES_URL not set)
            postgres_url: PostgreSQL connection string (overrides environment)
        """
        self.postgres_url = postgres_url or get_postgres_url()
        self.backend = "postgres" if self.postgres_url else "sqlite"

        if self.backend == "postgres":
            self.pg_pool: Optional[asyncpg.Pool] = None
            logger.info("[APIKeyService] Using PostgreSQL backend")
        else:
            if db_path is None:
                db_path = API_KEYS_DB_PATH
            self.db_path = db_path
            self.pool = get_async_pool(db_path)
            self.sync_pool = get_pool(db_path)
            logger.info(f"[APIKeyService] Using SQLite backend: {self.db_path}")

    async def _get_pg_pool(self) -> "asyncpg.Pool":
        """Get or create PostgreSQL connection pool."""
        import asyncpg

        if self.pg_pool is None:
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
        return self.pg_pool

    async def initialize(self):
        """Initialize the database schema."""
        if self.backend == "postgres":
            await self._init_postgres()
        else:
            await self._init_sqlite()

    async def _init_postgres(self):
        """Initialize PostgreSQL database schema."""
        pool = await self._get_pg_pool()

        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id SERIAL PRIMARY KEY,
                    key_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    is_active INTEGER DEFAULT 1,
                    base_url TEXT,
                    UNIQUE(key_type, name)
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_type ON api_keys(key_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active)")

        logger.info("[APIKeyService] PostgreSQL schema initialized")

    async def _init_sqlite(self):
        """Initialize SQLite database schema."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT,
                    is_active INTEGER DEFAULT 1,
                    base_url TEXT,
                    UNIQUE(key_type, name)
                )
            """)
            # Migration: add base_url column if it doesn't exist
            try:
                await conn.execute("ALTER TABLE api_keys ADD COLUMN base_url TEXT")
            except Exception:
                pass  # Column already exists
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_type
                ON api_keys(key_type)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_active
                ON api_keys(is_active)
            """)
            await conn.commit()

        logger.info(f"[APIKeyService] SQLite schema initialized: {self.db_path}")

    async def create_key(self, key_data: APIKeyCreate) -> APIKeyResponse:
        """Store a new API key.

        Args:
            key_data: The key data to store

        Returns:
            The created key metadata
        """
        # Encrypt the key
        encrypted_key = encrypt_value(key_data.api_key)

        # Get current timestamp
        created_at = datetime.utcnow().isoformat()

        if self.backend == "postgres":
            pool = await self._get_pg_pool()
            async with pool.acquire() as conn:
                key_id = await conn.fetchval(
                    """
                    INSERT INTO api_keys (key_type, name, encrypted_key, created_at, is_active, base_url)
                    VALUES ($1, $2, $3, $4, 1, $5)
                    RETURNING id
                    """,
                    key_data.key_type, key_data.name, encrypted_key, created_at, key_data.base_url
                )
        else:
            async with self.pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    INSERT INTO api_keys (key_type, name, encrypted_key, created_at, is_active, base_url)
                    VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    (key_data.key_type, key_data.name, encrypted_key, created_at, key_data.base_url)
                )
                await conn.commit()
                key_id = cursor.lastrowid

        logger.info(f"Created API key: {key_data.key_type}/{key_data.name} (ID: {key_id})")

        return APIKeyResponse(
            id=key_id,
            key_type=key_data.key_type,
            name=key_data.name,
            created_at=created_at,
            last_used=None,
            is_active=True,
            base_url=key_data.base_url,
            key_preview=self._mask_key(key_data.api_key)
        )

    async def get_key(self, key_type: str, name: str) -> Optional[str]:
        """Retrieve and decrypt an API key.

        Args:
            key_type: The type of key
            name: The name of the key

        Returns:
            The decrypted API key, or None if not found
        """
        if self.backend == "postgres":
            pool = await self._get_pg_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT encrypted_key FROM api_keys
                    WHERE key_type = $1 AND name = $2 AND is_active = 1
                    """,
                    key_type, name
                )
        else:
            async with self.pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    SELECT encrypted_key FROM api_keys
                    WHERE key_type = ? AND name = ? AND is_active = 1
                    """,
                    (key_type, name)
                )
                row = await cursor.fetchone()

        if row is None:
            return None

        encrypted_key = row[0] if self.backend == "postgres" else row["encrypted_key"]
        decrypted_key = decrypt_value(encrypted_key)

        # Update last_used timestamp
        await self._update_last_used(key_type, name)

        return decrypted_key

    def get_key_sync(self, key_type: str, name: str) -> Optional[str]:
        """Synchronously retrieve and decrypt an API key.

        Note: This method only supports SQLite backend. For PostgreSQL,
        use the async get_key() method instead.

        Args:
            key_type: The type of key
            name: The name of the key

        Returns:
            The decrypted API key, or None if not found

        Raises:
            RuntimeError: If called with PostgreSQL backend
        """
        if self.backend == "postgres":
            raise RuntimeError(
                "[APIKeyService] get_key_sync is not supported with PostgreSQL backend. "
                "Use 'await get_key()' instead. PostgreSQL requires async operations for proper connection pooling."
            )

        try:
            cursor = self.sync_pool.execute(
                """
                SELECT encrypted_key FROM api_keys
                WHERE key_type = ? AND name = ? AND is_active = 1
                """,
                (key_type, name)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            encrypted_key = row[0]
            decrypted_key = decrypt_value(encrypted_key)

            # Update last_used timestamp
            self._update_last_used_sync(key_type, name)

            return decrypted_key
        except Exception as e:
            logger.debug(f"[APIKeyService] Error in get_key_sync: {e}")
            return None

    async def list_keys(self, key_type: Optional[str] = None) -> List[APIKeyResponse]:
        """List all stored API keys (without decrypting them).

        Args:
            key_type: Optional filter by key type

        Returns:
            List of API key metadata
        """
        if self.backend == "postgres":
            pool = await self._get_pg_pool()
            if key_type:
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, key_type, name, created_at, last_used, is_active, encrypted_key, base_url
                        FROM api_keys
                        WHERE key_type = $1
                        ORDER BY created_at DESC
                        """,
                        key_type
                    )
            else:
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT id, key_type, name, created_at, last_used, is_active, encrypted_key, base_url
                        FROM api_keys
                        ORDER BY created_at DESC
                        """
                    )
        else:
            if key_type:
                async with self.pool.acquire() as conn:
                    cursor = await conn.execute(
                        """
                        SELECT id, key_type, name, created_at, last_used, is_active, encrypted_key, base_url
                        FROM api_keys
                        WHERE key_type = ?
                        ORDER BY created_at DESC
                        """,
                        (key_type,)
                    )
            else:
                async with self.pool.acquire() as conn:
                    cursor = await conn.execute(
                        """
                        SELECT id, key_type, name, created_at, last_used, is_active, encrypted_key, base_url
                        FROM api_keys
                        ORDER BY created_at DESC
                        """
                    )

            rows = await cursor.fetchall()

        keys = []

        for row in rows:
            # Create preview by decrypting just enough to show first 8 chars
            # SQLite returns tuples (access by index), PostgreSQL returns Record objects (also access by index)
            encrypted_key = row[6]
            try:
                decrypted = decrypt_value(encrypted_key)
                key_preview = self._mask_key(decrypted)
            except Exception:
                key_preview = "********"

            keys.append(APIKeyResponse(
                id=row[0],
                key_type=row[1],
                name=row[2],
                created_at=row[3],
                last_used=row[4],
                is_active=bool(row[5]),
                base_url=row[7],
                key_preview=key_preview
            ))

        return keys

    def list_keys_sync(self, key_type: Optional[str] = None) -> List[APIKeyResponse]:
        """Synchronously list all stored API keys (without decrypting them).

        Note: This method only supports SQLite backend. For PostgreSQL,
        use the async list_keys() method instead.

        Args:
            key_type: Optional filter by key type

        Returns:
            List of API key metadata

        Raises:
            RuntimeError: If called with PostgreSQL backend
        """
        if self.backend == "postgres":
            raise RuntimeError(
                "[APIKeyService] list_keys_sync is not supported with PostgreSQL backend. "
                "Use 'await list_keys()' instead. PostgreSQL requires async operations for proper connection pooling."
            )

        if key_type:
            cursor = self.sync_pool.execute(
                """
                SELECT id, key_type, name, created_at, last_used, is_active, encrypted_key, base_url
                FROM api_keys
                WHERE key_type = ?
                ORDER BY created_at DESC
                """,
                (key_type,)
            )
        else:
            cursor = self.sync_pool.execute(
                """
                SELECT id, key_type, name, created_at, last_used, is_active, encrypted_key, base_url
                FROM api_keys
                ORDER BY created_at DESC
                """
            )

        rows = cursor.fetchall()
        keys = []

        for row in rows:
            # Create preview by decrypting just enough to show first 8 chars
            encrypted_key = row[6]
            try:
                decrypted = decrypt_value(encrypted_key)
                key_preview = self._mask_key(decrypted)
            except Exception:
                key_preview = "********"

            keys.append(APIKeyResponse(
                id=row[0],
                key_type=row[1],
                name=row[2],
                created_at=row[3],
                last_used=row[4],
                is_active=bool(row[5]),
                base_url=row[7],
                key_preview=key_preview
            ))

        return keys

    async def update_key(self, key_type: str, name: str, key_data: APIKeyUpdate) -> bool:
        """Update an existing API key.

        Args:
            key_type: The type of key
            name: The name of the key
            key_data: New key data

        Returns:
            True if updated, False if not found
        """
        encrypted_key = encrypt_value(key_data.api_key)

        if self.backend == "postgres":
            pool = await self._get_pg_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE api_keys
                    SET encrypted_key = $1
                    WHERE key_type = $2 AND name = $3
                    """,
                    encrypted_key, key_type, name
                )
                updated = result == "UPDATE 1"
        else:
            async with self.pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    UPDATE api_keys
                    SET encrypted_key = ?
                    WHERE key_type = ? AND name = ?
                    """,
                    (encrypted_key, key_type, name)
                )
                await conn.commit()
                updated = cursor.rowcount > 0

        if updated:
            logger.info(f"Updated API key: {key_type}/{name}")

        return updated

    async def delete_key(self, key_type: str, name: str) -> bool:
        """Delete an API key.

        Args:
            key_type: The type of key
            name: The name of the key

        Returns:
            True if deleted, False if not found
        """
        if self.backend == "postgres":
            pool = await self._get_pg_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM api_keys
                    WHERE key_type = $1 AND name = $2
                    """,
                    key_type, name
                )
                deleted = result == "DELETE 1"
        else:
            async with self.pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    DELETE FROM api_keys
                    WHERE key_type = ? AND name = ?
                    """,
                    (key_type, name)
                )
                await conn.commit()
                deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted API key: {key_type}/{name}")

        return deleted

    async def deactivate_key(self, key_type: str, name: str) -> bool:
        """Deactivate an API key (without deleting it).

        Args:
            key_type: The type of key
            name: The name of the key

        Returns:
            True if deactivated, False if not found
        """
        if self.backend == "postgres":
            pool = await self._get_pg_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE api_keys
                    SET is_active = 0
                    WHERE key_type = $1 AND name = $2
                    """,
                    key_type, name
                )
                deactivated = result == "UPDATE 1"
        else:
            async with self.pool.acquire() as conn:
                cursor = await conn.execute(
                    """
                    UPDATE api_keys
                    SET is_active = 0
                    WHERE key_type = ? AND name = ?
                    """,
                    (key_type, name)
                )
                await conn.commit()
                deactivated = cursor.rowcount > 0

        if deactivated:
            logger.info(f"Deactivated API key: {key_type}/{name}")

        return deactivated

    async def _update_last_used(self, key_type: str, name: str):
        """Update the last_used timestamp for a key.

        Args:
            key_type: The type of key
            name: The name of the key
        """
        last_used = datetime.utcnow().isoformat()

        if self.backend == "postgres":
            pool = await self._get_pg_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE api_keys
                    SET last_used = $1
                    WHERE key_type = $2 AND name = $3
                    """,
                    last_used, key_type, name
                )
        else:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE api_keys
                    SET last_used = ?
                    WHERE key_type = ? AND name = ?
                    """,
                    (last_used, key_type, name)
                )
                await conn.commit()

    def _update_last_used_sync(self, key_type: str, name: str):
        """Update the last_used timestamp for a key (synchronous).

        Args:
            key_type: The type of key
            name: The name of the key
        """
        last_used = datetime.utcnow().isoformat()

        self.sync_pool.execute(
            """
            UPDATE api_keys
            SET last_used = ?
            WHERE key_type = ? AND name = ?
            """,
            (last_used, key_type, name)
        )
        self.sync_pool.commit()

    @staticmethod
    def _mask_key(key: str, visible_chars: int = 8) -> str:
        """Mask an API key for display.

        Args:
            key: The API key to mask
            visible_chars: Number of characters to show at the start

        Returns:
            Masked key (e.g., "sk-1234********")
        """
        if len(key) <= visible_chars:
            return key

        return key[:visible_chars] + "*" * (len(key) - visible_chars)


# =============================================================================
# Global service instance
# =============================================================================

_service: Optional[APIKeyService] = None


def get_api_key_service() -> APIKeyService:
    """Get the global API key service instance.

    Returns:
        APIKeyService instance
    """
    global _service

    if _service is None:
        _service = APIKeyService()

    return _service
