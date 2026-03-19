"""
Postgres initialization script for Context Engine.

Creates pgvector extension, tables, and indexes for artifact storage.
"""
import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
from typing import Optional
import logging

logger = logging.getLogger(__name__)


async def init_postgres(
    connection_string: str = "postgresql://ulli:ulli_password@localhost:5432/ulli_db",
    embedding_dim: int = 1536,  # Default for text-embedding-3-large
    artifact_table: str = "artifacts",
    embedding_table: str = "artifact_embeddings"
):
    """Initialize Postgres with pgvector extension and tables."""

    conn = None
    try:
        # Connect to database
        conn = await asyncpg.connect(connection_string)

        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✓ Enabled pgvector extension")

        # Create artifacts table with session isolation and deduplication
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {artifact_table} (
                id VARCHAR(255) PRIMARY KEY,
                content TEXT NOT NULL,
                artifact_type VARCHAR(50) NOT NULL,
                session_id VARCHAR(255) NOT NULL,  -- Session isolation (CRITICAL)
                user_id VARCHAR(255),               -- User ID for companion mode (NULL for research)
                source_ids TEXT[] DEFAULT '{{}}',
                content_hash VARCHAR(64),           -- SHA-256 hash for deduplication
                relevance_score FLOAT DEFAULT 1.0,
                decay_rate FLOAT DEFAULT 0.1,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        print(f"✓ Created table: {artifact_table}")

        # Create artifact embeddings table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {embedding_table} (
                artifact_id VARCHAR(255) PRIMARY KEY REFERENCES {artifact_table}(id) ON DELETE CASCADE,
                embedding vector({embedding_dim}) NOT NULL
            )
        """)
        print(f"✓ Created table: {embedding_table}")

        # Create indexes for common queries
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{artifact_table}_session_id
            ON {artifact_table}(session_id)
        """)
        print(f"✓ Created index: idx_{artifact_table}_session_id (CRITICAL for isolation)")

        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{artifact_table}_user_id
            ON {artifact_table}(user_id)
        """)
        print(f"✓ Created index: idx_{artifact_table}_user_id (for companion mode)")

        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{artifact_table}_artifact_type
            ON {artifact_table}(artifact_type)
        """)
        print(f"✓ Created index: idx_{artifact_table}_artifact_type")

        # Create composite index for session + artifact type (CRITICAL for performance)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{artifact_table}_session_type
            ON {artifact_table}(session_id, artifact_type)
        """)
        print(f"✓ Created index: idx_{artifact_table}_session_type")

        # Create composite index for user + created_at (for companion mode)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{artifact_table}_user_created
            ON {artifact_table}(user_id, created_at DESC)
        """)
        print(f"✓ Created index: idx_{artifact_table}_user_created (for companion mode)")

        # Create GIN index for metadata JSONB queries
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{artifact_table}_metadata
            ON {artifact_table} USING GIN (metadata)
        """)
        print(f"✓ Created index: idx_{artifact_table}_metadata (GIN)")

        # Create index for content_hash lookups (deduplication)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{artifact_table}_content_hash
            ON {artifact_table}(content_hash)
        """)
        print(f"✓ Created index: idx_{artifact_table}_content_hash (deduplication)")

        # Create unique constraint for session+source deduplication
        await conn.execute(f"""
            ALTER TABLE {artifact_table}
            ADD CONSTRAINT IF NOT EXISTS uq_{artifact_table}_session_source
            UNIQUE (session_id, source_ids, artifact_type)
        """)
        print(f"✓ Created constraint: uq_{artifact_table}_session_source (prevents duplicate sources)")

        # Create unique constraint for entity deduplication by name
        await conn.execute(f"""
            ALTER TABLE {artifact_table}
            ADD CONSTRAINT IF NOT EXISTS uq_entities_session_name
            UNIQUE (session_id, artifact_type, metadata->>'entity_name')
            WHERE artifact_type = 'entity'
        """)
        print(f"✓ Created constraint: uq_entities_session_name (prevents duplicate entities)")

        # Create vector similarity index (HNSW for fast approximate search)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{embedding_table}_embedding_hnsw
            ON {embedding_table} USING hnsw (embedding vector_cosine_ops)
        """)
        print(f"✓ Created index: idx_{embedding_table}_embedding_hnsw (HNSW)")

        print("\nPostgres initialization complete")

    except Exception as e:
        print(f"✗ Error initializing Postgres: {e}")
        raise
    finally:
        if conn:
            await conn.close()


async def cleanup_postgres(
    connection_string: str = "postgresql://ulli:ulli_password@localhost:5432/ulli_db",
    max_age_days: int = 30,
    min_relevance: float = 0.1
):
    """Clean up old artifacts with low relevance."""
    conn = None
    try:
        conn = await asyncpg.connect(connection_string)

        result = await conn.execute(f"""
            DELETE FROM artifacts
            WHERE created_at < NOW() - INTERVAL '{max_age_days} days'
              AND relevance_score < {min_relevance}
            RETURNING id
        """)

        deleted = int(result.split(" ")[-1]) if result else 0
        print(f"✓ Cleaned up {deleted} old artifacts")

    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    import sys

    conn_string = sys.argv[1] if len(sys.argv) > 1 else "postgresql://ulli:ulli_password@localhost:5432/ulli_db"
    embedding_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 1536

    asyncio.run(init_postgres(conn_string, embedding_dim))
