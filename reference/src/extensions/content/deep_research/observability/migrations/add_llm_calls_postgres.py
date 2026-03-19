"""Add llm_calls table to Postgres for V2 observability system.

This replaces the SQLite-based llm_calls table with a Postgres version.
Links to node_execution table via parent_node_id for hierarchical tracking.
"""

import asyncio
import asyncpg


async def upgrade(conn: asyncpg.Connection):
    """Create the llm_calls table in Postgres."""

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id TEXT NOT NULL,
            call_index INTEGER NOT NULL,
            phase TEXT NOT NULL,
            role TEXT NOT NULL,
            worker_id TEXT,

            -- Hierarchical tracking
            parent_node_id UUID,

            -- Model info
            model TEXT NOT NULL,
            provider TEXT NOT NULL,

            -- Timing and tokens
            duration_seconds REAL NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,

            -- Content
            messages JSONB NOT NULL,
            response_preview TEXT,
            thinking TEXT,

            -- Tool calls (JSON array)
            tool_calls JSONB DEFAULT '[]'::jsonb,

            -- Metadata
            created_at TIMESTAMPTZ DEFAULT NOW(),

            FOREIGN KEY (run_id) REFERENCES research_sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_node_id) REFERENCES node_execution(id) ON DELETE SET NULL
        );
    """)

    # Indexes for common queries
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id
        ON llm_calls(run_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_calls_phase
        ON llm_calls(phase);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_calls_parent_node
        ON llm_calls(parent_node_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_calls_run_phase
        ON llm_calls(run_id, phase);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at
        ON llm_calls(created_at);
    """)


async def downgrade(conn: asyncpg.Connection):
    """Drop the llm_calls table."""

    await conn.execute("DROP INDEX IF EXISTS idx_llm_calls_created_at;")
    await conn.execute("DROP INDEX IF EXISTS idx_llm_calls_run_phase;")
    await conn.execute("DROP INDEX IF EXISTS idx_llm_calls_parent_node;")
    await conn.execute("DROP INDEX IF EXISTS idx_llm_calls_phase;")
    await conn.execute("DROP INDEX IF EXISTS idx_llm_calls_run_id;")
    await conn.execute("DROP TABLE IF EXISTS llm_calls;")


if __name__ == "__main__":
    async def main():
        from dotenv import load_dotenv

        # Import centralized constants (absolute import avoids .... parent pattern)
        from src.extensions.content.deep_research.constants import DEFAULT_POSTGRES_CONNECTION_STRING

        load_dotenv()

        conn = await asyncpg.connect(DEFAULT_POSTGRES_CONNECTION_STRING)

        print("Creating llm_calls table in Postgres...")
        await upgrade(conn)
        print("Done!")

        await conn.close()

    asyncio.run(main())
