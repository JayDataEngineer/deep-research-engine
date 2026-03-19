"""Add node_execution table for hierarchical execution tracking.

This table tracks the execution hierarchy of nested agents:
Lab Architect → Flash Explorer → Deep Agent → Tools

Each row represents one node execution with parent-child relationships.
"""

import asyncio
import asyncpg


async def upgrade(conn: asyncpg.Connection):
    """Create the node_execution table."""

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS node_execution (
            id UUID PRIMARY KEY,
            run_id TEXT NOT NULL,
            parent_node_id UUID,

            node_name TEXT NOT NULL,
            node_type TEXT NOT NULL,

            start_time TIMESTAMPTZ DEFAULT NOW(),
            end_time TIMESTAMPTZ,
            duration_ms INTEGER,

            status TEXT DEFAULT 'running',

            llm_calls_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            total_tokens_per_sec FLOAT DEFAULT 0,

            metadata JSONB DEFAULT '{}'::jsonb,

            FOREIGN KEY (run_id) REFERENCES research_sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_node_id) REFERENCES node_execution(id) ON DELETE SET NULL
        );
    """)

    # Indexes for common queries
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_node_exec_run_id
        ON node_execution(run_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_node_exec_parent_id
        ON node_execution(parent_node_id);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_node_exec_status
        ON node_execution(status);
    """)

    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_node_exec_run_status
        ON node_execution(run_id, status);
    """)


async def downgrade(conn: asyncpg.Connection):
    """Drop the node_execution table."""

    await conn.execute("DROP INDEX IF EXISTS idx_node_exec_run_status;")
    await conn.execute("DROP INDEX IF EXISTS idx_node_exec_status;")
    await conn.execute("DROP INDEX IF EXISTS idx_node_exec_parent_id;")
    await conn.execute("DROP INDEX IF EXISTS idx_node_exec_run_id;")
    await conn.execute("DROP TABLE IF EXISTS node_execution;")


if __name__ == "__main__":
    async def main():
        from dotenv import load_dotenv

        # Import centralized constants (absolute import avoids .... parent pattern)
        from src.extensions.content.deep_research.constants import DEFAULT_POSTGRES_CONNECTION_STRING

        load_dotenv()

        conn = await asyncpg.connect(DEFAULT_POSTGRES_CONNECTION_STRING)

        print("Creating node_execution table...")
        await upgrade(conn)
        print("Done!")

        await conn.close()

    asyncio.run(main())
