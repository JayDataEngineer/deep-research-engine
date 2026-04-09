"""Neo4j connection factory — creates Neo4jGraph + async driver from config.

Centralizes Neo4j connection management so that nodes, tools, and storage
backends don't each hand-roll their own driver setup.

Usage:
    conn = Neo4jConnection.from_config(config_data)
    graph = conn.graph          # Neo4jGraph (sync, schema-aware)
    driver = conn.driver        # AsyncGraphDatabase driver

    # ... use ...

    await conn.close()
"""

from __future__ import annotations

from typing import Any

from ...utils.logging import get_loguru_logger

logger = get_loguru_logger("shared.neo4j.connection")


class Neo4jConnection:
    """Centralized Neo4j connection manager.

    Holds both a ``Neo4jGraph`` (sync schema-aware wrapper) and an
    ``AsyncGraphDatabase`` driver, created from the same config dict.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

        self._graph: Any | None = None
        self._driver: Any | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_data: dict[str, Any]) -> Neo4jConnection | None:
        """Create from workflow config.  Returns *None* if Neo4j is disabled."""
        neo4j_cfg = config_data.get("storage", {}).get("neo4j", {})
        if not neo4j_cfg.get("enabled"):
            return None

        uri = neo4j_cfg.get("uri", "bolt://localhost:7687")
        username = neo4j_cfg.get("username", "neo4j")
        password = neo4j_cfg.get("password", "password")
        database = neo4j_cfg.get("database", "neo4j")

        if not uri or not password or (password == "password" and "neo4j" not in uri):
            # Likely placeholder config — skip silently
            logger.debug("Neo4j config looks like placeholder, skipping")
            return None

        return cls(uri=uri, username=username, password=password, database=database)

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Any:
        """Return (and lazily create) a ``Neo4jGraph`` instance.

        ``Neo4jGraph`` is synchronous and provides ``query()``,
        ``refresh_schema()``, ``structured_schema``, etc.
        """
        if self._graph is None:
            from langchain_neo4j import Neo4jGraph

            self._graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
                refresh_schema=False,  # avoid APOC dependency on init
            )
            logger.info(f"Neo4jGraph initialized at {self.uri}")
        return self._graph

    @property
    def driver(self) -> Any:
        """Return (and lazily create) an ``AsyncGraphDatabase`` driver."""
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=50,
            )
            logger.info(f"Async Neo4j driver initialized at {self.uri}")
        return self._driver

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close both connections."""
        import contextlib

        if self._graph is not None:
            with contextlib.suppress(Exception):
                self._graph.close()
            self._graph = None

        if self._driver is not None:
            with contextlib.suppress(Exception):
                await self._driver.close()
            self._driver = None

        logger.debug("Neo4j connections closed")

    async def __aenter__(self) -> Neo4jConnection:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
