"""
Context Engine Extension - Main extension class.

Provides artifact-centric context management for the Unified LLM Interface.
Implements a tiered storage architecture with health checks and conditional tool inclusion.
"""
from typing import Optional, List, Any, Dict
import logging

from src.extensions.core.base import LogicExtension
from src.extensions.core.context import ExtensionContext

from .health_check import (
    get_health_registry,
    HealthCheckRegistry,
    HealthCheckResult,
    HealthStatus
)
from .storage import (
    PostgresVectorStore,
    S3Backend,
    Neo4jStore,
    compute_content_hash
)
from .tools import ALL_CONTEXT_ENGINE_TOOLS
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class ContextEngineExtension(LogicExtension):
    """Context Engine extension for artifact-centric context management.

    Provides:
    - Multi-tier storage (Postgres, MinIO, Neo4j)
    - Health check monitoring
    - Conditional tool inclusion based on backend availability
    - Session isolation for research and companion modes
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Configuration
        self.enabled = config.get("enabled", False)
        self.mode = config.get("mode", "research")  # research or companion
        self.isolation = config.get("isolation", "session")

        # Health registry
        self.health_registry: HealthCheckRegistry = get_health_registry()

        # Storage backends (initialized during startup)
        self.postgres_store: Optional[PostgresVectorStore] = None
        self.s3_backend: Optional[S3Backend] = None
        self.neo4j_store: Optional[Neo4jStore] = None

        # Neo4j vector store for hybrid search (initialized after Neo4j)
        self.neo4j_vector_store: Optional["Neo4jVectorStore"] = None

        # Embedding model
        self.embedding_model: Optional[Embeddings] = None

    async def initialize(self, context: ExtensionContext) -> None:
        """Initialize the Context Engine extension.

        Sets up storage backends, health checks, and validates configuration.
        """
        if not self.enabled:
            logger.info("Context Engine is disabled in configuration")
            return

        logger.info(f"Initializing Context Engine in {self.mode} mode")

        # Get embedding model - prefer config override, then model_provider
        embedding_model_name = self.config.get("embedding_model")

        if embedding_model_name:
            # Use configured embedding model
            try:
                from src.shared.embeddings import get_embeddings
                self.embedding_model = get_embeddings(model_name=embedding_model_name)
                logger.info(f"Using configured embedding model: {embedding_model_name}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize configured embedding model '{embedding_model_name}': {e}"
                ) from e
        else:
            # Fall back to model_provider service
            try:
                model_provider = context.get_service("model_provider")
                self.embedding_model = model_provider.get_embeddings()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get embedding model from model_provider: {e}. "
                    f"Set 'embedding_model' in config to specify a model."
                ) from e

        # Initialize Postgres (required)
        await self._init_postgres(context.config)

        # Initialize MinIO (optional)
        await self._init_minio(context.config)

        # Initialize Neo4j (optional)
        await self._init_neo4j(context.config)

        # Run initial health checks
        await self._run_health_checks()

        logger.info("Context Engine initialization complete")

    async def _init_postgres(self, config: dict) -> None:
        """Initialize Postgres vector store."""
        postgres_config = self.config.get("postgres", {})

        self.postgres_store = PostgresVectorStore(
            connection_string=postgres_config.get(
                "connection_string",
                "postgresql://ulli:ulli_password@localhost:5432/ulli_db"
            ),
            embedding_model=self.embedding_model,
            artifact_table=postgres_config.get("artifact_table", "artifacts"),
            embedding_table=postgres_config.get("embedding_table", "artifact_embeddings"),
            pool_min_size=postgres_config.get("pool_min_size", 5),
            pool_max_size=postgres_config.get("pool_max_size", 20)
        )

        # Register health check
        self.health_registry.register("postgres", self.postgres_store.health_check)

        logger.info("Postgres vector store initialized")

    async def _init_minio(self, config: dict) -> None:
        """Initialize MinIO/S3 backend."""
        if not self.config.get("minio", {}).get("enabled", False):
            return

        minio_config = self.config["minio"]

        self.s3_backend = S3Backend(
            bucket=minio_config.get("bucket", "context-engine"),
            endpoint_url=minio_config.get("endpoint_url", "http://localhost:9000"),
            aws_access_key_id=minio_config.get("access_key", "minioadmin"),
            aws_secret_access_key=minio_config.get("secret_key", "minioadmin"),
            connect_timeout=minio_config.get("connect_timeout", 5),
            read_timeout=minio_config.get("read_timeout", 10)
        )

        # Register health check
        self.health_registry.register("minio", self.s3_backend.health_check)

        logger.info("MinIO backend initialized")

    async def _init_neo4j(self, config: dict) -> None:
        """Initialize Neo4j graph store and vector search."""
        if not self.config.get("neo4j", {}).get("enabled", False):
            return

        neo4j_config = self.config["neo4j"]

        self.neo4j_store = Neo4jStore(
            url=neo4j_config.get("url", "bolt://localhost:7687"),
            username=neo4j_config.get("username", "neo4j"),
            password=neo4j_config.get("password", "password")
        )

        # Register health check
        self.health_registry.register("neo4j", self.neo4j_store.health_check)

        logger.info("Neo4j store initialized")

        # Initialize Neo4jVectorStore for hybrid vector+graph search
        # CRASH-ONLY: If Neo4j is enabled, vector store MUST initialize
        from .graph_vector_store import Neo4jVectorStore

        self.neo4j_vector_store = Neo4jVectorStore(
            url=neo4j_config.get("url", "bolt://localhost:7687"),
            username=neo4j_config.get("username", "neo4j"),
            password=neo4j_config.get("password", "password"),
            embedding_model=self.embedding_model
        )

        # Create vector index if it doesn't exist
        await self.neo4j_vector_store.create_vector_index()

        logger.info("Neo4j vector store initialized")

    async def _run_health_checks(self) -> None:
        """Run health checks on all backends and log results."""
        logger.info("Running health checks...")

        results = await self.health_registry.check_all(use_cache=False)

        for name, result in results.items():
            if result.status == HealthStatus.HEALTHY:
                logger.info(f"  ✓ {name}: {result.message}")
            else:
                logger.warning(f"  ✗ {name}: {result.message}")

    async def get_available_tools(
        self,
        state: Dict[str, Any],
        include_arxiv: bool = True
    ) -> List:
        """Get tools based on available (healthy) backends.

        Only includes tools for backends that are healthy. This prevents
        runtime errors when optional backends are unavailable.

        Args:
            state: Current state/context
            include_arxiv: Whether to include ArXiv tools (no backend dependency)

        Returns:
            List of available @tool functions
        """
        tools = []

        # Memory tools (LangGraph Store semantic search)
        if self.health_registry.is_healthy("postgres"):
            from .tools import search_memories
            tools.append(search_memories)

        # ArXiv tools (no backend dependency)
        if include_arxiv:
            from .tools import search_arxiv, get_arxiv_abstract
            tools.extend([search_arxiv, get_arxiv_abstract])

        logger.info(f"Providing {len(tools)} Context Engine tools")
        return tools

    async def add_artifact(self, artifact) -> str:
        """Add an artifact to storage.

        Routes to appropriate backend based on artifact type.
        Handles deduplication automatically.

        Args:
            artifact: BaseArtifact instance

        Returns:
            Artifact ID
        """
        if not self.postgres_store:
            raise RuntimeError("Postgres store not available")

        # Store in Postgres (with deduplication)
        artifact_id, is_new = await self.postgres_store.add(artifact)

        # Store in Neo4j if it's an EntityArtifact and Neo4j is available
        if (artifact.artifact_type.value == "entity" and
            self.neo4j_store and
            self.health_registry.is_healthy("neo4j")):
            from .artifacts.osint import EntityArtifact
            if isinstance(artifact, EntityArtifact):
                await self.neo4j_store.add(artifact)
                logger.debug(f"Added entity to Neo4j: {artifact.entity_name}")

        return artifact_id

    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        session_id: Optional[str] = None,
        artifact_type: Optional[str] = None
    ) -> List:
        """Search for semantically similar artifacts.

        Args:
            query: Search query
            k: Number of results
            session_id: Session ID for isolation
            artifact_type: Optional artifact type filter

        Returns:
            List of BaseArtifact objects
        """
        if not self.postgres_store:
            raise RuntimeError("Postgres store not available")

        return await self.postgres_store.search(
            query=query,
            k=k,
            session_id=session_id,
            artifact_type=artifact_type
        )

    # ========================================================================
    # Entity-Aware Search (Neo4j Vector + Graph Traversal)
    # ========================================================================

    async def search_entities_with_sources(
        self,
        query: str,
        k: int = 5,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search entities with vector similarity + source provenance.

        This is the KILLER FEATURE for citations:
        - Finds entities similar to query
        - Traverses to chunks that mention them
        - Traverses to documents containing those chunks
        - Returns everything in ONE call

        Uses Neo4jVector with retrieval_query for hybrid vector+graph search.

        CRASH-ONLY: Will crash if Neo4j vector store is not available.
        No degraded fallback mode - entity context requires graph database.

        Args:
            query: Search query (e.g., "Elon Musk companies")
            k: Number of results
            filter_params: Optional filters (entity_type, etc.)

        Returns:
            List of dicts with entity + chunk + document info:
            {
                "entity_name": "Elon Musk",
                "entity_type": "Person",
                "chunk_text": "Elon Musk founded SpaceX...",
                "chunk_index": 0,
                "doc_id": "research_123",
                "doc_url": "wikipedia.org/elon-musk",
                "score": 0.95
            }

        Raises:
            RuntimeError: If Neo4j vector store is not available
        """
        if not self.neo4j_vector_store:
            raise RuntimeError(
                "╔═══════════════════════════════════════════════════════════════════╗\n"
                "║         NEO4J VECTOR STORE NOT AVAILABLE - CRITICAL                ║\n"
                "╚═══════════════════════════════════════════════════════════════════╝\n"
                "\n"
                "search_entities_with_sources requires Neo4j vector store.\n"
                "\n"
                "CRASH-ONLY: No degraded fallback mode.\n"
                "Entity context with source provenance REQUIRES graph database.\n"
                "\n"
                "FIX:\n"
                "  1. Enable Neo4j in context_engine config:\n"
                "     neo4j:\n"
                "       enabled: true\n"
                "       url: bolt://localhost:7687\n"
                "       username: neo4j\n"
                "       password: your_password\n"
                "  2. Ensure Neo4j server is running\n"
                "  3. Verify Neo4jVectorStore initialized correctly\n"
            )

        return await self.neo4j_vector_store.search_entities(
            query=query,
            k=k,
            filter_params=filter_params
        )

    async def search_findings_with_entities(
        self,
        query: str,
        k: int = 10,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search findings with entity context.

        CRASH-ONLY: Will crash if Neo4j vector store is not available.
        No degraded fallback mode - entity context requires graph database.

        Args:
            query: Search query
            k: Number of results
            filter_params: Optional filters

        Returns:
            List of dicts with finding + entities + document info

        Raises:
            RuntimeError: If Neo4j vector store is not available
        """
        if not self.neo4j_vector_store:
            raise RuntimeError(
                "╔═══════════════════════════════════════════════════════════════════╗\n"
                "║         NEO4J VECTOR STORE NOT AVAILABLE - CRITICAL                ║\n"
                "╚═══════════════════════════════════════════════════════════════════╝\n"
                "\n"
                "search_findings_with_entities requires Neo4j vector store.\n"
                "\n"
                "CRASH-ONLY: No degraded fallback mode.\n"
                "Findings with entity context REQUIRES graph database.\n"
                "\n"
                "FIX:\n"
                "  1. Enable Neo4j in context_engine config:\n"
                "     neo4j:\n"
                "       enabled: true\n"
                "       url: bolt://localhost:7687\n"
                "       username: neo4j\n"
                "       password: your_password\n"
                "  2. Ensure Neo4j server is running\n"
                "  3. Verify Neo4jVectorStore initialized correctly\n"
            )

        return await self.neo4j_vector_store.search_findings(
            query=query,
            k=k,
            filter_params=filter_params
        )

    async def find_entity_sources(
        self,
        entity_name: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Find all sources that mention an entity (graph traversal).

        Args:
            entity_name: Entity name to search
            max_depth: Maximum traversal depth

        Returns:
            List of source documents with chunks
        """
        if not self.neo4j_store:
            raise RuntimeError("Neo4j store not available")

        return await self.neo4j_vector_store.find_entity_sources(
            entity_name=entity_name,
            max_depth=max_depth
        )

    async def get_entity_context(
        self,
        entity_name: str,
        context_width: int = 1
    ) -> Dict[str, Any]:
        """Get rich context for an entity: sources, connections, mentions.

        Args:
            entity_name: Entity name
            context_width: Number of chunks before/after to include

        Returns:
            Dict with entity info, sources, connections
        """
        if not self.neo4j_vector_store:
            raise RuntimeError("Neo4j vector store not available")

        return await self.neo4j_vector_store.get_entity_context(
            entity_name=entity_name,
            context_width=context_width
        )

    async def find_connected_entities(
        self,
        entity_name: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find entities connected to a given entity.

        Args:
            entity_name: Starting entity name
            max_depth: Maximum traversal depth
            relationship_types: Filter to specific relationship types

        Returns:
            List of connected entities with paths
        """
        if not self.neo4j_vector_store:
            raise RuntimeError("Neo4j vector store not available")

        return await self.neo4j_vector_store.find_connected_entities(
            entity_name=entity_name,
            max_depth=max_depth,
            relationship_types=relationship_types
        )

    # ========================================================================
    # Natural Language Graph Queries (GraphCypherQAChain)
    # ========================================================================

    async def natural_language_query(
        self,
        question: str,
        llm=None,
        use_enhanced_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Execute natural language query against the knowledge graph.

        Uses GraphCypherQAChain to convert natural language to Cypher.

        Args:
            question: Natural language question (e.g., "Who works for SpaceX?")
            llm: LangChain LLM instance (uses default if not provided)
            use_enhanced_schema: Whether to use enhanced schema with example values

        Returns:
            Dict with:
                - answer: Natural language answer
                - cypher: Generated Cypher query
                - context: Query results
        """
        if not self.neo4j_store:
            raise RuntimeError("Neo4j store not available")

        # Get LLM if not provided
        # Get model from config with fallback
        if llm is None:
            model_name = self.config.get("graph_query_model", "GLM-4.7-Flash-UD-Q4_K_XL")
            from src.shared.chat_llm import get_chat_llm
            llm = get_chat_llm(model=model_name)

        # Get enhanced schema if requested
        enhanced_schema = None
        if use_enhanced_schema:
            try:
                from .graph_transformer import ResearchGraphTransformer

                # Create a minimal transformer for schema enhancement
                transformer = ResearchGraphTransformer(llm=llm)

                # Create a base schema to enhance
                from .graph_transformer import DynamicGraphSchema
                base_schema = DynamicGraphSchema(node_types=[], relationship_types=[], patterns=[])

                # Enhance with actual graph values
                enhanced_schema = await transformer.enhance_schema_with_values(
                    base_schema,
                    self.neo4j_store
                )

                # Convert to dict
                from .graph_qa_chain import schema_to_dict
                enhanced_schema = schema_to_dict(enhanced_schema)

            except Exception as e:
                logger.warning(f"[CONTEXT_ENGINE] Failed to get enhanced schema: {e}")

        # Create query chain and execute
        from .graph_qa_chain import NaturalLanguageGraphQuery

        query_chain = NaturalLanguageGraphQuery(
            llm=llm,
            neo4j_url=self.neo4j_store.url,
            neo4j_username=self.neo4j_store.username,
            neo4j_password=self.neo4j_store.password,
            neo4j_database=self.neo4j_store.database,
            enhanced_schema=enhanced_schema
        )

        return await query_chain.query(question)

    async def batch_natural_language_queries(
        self,
        questions: List[str],
        llm=None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple natural language queries.

        Args:
            questions: List of natural language questions
            llm: LangChain LLM instance

        Returns:
            List of result dicts
        """
        if not self.neo4j_store:
            raise RuntimeError("Neo4j store not available")

        results = []
        for question in questions:
            result = await self.natural_language_query(question, llm)
            results.append(result)
        return results

    def get_graph_schema_for_llm(self, include_examples: bool = True) -> str:
        """
        Get formatted graph schema for LLM consumption.

        Args:
            include_examples: Whether to include example values

        Returns:
            Formatted schema string
        """
        if not self.neo4j_store:
            raise RuntimeError("Neo4j store not available")

        if include_examples:
            # Get enhanced schema with examples
            from .graph_qa_chain import format_enhanced_schema_for_llm
            # Note: This would require building the schema first
            # For now, return standard schema
            return self.neo4j_store.get_schema()
        else:
            return self.neo4j_store.get_schema()

    async def health_check_endpoint(self) -> Dict[str, Any]:
        """Health check for API/monitoring.

        Returns:
            Dict with overall health and individual backend status
        """
        results = await self.health_registry.check_all(use_cache=False)

        # Determine overall health
        all_healthy = all(
            r.status == HealthStatus.HEALTHY
            for r in results.values()
        )

        return {
            "status": "healthy" if all_healthy else "degraded",
            "mode": self.mode,
            "backends": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "latency_ms": result.latency_ms
                }
                for name, result in results.items()
            }
        }

    async def shutdown(self, context: ExtensionContext) -> None:
        """Cleanup and close connections."""
        logger.info("Shutting down Context Engine")

        if self.postgres_store:
            await self.postgres_store.close()

        if self.neo4j_vector_store:
            await self.neo4j_vector_store.close()

        if self.neo4j_store:
            await self.neo4j_store.close()

        logger.info("Context Engine shutdown complete")

    def get_config_schema(self) -> Dict:
        """Return configuration schema for this extension."""
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "title": "Enable Context Engine",
                    "default": False
                },
                "mode": {
                    "type": "string",
                    "enum": ["research", "companion"],
                    "title": "Operation Mode",
                    "default": "research"
                },
                "embedding_model": {
                    "type": "string",
                    "title": "Embedding Model",
                    "description": "Embedding model for vector search (optional, uses model_provider if not set)"
                },
                "entity_extraction_model": {
                    "type": "string",
                    "title": "Entity Extraction Model",
                    "description": "LLM model for entity extraction from research findings",
                    "default": "GLM-4.7-Flash-UD-Q4_K_XL"
                },
                "graph_query_model": {
                    "type": "string",
                    "title": "Graph Query Model",
                    "description": "LLM model for natural language graph queries",
                    "default": "GLM-4.7-Flash-UD-Q4_K_XL"
                },
                "postgres": {
                    "type": "object",
                    "title": "Postgres Configuration",
                    "properties": {
                        "connection_string": {
                            "type": "string",
                            "title": "PostgreSQL Connection String"
                        },
                        "artifact_table": {
                            "type": "string",
                            "title": "Artifacts Table Name",
                            "default": "artifacts"
                        },
                        "embedding_table": {
                            "type": "string",
                            "title": "Embeddings Table Name",
                            "default": "artifact_embeddings"
                        },
                        "pool_min_size": {
                            "type": "integer",
                            "title": "Connection Pool Min Size",
                            "default": 5
                        },
                        "pool_max_size": {
                            "type": "integer",
                            "title": "Connection Pool Max Size",
                            "default": 20
                        }
                    }
                },
                "minio": {
                    "type": "object",
                    "title": "MinIO/S3 Configuration",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "title": "Enable MinIO",
                            "default": False
                        },
                        "endpoint_url": {
                            "type": "string",
                            "title": "MinIO Endpoint",
                            "default": "http://localhost:9000"
                        },
                        "bucket": {
                            "type": "string",
                            "title": "S3 Bucket Name",
                            "default": "context-engine"
                        },
                        "access_key": {
                            "type": "string",
                            "title": "Access Key",
                            "default": "minioadmin"
                        },
                        "secret_key": {
                            "type": "string",
                            "title": "Secret Key",
                            "default": "minioadmin"
                        }
                    }
                },
                "neo4j": {
                    "type": "object",
                    "title": "Neo4j Configuration",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "title": "Enable Neo4j",
                            "default": False
                        },
                        "url": {
                            "type": "string",
                            "title": "Neo4j Bolt URL",
                            "default": "bolt://localhost:7687"
                        },
                        "username": {
                            "type": "string",
                            "title": "Neo4j Username",
                            "default": "neo4j"
                        },
                        "password": {
                            "type": "string",
                            "title": "Neo4j Password",
                            "default": "password"
                        }
                    }
                }
            }
        }
