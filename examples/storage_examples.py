"""
Storage Backends - Practical Examples

This module demonstrates practical usage of all storage backends
with the shared Docker infrastructure.

Run with: uv run python -m examples.storage_examples
"""

import asyncio
from typing import Dict, Any
import json


async def example_postgres_vectors():
    """Example: PostgreSQL+pgvector for embeddings and RAG."""
    from deep_research_engine.storage import PostgresStorage

    print("\n" + "=" * 60)
    print("PostgreSQL+pgvector Example")
    print("=" * 60)

    storage = PostgresStorage({
        "url": "postgresql://postgres:e2e-test-postgres-password@localhost:25432/shared_db",
        "embedding_dimension": 768,
        "pool_size": 5,
    })

    try:
        await storage.initialize()
        print("✓ PostgreSQL initialized")

        # Store documents with embeddings
        documents = [
            {
                "id": "doc-1",
                "content": "Machine learning is a subset of AI",
                "embedding": [0.1] * 768,
                "metadata": {"source": "wikipedia", "year": 2024},
            },
            {
                "id": "doc-2",
                "content": "Deep learning uses neural networks",
                "embedding": [0.2] * 768,
                "metadata": {"source": "arxiv", "year": 2023},
            },
        ]

        for doc in documents:
            result = await storage.store(doc, namespace="rag-docs")
            print(f"✓ Stored: {result.id}")

        # Vector similarity search
        print("\nSearching for similar documents...")
        results = await storage.query(
            query=",".join(map(str, [0.15] * 768)),
            namespace="rag-docs",
            limit=5,
            embedding_query=True,
            similarity_threshold=0.5,
        )

        for doc in results:
            print(f"  Similarity: {doc.get('similarity', 0):.3f} - {doc.get('content', '')[:50]}")

        # Batch insert
        batch_docs = [
            {
                "id": f"batch-{i}",
                "content": f"Batch document {i}",
                "embedding": [float(i)] * 768,
            }
            for i in range(10)
        ]
        results = await storage.store_batch(batch_docs, namespace="batch-test")
        print(f"\n✓ Batch stored {len(results)} documents")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await storage.cleanup()
        print("✓ PostgreSQL cleanup complete")


async def example_neo4j_knowledge_graph():
    """Example: Neo4j for knowledge graph construction."""
    from deep_research_engine.storage import Neo4jStorage

    print("\n" + "=" * 60)
    print("Neo4j Knowledge Graph Example")
    print("=" * 60)

    storage = Neo4jStorage({
        "uri": "bolt://localhost:37687",
        "username": "neo4j",
        "password": "e2e-test-neo4j-password",
        "database": "neo4j",
    })

    try:
        await storage.initialize()
        print("✓ Neo4j initialized")

        # Create entities
        entities = [
            {
                "id": "person-1",
                "labels": ["Person", "Researcher"],
                "name": "Alice Smith",
                "affiliation": "MIT",
            },
            {
                "id": "person-2",
                "labels": ["Person", "Researcher"],
                "name": "Bob Johnson",
                "affiliation": "Stanford",
            },
            {
                "id": "concept-1",
                "labels": ["Concept"],
                "name": "Machine Learning",
            },
        ]

        for entity in entities:
            result = await storage.store(entity, namespace="research-graph")
            print(f"✓ Created entity: {result.id}")

        # Create relationships
        rel_result = await storage.create_relationship(
            source_id="person-1",
            target_id="person-2",
            rel_type="COLLABORATED_WITH",
            namespace="research-graph",
            properties={"since": "2020", "projects": 5},
        )
        print(f"✓ Created relationship: {rel_result.id}")

        # Query with Cypher
        print("\nQuerying graph...")
        results = await storage.query(
            "MATCH (n:Person {namespace: $namespace}) RETURN n.name as name LIMIT $limit",
            namespace="research-graph",
            limit=10,
            cypher=True,
        )

        for result in results:
            print(f"  Found: {result.get('name', 'Unknown')}")

        # Find neighbors
        neighbors = await storage.get_neighbors(
            node_id="person-1",
            namespace="research-graph",
            direction="outgoing",
        )
        print(f"\n✓ Found {len(neighbors)} neighbors for person-1")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await storage.cleanup()
        print("✓ Neo4j cleanup complete")


async def example_mongodb_documents():
    """Example: MongoDB for document storage."""
    from deep_research_engine.storage import MongoDBStorage

    print("\n" + "=" * 60)
    print("MongoDB Document Storage Example")
    print("=" * 60)

    storage = MongoDBStorage({
        "uri": "mongodb://admin:e2e-test-mongodb-password@localhost:37017",
        "database": "deep_research",
    })

    try:
        await storage.initialize()
        print("✓ MongoDB initialized")

        # Store documents
        documents = [
            {
                "id": "paper-1",
                "type": "research_paper",
                "title": "Advances in Deep Learning",
                "authors": ["Alice Smith", "Bob Johnson"],
                "abstract": "This paper presents novel approaches...",
                "metadata": {
                    "journal": "Nature ML",
                    "year": 2024,
                    "citations": 42,
                },
            },
            {
                "id": "paper-2",
                "type": "research_paper",
                "title": "Neural Network Optimization",
                "authors": ["Carol White"],
                "abstract": "We propose a new optimization method...",
                "metadata": {
                    "journal": "ICML",
                    "year": 2023,
                    "citations": 128,
                },
            },
        ]

        for doc in documents:
            result = await storage.store(doc, namespace="papers")
            print(f"✓ Stored: {result.id}")

        # Query documents
        print("\nSearching for papers...")
        results = await storage.query(
            query="Learning",
            namespace="papers",
            limit=10,
        )

        for doc in results:
            print(f"  Found: {doc.get('title', 'Unknown')}")

        # Store binary data
        sample_binary = b"This is simulated binary data for a PDF file"
        file_result = await storage.store_file(
            file_data=sample_binary,
            filename="sample.pdf",
            content_type="application/pdf",
            namespace="files",
            metadata={"source": "test"},
        )
        print(f"\n✓ Stored file: {file_result.id} ({file_result.metadata.get('size_bytes')} bytes)")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await storage.cleanup()
        print("✓ MongoDB cleanup complete")


async def example_minio_object_storage():
    """Example: MinIO for object storage."""
    from deep_research_engine.storage import MinIOStorage

    print("\n" + "=" * 60)
    print("MinIO Object Storage Example")
    print("=" * 60)

    storage = MinIOStorage({
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "e2e-test-minio-password",
        "secure": False,
        "default_bucket": "deep-research",
    })

    try:
        await storage.initialize()
        print("✓ MinIO initialized")

        # Store JSON object
        config_data = {
            "id": "workflow-config-1",
            "name": "Research Pipeline",
            "settings": {
                "max_workers": 10,
                "timeout": 300,
                "retry_count": 3,
            },
        }
        result = await storage.store(config_data, namespace="configs")
        print(f"✓ Stored config: {result.id}")

        # Store file
        sample_file = b"Simulated PDF content for testing"
        file_result = await storage.store_file(
            file_data=sample_file,
            filename="research-paper.pdf",
            namespace="documents",
            content_type="application/pdf",
        )
        print(f"✓ Stored file: {file_result.metadata['key']}")

        # Generate presigned URL
        url = await storage.get_presigned_url(
            key="documents/research-paper.pdf",
            expires_in=3600,
        )
        print(f"✓ Generated presigned URL: {url[:50]}...")

        # List objects
        print("\nListing objects in namespace 'configs':")
        items = await storage.list_items(namespace="configs", limit=10)
        for item in items:
            print(f"  {item['key']} ({item['size']} bytes)")

        # List buckets
        print("\nAvailable buckets:")
        buckets = await storage.list_buckets()
        for bucket in buckets:
            print(f"  - {bucket}")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await storage.cleanup()
        print("✓ MinIO cleanup complete")


async def example_storage_manager():
    """Example: Using StorageManager for multiple backends."""
    from deep_research_engine.storage import StorageManager

    print("\n" + "=" * 60)
    print("Storage Manager - Multi-Backend Example")
    print("=" * 60)

    manager = StorageManager()

    try:
        # Initialize multiple backends
        print("Initializing backends...")
        
        await manager.initialize("postgres", {
            "url": "postgresql://postgres:e2e-test-postgres-password@localhost:25432/shared_db",
            "embedding_dimension": 768,
        })
        print("✓ PostgreSQL initialized")

        await manager.initialize("neo4j", {
            "uri": "bolt://localhost:37687",
            "username": "neo4j",
            "password": "e2e-test-neo4j-password",
        })
        print("✓ Neo4j initialized")

        await manager.initialize("mongodb", {
            "uri": "mongodb://admin:e2e-test-mongodb-password@localhost:37017",
            "database": "deep_research",
        })
        print("✓ MongoDB initialized")

        await manager.initialize("minio", {
            "endpoint": "localhost:9000",
            "access_key": "minioadmin",
            "secret_key": "e2e-test-minio-password",
            "secure": False,
            "default_bucket": "deep-research",
        })
        print("✓ MinIO initialized")

        # Store in different backends
        print("\nStoring data in different backends...")

        postgres_id = await manager.store(
            "postgres",
            {
                "content": "Vector embedding data",
                "embedding": [0.1] * 768,
            },
            namespace="vectors",
        )
        print(f"✓ Stored in PostgreSQL: {postgres_id}")

        neo4j_id = await manager.store(
            "neo4j",
            {
                "id": "concept-ml",
                "labels": ["Concept"],
                "name": "Machine Learning",
            },
            namespace="graph",
        )
        print(f"✓ Stored in Neo4j: {neo4j_id}")

        mongodb_id = await manager.store(
            "mongodb",
            {
                "type": "document",
                "title": "Test Document",
                "content": "Sample content",
            },
            namespace="documents",
        )
        print(f"✓ Stored in MongoDB: {mongodb_id}")

        minio_id = await manager.store(
            "minio",
            {
                "config": "workflow-settings",
                "value": {"key": "value"},
            },
            namespace="configs",
        )
        print(f"✓ Stored in MinIO: {minio_id}")

        # Retrieve from different backends
        print("\nRetrieving data...")
        
        postgres_data = await manager.retrieve("postgres", postgres_id, namespace="vectors")
        print(f"✓ Retrieved from PostgreSQL: {type(postgres_data)}")

        neo4j_data = await manager.retrieve("neo4j", neo4j_id, namespace="graph")
        print(f"✓ Retrieved from Neo4j: {type(neo4j_data)}")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        await manager.cleanup()
        print("✓ Storage Manager cleanup complete")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Storage Backends - Practical Examples")
    print("=" * 60)
    print("\nMake sure Docker infrastructure is running:")
    print("  cd ~/Documents/programs/shared-docker-infra")
    print("  make db-up")
    print("\n" + "=" * 60)

    # Run examples
    examples = [
        ("PostgreSQL+pgvector", example_postgres_vectors),
        ("Neo4j", example_neo4j_knowledge_graph),
        ("MongoDB", example_mongodb_documents),
        ("MinIO", example_minio_object_storage),
        ("Storage Manager", example_storage_manager),
    ]

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\n✗ {name} example failed: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
