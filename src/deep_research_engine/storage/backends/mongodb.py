"""MongoDB storage backend for media files and documents."""

import base64
import contextlib
import uuid
from datetime import datetime
from typing import Any

from ..base import StorageBackend, StorageResult
from ..errors import StorageInitializationError
from ..registry import StorageRegistry


@StorageRegistry.register("mongodb")
class MongoDBStorage(StorageBackend):
    """MongoDB storage backend for media files and documents.

    Features:
    - Document storage with flexible schema
    - GridFS for large file storage
    - Automatic indexing
    - Binary data support

    Configuration:
    ```yaml
    storage:
      mongodb:
        uri: "mongodb://localhost:27017"
        database: "deep_research"
        username: "admin"
        password: "your-password"
    ```

    Use Cases:
    - Media file metadata
    - Large JSON documents
    - Binary data storage
    - Unstructured data
    """

    backend_name = "mongodb"
    description = "MongoDB for documents and media file metadata"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.uri = config.get("uri", "mongodb://localhost:27017")
        self.database_name = config.get("database", "deep_research")
        self.username = config.get("username")
        self.password = config.get("password")
        self.client = None
        self.database = None
        self._collection = None

    async def initialize(self) -> None:
        """Initialize MongoDB client and create indexes."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as e:
            raise StorageInitializationError(
                f"Missing dependencies for MongoDB: {e}. Install with: pip install motor pymongo"
            )

        try:
            # Build connection string with auth if provided
            if self.username and self.password:
                # Parse URI and inject credentials
                if "://" in self.uri:
                    prefix, rest = self.uri.split("://", 1)
                    self.uri = f"{prefix}://{self.username}:{self.password}@{rest}"

            self.client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,
            )

            # Verify connection
            await self.client.admin.command("ping")

            self.database = self.client[self.database_name]

            # Create indexes
            await self._create_indexes()

            self.initialized = True
            self.health_status = "healthy"
            self._log(f"Initialized MongoDB storage at {self.uri}")

        except Exception as e:
            self.health_status = "unhealthy"
            raise StorageInitializationError(f"Failed to initialize MongoDB: {e}")

    async def _create_indexes(self) -> None:
        """Create indexes on collection."""
        collection = self.database.storage_items

        # Create indexes
        await collection.create_index([("id", 1)], unique=True)
        await collection.create_index([("namespace", 1)])
        await collection.create_index([("created_at", -1)])
        await collection.create_index([("type", 1)])
        await collection.create_index([("metadata.contentType", 1)])

        self._log("Indexes created")

    async def cleanup(self) -> None:
        """Close MongoDB client."""
        if self.client:
            self.client.close()
            self.initialized = False
            self._log("MongoDB client closed")

    async def health_check(self) -> bool:
        """Check MongoDB connection health."""
        if not self.client or not self.initialized:
            return False

        try:
            await self.client.admin.command("ping")
            return True
        except Exception:
            self.health_status = "unhealthy"
            return False

    @property
    def collection(self):
        """Get storage collection."""
        if self._collection is None:
            self._collection = self.database.storage_items
        return self._collection

    async def store(
        self,
        data: dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        """Store data in MongoDB.

        Supports:
        - Regular JSON documents
        - Binary data (stored as base64)
        - GridFS for large files (>16MB)
        """
        if not self.initialized:
            return StorageResult(
                success=False,
                error="MongoDB not initialized",
            )

        try:
            resource_id = data.get("id", str(uuid.uuid4()))
            document = data.copy()

            # Handle binary data
            if "binary_data" in document:
                binary_data = document.pop("binary_data")
                if isinstance(binary_data, bytes):
                    document["binary_data"] = base64.b64encode(binary_data).decode("utf-8")
                    document["metadata"] = document.get("metadata", {})
                    document["metadata"]["has_binary"] = True

            # Add standard fields
            document["id"] = resource_id
            document["namespace"] = namespace
            document["created_at"] = document.get("created_at", datetime.now())
            document["updated_at"] = datetime.now()

            # Upsert
            await self.collection.update_one(
                {"id": resource_id, "namespace": namespace},
                {"$set": document},
                upsert=True,
            )

            self._log(f"Stored document {resource_id} in namespace {namespace}")

            return StorageResult(
                success=True,
                id=resource_id,
                metadata={"namespace": namespace, "size_bytes": len(str(document))},
            )

        except Exception as e:
            self._log(f"Store failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve document by ID."""
        if not self.initialized:
            return None

        try:
            document = await self.collection.find_one(
                {
                    "id": id,
                    "namespace": namespace,
                }
            )

            if not document:
                return None

            # Remove MongoDB _id
            document.pop("_id", None)

            # Decode binary data if present
            if "binary_data" in document:
                with contextlib.suppress(Exception):
                    document["binary_data_decoded"] = base64.b64decode(document["binary_data"])

            return document

        except Exception as e:
            self._log(f"Retrieve failed: {e}", level="error")
            return None

    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        """Delete document by ID."""
        if not self.initialized:
            return False

        try:
            result = await self.collection.delete_one(
                {
                    "id": id,
                    "namespace": namespace,
                }
            )

            deleted = result.deleted_count > 0
            if deleted:
                self._log(f"Deleted document {id} from namespace {namespace}")
            return deleted

        except Exception as e:
            self._log(f"Delete failed: {e}", level="error")
            return False

    async def query(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 10,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Query documents with text search or filters.

        Args:
            query: Search query (text search or JSON filter)
            namespace: Namespace to query
            limit: Maximum results
            **kwargs: Additional filters:
                - type: Filter by document type
                - metadata: Filter by metadata fields
                - sort: Sort field (default: created_at)
                - order: Sort order (1 for asc, -1 for desc)
        """
        if not self.initialized:
            return []

        try:
            # Build query filter
            filter_dict = {"namespace": namespace}

            # Add type filter if provided
            doc_type = kwargs.get("type")
            if doc_type:
                filter_dict["type"] = doc_type

            # Try to parse query as JSON for exact matching
            import json

            try:
                json_query = json.loads(query)
                filter_dict.update(json_query)
                documents = (
                    await self.collection.find(filter_dict).limit(limit).to_list(length=limit)
                )
            except (json.JSONDecodeError, TypeError):
                # Text search using regex
                filter_dict["$or"] = [
                    {"content": {"$regex": query, "$options": "i"}},
                    {"metadata": {"$regex": query, "$options": "i"}},
                    {"title": {"$regex": query, "$options": "i"}},
                ]

                sort_field = kwargs.get("sort", "created_at")
                sort_order = kwargs.get("order", -1)

                documents = (
                    await self.collection.find(filter_dict)
                    .sort(sort_field, sort_order)
                    .limit(limit)
                    .to_list(length=limit)
                )

            # Clean up documents
            results = []
            for doc in documents:
                doc.pop("_id", None)
                results.append(doc)

            return results

        except Exception as e:
            self._log(f"Query failed: {e}", level="error")
            return []

    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List items in namespace."""
        if not self.initialized:
            return []

        try:
            cursor = (
                self.collection.find(
                    {"namespace": namespace},
                    {"id": 1, "namespace": 1, "type": 1, "created_at": 1, "metadata": 1},
                )
                .sort("created_at", -1)
                .limit(limit)
            )

            items = []
            async for doc in cursor:
                doc.pop("_id", None)
                items.append(doc)

            return items

        except Exception as e:
            self._log(f"List failed: {e}", level="error")
            return []

    async def store_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> StorageResult:
        """Store file using GridFS for large files."""
        if not self.initialized:
            return StorageResult(success=False, error="MongoDB not initialized")

        try:
            from gridfs import AsyncGridFS

            # Get or create GridFS
            fs = AsyncGridFS(self.database, "fs")

            resource_id = str(uuid.uuid4())

            # Store file
            file_id = await fs.put(
                file_data,
                filename=filename,
                content_type=content_type,
                _id=resource_id,
                namespace=namespace,
                metadata=metadata or {},
                created_at=datetime.now(),
            )

            self._log(f"Stored file {filename} ({len(file_data)} bytes) in GridFS")

            return StorageResult(
                success=True,
                id=str(file_id),
                metadata={
                    "namespace": namespace,
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": len(file_data),
                },
            )

        except Exception as e:
            self._log(f"Store file failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def retrieve_file(
        self,
        file_id: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve file from GridFS."""
        if not self.initialized:
            return None

        try:
            from gridfs import AsyncGridFS, NoFile

            fs = AsyncGridFS(self.database, "fs")

            try:
                file = await fs.get(file_id)

                # Check namespace
                if file.namespace != namespace:
                    return None

                return {
                    "id": str(file._id),
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "data": file.read(),
                    "metadata": file.metadata,
                    "created_at": file.created_at.isoformat() if file.created_at else None,
                }

            except NoFile:
                return None

        except Exception as e:
            self._log(f"Retrieve file failed: {e}", level="error")
            return None

    async def delete_file(
        self,
        file_id: str,
        namespace: str = "default",
    ) -> bool:
        """Delete file from GridFS."""
        if not self.initialized:
            return False

        try:
            from gridfs import AsyncGridFS, NoFile

            fs = AsyncGridFS(self.database, "fs")

            # Verify namespace before deleting
            try:
                file = await fs.get(file_id)
                if file.namespace != namespace:
                    return False
            except NoFile:
                return False

            await fs.delete(file_id)
            self._log(f"Deleted file {file_id} from namespace {namespace}")
            return True

        except Exception as e:
            self._log(f"Delete file failed: {e}", level="error")
            return False

    async def aggregate(
        self,
        pipeline: list[dict[str, Any]],
        namespace: str = "default",
    ) -> list[dict[str, Any]]:
        """Run aggregation pipeline."""
        if not self.initialized:
            return []

        try:
            # Prepend namespace filter
            match_stage = {"$match": {"namespace": namespace}}
            full_pipeline = [match_stage, *pipeline]

            cursor = self.collection.aggregate(full_pipeline)
            return [doc async for doc in cursor]

        except Exception as e:
            self._log(f"Aggregate failed: {e}", level="error")
            return []
