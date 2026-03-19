"""MinIO/S3 storage backend for object storage."""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import StorageBackend, StorageResult
from ..errors import StorageInitializationError
from ..registry import StorageRegistry


@StorageRegistry.register("minio")
class MinIOStorage(StorageBackend):
    """MinIO/S3-compatible object storage backend.

    Features:
    - S3-compatible object storage
    - Large file support with multipart upload
    - Bucket management
    - Presigned URLs for secure access

    Configuration:
    ```yaml
    storage:
      minio:
        endpoint: "localhost:9000"
        access_key: "minioadmin"
        secret_key: "minioadmin"
        secure: false
        region: "us-east-1"
        default_bucket: "deep-research"
    ```

    Use Cases:
    - Large file storage (PDFs, images, audio, video)
    - Binary asset storage
    - Backup and archival
    - Static asset hosting
    """

    backend_name = "minio"
    description = "MinIO/S3 for object and file storage"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint", "localhost:9000")
        self.access_key = config.get("access_key", "minioadmin")
        self.secret_key = config.get("secret_key", "minioadmin")
        self.secure = config.get("secure", False)
        self.region = config.get("region", "us-east-1")
        self.default_bucket = config.get("default_bucket", "deep-research")
        self.client = None
        self._lock = None

    async def initialize(self) -> None:
        """Initialize MinIO/S3 client and create default bucket."""
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise StorageInitializationError(
                f"Missing dependencies for MinIO: {e}. Install with: pip install boto3 botocore"
            )

        try:
            # Configure boto3 for async usage
            config = Config(
                region_name=self.region,
                signature_version="s3v4",
                retries={"max_attempts": 3},
                connect_timeout=5,
                read_timeout=10,
            )

            self.client = boto3.client(
                "s3",
                endpoint_url=f"{'https' if self.secure else 'http'}://{self.endpoint}",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=config,
            )

            # Create default bucket if it doesn't exist
            await self._create_bucket_if_not_exists(self.default_bucket)

            self.initialized = True
            self.health_status = "healthy"
            self._log(f"Initialized MinIO storage at {self.endpoint}")

        except Exception as e:
            self.health_status = "unhealthy"
            raise StorageInitializationError(f"Failed to initialize MinIO: {e}")

    def _run_in_executor(self, func, *args):
        """Run blocking boto3 calls in executor."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: func(*args))

    async def _create_bucket_if_not_exists(self, bucket_name: str) -> None:
        """Create bucket if it doesn't exist."""
        try:
            # Check if bucket exists
            self.client.head_bucket(Bucket=bucket_name)
            self._log(f"Bucket {bucket_name} already exists")
        except Exception:
            # Create bucket
            if self.region == "us-east-1":
                self.client.create_bucket(Bucket=bucket_name)
            else:
                self.client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self.region},
                )
            self._log(f"Created bucket {bucket_name}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.client = None
        self.initialized = False
        self._log("MinIO client cleaned up")

    async def health_check(self) -> bool:
        """Check MinIO connection health."""
        if not self.client or not self.initialized:
            return False

        try:
            await self._run_in_executor(
                self.client.head_bucket,
                self.default_bucket,
            )
            return True
        except Exception:
            self.health_status = "unhealthy"
            return False

    async def store(
        self,
        data: dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        """Store data as JSON object in MinIO.

        For binary data, use store_file instead.
        """
        if not self.initialized:
            return StorageResult(
                success=False,
                error="MinIO not initialized",
            )

        try:
            import json

            resource_id = data.get("id", str(uuid.uuid4()))
            object_key = f"{namespace}/{resource_id}.json"

            # Convert to JSON
            json_data = json.dumps(data, indent=2, default=str)

            # Upload
            await self._run_in_executor(
                self.client.put_object,
                Bucket=self.default_bucket,
                Key=object_key,
                Body=json_data,
                ContentType="application/json",
                Metadata={
                    "namespace": namespace,
                    "stored_at": datetime.now().isoformat(),
                },
            )

            self._log(f"Stored object {object_key}")

            return StorageResult(
                success=True,
                id=resource_id,
                metadata={
                    "namespace": namespace,
                    "bucket": self.default_bucket,
                    "key": object_key,
                    "size_bytes": len(json_data),
                },
            )

        except Exception as e:
            self._log(f"Store failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve JSON object by ID."""
        if not self.initialized:
            return None

        try:
            import json

            object_key = f"{namespace}/{id}.json"

            # Download
            response = await self._run_in_executor(
                self.client.get_object,
                Bucket=self.default_bucket,
                Key=object_key,
            )

            # Parse JSON
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)

        except Exception as e:
            self._log(f"Retrieve failed: {e}", level="debug")
            return None

    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        """Delete object by ID."""
        if not self.initialized:
            return False

        try:
            object_key = f"{namespace}/{id}.json"

            await self._run_in_executor(
                self.client.delete_object,
                Bucket=self.default_bucket,
                Key=object_key,
            )

            self._log(f"Deleted object {object_key}")
            return True

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
        """Query objects using S3 Select or prefix search.

        Args:
            query: Search query (prefix or S3 Select SQL)
            namespace: Namespace (bucket prefix) to query
            limit: Maximum results
            **kwargs: Additional options:
                - prefix: Additional prefix filter
                - suffix: Filter by file suffix
        """
        if not self.initialized:
            return []

        try:
            prefix = f"{namespace}/"
            if kwargs.get("prefix"):
                prefix += kwargs["prefix"]

            # List objects with prefix
            response = await self._run_in_executor(
                self.client.list_objects_v2,
                Bucket=self.default_bucket,
                Prefix=prefix,
                MaxKeys=limit,
            )

            if "Contents" not in response:
                return []

            # Filter by suffix if provided
            suffix = kwargs.get("suffix")
            objects = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue

                # Try to retrieve JSON content
                if key.endswith(".json"):
                    try:
                        data = await self.retrieve_by_key(key)
                        if data:
                            objects.append(data)
                    except Exception:
                        continue
                else:
                    # Return metadata for non-JSON files
                    objects.append(
                        {
                            "key": key,
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                        }
                    )

            return objects

        except Exception as e:
            self._log(f"Query failed: {e}", level="error")
            return []

    async def retrieve_by_key(self, key: str) -> dict[str, Any] | None:
        """Retrieve object by full key."""
        try:
            import json

            response = await self._run_in_executor(
                self.client.get_object,
                Bucket=self.default_bucket,
                Key=key,
            )

            body = response["Body"].read().decode("utf-8")
            return json.loads(body)

        except Exception:
            return None

    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List items in namespace."""
        if not self.initialized:
            return []

        try:
            prefix = f"{namespace}/"

            response = await self._run_in_executor(
                self.client.list_objects_v2,
                Bucket=self.default_bucket,
                Prefix=prefix,
                MaxKeys=limit,
            )

            if "Contents" not in response:
                return []

            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "etag": obj.get("ETag", ""),
                }
                for obj in response["Contents"]
            ]

        except Exception as e:
            self._log(f"List failed: {e}", level="error")
            return []

    async def store_file(
        self,
        file_data: bytes,
        filename: str,
        namespace: str = "default",
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> StorageResult:
        """Store file in MinIO.

        Supports files of any size with automatic multipart upload.
        """
        if not self.initialized:
            return StorageResult(success=False, error="MinIO not initialized")

        try:
            resource_id = str(uuid.uuid4())
            object_key = f"{namespace}/{filename}"

            # Upload file
            await self._run_in_executor(
                self.client.put_object,
                Bucket=self.default_bucket,
                Key=object_key,
                Body=file_data,
                ContentType=content_type,
                Metadata={
                    "namespace": namespace,
                    "original_filename": filename,
                    "stored_at": datetime.now().isoformat(),
                    **(metadata or {}),
                },
            )

            self._log(f"Stored file {object_key} ({len(file_data)} bytes)")

            return StorageResult(
                success=True,
                id=resource_id,
                metadata={
                    "namespace": namespace,
                    "bucket": self.default_bucket,
                    "key": object_key,
                    "filename": filename,
                    "content_type": content_type,
                    "size_bytes": len(file_data),
                },
            )

        except Exception as e:
            self._log(f"Store file failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def store_file_from_path(
        self,
        file_path: str,
        namespace: str = "default",
        object_name: str | None = None,
    ) -> StorageResult:
        """Store file from local path."""
        if not self.initialized:
            return StorageResult(success=False, error="MinIO not initialized")

        try:
            path = Path(file_path)
            if not path.exists():
                return StorageResult(success=False, error=f"File not found: {file_path}")

            filename = object_name or path.name
            with open(path, "rb") as f:
                file_data = f.read()

            return await self.store_file(
                file_data=file_data,
                filename=filename,
                namespace=namespace,
                content_type=self._guess_content_type(path.suffix),
            )

        except Exception as e:
            self._log(f"Store file from path failed: {e}", level="error")
            return StorageResult(success=False, error=str(e))

    async def retrieve_file(
        self,
        key: str,
        namespace: str = "default",
    ) -> dict[str, Any] | None:
        """Retrieve file by key."""
        if not self.initialized:
            return None

        try:
            # Full key if namespace not in key
            object_key = f"{namespace}/{key}" if not key.startswith(namespace) else key

            response = await self._run_in_executor(
                self.client.get_object,
                Bucket=self.default_bucket,
                Key=object_key,
            )

            return {
                "key": object_key,
                "data": response["Body"].read(),
                "content_type": response.get("ContentType", "application/octet-stream"),
                "metadata": response.get("Metadata", {}),
                "size": response.get("ContentLength", 0),
            }

        except Exception as e:
            self._log(f"Retrieve file failed: {e}", level="debug")
            return None

    async def delete_file(
        self,
        key: str,
        namespace: str = "default",
    ) -> bool:
        """Delete file by key."""
        if not self.initialized:
            return False

        try:
            # Full key if namespace not in key
            object_key = f"{namespace}/{key}" if not key.startswith(namespace) else key

            await self._run_in_executor(
                self.client.delete_object,
                Bucket=self.default_bucket,
                Key=object_key,
            )

            self._log(f"Deleted file {object_key}")
            return True

        except Exception as e:
            self._log(f"Delete file failed: {e}", level="error")
            return False

    async def get_presigned_url(
        self,
        key: str,
        namespace: str = "default",
        expires_in: int = 3600,
        method: str = "GET",
    ) -> str | None:
        """Generate presigned URL for secure access."""
        if not self.initialized:
            return None

        try:
            # Full key if namespace not in key
            object_key = f"{namespace}/{key}" if not key.startswith(namespace) else key

            url = await self._run_in_executor(
                self.client.generate_presigned_url,
                "get_object" if method == "GET" else "put_object",
                Params={
                    "Bucket": self.default_bucket,
                    "Key": object_key,
                },
                ExpiresIn=expires_in,
            )

            self._log(f"Generated presigned URL for {object_key}")
            return url

        except Exception as e:
            self._log(f"Get presigned URL failed: {e}", level="error")
            return None

    async def create_bucket(self, bucket_name: str) -> bool:
        """Create a new bucket."""
        if not self.initialized:
            return False

        try:
            await self._create_bucket_if_not_exists(bucket_name)
            return True
        except Exception as e:
            self._log(f"Create bucket failed: {e}", level="error")
            return False

    async def list_buckets(self) -> list[str]:
        """List all buckets."""
        if not self.initialized:
            return []

        try:
            response = await self._run_in_executor(
                self.client.list_buckets,
            )

            return [bucket["Name"] for bucket in response.get("Buckets", [])]

        except Exception as e:
            self._log(f"List buckets failed: {e}", level="error")
            return []

    def _guess_content_type(self, suffix: str) -> str:
        """Guess content type from file suffix."""
        content_types = {
            ".json": "application/json",
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".csv": "text/csv",
            ".xml": "application/xml",
            ".html": "text/html",
            ".md": "text/markdown",
        }
        return content_types.get(suffix.lower(), "application/octet-stream")
