"""
S3/MinIO backend for raw data storage (Level 1 VFS).

Stores large files (300k+ token outputs) that should NOT go in the database.
"""
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from botocore.config import Config
from typing import Optional, List
import logging

from .base import StorageBackend, StorageCapability
from ..health_check import HealthCheckResult, HealthStatus

logger = logging.getLogger(__name__)


class S3Backend(StorageBackend):
    """S3/MinIO backend for raw data storage (Level 1).

    This is where 300k+ token firehose data lives. The LLM never sees
    this data directly - it only accesses via read_vfs_raw @tool when
    specifically verifying a claim.
    """

    def __init__(
        self,
        name: str = "minio",
        bucket: str = "context-engine",
        endpoint_url: str = "http://localhost:9000",
        aws_access_key_id: str = "minioadmin",
        aws_secret_access_key: str = "minioadmin",
        region_name: str = "us-east-1",
        connect_timeout: int = 5,
        read_timeout: int = 10,
        config: dict = None
    ):
        super().__init__(name, config or {})

        # Configure boto3 client with timeouts to prevent hanging
        from botocore.config import Config
        boto_config = Config(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={'max_attempts': 3},
            # Use path-style addressing for reverse proxy compatibility
            s3={'addressing_style': 'path'}
        )

        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=boto_config,
            # Don't use SSL for local development
            use_ssl=False if endpoint_url.startswith('http://') else True,
        )
        self.bucket = bucket
        self._ensure_bucket_exists()

    @property
    def capabilities(self) -> set[StorageCapability]:
        """S3 supports add, get, raw storage."""
        return {
            StorageCapability.ADD,
            StorageCapability.GET,
            StorageCapability.RAW_STORAGE,
            StorageCapability.DELETE
        }

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except self.s3.exceptions.NoSuchBucket:
            # NoSuchBucket - bucket doesn't exist, create it
            try:
                self.s3.create_bucket(Bucket=self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
            except (BotoCoreError, ClientError) as e:
                logger.error(f"Failed to create bucket {self.bucket}: {e}")
                raise
        except ClientError as e:
            # MinIO returns 404 as generic ClientError, not NoSuchBucket
            # Check if it's a 404 error (bucket not found)
            if e.response.get('Error', {}).get('Code') == '404' or e.response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 404:
                try:
                    self.s3.create_bucket(Bucket=self.bucket)
                    logger.info(f"Created bucket: {self.bucket} (detected via 404)")
                except (BotoCoreError, ClientError) as create_error:
                    logger.error(f"Failed to create bucket {self.bucket}: {create_error}")
                    raise
            else:
                # Some other ClientError, re-raise
                logger.error(f"Error checking bucket {self.bucket}: {e}")
                raise

    async def _add_internal(self, artifact):
        """S3 doesn't support artifacts in the same way.

        Use write() method directly for file operations.
        """
        raise NotImplementedError("Use write() method for S3 operations")

    async def _get_internal(self, artifact_id: str):
        """S3 doesn't support artifact lookup by ID.

        Use read() method with file_path for S3 operations.
        """
        raise NotImplementedError("Use read() method for S3 operations")

    async def _search_internal(self, query, k, session_id, artifact_type, **filters):
        """S3 doesn't support search operations."""
        return []

    async def _delete_internal(self, artifact_id: str) -> bool:
        """S3 doesn't support deletion by artifact ID."""
        raise NotImplementedError("Use delete_file() method for S3 operations")

    def write(self, file_path: str, content: str, overwrite: bool = True) -> dict:
        """Write raw content to S3/MinIO.

        Args:
            file_path: Path within bucket (e.g., /raw/worker_1_output.txt)
            content: Content to write
            overwrite: If True, overwrite existing files. If False, return error if file exists.

        Returns:
            Dict with success status or error
        """
        key = file_path.lstrip('/')

        # If overwrite is False, check if file exists first
        if not overwrite:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=key)
                return {"error": f"File '{file_path}' already exists"}
            except self.s3.exceptions.NoSuchKey:
                pass  # File doesn't exist, proceed with write
            except ClientError as e:
                # MinIO returns 404 as generic ClientError, not NoSuchKey
                if e.response.get('Error', {}).get('Code') == '404' or e.response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 404:
                    pass  # File doesn't exist, proceed with write
                else:
                    logger.error(f"S3 head_object error for {key}: {e}")
                    return {"error": f"S3 error checking file: {str(e)}"}
            except (BotoCoreError,) as e:
                logger.error(f"S3 head_object error for {key}: {e}")
                return {"error": f"S3 error checking file: {str(e)}"}

        # Write content (overwrites if file exists and overwrite=True)
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content.encode('utf-8')
            )
            return {"path": file_path, "success": True}
        except (BotoCoreError, ClientError) as e:
            logger.error(f"S3 put_object error for {key}: {e}")
            return {"error": f"S3 error writing file: {str(e)}"}

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with pagination for large files.

        Args:
            file_path: Path within bucket
            offset: Starting line number (0-indexed)
            limit: Number of lines to return

        Returns:
            File content with line numbers
        """
        key = file_path.lstrip('/')

        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=key
            )
            content = response['Body'].read().decode('utf-8')
        except self.s3.exceptions.NoSuchKey:
            logger.error(f"File not found: {key}")
            return f"Error: File '{file_path}' not found"
        except (BotoCoreError, ClientError) as e:
            logger.error(f"S3 get_object error for {key}: {e}")
            return f"Error: S3 read failed: {str(e)}"

        lines = content.split('\n')

        # Apply offset and limit for pagination
        start_line = offset
        end_line = offset + limit
        paginated = lines[start_line:end_line]

        # Add line numbers
        return '\n'.join(
            f"{i + start_line + 1}: {line}"
            for i, line in enumerate(paginated)
        )

    def delete_file(self, file_path: str) -> bool:
        """Delete a file from S3.

        Args:
            file_path: Path within bucket

        Returns:
            True if deleted, False otherwise
        """
        key = file_path.lstrip('/')

        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted S3 object: {key}")
            return True
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to delete S3 object {key}: {e}")
            return False

    def list_files(self, prefix: str = "", max_keys: int = 1000) -> List[dict]:
        """List objects in bucket with optional prefix.

        Args:
            prefix: Filter by prefix
            max_keys: Maximum number of objects to return

        Returns:
            List of file info dicts
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix.lstrip('/'),
                MaxKeys=max_keys
            )
        except (BotoCoreError, ClientError) as e:
            logger.error(f"S3 list_objects_v2 error: {e}")
            return []

        return [
            {
                "path": obj['Key'],
                "size": obj['Size'],
                "modified_at": obj['LastModified'].timestamp()
            }
            for obj in response.get('Contents', [])
        ]

    def ls(self, prefix: str = "", limit: int = 1000) -> List[dict]:
        """Alias for list_files for compatibility.

        Also adds 'name' field for compatibility with minio_routes.

        Args:
            prefix: Filter by prefix (e.g., /logs/session_id/)
            limit: Maximum number of objects to return

        Returns:
            List of file info dicts with 'name', 'path', 'size', 'modified_at', 'last_modified'
        """
        files = self.list_files(prefix, limit)
        # Add 'name' and 'last_modified' fields for compatibility
        for f in files:
            f['name'] = f['path']
            f['last_modified'] = f.get('modified_at', 0)
        return files

    async def health_check(self) -> HealthCheckResult:
        """S3/MinIO-specific health check."""
        import time
        start = time.time()

        try:
            # Test by listing bucket
            response = self.s3.head_bucket(Bucket=self.bucket)

            latency_ms = (time.time() - start) * 1000

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"MinIO accessible, bucket: {self.bucket}, latency: {latency_ms:.1f}ms",
                latency_ms=latency_ms,
                metadata={"bucket": self.bucket}
            )

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"MinIO bucket check failed: {error_code}"
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"MinIO connection failed: {str(e)}"
            )
