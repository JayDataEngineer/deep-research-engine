"""Unit Tests for Storage.

Tests for:
- StorageBackend ABC
- StorageResult dataclass
- StorageRegistry
- StorageManager
- InMemoryStorage backend
- Storage error types
"""

import pytest
import asyncio
from typing import Dict, Any

from deep_research_engine.storage import (
    StorageBackend,
    StorageRegistry,
    StorageManager,
    StorageResult,
    StorageError,
    StorageInitializationError,
    StorageNotFoundError,
    InMemoryStorage,
)


# =============================================================================
# STORAGE RESULT TESTS
# =============================================================================

class TestStorageResult:
    """Tests for StorageResult dataclass."""
    
    @pytest.mark.unit
    def test_result_success(self):
        """Test successful storage result."""
        result = StorageResult(
            success=True,
            id="doc-123",
            metadata={"namespace": "test", "size": 1024}
        )
        
        assert result.success is True
        assert result.id == "doc-123"
        assert result.metadata["namespace"] == "test"
        assert result.error is None
    
    @pytest.mark.unit
    def test_result_failure(self):
        """Test failed storage result."""
        result = StorageResult(
            success=False,
            error="Connection timeout"
        )
        
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.id is None
    
    @pytest.mark.unit
    def test_result_minimal(self):
        """Test minimal storage result."""
        result = StorageResult(success=True)
        
        assert result.success is True
        assert result.id is None
        assert result.metadata is None
        assert result.error is None


# =============================================================================
# STORAGE BACKEND ABC TESTS
# =============================================================================

class TestStorageBackendABC:
    """Tests for StorageBackend ABC."""
    
    @pytest.mark.unit
    def test_backend_requires_initialization(self):
        """Test that backends require implementation."""
        with pytest.raises(TypeError):
            StorageBackend({})  # type: ignore
    
    @pytest.mark.unit
    def test_backend_common_attributes(self, test_storage: StorageBackend):
        """Test backend common attributes."""
        assert test_storage.backend_name == "test_storage"
        assert test_storage.description == "Test storage backend"
        assert test_storage.initialized is False
        assert test_storage.health_status == "unknown"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backend_initialization(self, test_storage: StorageBackend):
        """Test backend initialization."""
        await test_storage.initialize()
        
        assert test_storage.initialized is True
        assert test_storage.health_status == "healthy"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backend_cleanup(self, test_storage: StorageBackend):
        """Test backend cleanup."""
        await test_storage.initialize()
        await test_storage.store({"id": "test"}, namespace="test")
        
        await test_storage.cleanup()
        
        assert test_storage.initialized is False
        # Data should be cleared
        retrieved = await test_storage.retrieve("test", namespace="test")
        assert retrieved is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_backend_health_check(self, test_storage: StorageBackend):
        """Test backend health check."""
        # Before initialization
        healthy = await test_storage.health_check()
        assert healthy is False
        
        # After initialization
        await test_storage.initialize()
        healthy = await test_storage.health_check()
        assert healthy is True


# =============================================================================
# IN-MEMORY STORAGE TESTS
# =============================================================================

class TestInMemoryStorage:
    """Tests for InMemoryStorage backend."""
    
    @pytest.fixture
    def memory_storage(self) -> InMemoryStorage:
        """Create in-memory storage instance."""
        return InMemoryStorage({"max_size_mb": 100})
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_initialization(self, memory_storage: InMemoryStorage):
        """Test in-memory storage initialization."""
        await memory_storage.initialize()
        
        assert memory_storage.initialized is True
        assert memory_storage.health_status == "healthy"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_store(self, memory_storage: InMemoryStorage):
        """Test storing in in-memory storage."""
        await memory_storage.initialize()
        
        data = {"content": "test document", "metadata": {"source": "test"}}
        result = await memory_storage.store(data, namespace="test")
        
        assert result.success is True
        assert result.id is not None
        assert result.metadata["namespace"] == "test"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_retrieve(self, memory_storage: InMemoryStorage):
        """Test retrieving from in-memory storage."""
        await memory_storage.initialize()
        
        # Store
        data = {"content": "test content"}
        result = await memory_storage.store(data, namespace="docs")
        
        # Retrieve
        retrieved = await memory_storage.retrieve(result.id, namespace="docs")
        
        assert retrieved is not None
        assert retrieved["content"] == "test content"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_retrieve_wrong_namespace(self, memory_storage: InMemoryStorage):
        """Test retrieving from wrong namespace returns None."""
        await memory_storage.initialize()
        
        # Store in namespace A
        data = {"content": "test"}
        result = await memory_storage.store(data, namespace="A")
        
        # Try to retrieve from namespace B
        retrieved = await memory_storage.retrieve(result.id, namespace="B")
        
        assert retrieved is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_retrieve_nonexistent(self, memory_storage: InMemoryStorage):
        """Test retrieving nonexistent ID returns None."""
        await memory_storage.initialize()
        
        retrieved = await memory_storage.retrieve("nonexistent-id", namespace="test")
        
        assert retrieved is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_delete(self, memory_storage: InMemoryStorage):
        """Test deleting from in-memory storage."""
        await memory_storage.initialize()
        
        # Store
        data = {"content": "test"}
        result = await memory_storage.store(data, namespace="test")
        
        # Delete
        deleted = await memory_storage.delete(result.id, namespace="test")
        assert deleted is True
        
        # Verify deleted
        retrieved = await memory_storage.retrieve(result.id, namespace="test")
        assert retrieved is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_delete_nonexistent(self, memory_storage: InMemoryStorage):
        """Test deleting nonexistent ID returns False."""
        await memory_storage.initialize()
        
        deleted = await memory_storage.delete("nonexistent", namespace="test")
        assert deleted is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_list_items(self, memory_storage: InMemoryStorage):
        """Test listing items in namespace."""
        await memory_storage.initialize()
        
        # Store multiple items
        for i in range(5):
            await memory_storage.store({"id": f"doc-{i}"}, namespace="test")
        
        items = await memory_storage.list_items(namespace="test", limit=10)
        
        assert len(items) == 5
        assert all("id" in item for item in items)
        assert all("created_at" in item for item in items)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_list_items_limit(self, memory_storage: InMemoryStorage):
        """Test listing items with limit."""
        await memory_storage.initialize()
        
        # Store 10 items
        for i in range(10):
            await memory_storage.store({"id": f"doc-{i}"}, namespace="test")
        
        items = await memory_storage.list_items(namespace="test", limit=5)
        
        assert len(items) == 5
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, memory_storage: InMemoryStorage):
        """Test cleanup clears all data."""
        await memory_storage.initialize()
        
        # Store data
        for i in range(5):
            await memory_storage.store({"id": f"doc-{i}"}, namespace="test")
        
        # Cleanup
        await memory_storage.cleanup()
        
        # Verify cleared
        items = await memory_storage.list_items(namespace="test")
        assert len(items) == 0
        assert memory_storage.initialized is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_multiple_namespaces(self, memory_storage: InMemoryStorage):
        """Test multiple namespaces."""
        await memory_storage.initialize()
        
        # Store in different namespaces
        await memory_storage.store({"content": "A"}, namespace="ns_a")
        await memory_storage.store({"content": "B"}, namespace="ns_b")
        
        # List each namespace
        items_a = await memory_storage.list_items(namespace="ns_a")
        items_b = await memory_storage.list_items(namespace="ns_b")
        
        assert len(items_a) == 1
        assert len(items_b) == 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_default_namespace(self, memory_storage: InMemoryStorage):
        """Test default namespace."""
        await memory_storage.initialize()
        
        # Store without specifying namespace
        result = await memory_storage.store({"content": "default"})
        
        # Should use "default" namespace
        retrieved = await memory_storage.retrieve(result.id)
        assert retrieved is not None
        assert retrieved["content"] == "default"


# =============================================================================
# STORAGE REGISTRY TESTS
# =============================================================================

class TestStorageRegistry:
    """Tests for StorageRegistry."""
    
    @pytest.mark.unit
    def test_registry_register_decorator(self):
        """Test backend registration via decorator."""
        
        @StorageRegistry.register("custom_backend")
        class CustomBackend(StorageBackend):
            backend_name = "custom_backend"
            description = "Custom test backend"
            
            async def initialize(self) -> None:
                self.initialized = True
            
            async def cleanup(self) -> None:
                self.initialized = False
            
            async def health_check(self) -> bool:
                return self.initialized
            
            async def store(self, data, namespace="default"):
                return StorageResult(success=True)
            
            async def retrieve(self, id, namespace="default"):
                return None
            
            async def delete(self, id, namespace="default"):
                return True
        
        # Should be registered
        assert "custom_backend" in StorageRegistry.list_backends()
        
        # Should be retrievable
        backend_class = StorageRegistry.get("custom_backend")
        assert backend_class == CustomBackend
    
    @pytest.mark.unit
    def test_registry_get_nonexistent(self):
        """Test getting nonexistent backend."""
        with pytest.raises(StorageNotFoundError) as exc_info:
            StorageRegistry.get("nonexistent_backend")
        
        assert "not found" in str(exc_info.value).lower()
    
    @pytest.mark.unit
    def test_registry_list_backends(self):
        """Test listing registered backends."""
        backends = StorageRegistry.list_backends()
        
        assert isinstance(backends, list)
        assert "memory" in backends  # InMemoryStorage is registered
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_registry_instantiation(self):
        """Test instantiating backend from registry."""
        backend_class = StorageRegistry.get("memory")
        
        backend = backend_class({"max_size_mb": 100})
        await backend.initialize()
        
        assert backend.backend_name == "memory"
        assert backend.initialized is True


# =============================================================================
# STORAGE MANAGER TESTS
# =============================================================================

class TestStorageManager:
    """Tests for StorageManager."""
    
    @pytest.fixture
    def storage_manager(self) -> StorageManager:
        """Create storage manager instance."""
        return StorageManager()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manager_initialize_backend(self, storage_manager: StorageManager):
        """Test manager initializes backend."""
        await storage_manager.initialize("memory", {"max_size_mb": 100})
        
        assert "memory" in storage_manager.backends
        assert storage_manager.initialized is False  # Manager doesn't track its own initialized state
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manager_initialize_multiple_backends(self, storage_manager: StorageManager):
        """Test manager initializes multiple backends."""
        await storage_manager.initialize("memory", {"max_size_mb": 100})
        
        # Memory is the only backend we can reliably test
        assert "memory" in storage_manager.backends
        assert storage_manager.backends["memory"].initialized is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manager_cleanup(self, storage_manager: StorageManager):
        """Test manager cleanup."""
        await storage_manager.initialize("memory", {"max_size_mb": 100})
        
        # Store data
        backend = storage_manager.backends["memory"]
        await backend.store({"content": "test"}, namespace="test")
        
        # Cleanup
        await storage_manager.cleanup()
        
        # Verify backend cleaned up
        assert storage_manager.backends["memory"].initialized is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manager_store(self, storage_manager: StorageManager):
        """Test storing via manager."""
        await storage_manager.initialize("memory", {"max_size_mb": 100})
        
        doc_id = await storage_manager.store(
            "memory",
            {"content": "test document"},
            namespace="test"
        )
        
        assert doc_id is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manager_store_uninitialized_backend(self, storage_manager: StorageManager):
        """Test storing to uninitialized backend."""
        # Don't initialize
        doc_id = await storage_manager.store(
            "memory",
            {"content": "test"},
            namespace="test"
        )
        
        assert doc_id is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manager_retrieve(self, storage_manager: StorageManager):
        """Test retrieving via manager."""
        await storage_manager.initialize("memory", {"max_size_mb": 100})
        
        # Store
        doc_id = await storage_manager.store(
            "memory",
            {"content": "test"},
            namespace="test"
        )
        
        # Retrieve
        retrieved = await storage_manager.retrieve("memory", doc_id, namespace="test")
        
        assert retrieved is not None
        assert retrieved["content"] == "test"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_manager_retrieve_uninitialized(self, storage_manager: StorageManager):
        """Test retrieving from uninitialized backend."""
        retrieved = await storage_manager.retrieve(
            "memory",
            "some-id",
            namespace="test"
        )
        
        assert retrieved is None


# =============================================================================
# STORAGE ERROR TESTS
# =============================================================================

class TestStorageErrors:
    """Tests for storage error types."""
    
    @pytest.mark.unit
    def test_storage_error_base(self):
        """Test base StorageError."""
        error = StorageError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    @pytest.mark.unit
    def test_storage_initialization_error(self):
        """Test StorageInitializationError."""
        error = StorageInitializationError("Initialization failed")
        assert str(error) == "Initialization failed"
        assert isinstance(error, StorageError)
    
    @pytest.mark.unit
    def test_storage_not_found_error(self):
        """Test StorageNotFoundError."""
        error = StorageNotFoundError("Backend not found")
        assert str(error) == "Backend not found"
        assert isinstance(error, StorageError)
    
    @pytest.mark.unit
    def test_error_inheritance(self):
        """Test error inheritance chain."""
        assert issubclass(StorageInitializationError, StorageError)
        assert issubclass(StorageNotFoundError, StorageError)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestStorageIntegration:
    """Integration tests for storage."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_crud_cycle(self):
        """Test full CRUD cycle."""
        storage = InMemoryStorage({})
        await storage.initialize()
        
        # Create
        data = {
            "title": "Test Document",
            "content": "This is test content",
            "tags": ["test", "example"]
        }
        result = await storage.store(data, namespace="docs")
        assert result.success is True
        doc_id = result.id
        
        # Read
        retrieved = await storage.retrieve(doc_id, namespace="docs")
        assert retrieved is not None
        assert retrieved["title"] == "Test Document"
        
        # Update (store with same ID pattern - in real storage this would update)
        # In-memory storage doesn't support update by ID, so we skip
        
        # Delete
        deleted = await storage.delete(doc_id, namespace="docs")
        assert deleted is True
        
        # Verify deleted
        retrieved = await storage.retrieve(doc_id, namespace="docs")
        assert retrieved is None
        
        await storage.cleanup()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_storage_with_complex_data(self):
        """Test storage with complex nested data."""
        storage = InMemoryStorage({})
        await storage.initialize()
        
        complex_data = {
            "metadata": {
                "title": "Complex Document",
                "author": {"name": "Test User", "email": "test@example.com"},
                "tags": ["tag1", "tag2", "tag3"]
            },
            "content": {
                "sections": [
                    {"heading": "Section 1", "body": "Content 1"},
                    {"heading": "Section 2", "body": "Content 2"}
                ]
            },
            "attachments": [
                {"name": "file1.txt", "size": 1024},
                {"name": "file2.txt", "size": 2048}
            ]
        }
        
        result = await storage.store(complex_data, namespace="complex")
        assert result.success is True
        
        retrieved = await storage.retrieve(result.id, namespace="complex")
        assert retrieved is not None
        assert retrieved["metadata"]["author"]["name"] == "Test User"
        assert len(retrieved["content"]["sections"]) == 2
        assert len(retrieved["attachments"]) == 2
        
        await storage.cleanup()
