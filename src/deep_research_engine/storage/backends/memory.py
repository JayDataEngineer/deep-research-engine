"""In-memory storage backend (default fallback)."""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..base import StorageBackend, StorageResult


class InMemoryStorage(StorageBackend):
    """In-memory storage backend.
    
    Default fallback when no other backends are configured.
    Data is lost on restart.
    """
    
    backend_name = "memory"
    description = "In-memory storage (ephemeral)"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._store: Dict[str, Dict[str, Any]] = {}
        self._namespaces: Dict[str, set] = {}
        self.max_size_mb = config.get("max_size_mb", 500)
    
    async def initialize(self) -> None:
        self.initialized = True
        self.health_status = "healthy"
        self._log("Initialized in-memory storage")
    
    async def cleanup(self) -> None:
        self._store.clear()
        self._namespaces.clear()
        self.initialized = False
    
    async def health_check(self) -> bool:
        return self.initialized
    
    async def store(
        self,
        data: Dict[str, Any],
        namespace: str = "default",
    ) -> StorageResult:
        resource_id = str(uuid.uuid4())
        
        self._store[resource_id] = {
            "data": data,
            "namespace": namespace,
            "created_at": datetime.now().isoformat(),
        }
        
        if namespace not in self._namespaces:
            self._namespaces[namespace] = set()
        self._namespaces[namespace].add(resource_id)
        
        self._log(f"Stored {resource_id} in namespace {namespace}")
        
        return StorageResult(
            success=True,
            id=resource_id,
            metadata={"namespace": namespace},
        )
    
    async def retrieve(
        self,
        id: str,
        namespace: str = "default",
    ) -> Optional[Dict[str, Any]]:
        item = self._store.get(id)
        if not item:
            return None
        
        if item["namespace"] != namespace:
            return None
        
        return item["data"]
    
    async def delete(
        self,
        id: str,
        namespace: str = "default",
    ) -> bool:
        if id in self._store:
            del self._store[id]
            if namespace in self._namespaces:
                self._namespaces[namespace].discard(id)
            return True
        return False
    
    async def list_items(
        self,
        namespace: str = "default",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        items = []
        for id in self._namespaces.get(namespace, set()):
            if len(items) >= limit:
                break
            item = self._store.get(id)
            if item:
                items.append({
                    "id": id,
                    "namespace": namespace,
                    "created_at": item["created_at"],
                })
        return items
