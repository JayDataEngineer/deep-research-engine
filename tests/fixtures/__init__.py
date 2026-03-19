"""Test Fixtures - Shared fixtures for all tests.

This module provides pytest fixtures for testing the deep research engine.

Fixtures are organized by category:
- Config fixtures: sample_config_dict, rag_config_path, config_with_env_vars
- State fixtures: initial_state, state_with_results
- Worker fixtures: worker_config, sample_worker_result, TestWorker, test_worker
- Storage fixtures: storage_config, sample_storage_data, TestStorageBackend, test_storage
- Utility fixtures: temp_output_dir, sample_markdown_output

Usage:
    from tests.conftest import sample_config_dict, initial_state
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from deep_research_engine.state import EngineState, create_initial_state, WorkerResult
from deep_research_engine.workers import BaseWorker, WorkerConfig, WorkerRegistry
from deep_research_engine.storage import StorageBackend, StorageRegistry, StorageResult
from deep_research_engine.config import WorkflowConfig


# =============================================================================
# CONFIG FIXTURES
# =============================================================================

@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "workflow": {
            "name": "test_workflow",
            "type": "research",
            "version": "1.0.0",
            "description": "Test workflow"
        },
        "input": {
            "type": "query",
            "source": "test query"
        },
        "phases": [
            {
                "name": "boss",
                "node": "BossNode",
                "config": {
                    "strategy": "complexity_based",
                    "workers": {"min": 1, "max": 3}
                }
            },
            {
                "name": "workers",
                "node": "WorkerNode",
                "config": {
                    "worker_type": "web_researcher",
                    "parallel": True
                }
            }
        ],
        "storage": {
            "memory": {"enabled": True},
            "postgres": {"enabled": False}
        },
        "llm": {
            "models": {
                "main": "test-model",
                "tool": "test-tool-model"
            }
        },
        "tools": {
            "enabled": ["web_search"]
        },
        "execution": {
            "max_steps": 50,
            "timeout": 300
        },
        "output": {
            "destination": {"type": "memory"},
            "format": "markdown"
        }
    }


@pytest.fixture
def rag_config_path(tmp_path: Path) -> Path:
    """Create a temporary RAG config file."""
    config_content = """
workflow:
  name: rag_test
  type: research
  version: 1.0.0

input:
  type: query
  source: "Test RAG query"

phases:
  - name: boss
    node: BossNode
    config:
      strategy: complexity_based

storage:
  memory:
    enabled: true

llm:
  models:
    main: test-model

execution:
  max_steps: 50
"""
    config_file = tmp_path / "test_rag.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def config_with_env_vars(tmp_path: Path) -> Path:
    """Create a config with environment variable placeholders."""
    config_content = """
workflow:
  name: ${WORKFLOW_NAME-test_workflow}
  type: research

input:
  type: query
  source: "${QUERY_SOURCE-default query}"

phases:
  - name: boss
    node: BossNode

storage:
  postgres:
    connection: "${POSTGRES_URL-postgresql://localhost:5432/test}"
"""
    config_file = tmp_path / "test_env.yaml"
    config_file.write_text(config_content)
    return config_file


# =============================================================================
# STATE FIXTURES
# =============================================================================

@pytest.fixture
def initial_state() -> EngineState:
    """Create an initial engine state for testing."""
    return create_initial_state(
        workflow_name="test_workflow",
        workflow_type="research",
        input_data={"query": "test query"},
        session_id="test-session-123",
        run_id="test-run-456",
        max_iterations=3,
        context={"custom_key": "custom_value"}
    )


@pytest.fixture
def state_with_results(initial_state: EngineState) -> EngineState:
    """Create a state with some worker results."""
    state = initial_state.copy()
    state["worker_results"] = [
        {
            "worker_id": "worker_1",
            "worker_type": "web_researcher",
            "task": "research topic A",
            "success": True,
            "output": {"findings": ["finding 1", "finding 2"]},
            "confidence": 0.85
        },
        {
            "worker_id": "worker_2",
            "worker_type": "web_researcher",
            "task": "research topic B",
            "success": True,
            "output": {"findings": ["finding 3"]},
            "confidence": 0.75
        }
    ]
    state["current_phase"] = "critic"
    state["phase_history"] = ["boss", "workers", "critic"]
    return state


# =============================================================================
# WORKER FIXTURES
# =============================================================================

@pytest.fixture
def worker_config() -> WorkerConfig:
    """Create a sample worker configuration."""
    return WorkerConfig(
        worker_id="test_worker_1",
        worker_type="test_worker",
        task="test task",
        tools=["tool1", "tool2"],
        max_retries=3,
        timeout=60,
        enable_critic=True,
        critic_threshold=0.7
    )


@pytest.fixture
def sample_worker_result() -> WorkerResult:
    """Create a sample worker result."""
    sources = [{"url": "http://example.com", "title": "Example"}]
    return WorkerResult(
        worker_id="test_worker_1",
        worker_type="test_worker",
        task="test task",
        success=True,
        output={"data": "test output"},
        output_type="generic",
        confidence=0.85,
        timing={"start": 0.0, "end": 1.0, "duration": 1.0},
        sources=sources,
        source_count=len(sources),
        retry_count=0
    )


class TestWorker(BaseWorker):
    """Test worker implementation for testing."""
    
    worker_type = "test_worker"
    description = "Test worker for unit tests"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute test worker task."""
        self._start_timing()
        
        # Simulate work
        await asyncio.sleep(0.01)
        
        return self._create_result(
            success=True,
            output={"test_data": "test output"},
            output_type="test",
            confidence=0.9
        )


@pytest.fixture
def test_worker(worker_config: WorkerConfig) -> TestWorker:
    """Create a test worker instance."""
    return TestWorker(worker_config)


# =============================================================================
# STORAGE FIXTURES
# =============================================================================

@pytest.fixture
def storage_config() -> Dict[str, Any]:
    """Create a sample storage configuration."""
    return {
        "namespace": "test",
        "connection": "memory://test",
        "options": {"max_size": 100}
    }


@pytest.fixture
def sample_storage_data() -> Dict[str, Any]:
    """Create sample data for storage testing."""
    return {
        "id": "test-doc-1",
        "content": "Test document content",
        "metadata": {
            "source": "test",
            "created_at": "2026-03-18T00:00:00Z"
        }
    }


class TestStorageBackend(StorageBackend):
    """Test storage backend for testing."""
    
    backend_name = "test_storage"
    description = "Test storage backend"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._storage: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialize test storage."""
        self.initialized = True
        self.health_status = "healthy"
    
    async def cleanup(self) -> None:
        """Cleanup test storage."""
        self._storage.clear()
        self.initialized = False
    
    async def health_check(self) -> bool:
        """Check health."""
        return self.initialized
    
    async def store(self, data: Dict[str, Any], namespace: str = "default") -> StorageResult:
        """Store data."""
        doc_id = data.get("id", f"doc-{len(self._storage)}")
        self._storage[f"{namespace}:{doc_id}"] = data
        return StorageResult(success=True, id=doc_id)
    
    async def retrieve(self, id: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """Retrieve data."""
        return self._storage.get(f"{namespace}:{id}")
    
    async def delete(self, id: str, namespace: str = "default") -> bool:
        """Delete data."""
        key = f"{namespace}:{id}"
        if key in self._storage:
            del self._storage[key]
            return True
        return False
    
    async def list_items(self, namespace: str = "default", limit: int = 100) -> list:
        """List items."""
        prefix = f"{namespace}:"
        items = [v for k, v in self._storage.items() if k.startswith(prefix)]
        return items[:limit]


@pytest.fixture
def test_storage(storage_config: Dict[str, Any]) -> TestStorageBackend:
    """Create a test storage backend."""
    return TestStorageBackend(storage_config)


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_markdown_output() -> str:
    """Sample markdown output for testing."""
    return """# Test Report

## Summary
This is a test report with findings.

## Findings
1. Finding 1
2. Finding 2
3. Finding 3

## Sources
- [Source 1](http://example.com/1)
- [Source 2](http://example.com/2)
"""
