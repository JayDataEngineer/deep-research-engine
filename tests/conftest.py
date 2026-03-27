"""Pytest Configuration and Shared Fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

from deep_research_engine.state import EngineState, WorkerResult
from deep_research_engine.workers import (
    BaseWorker,
    WorkerConfig,
    WorkerError,
)
from deep_research_engine.storage import StorageBackend, StorageRegistry, StorageResult

# Import mock fixtures from fixtures.mocks
from tests.fixtures.mocks import (
    MockLLMProvider,
    MockWebSearch,
    MockDatabase,
    MockHTTPClient,
)

# Import test fixtures from fixtures
from tests.fixtures import (
    TestStorageBackend,
    TestWorker,
)


# =============================================================================
# Storage Fixtures (from tests.fixtures)
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
def test_storage(storage_config: Dict[str, Any]) -> TestStorageBackend:
    """Create a test storage backend."""
    return TestStorageBackend(storage_config)


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


# =============================================================================
# Worker Fixtures (from tests.fixtures)
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
    from deep_research_engine.workers.base import WorkerResult as WorkerResultDict
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


@pytest.fixture
def test_worker(worker_config: WorkerConfig) -> TestWorker:
    """Create a test worker instance."""
    return TestWorker(worker_config)


# =============================================================================
# Mock Fixtures (from tests.fixtures.mocks)
# =============================================================================


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Create mock LLM provider."""
    llm = MockLLMProvider()
    llm.set_default_response("This is a mock LLM response for testing.")
    return llm


@pytest.fixture
def mock_llm_with_delay() -> MockLLMProvider:
    """Create mock LLM provider with simulated delay."""
    llm = MockLLMProvider()
    llm.set_delay(0.1)  # 100ms delay
    llm.set_default_response("Delayed mock response")
    return llm


@pytest.fixture
def mock_llm_with_errors() -> MockLLMProvider:
    """Create mock LLM provider with error simulation."""
    llm = MockLLMProvider()
    llm.set_error_rate(0.3)  # 30% error rate
    llm.set_default_response("Response (may error)")
    return llm


@pytest.fixture
def mock_web_search() -> MockWebSearch:
    """Create mock web search."""
    search = MockWebSearch()
    search.set_default_results([
        {"title": "Default Result", "url": "http://example.com", "snippet": "Default snippet"}
    ])
    return search


@pytest.fixture
def mock_web_search_populated() -> MockWebSearch:
    """Create mock web search with pre-populated results."""
    search = MockWebSearch()
    search.add_results("AI", [
        {"title": "AI Breakthrough", "url": "http://ai-news.com/breakthrough", "snippet": "Major AI advancement"},
        {"title": "Machine Learning", "url": "http://ml-journal.com/learn", "snippet": "ML fundamentals"},
        {"title": "Neural Networks", "url": "http://deep-learning.org/nn", "snippet": "NN architecture"},
    ])
    search.add_results("climate", [
        {"title": "Climate Report", "url": "http://climate.gov/report", "snippet": "Climate change data"},
        {"title": "Climate Action", "url": "http://climate.org/action", "snippet": "Climate action plan"},
    ])
    return search


@pytest.fixture
def mock_database() -> MockDatabase:
    """Create mock database."""
    db = MockDatabase()
    db.initialize()
    return db


@pytest.fixture
def mock_database_populated() -> MockDatabase:
    """Create mock database with pre-populated tables."""
    db = MockDatabase()
    db.initialize()
    db.create_table("users", {"id": "INTEGER", "name": "TEXT", "email": "TEXT"})
    db.insert("users", {"name": "Alice", "email": "alice@example.com"})
    db.insert("users", {"name": "Bob", "email": "bob@example.com"})
    db.insert("users", {"name": "Charlie", "email": "charlie@example.com"})
    db.create_table("documents", {"id": "INTEGER", "title": "TEXT", "content": "TEXT"})
    db.insert("documents", {"title": "Doc 1", "content": "Content 1"})
    db.insert("documents", {"title": "Doc 2", "content": "Content 2"})
    db.insert("documents", {"title": "Doc 3", "content": "Content 3"})
    return db


@pytest.fixture
def mock_http_client() -> MockHTTPClient:
    """Create mock HTTP client."""
    client = MockHTTPClient()
    return client


@pytest.fixture
def mock_http_client_api() -> MockHTTPClient:
    """Create mock HTTP client with API responses."""
    client = MockHTTPClient()
    client.add_response("/api/v1/status", json_data={
        "status": "healthy",
        "version": "1.0.0"
    }, headers={"Content-Type": "application/json"})
    client.add_response("/api/v1/users", json_data={
        "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    }, headers={"Content-Type": "application/json"})
    client.add_response("/api/v1/search", json_data={
        "results": []
    }, headers={"Content-Type": "application/json"})
    return client


@pytest.fixture
def mock_all_services() -> Dict[str, Any]:
    """Create all mock services together."""
    return {
        "llm": MockLLMProvider(),
        "web_search": MockWebSearch(),
        "database": MockDatabase(),
        "http": MockHTTPClient(),
    }


# =============================================================================
# Test Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# =============================================================================
# Stub Classes for Testing
# =============================================================================


class StubWorker(BaseWorker):
    """Stub worker for testing."""
    
    worker_type = "stub"
    description = "Stub worker for testing"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute stub task."""
        self._start_timing()
        return self._create_result(
            success=True,
            output={"stub": "output"},
            output_type="stub"
        )


class FailingWorker(BaseWorker):
    """Worker that always fails."""
    
    worker_type = "failing"
    description = "Failing worker for testing"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Always fail."""
        self._start_timing()
        raise WorkerError("Intentional failure")


class SlowWorker(BaseWorker):
    """Worker that simulates slow execution."""
    
    worker_type = "slow"
    description = "Slow worker for testing"
    
    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Simulate slow execution."""
        self._start_timing()
        await asyncio.sleep(0.1)
        return self._create_result(
            success=True,
            output={"slow": "output"},
            output_type="slow"
        )


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def sample_worker_results():
    """Sample worker results for testing."""
    return [
        {
            "worker_id": "researcher_0",
            "worker_type": "web_researcher",
            "task": "Search for information",
            "success": True,
            "output": "Research findings...",
            "confidence": 0.85,
            "source_count": 5,
            "timing": {
                "start": 1234567890.0,
                "end": 1234567895.0,
                "duration": 5.0
            }
        },
        {
            "worker_id": "researcher_1",
            "worker_type": "chunk_processor",
            "task": "Process PDF chunks",
            "success": True,
            "output": "PDF analysis results...",
            "confidence": 0.92,
            "source_count": 3,
            "timing": {
                "start": 1234567890.0,
                "end": 1234567893.0,
                "duration": 3.0
            }
        }
    ]


@pytest.fixture
def sample_plan():
    """Sample research plan."""
    return [
        "Search web for primary sources",
        "Analyze findings",
        "Synthesize report"
    ]


@pytest.fixture
def sample_state(sample_worker_results, sample_plan) -> EngineState:
    """Sample engine state for testing."""
    from deep_research_engine.state import create_initial_state
    
    state = create_initial_state(
        workflow_name="test_workflow",
        workflow_type="research",
        input_data={"query": "Test query"},
        session_id="test_session",
        run_id="test_run"
    )
    
    state["worker_results"] = sample_worker_results
    state["plan"] = sample_plan
    state["context"] = {"config": {}}
    
    return state


@pytest.fixture
def initial_state() -> EngineState:
    """Create an initial engine state for testing (alias for sample_state)."""
    from deep_research_engine.state import create_initial_state
    
    return create_initial_state(
        workflow_name="test_workflow",
        workflow_type="research",
        input_data={"query": "Test query"},
        session_id="test_session",
        run_id="test_run"
    )


@pytest.fixture
def sample_critic_evaluation():
    """Sample critic evaluation for testing."""
    from deep_research_engine.state import CriticEvaluation
    
    return CriticEvaluation(
        passed=True,
        score=0.85,
        completeness=0.9,
        accuracy=0.8,
        citation_quality=0.85,
        coherence=0.85,
        issues=["Minor formatting issue"],
        suggestions=["Add more citations"],
        recommendation="pass",
        feedback_to="artifact_maker"
    )


@pytest.fixture
def mock_deep_agent():
    """Mock deep agent for testing."""
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Mock response")]
    })
    return mock_agent


@pytest.fixture
def mock_structured_output():
    """Mock structured output for testing."""
    def _create_mock(evaluation_data=None):
        if evaluation_data is None:
            evaluation_data = {
                "passed": True,
                "score": 0.85,
                "issues": [],
                "suggestions": [],
                "recommendation": "pass"
            }
        
        mock_evaluation = MagicMock()
        for key, value in evaluation_data.items():
            setattr(mock_evaluation, key, value)
        mock_evaluation.model_dump.return_value = evaluation_data
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_evaluation)
        
        return mock_llm
    
    return _create_mock


@pytest.fixture
def tmp_config_file(tmp_path):
    """Create temporary config file for testing."""
    def _create_config(content: str, name: str = "test_config.yaml"):
        config_file = tmp_path / name
        config_file.write_text(content)
        return config_file
    
    return _create_config


@pytest.fixture
def sample_video_file(tmp_path):
    """Create sample video file for testing."""
    # Note: This creates an empty file, not a real video
    # Real video tests would need actual video files
    video_file = tmp_path / "test_video.mp4"
    video_file.touch()
    return video_file


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create sample audio file for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.touch()
    return audio_file


@pytest.fixture
def sample_image_file(tmp_path):
    """Create sample image file for testing."""
    image_file = tmp_path / "test_image.jpg"
    image_file.touch()
    return image_file


@pytest.fixture
def ffmpeg_metadata_file(tmp_path):
    """Create FFmpeg metadata file for testing."""
    def _create_metadata(format_type="new"):
        metadata_file = tmp_path / "time.txt"
        
        if format_type == "new":
            content = (
                "[Parsed_metadata_1 @ 0x7f8b8c0d4e00] pts:108859 pts_time:1.20954\n"
                "lavfi.scene_score=0.436456\n"
                "[Parsed_metadata_1 @ 0x7f8b8c0d4e00] pts:217718 pts_time:2.41908\n"
                "lavfi.scene_score=0.523412\n"
            )
        else:  # old format
            content = (
                "frame=    1 pts=      1 time=00:00:01.23\n"
                "frame=    2 pts=     45 time=00:00:04.50\n"
            )
        
        metadata_file.write_text(content)
        return metadata_file
    
    return _create_metadata


# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_worker_result(
    worker_type="web_researcher",
    success=True,
    output="Mock output",
    confidence=0.85
):
    """Helper to create mock worker result."""
    return {
        "worker_id": f"{worker_type}_0",
        "worker_type": worker_type,
        "task": "Mock task",
        "success": success,
        "output": output,
        "confidence": confidence,
        "source_count": 3
    }


def create_mock_state_with_results(worker_results):
    """Helper to create mock state with worker results."""
    from deep_research_engine.state import create_initial_state
    
    state = create_initial_state(
        workflow_name="test",
        workflow_type="research",
        input_data={"query": "Test"},
        session_id="test",
        run_id="test"
    )
    
    state["worker_results"] = worker_results
    state["plan"] = ["Test plan"]
    state["context"] = {"config": {}}
    
    return state
