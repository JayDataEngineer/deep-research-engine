"""Shared pytest fixtures and configuration for Deep Research Engine tests."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from deep_research_engine.state import EngineState


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CONFIGS_DIR = Path(__file__).parent / "configs"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Return the event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def test_configs_dir() -> Path:
    """Return the path to the test configs directory."""
    TEST_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_CONFIGS_DIR


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture
def tmp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config_content = """
workflow:
  name: test_workflow
  type: research
  max_iterations: 2

phases:
  - name: ingestion
    enabled: true
  - name: drafter
    enabled: true
  - name: critic
    enabled: false
  - name: synthesizer
    enabled: true
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_llm_response() -> MagicMock:
    """Create a mock LLM response."""
    response = MagicMock()
    response.content = "This is a mock LLM response."
    response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    return response


@pytest.fixture
def mock_async_llm_response() -> AsyncMock:
    """Create a mock async LLM response."""
    response = AsyncMock()
    response.content = "This is a mock async LLM response."
    response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    return response


@pytest.fixture
def sample_worker_result() -> dict[str, Any]:
    """Create a mock worker result."""
    return {
        "worker_id": "test_worker_1",
        "worker_type": "test_worker",
        "task": "Test task",
        "success": True,
        "output": {"data": "test output"},
        "output_type": "generic",
        "confidence": 0.9,
        "timing": {
            "start": 1000.0,
            "end": 1001.0,
            "duration": 1.0,
        },
        "token_usage": {
            "prompt": 100,
            "completion": 50,
            "total": 150,
        },
        "sources": [],
        "source_count": 0,
    }


@pytest.fixture
def mock_storage_backend() -> MagicMock:
    """Create a mock storage backend."""
    backend = MagicMock()
    backend.backend_name = "mock_storage"
    backend.initialized = True
    backend.health_status = "healthy"
    backend.store = AsyncMock(return_value={"success": True, "id": "test-id"})
    backend.retrieve = AsyncMock(return_value={"data": "test data"})
    backend.delete = AsyncMock(return_value=True)
    backend.query = AsyncMock(return_value=[])
    return backend


@pytest.fixture
def mock_state() -> dict[str, Any]:
    """Create a mock engine state."""
    return {
        "workflow_id": "test-workflow-123",
        "workflow_name": "test_workflow",
        "workflow_type": "research",
        "current_phase": "ingestion",
        "phase_history": ["ingestion"],
        "input_data": {"query": "Test query"},
        "plan": ["Task 1", "Task 2"],
        "plan_metadata": {},
        "worker_results": [],
        "worker_errors": [],
        "critique": None,
        "critique_history": [],
        "critic_passed": False,
        "synthesis_output": None,
        "synthesis_metadata": {},
        "remaining_steps": 100,
        "iteration_count": 0,
        "max_iterations": 3,
        "context": {},
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": None,
        "session_id": "test-session-123",
        "run_id": "test-run-123",
        "errors": [],
        "fatal_error": None,
    }
@pytest.fixture
def initial_state(mock_state: dict[str, Any]) -> EngineState:
    """Fixture for initial engine state."""
    return mock_state.copy()


@pytest.fixture
def state_with_results(initial_state: EngineState, sample_worker_result: dict[str, Any]) -> EngineState:
    """Fixture for state with some worker results."""
    state = initial_state.copy()
    state["worker_results"] = [sample_worker_result.copy()]
    return state


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Temporarily clear environment variables that might affect tests."""
    # Store original environment
    original_env = os.environ.copy()

    # Clear relevant environment variables
    env_vars_to_clear = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEP_RESEARCH_CONFIG",
        "LITELLM_API_KEY",
    ]
    for var in env_vars_to_clear:
        os.environ.pop(var, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_env_vars() -> Generator[None, None, None]:
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()

    os.environ.update(
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "LITELLM_API_KEY": "test-litellm-key",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def sample_text() -> str:
    """Return sample text for testing."""
    return """
    This is a sample text for testing purposes.
    It contains multiple lines and sentences.
    The quick brown fox jumps over the lazy dog.
    """


@pytest.fixture
def sample_json_data() -> dict[str, Any]:
    """Return sample JSON data for testing."""
    return {
        "name": "Test Data",
        "value": 42,
        "nested": {
            "key1": "value1",
            "key2": "value2",
        },
        "list": [1, 2, 3, 4, 5],
    }


# =============================================================================
# MARKERS
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring external services",
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests",
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests",
    )
    config.addinivalue_line(
        "markers",
        "requires_api: marks tests requiring API keys",
    )


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Return a sample configuration dictionary."""
    return {
        "workflow": {"name": "test_workflow", "type": "research"},
        "input": {"type": "query", "source": "test"},
        "phases": [
            {"name": "ingestion", "enabled": True},
            {"name": "drafter", "enabled": True}
        ],
        "storage": {"memory": {"enabled": True}, "postgres": {"enabled": False}},
        "llm": {"models": {"main": "test-model"}},
        "tools": {"enabled": ["web_search"]},
        "execution": {"max_steps": 50}
    }


@pytest.fixture
def rag_config_path(tmp_path: Path) -> Path:
    """Create a temporary RAG config file."""
    config_content = """
workflow:
  name: rag_test
  type: research

input:
  type: query
  source: test

phases:
  - name: ingestion
  - name: drafter
  - name: critic
  - name: synthesizer
"""
    config_file = tmp_path / "rag_test.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def config_with_env_vars(tmp_path: Path) -> Path:
    """Create a temporary config file with environment variables."""
    config_content = """
workflow:
  name: ${WORKFLOW_NAME}
  type: research

input:
  type: query
  source: ${QUERY_SOURCE}

phases:
  - name: ingestion
"""
    config_file = tmp_path / "env_test.yaml"
    config_file.write_text(config_content)
    return config_file
