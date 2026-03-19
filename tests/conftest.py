"""Pytest Configuration and Shared Fixtures."""

import pytest
import asyncio
from typing import Generator
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (multiple components)"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (full workflow)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "mock: Tests using mock services"
    )


# =============================================================================
# EVENT LOOP FIXTURE
# =============================================================================

@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Use default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()


# =============================================================================
# IMPORT FIXTURES FROM fixtures MODULE
# =============================================================================

from .fixtures import (
    # Config fixtures
    sample_config_dict,
    rag_config_path,
    config_with_env_vars,
    
    # State fixtures
    initial_state,
    state_with_results,
    
    # Worker fixtures
    worker_config,
    sample_worker_result,
    TestWorker,
    test_worker,
    
    # Storage fixtures
    storage_config,
    sample_storage_data,
    TestStorageBackend,
    test_storage,
    
    # Utility fixtures
    temp_output_dir,
    sample_markdown_output,
)

# Import mock fixtures
from .fixtures.mocks import (
    # Mock classes
    MockLLMProvider,
    MockWebSearch,
    MockDatabase,
    MockHTTPClient,
    LLMResponse,
    SearchResult,
    
    # Mock fixtures
    mock_llm,
    mock_llm_with_delay,
    mock_llm_with_errors,
    mock_web_search,
    mock_web_search_populated,
    mock_database,
    mock_database_populated,
    mock_http_client,
    mock_http_client_api,
    mock_all_services,
    
    # Context managers
    llm_patcher,
    http_patcher,
)

__all__ = [
    # Config
    "sample_config_dict",
    "rag_config_path",
    "config_with_env_vars",
    
    # State
    "initial_state",
    "state_with_results",
    
    # Worker
    "worker_config",
    "sample_worker_result",
    "TestWorker",
    "test_worker",
    
    # Storage
    "storage_config",
    "sample_storage_data",
    "TestStorageBackend",
    "test_storage",
    
    # Utility
    "temp_output_dir",
    "sample_markdown_output",
    
    # Mocks
    "MockLLMProvider",
    "MockWebSearch",
    "MockDatabase",
    "MockHTTPClient",
    "LLMResponse",
    "SearchResult",
    "mock_llm",
    "mock_llm_with_delay",
    "mock_llm_with_errors",
    "mock_web_search",
    "mock_web_search_populated",
    "mock_database",
    "mock_database_populated",
    "mock_http_client",
    "mock_http_client_api",
    "mock_all_services",
    "llm_patcher",
    "http_patcher",
]
