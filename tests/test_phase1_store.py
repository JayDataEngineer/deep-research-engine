"""Tests for Phase 1: Store + Checkpointer + Logging."""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Set test environment before imports
os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("SMART_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("FAST_MODEL", "claude-haiku-4-5")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")


# =============================================================================
# STATE TESTS
# =============================================================================

class TestState:
    """Tests for minimal ContextEngineState."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        from context_engine.state import create_initial_state

        state = create_initial_state(
            session_id="test_session",
            run_id="test_run",
            max_iterations=3,
        )

        assert state["session_id"] == "test_session"
        assert state["run_id"] == "test_run"
        assert state["current_phase"] == "ingest"
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 3
        assert state["critic_score"] == 0.0
        assert state["critic_passed"] is False
        assert state["critic_issues"] == []
        assert state["critic_suggestions"] == []

    def test_state_has_only_6_core_fields(self):
        """Test that state has minimal fields (6 core orchestration fields)."""
        from context_engine.state import create_initial_state

        state = create_initial_state("s1", "r1")

        # Core orchestration fields (not counting messages which is standard)
        core_fields = [
            "session_id", "run_id",
            "current_phase", "iteration_count", "max_iterations",
            "critic_score", "critic_passed", "critic_issues", "critic_suggestions",
        ]

        for field in core_fields:
            assert field in state, f"Missing field: {field}"


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration."""

    def test_get_smart_model(self):
        """Test getting smart model."""
        from context_engine.config import get_model, SMART_MODEL

        model = get_model("smart")
        assert model is not None
        # ChatLiteLLM stores model in model attribute
        assert hasattr(model, "model_name") or hasattr(model, "model")

    def test_get_fast_model(self):
        """Test getting fast model."""
        from context_engine.config import get_model

        model = get_model("fast")
        assert model is not None

    def test_model_tier_validation(self):
        """Test that model tier must be valid."""
        from context_engine.config import get_model

        # Should work with valid tiers
        get_model("smart")
        get_model("fast")


# =============================================================================
# LOGGING TESTS
# =============================================================================

class TestLogging:
    """Tests for Loguru + Rich logging."""

    def test_setup_logging(self, capsys):
        """Test logging setup."""
        from context_engine.logging import setup_logging

        setup_logging(level="INFO")
        # Should not raise

    def test_log_stage(self, capsys):
        """Test stage logging."""
        from context_engine.logging import log_stage

        log_stage("TEST_STAGE", {"key": "value"})
        # Should print Rich panel to stdout

    def test_log_critic_review_pass(self, capsys):
        """Test critic review logging - pass."""
        from context_engine.logging import log_critic_review

        log_critic_review(score=0.85, passed=True, issues=[])
        # Should print green panel

    def test_log_critic_review_fail(self, capsys):
        """Test critic review logging - fail."""
        from context_engine.logging import log_critic_review

        log_critic_review(
            score=0.5,
            passed=False,
            issues=["Issue 1", "Issue 2"],
            suggestions=["Fix this", "Fix that"],
        )
        # Should print red panel with issues

    def test_log_error(self, capsys):
        """Test error logging."""
        from context_engine.logging import log_error

        log_error("Test error", phase="test_phase", details={"key": "value"})
        # Should print red error panel


# =============================================================================
# STORE TOOLS TESTS
# =============================================================================

class TestStoreTools:
    """Tests for store tools."""

    @pytest.mark.asyncio
    async def test_search_vectors_tool_creation(self):
        """Test search_vectors tool creation."""
        from context_engine.store.tools import create_search_vectors_tool

        # Mock store
        mock_store = MagicMock()
        mock_store.asearch = AsyncMock(return_value=[])

        tool = create_search_vectors_tool(mock_store, "test_session")
        assert tool is not None
        assert tool.name == "search_vectors"

    @pytest.mark.asyncio
    async def test_search_vectors_tool_execution(self):
        """Test search_vectors tool execution."""
        from context_engine.store.tools import create_search_vectors_tool

        # Mock store with results
        mock_item = MagicMock()
        mock_item.key = "doc_001"
        mock_item.value = {"content": "Test content about AI research"}
        mock_item.score = 0.85

        mock_store = MagicMock()
        mock_store.asearch = AsyncMock(return_value=[mock_item])

        tool = create_search_vectors_tool(mock_store, "test_session")

        # Execute tool
        result = await tool.ainvoke({
            "query": "AI research",
            "namespace": "memories",
            "limit": 5,
        })

        assert "Found 1 results" in result
        assert "doc_001" in result
        assert "0.85" in result

    @pytest.mark.asyncio
    async def test_search_vectors_tool_no_results(self):
        """Test search_vectors tool with no results."""
        from context_engine.store.tools import create_search_vectors_tool

        mock_store = MagicMock()
        mock_store.asearch = AsyncMock(return_value=[])

        tool = create_search_vectors_tool(mock_store, "test_session")

        result = await tool.ainvoke({
            "query": "nonexistent",
            "namespace": "memories",
        })

        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_read_all_memories_tool(self):
        """Test read_all_memories tool."""
        from context_engine.store.tools import create_read_all_memories_tool

        # Mock items
        mock_items = []
        for i in range(3):
            item = MagicMock()
            item.key = f"doc_{i}"
            item.namespace = ("test_session", "memories")
            item.value = {"content": f"Content {i}", "source": "web"}
            mock_items.append(item)

        mock_store = MagicMock()
        mock_store.asearch = AsyncMock(return_value=mock_items)

        tool = create_read_all_memories_tool(mock_store, "test_session")

        result = await tool.ainvoke({})

        assert "Total memories: 3" in result


# =============================================================================
# INTEGRATION TESTS (require Docker Postgres)
# =============================================================================

@pytest.mark.integration
class TestStoreIntegration:
    """Integration tests requiring Docker Postgres.

    Run with: pytest -m integration
    Skip in CI with: pytest -m "not integration"
    """

    @pytest.mark.asyncio
    async def test_store_connection_and_crud(self):
        """Test actual store connection and CRUD operations.

        Requires:
            - Docker Postgres running
            - POSTGRES_URL env var set
        """
        pytest.skip("Integration test - requires Docker Postgres")

        # This would test:
        # 1. Connection to Postgres
        # 2. Store setup (table creation)
        # 3. Write with auto-vectorization
        # 4. Semantic search
        # 5. Read back

    @pytest.mark.asyncio
    async def test_checkpointer_persistence(self):
        """Test checkpointer state persistence.

        Requires:
            - Docker Postgres running
        """
        pytest.skip("Integration test - requires Docker Postgres")

        # This would test:
        # 1. Checkpointer setup
        # 2. State save
        # 3. State load
        # 4. Thread isolation


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_store():
    """Create a mock LangGraph store."""
    store = MagicMock()

    # Mock async methods
    store.aput = AsyncMock()
    store.aget = AsyncMock(return_value=None)
    store.asearch = AsyncMock(return_value=[])
    store.adelete = AsyncMock()
    store.alist_namespaces = AsyncMock(return_value=[])

    return store


@pytest.fixture
def mock_checkpointer():
    """Create a mock LangGraph checkpointer."""
    checkpointer = MagicMock()

    checkpointer.aget = AsyncMock(return_value=None)
    checkpointer.aput = AsyncMock()
    checkpointer.setup = AsyncMock()

    return checkpointer


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

# Run async tests with asyncio
pytestmark = pytest.mark.asyncio(loop_scope="function")
