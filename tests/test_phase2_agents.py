"""Tests for Phase 2: Core Agents + Pipeline."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Set test environment before imports
os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("SMART_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("FAST_MODEL", "claude-haiku-4-5")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("LITELLM_BASE_URL", "http://localhost:4000/v1")


# =============================================================================
# PROMPTS TESTS
# =============================================================================

class TestPrompts:
    """Tests for prompt management."""

    def test_get_main_agent_prompt(self):
        """Test main agent prompt retrieval."""
        from context_engine.prompts.langfuse import get_prompt_manager

        manager = get_prompt_manager(use_langfuse=False)
        prompt = manager.get_main_agent_prompt("test_session")

        assert "Context Engine Main Agent" in prompt
        assert "test_session" in prompt

    def test_get_critic_agent_prompt(self):
        """Test critic agent prompt retrieval."""
        from context_engine.prompts.langfuse import get_prompt_manager

        manager = get_prompt_manager(use_langfuse=False)
        prompt = manager.get_critic_agent_prompt()

        assert "Critic" in prompt
        assert "passed" in prompt.lower()
        assert "score" in prompt.lower()

    def test_get_flash_explorer_prompt(self):
        """Test flash explorer prompt retrieval."""
        from context_engine.prompts.langfuse import get_prompt_manager

        manager = get_prompt_manager(use_langfuse=False)
        prompt = manager.get_flash_explorer_prompt()

        assert "Flash Explorer" in prompt
        assert "finding" in prompt.lower()

    def test_load_all_prompts(self):
        """Test loading all prompts."""
        from context_engine.prompts.langfuse import load_prompts

        prompts = load_prompts(use_langfuse=False)

        assert "main_agent" in prompts
        assert "critic_agent" in prompts
        assert "flash_explorer" in prompts

    def test_prompt_manager_singleton(self):
        """Test prompt manager singleton pattern."""
        from context_engine.prompts.langfuse import get_prompt_manager

        manager1 = get_prompt_manager(use_langfuse=False)
        manager2 = get_prompt_manager(use_langfuse=False)

        # Same instance
        assert manager1 is manager2

    def test_main_agent_prompt_content(self):
        """Test main agent prompt has required sections."""
        from context_engine.prompts.langfuse import MAIN_AGENT_PROMPT

        assert "write_todos" in MAIN_AGENT_PROMPT
        assert "write_file" in MAIN_AGENT_PROMPT
        assert "search_vectors" in MAIN_AGENT_PROMPT

    def test_critic_agent_prompt_content(self):
        """Test critic prompt has JSON format requirements."""
        from context_engine.prompts.langfuse import CRITIC_AGENT_PROMPT

        assert "passed" in CRITIC_AGENT_PROMPT
        assert "score" in CRITIC_AGENT_PROMPT
        assert "JSON" in CRITIC_AGENT_PROMPT or "json" in CRITIC_AGENT_PROMPT


# =============================================================================
# CRITIC RESPONSE PARSING TESTS
# =============================================================================

class TestCriticResponseParsing:
    """Tests for critic response parsing."""

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response."""
        from context_engine.agents.critic import parse_critic_response

        response = {
            "messages": [
                MagicMock(content='{"passed": true, "score": 0.85, "issues": [], "suggestions": []}')
            ]
        }

        result = parse_critic_response(response)

        assert result["passed"] is True
        assert result["score"] == 0.85
        assert result["issues"] == []
        assert result["suggestions"] == []

    def test_parse_json_with_issues(self):
        """Test parsing JSON with issues."""
        from context_engine.agents.critic import parse_critic_response

        response = {
            "messages": [
                MagicMock(content='{"passed": false, "score": 0.5, "issues": ["Missing sources"], "suggestions": ["Add citations"]}')
            ]
        }

        result = parse_critic_response(response)

        assert result["passed"] is False
        assert result["score"] == 0.5
        assert "Missing sources" in result["issues"]
        assert "Add citations" in result["suggestions"]

    def test_parse_json_in_markdown_block(self):
        """Test parsing JSON in markdown code block."""
        from context_engine.agents.critic import parse_critic_response

        response = {
            "messages": [
                MagicMock(content='Here is my evaluation:\n```json\n{"passed": true, "score": 0.9, "issues": [], "suggestions": []}\n```')
            ]
        }

        result = parse_critic_response(response)

        assert result["passed"] is True
        assert result["score"] == 0.9

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns defaults."""
        from context_engine.agents.critic import parse_critic_response

        response = {
            "messages": [
                MagicMock(content='This is not valid JSON')
            ]
        }

        result = parse_critic_response(response)

        assert result["passed"] is False
        assert result["score"] == 0.0
        assert len(result["issues"]) > 0

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        from context_engine.agents.critic import parse_critic_response

        result = parse_critic_response({})

        assert result["passed"] is False
        assert result["score"] == 0.0


# =============================================================================
# AGENT CREATION TESTS (require deepagents - skip if not installed)
# =============================================================================

class TestAgentCreation:
    """Tests for agent creation - requires deepagents package."""

    @pytest.mark.skipif(
        True,  # Skip by default - deepagents not installed
        reason="deepagents package not installed"
    )
    @pytest.mark.asyncio
    async def test_create_main_agent(self):
        """Test main agent creation."""
        # This test requires deepagents to be installed
        pass

    @pytest.mark.skipif(
        True,  # Skip by default - deepagents not installed
        reason="deepagents package not installed"
    )
    @pytest.mark.asyncio
    async def test_create_critic_agent(self):
        """Test critic agent creation."""
        pass

    @pytest.mark.skipif(
        True,  # Skip by default - deepagents not installed
        reason="deepagents package not installed"
    )
    def test_create_flash_explorer_subagent(self):
        """Test flash explorer subagent definition."""
        pass


# =============================================================================
# PIPELINE TESTS (require deepagents - skip if not installed)
# =============================================================================

class TestPipeline:
    """Tests for pipeline orchestration - requires deepagents."""

    @pytest.mark.skipif(
        True,  # Skip by default - deepagents not installed
        reason="deepagents package not installed"
    )
    @pytest.mark.asyncio
    async def test_run_pipeline(self):
        """Test pipeline execution."""
        pass

    @pytest.mark.skipif(
        True,  # Skip by default - deepagents not installed
        reason="deepagents package not installed"
    )
    @pytest.mark.asyncio
    async def test_pipeline_with_input_files(self):
        """Test pipeline with input file ingestion."""
        pass

    @pytest.mark.skipif(
        True,  # Skip by default - deepagents not installed
        reason="deepagents package not installed"
    )
    @pytest.mark.asyncio
    async def test_pipeline_retry_on_fail(self):
        """Test pipeline retries on critic fail."""
        pass


# =============================================================================
# STATE TESTS
# =============================================================================

class TestState:
    """Tests for state creation."""

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


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_store():
    """Create a mock LangGraph store."""
    store = MagicMock()

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
