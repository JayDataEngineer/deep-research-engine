"""Tests for Phase 3: Flash Explorers + Specialists."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

# Set test environment before imports
os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("SMART_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("FAST_MODEL", "claude-haiku-4-5")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("LITELLM_BASE_URL", "http://localhost:4000/v1")


# Check if deepagents is available
try:
    import deepagents
    HAS_DEEPAGENTS = True
except ImportError:
    HAS_DEEPAGENTS = False


# =============================================================================
# MCP TOOLS TESTS
# =============================================================================

class TestMCPTools:
    """Tests for MCP tool wrappers."""

    def test_create_mcp_web_search_tool(self):
        """Test MCP web search tool creation."""
        from context_engine.specialists.web_search import create_mcp_web_search_tool

        tool = create_mcp_web_search_tool()

        assert tool is not None
        assert tool.name == "search_web"
        assert "search" in tool.description.lower()

    def test_create_mcp_scrape_tool(self):
        """Test MCP scrape tool creation."""
        from context_engine.specialists.web_search import create_mcp_scrape_tool

        tool = create_mcp_scrape_tool()

        assert tool is not None
        assert tool.name == "scrape_url"
        assert "scrape" in tool.description.lower() or "url" in tool.description.lower()

    def test_create_all_mcp_tools(self):
        """Test all MCP tools creation."""
        from context_engine.specialists.web_search import create_all_mcp_tools

        tools = create_all_mcp_tools()

        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "search_web" in tool_names
        assert "scrape_url" in tool_names
        assert "crawl_site" in tool_names
        assert "map_domain" in tool_names
        assert "scrape_structured" in tool_names


# =============================================================================
# DOCUMENT TOOLS TESTS
# =============================================================================

class TestDocumentTools:
    """Tests for document processing tools."""

    def test_create_summarize_document_tool(self):
        """Test summarize document tool creation."""
        mock_store = MagicMock()
        mock_store.aget = AsyncMock(return_value=MagicMock(
            value={"content": "Test content"}
        ))

        from context_engine.specialists.document import create_summarize_document_tool

        tool = create_summarize_document_tool(mock_store, "test_session")

        assert tool is not None
        assert tool.name == "summarize_document"

    def test_create_list_documents_tool(self):
        """Test list documents tool creation."""
        mock_store = MagicMock()
        mock_store.alist = AsyncMock(return_value=[])

        from context_engine.specialists.document import create_list_documents_tool

        tool = create_list_documents_tool(mock_store, "test_session")

        assert tool is not None
        assert tool.name == "list_documents"

    def test_create_all_document_tools(self):
        """Test all document tools creation."""
        mock_store = MagicMock()

        from context_engine.specialists.document import create_all_document_tools

        tools = create_all_document_tools(mock_store, "test_session")

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "summarize_document" in tool_names
        assert "extract_key_points" in tool_names
        assert "list_documents" in tool_names


# =============================================================================
# FLASH EXPLORER TESTS
# =============================================================================

class TestFlashExplorer:
    """Tests for flash explorer creation."""

    def test_create_flash_explorer_subagent(self):
        """Test flash explorer subagent definition."""
        mock_store = MagicMock()

        from context_engine.agents.flash_explorer import create_flash_explorer_subagent

        subagent = create_flash_explorer_subagent(
            store=mock_store,
            session_id="test_session",
            use_mcp=True,
        )

        assert subagent["name"] == "flash-explorer"
        assert "description" in subagent
        assert "system_prompt" in subagent
        assert "model" in subagent
        assert "tools" in subagent
        assert len(subagent["tools"]) > 1  # search_vectors + MCP tools

    def test_create_flash_explorer_subagent_without_mcp(self):
        """Test flash explorer without MCP tools."""
        mock_store = MagicMock()

        from context_engine.agents.flash_explorer import create_flash_explorer_subagent

        subagent = create_flash_explorer_subagent(
            store=mock_store,
            session_id="test_session",
            use_mcp=False,
        )

        assert subagent["name"] == "flash-explorer"
        # Should only have search_vectors tool
        assert len(subagent["tools"]) == 1

    def test_create_web_research_flash_explorer(self):
        """Test web research flash explorer."""
        mock_store = MagicMock()

        from context_engine.agents.flash_explorer import create_web_research_flash_explorer

        subagent = create_web_research_flash_explorer(
            store=mock_store,
            session_id="test_session",
        )

        assert subagent["name"] == "flash-explorer"
        assert "web" in subagent["description"].lower() or "search" in subagent["description"].lower()

    @pytest.mark.skipif(not HAS_DEEPAGENTS, reason="deepagents package not installed")
    def test_create_flash_explorer_agent(self):
        """Test flash explorer agent compilation."""
        mock_store = MagicMock()

        from context_engine.agents.flash_explorer import create_flash_explorer_agent

        agent = create_flash_explorer_agent(
            store=mock_store,
            session_id="test_session",
            use_mcp=True,
        )

        assert agent is not None


# =============================================================================
# SPECIALIST SUBAGENT TESTS
# =============================================================================

class TestSpecialistSubagents:
    """Tests for specialist subagent definitions."""

    def test_create_web_search_specialist(self):
        """Test web search specialist subagent."""
        mock_store = MagicMock()

        from context_engine.specialists.web_search import create_web_search_specialist

        specialist = create_web_search_specialist(
            store=mock_store,
            session_id="test_session",
        )

        assert specialist["name"] == "web-search-specialist"
        assert "web" in specialist["description"].lower()
        assert len(specialist["tools"]) >= 5  # All MCP tools + search_vectors

    def test_create_document_specialist(self):
        """Test document specialist subagent."""
        mock_store = MagicMock()

        from context_engine.specialists.document import create_document_specialist

        specialist = create_document_specialist(
            store=mock_store,
            session_id="test_session",
        )

        assert specialist["name"] == "document-specialist"
        assert "document" in specialist["description"].lower()
        assert len(specialist["tools"]) >= 4  # search_vectors + document tools


# =============================================================================
# MAIN AGENT INTEGRATION TESTS
# =============================================================================

class TestMainAgentIntegration:
    """Tests for main agent with specialists."""

    def test_create_default_specialists(self):
        """Test default specialist creation."""
        mock_store = MagicMock()

        from context_engine.agents.main import create_default_specialists

        specialists = create_default_specialists(
            store=mock_store,
            session_id="test_session",
        )

        assert len(specialists) == 3
        names = [s["name"] for s in specialists]
        assert "flash-explorer" in names
        assert "web-search-specialist" in names
        assert "document-specialist" in names

    def test_create_minimal_specialists(self):
        """Test minimal specialist creation."""
        mock_store = MagicMock()

        from context_engine.agents.main import create_minimal_specialists

        specialists = create_minimal_specialists(
            store=mock_store,
            session_id="test_session",
        )

        assert len(specialists) == 1
        assert specialists[0]["name"] == "flash-explorer"

    @pytest.mark.skipif(not HAS_DEEPAGENTS, reason="deepagents package not installed")
    def test_create_main_agent_with_specialists(self):
        """Test main agent creation with specialists."""
        mock_store = MagicMock()

        from context_engine.agents.main import create_main_agent
        from context_engine.prompts.langfuse import load_prompts

        prompts = load_prompts(use_langfuse=False)

        agent = create_main_agent(
            store=mock_store,
            session_id="test_session",
            prompts=prompts,
        )

        assert agent is not None


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
    store.alist = AsyncMock(return_value=[])

    return store


@pytest.fixture
def mock_checkpointer():
    """Create a mock LangGraph checkpointer."""
    checkpointer = MagicMock()

    checkpointer.aget = AsyncMock(return_value=None)
    checkpointer.aput = AsyncMock()
    checkpointer.setup = AsyncMock()

    return checkpointer
