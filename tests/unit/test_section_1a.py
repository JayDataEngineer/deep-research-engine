"""Unit Tests for Section 1A - Drafter Node & Workers."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from deep_research_engine.state import EngineState, create_initial_state
from deep_research_engine.workers.base import WorkerConfig
from deep_research_engine.workers.implementations.social_archive import SocialArchiveWorker
from deep_research_engine.workers.implementations.web_researcher import WebResearcherWorker
from deep_research_engine.workers.implementations.chunk_processor import ChunkProcessorWorker
from deep_research_engine.nodes.drafter import drafter_node

@pytest.fixture
def initial_state():
    return create_initial_state(
        workflow_name="test_workflow",
        workflow_type="research",
        input_data={"query": "test query"},
        session_id="test_session",
        run_id="test_run"
    )

@pytest.mark.asyncio
async def test_social_archive_worker_execute(initial_state):
    # Setup mock content
    with patch("builtins.open", MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value="test content")))))):
        initial_state["context"]["big_text_dump"] = "fake_path.md"

        config = WorkerConfig(
            worker_id="social_1",
            worker_type="social_archive",
            task="Analyze social data"
        )
        worker = SocialArchiveWorker(config)

        # Mock LLM calls
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AsyncMock(content="Vibe: Cool"),  # _extract_vibe
            AsyncMock(content='[{"person_a": "A", "person_b": "B", "relationship": "Friends"}]'),  # _map_social_graph
            AsyncMock(content='[{"name": "A", "traits": ["nice"], "summary": "A good person"}]'),  # _extract_friend_profiles
            AsyncMock(content='[{"event": "Party", "description": "Fun time"}]')  # _find_key_memories
        ]

        with patch.object(SocialArchiveWorker, "_get_llm", return_value=mock_llm):
            result = await worker.execute(initial_state, config)

            assert result["success"] is True
            assert result["output"]["vibe_summary"] == "Vibe: Cool"
            assert len(result["output"]["social_graph"]) == 1
            assert result["output"]["social_graph"][0]["person_a"] == "A"
            assert len(result["output"]["friend_profiles"]) == 1
            assert len(result["output"]["key_memories"]) == 1

@pytest.mark.asyncio
async def test_web_researcher_worker_execute(initial_state):
    config = WorkerConfig(
        worker_id="web_1",
        worker_type="web_researcher",
        task="Research AI"
    )
    worker = WebResearcherWorker(config)

    # Mock deep agents - use AsyncMock for awaitable returns
    with patch.object(WebResearcherWorker, "_run_web_agent", new_callable=AsyncMock) as mock_web, \
         patch.object(WebResearcherWorker, "_run_pdf_agent", new_callable=AsyncMock) as mock_pdf, \
         patch.object(WebResearcherWorker, "_run_community_agent", new_callable=AsyncMock) as mock_comm, \
         patch.object(WebResearcherWorker, "_needs_community_sentiment", return_value=True):

        mock_web.return_value = "Web findings"
        mock_pdf.return_value = "PDF findings"
        mock_comm.return_value = "Community Sentiment: People are excited about AI"

        # Test with RAG enabled to trigger PDF agent
        worker.enable_rag = True
        worker.documents = ["doc1.pdf"]

        # Test with query that triggers community sentiment
        query = "What do people think about AI?"
        initial_state["input_data"]["query"] = query

        # Ensure asyncio is available for mocking sleep
        import asyncio

        # Manually ensure enable_community is True
        worker.enable_community = True

        result = await worker.execute(initial_state, config)

        assert result["success"] is True
        assert "Web Research" in result["output"]
        assert "Document Research" in result["output"]
        assert "Community Sentiment" in result["output"]
        assert mock_web.called
        assert mock_pdf.called
        assert mock_comm.called

@pytest.mark.asyncio
async def test_chunk_processor_worker_execute(initial_state):
    config = WorkerConfig(
        worker_id="chunk_1",
        worker_type="chunk_processor",
        task="Process chunk",
        custom={"chunk": "This is a substantive chunk of text about technology."}
    )
    worker = ChunkProcessorWorker(config)

    # Mock the classification and extraction methods
    with patch.object(ChunkProcessorWorker, "_classify_chunk", new_callable=AsyncMock) as mock_classify, \
         patch.object(ChunkProcessorWorker, "_extract_core_content", new_callable=AsyncMock) as mock_extract:

        mock_classify.return_value = {
            "category": "substantive",
            "confidence": 0.9,
            "reasoning": "Looks good",
            "is_filler": False
        }
        mock_extract.return_value = {
            "core_content": "Extracted text",
            "key_points": ["p1"],
            "topics": ["t1"],
            "entities": {},
            "summary": "sum"
        }

        result = await worker.execute(initial_state, config)

        assert result["success"] is True
        assert result["output"]["classification"]["category"] == "substantive"
        assert result["output"]["extraction"]["core_content"] == "Extracted text"
        assert mock_classify.called
        assert mock_extract.called

@pytest.mark.asyncio
async def test_drafter_node_orchestration(initial_state):
    # Setup state with config for drafter
    initial_state["context"]["config"] = {
        "phases": {
            "drafter": {
                "enabled": True,
                "workers": [
                    {"type": "web_researcher", "task": "Search web", "enabled": True}
                ]
            }
        }
    }

    # Mock King Drafter agent
    mock_king = AsyncMock()
    mock_king.ainvoke.return_value = {
        "messages": [MagicMock(content="Final synthesized draft")]
    }

    with patch("deep_research_engine.nodes.drafter.create_deep_agent", return_value=mock_king) as mock_create, \
         patch("deep_research_engine.nodes.drafter.initialize_storage_tools"):

        # drafter_node is wrapped by @node_context which returns Command
        result = await drafter_node(initial_state)

        from langgraph.types import Command
        assert isinstance(result, Command)
        assert "worker_results" in result.update
        assert result.update["worker_results"][0]["output"] == "Final synthesized draft"
        assert result.goto == "critic"
        assert mock_create.called
        # Verify subagents were passed correctly
        args, kwargs = mock_create.call_args
        subagents = kwargs.get("subagents", [])
        assert any(s["name"] == "web_researcher" for s in subagents)
