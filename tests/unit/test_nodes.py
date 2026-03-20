"""Unit Tests for LangGraph Nodes.

Tests for ingestion, drafter, critic, neo4j_builder, and artifact_maker nodes.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

from deep_research_engine.state import EngineState, create_initial_state, WorkerResult
from deep_research_engine.nodes import (
    ingestion_node,
    drafter_node,
    critic_node,
    neo4j_builder_node,
    artifact_maker_node,
)
from deep_research_engine.workers import WorkerConfig, WorkerRegistry

# Patch locations
DEEPAGENTS_PATCH = 'deep_research_engine.nodes.drafter.create_deep_agent'

# Check if deepagents is available
try:
    import deepagents
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False


# =============================================================================
# INGESTION NODE TESTS
# =============================================================================

class TestIngestionNode:
    """Tests for ingestion_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingestion_node_success(self, initial_state: EngineState, tmp_path):
        """Test that ingestion node processes data correctly."""
        state = initial_state.copy()
        
        # Mock the context for ingestion
        state["context"] = {
            "config": {
                "ingestion": {
                    "enabled": True,
                    "source": str(tmp_path),
                    "output_dir": str(tmp_path / "output")
                }
            }
        }

        # Create dummy input
        (tmp_path / "messages.html").write_text("<html><body>Test message</body></html>")

        command = await ingestion_node(state)

        assert command.goto == "drafter"
        assert "ingested_items" in command.update
        # History should include ingestion
        assert "ingestion" in command.update["phase_history"]


# =============================================================================
# DRAFTER NODE TESTS
# =============================================================================

class TestDrafterNode:
    """Tests for drafter_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_drafter_node_creates_results(self, initial_state: EngineState):
        """Test that drafter node creates worker results via King Drafter."""
        # Setup
        state = initial_state.copy()
        state["input_data"] = {"query": "What is machine learning?"}
        state["context"] = {"config": {"drafter": {"enabled": True}}}

        # Mock the LLM call
        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [
                    MagicMock(content="Final research draft content")
                ]
            })
            mock_agent_factory.return_value = mock_agent

            # Execute
            command = await drafter_node(state)

            # Verify
            assert "worker_results" in command.update
            assert len(command.update["worker_results"]) > 0
            assert command.update["worker_results"][0]["worker_id"] == "king_drafter"
            assert command.goto == "critic"
            assert "drafter" in command.update["phase_history"]


# =============================================================================
# CRITIC NODE TESTS
# =============================================================================

class TestCriticNode:
    """Tests for critic_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_critic_node_passes_good_results(self, state_with_results: EngineState):
        """Test critic passes."""
        state = state_with_results.copy()
        state["context"] = {"config": {"critic": {"enabled": True}}}

        command = await critic_node(state)

        assert command.update["critic_passed"] is True
        assert command.goto == "neo4j_builder"


# =============================================================================
# NEO4J BUILDER NODE TESTS
# =============================================================================

class TestNeo4jBuilderNode:
    """Tests for neo4j_builder_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_neo4j_builder_node_handles_error(self, initial_state: EngineState):
        """Test that it handles errors gracefully and continues to artifact_maker."""
        state = initial_state.copy()
        state["context"] = {"config": {"neo4j_builder": {"enabled": True}}}
        state["worker_results"] = [{"output": "Test content", "success": True}]
        
        # Mock triplet extraction to raise error
        with patch('deep_research_engine.nodes.neo4j_builder._extract_triplets', side_effect=Exception("LLM Error")):
            command = await neo4j_builder_node(state)
            assert command.goto == "artifact_maker"
            assert "neo4j_error" in command.update


# =============================================================================
# ARTIFACT MAKER NODE TESTS
# =============================================================================

class TestArtifactMakerNode:
    """Tests for artifact_maker_node function."""

    @pytest.fixture
    def mock_registry(self):
        """Clear the registry before and after tests."""
        original_registry = WorkerRegistry._workers.copy()
        WorkerRegistry._workers.clear()
        yield WorkerRegistry
        WorkerRegistry._workers = original_registry

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_artifact_maker_node_creates_report(self, state_with_results: EngineState, mock_registry):
        """Test artifact maker node creates final report."""
        state = state_with_results.copy()
        state["context"] = {
            "config": {
                "artifact_maker": {
                    "format": "research_report"
                }
            }
        }

        # Define and Register Mock Worker
        class TestArtifactMakerWorker:
            def __init__(self, config=None): pass
            async def execute(self, state, config):
                state["artifact_output"] = "# Research Report\n\nTest findings"
                state["artifact_metadata"] = {"format": "research_report"}
                return MagicMock(success=True)

        WorkerRegistry.register("artifact_maker")(TestArtifactMakerWorker)

        command = await artifact_maker_node(state)

        assert "artifact_output" in command.update
        assert "# Research Report" in command.update["artifact_output"]
        assert command.goto == "complete"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_artifact_maker_node_handles_failure(self, state_with_results: EngineState, mock_registry):
        """Test artifact maker node handles worker failure by falling back."""
        state = state_with_results.copy()
        state["context"] = {"config": {"artifact_maker": {"enabled": True}}}

        class FailingArtifactMakerWorker:
            def __init__(self, config=None): pass
            async def execute(self, state, config):
                return MagicMock(success=False, error="Artifact generation failed")

        WorkerRegistry.register("artifact_maker")(FailingArtifactMakerWorker)

        command = await artifact_maker_node(state)

        # Should fallback to basic aggregation
        assert "artifact_output" in command.update
        assert "# Final Research Artifact" in command.update["artifact_output"]
        assert command.goto == "complete"
