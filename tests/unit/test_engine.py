"""Unit Tests for Engine.

Tests for the main workflow engine including run_workflow function.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from deep_research_engine.engine import run_workflow
from deep_research_engine.config import WorkflowConfig
from deep_research_engine.state import EngineState


# =============================================================================
# RUN WORKFLOW TESTS
# =============================================================================

class TestRunWorkflow:
    """Tests for run_workflow main entry point."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_streaming(self, sample_config_dict: Dict[str, Any]):
        """Test workflow execution with streaming enabled."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                with patch('deep_research_engine.engine.critic_node') as mock_critic:
                    with patch('deep_research_engine.engine.synthesizer_node') as mock_synthesis:
                        # Setup mock nodes
                        mock_drafter.return_value = {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}
                        mock_critic.return_value = {"critique": {"passed": True}, "current_phase": "synthesizer", "phase_history": ["drafter", "critic", "synthesizer"]}
                        mock_synthesis.return_value = {"synthesis_output": "Final report", "current_phase": "complete"}

                        events = []
                        async for event in run_workflow(
                            input_data="Test query",
                            config=config,
                            stream=True
                        ):
                            events.append(event)

                        # Should have events for each phase
                        assert len(events) > 0
                        assert any(e["type"] == "phase_start" for e in events)
                        assert any(e["type"] == "phase_complete" for e in events)
                        assert any(e["type"] == "complete" for e in events)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_non_streaming(self, sample_config_dict: Dict[str, Any]):
        """Test workflow execution without streaming."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                with patch('deep_research_engine.engine.critic_node') as mock_critic:
                    with patch('deep_research_engine.engine.synthesizer_node') as mock_synthesis:
                        mock_drafter.return_value = {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}
                        mock_critic.return_value = {"critique": {"passed": True}, "current_phase": "synthesizer", "phase_history": ["drafter", "critic", "synthesizer"]}
                        mock_synthesis.return_value = {"synthesis_output": "Final report", "current_phase": "complete"}

                        events = []
                        async for event in run_workflow(
                            input_data="Test query",
                            config=config,
                            stream=False
                        ):
                            events.append(event)

                        # Should have single complete event
                        assert len(events) == 1
                        assert events[0]["type"] == "complete"
                        assert events[0]["success"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_initializes_storage(self, sample_config_dict: Dict[str, Any]):
        """Test workflow initializes storage backends."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                mock_drafter.return_value = {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}

                try:
                    await run_workflow(
                        input_data="Test",
                        config=config,
                        stream=False
                    ).__anext__()
                except StopAsyncIteration:
                    pass

                # Should initialize memory storage (from config)
                mock_storage.initialize.assert_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_handles_storage_failure(self, sample_config_dict: Dict[str, Any]):
        """Test workflow handles storage initialization failure."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.initialize.side_effect = Exception("Storage failed")
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                mock_drafter.return_value = {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}

                # Should continue despite storage failure
                events = []
                async for event in run_workflow(
                    input_data="Test",
                    config=config,
                    stream=False
                ):
                    events.append(event)

                assert len(events) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_with_session_id(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with custom session ID."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                mock_drafter.return_value = {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}

                try:
                    async for event in run_workflow(
                        input_data="Test",
                        config=config,
                        session_id="custom-session-123",
                        stream=False
                    ):
                        pass
                except StopAsyncIteration:
                    pass

                # Verify session ID is used (check via drafter_node call)
                call_args = mock_drafter.call_args[0][0]
                assert call_args.get("session_id") == "custom-session-123"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_with_dict_input(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with dictionary input."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                mock_drafter.return_value = {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}

                try:
                    async for event in run_workflow(
                        input_data={"query": "Test", "extra": "data"},
                        config=config,
                        stream=False
                    ):
                        pass
                except StopAsyncIteration:
                    pass

                # Should pass dict to state
                call_args = mock_drafter.call_args[0][0]
                assert call_args["input_data"]["extra"] == "data"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_handles_node_error(self, sample_config_dict: Dict[str, Any]):
        """Test workflow handles node execution error."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                mock_drafter.side_effect = Exception("Drafter node failed")

                events = []
                async for event in run_workflow(
                    input_data="Test",
                    config=config,
                    stream=True
                ):
                    events.append(event)

                # Should have error event
                assert any(e["type"] == "error" for e in events)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_cleanup_storage(self, sample_config_dict: Dict[str, Any]):
        """Test workflow cleans up storage on completion."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.drafter_node') as mock_drafter:
                mock_drafter.return_value = {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}

                try:
                    async for _ in run_workflow(
                        input_data="Test",
                        config=config,
                        stream=False
                    ):
                        pass
                except StopAsyncIteration:
                    pass

                # Should cleanup
                mock_storage.cleanup.assert_called()


# =============================================================================
# ENGINE INTEGRATION TESTS
# =============================================================================

class TestEngineIntegration:
    """Integration tests for engine components."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_workflow_with_mock_nodes(self, sample_config_dict: Dict[str, Any]):
        """Test complete workflow with mocked nodes."""
        config = WorkflowConfig(**sample_config_dict)

        # Create state transitions
        state_sequence = [
            {"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]},
            {"critique": {"passed": True}, "current_phase": "synthesizer", "phase_history": ["drafter", "critic", "synthesizer"]},
            {"synthesis_output": "Final report", "current_phase": "complete"},
        ]

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.drafter_node', return_value=state_sequence[0]):
                with patch('deep_research_engine.engine.critic_node', return_value=state_sequence[1]):
                    with patch('deep_research_engine.engine.synthesizer_node', return_value=state_sequence[2]):
                        events = []
                        async for event in run_workflow(
                            input_data="Test query",
                            config=config,
                            stream=True
                        ):
                            events.append(event)

                        # Verify workflow completed
                        assert any(e["type"] == "complete" for e in events)
                        assert any(e["success"] for e in events if e["type"] == "complete")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workflow_with_custom_storage_backends(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with custom storage backends."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.drafter_node', return_value={"plan": ["task"], "current_phase": "critic", "phase_history": ["drafter", "critic"]}):
                try:
                    await run_workflow(
                        input_data="Test",
                        config=config,
                        storage_backends=["memory", "postgres"],
                        stream=False
                    ).__anext__()
                except StopAsyncIteration:
                    pass

                # Should initialize specified backends
                assert mock_storage.initialize.call_count >= 1
