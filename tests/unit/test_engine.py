"""Unit Tests for Engine.

Tests for the main workflow engine including run_workflow function.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from deep_research_engine.engine import run_workflow, _run_graph_streaming, _run_graph_sync
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

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                with patch('deep_research_engine.engine.workers_node') as mock_workers:
                    with patch('deep_research_engine.engine.critic_node') as mock_critic:
                        with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                            # Setup mock nodes
                            mock_boss.return_value = {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}
                            mock_workers.return_value = {"worker_results": [{}], "current_phase": "critic", "phase_history": ["boss", "workers", "critic"]}
                            mock_critic.return_value = {"critique": {"passed": True}, "current_phase": "synthesis", "phase_history": ["boss", "workers", "critic", "synthesis"]}
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

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                with patch('deep_research_engine.engine.workers_node') as mock_workers:
                    with patch('deep_research_engine.engine.critic_node') as mock_critic:
                        with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                            mock_boss.return_value = {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}
                            mock_workers.return_value = {"worker_results": [{}], "current_phase": "critic", "phase_history": ["boss", "workers", "critic"]}
                            mock_critic.return_value = {"critique": {"passed": True}, "current_phase": "synthesis", "phase_history": ["boss", "workers", "critic", "synthesis"]}
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

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                mock_boss.return_value = {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}

                await run_workflow(
                    input_data="Test",
                    config=config,
                    stream=False
                ).__anext__()

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

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                mock_boss.return_value = {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}

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
            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                mock_boss.return_value = {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}

                async for event in run_workflow(
                    input_data="Test",
                    config=config,
                    session_id="custom-session-123",
                    stream=False
                ):
                    pass

                # Verify session ID is used (check via boss_node call)
                call_args = mock_boss.call_args[0][0]
                assert call_args.get("session_id") == "custom-session-123"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_with_dict_input(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with dictionary input."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                mock_boss.return_value = {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}

                async for event in run_workflow(
                    input_data={"query": "Test", "extra": "data"},
                    config=config,
                    stream=False
                ):
                    pass

                # Should pass dict to state
                call_args = mock_boss.call_args[0][0]
                assert call_args["input_data"]["extra"] == "data"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_handles_node_error(self, sample_config_dict: Dict[str, Any]):
        """Test workflow handles node execution error."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                mock_boss.side_effect = Exception("Boss node failed")

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

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                mock_boss.return_value = {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}

                async for _ in run_workflow(
                    input_data="Test",
                    config=config,
                    stream=False
                ):
                    pass

                # Should cleanup
                mock_storage.cleanup.assert_called()


# =============================================================================
# STREAMING GRAPH TESTS
# =============================================================================

class TestRunGraphStreaming:
    """Tests for _run_graph_streaming function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_streaming_executes_all_phases(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test streaming executes all workflow phases."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.boss_node') as mock_boss:
            with patch('deep_research_engine.engine.workers_node') as mock_workers:
                with patch('deep_research_engine.engine.critic_node') as mock_critic:
                    with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                        mock_boss.return_value = initial_state.copy()
                        mock_workers.return_value = initial_state.copy()
                        mock_critic.return_value = initial_state.copy()
                        mock_synthesis.return_value = initial_state.copy()

                        events = []
                        async for event in _run_graph_streaming(initial_state, config):
                            events.append(event)

                        # Should have events for all phases
                        phases = [e.get("phase") for e in events if e["type"] == "phase_start"]
                        assert "boss" in phases
                        assert "workers" in phases
                        assert "critic" in phases
                        assert "synthesis" in phases

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_streaming_handles_critic_retry(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test streaming handles critic retry loop."""
        config = WorkflowConfig(**sample_config_dict)

        call_count = {"workers": 0, "critic": 0}

        async def mock_workers_side_effect(state):
            call_count["workers"] += 1
            return state

        async def mock_critic_side_effect(state):
            call_count["critic"] += 1
            if call_count["critic"] == 1:
                # First call: request retry
                return {**state, "critique": {"recommendation": "revise"}, "iteration_count": 1}
            else:
                # Second call: pass
                return {**state, "critique": {"recommendation": "pass"}}

        with patch('deep_research_engine.engine.boss_node', return_value=initial_state.copy()):
            with patch('deep_research_engine.engine.workers_node', side_effect=mock_workers_side_effect):
                with patch('deep_research_engine.engine.critic_node', side_effect=mock_critic_side_effect):
                    with patch('deep_research_engine.engine.synthesis_node', return_value=initial_state.copy()):
                        events = []
                        async for event in _run_graph_streaming(initial_state, config):
                            events.append(event)

                        # Workers should be called twice
                        assert call_count["workers"] == 2
                        # Should have retry event
                        assert any(e["type"] == "phase_retry" for e in events)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_streaming_handles_phase_error(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test streaming handles phase execution error."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.boss_node', side_effect=Exception("Boss failed")):
            events = []
            async for event in _run_graph_streaming(initial_state, config):
                events.append(event)

            # Should have error event
            assert any(e["type"] == "error" for e in events)
            assert "Boss failed" in str([e.get("error") for e in events])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_streaming_yields_duration(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test streaming yields phase duration."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.boss_node', return_value=initial_state.copy()):
            with patch('deep_research_engine.engine.workers_node', return_value=initial_state.copy()):
                with patch('deep_research_engine.engine.critic_node', return_value=initial_state.copy()):
                    with patch('deep_research_engine.engine.synthesis_node', return_value=initial_state.copy()):
                        events = []
                        async for event in _run_graph_streaming(initial_state, config):
                            events.append(event)

                        # Phase complete events should have duration
                        complete_events = [e for e in events if e["type"] == "phase_complete"]
                        for event in complete_events:
                            assert "duration" in event


# =============================================================================
# SYNC GRAPH TESTS
# =============================================================================

class TestRunGraphSync:
    """Tests for _run_graph_sync function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_executes_all_phases(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test sync execution runs all phases."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.boss_node', return_value=initial_state.copy()):
            with patch('deep_research_engine.engine.workers_node', return_value=initial_state.copy()):
                with patch('deep_research_engine.engine.critic_node', return_value=initial_state.copy()):
                    with patch('deep_research_engine.engine.synthesis_node', return_value=initial_state.copy()):
                        result = await _run_graph_sync(initial_state, config)

                        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_handles_boss_failure(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test sync handles boss node failure."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.boss_node') as mock_boss:
            mock_boss.return_value = {**initial_state, "fatal_error": "Boss failed"}

            with pytest.raises(RuntimeError, match="Boss node failed"):
                await _run_graph_sync(initial_state, config)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_handles_workers_failure(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test sync handles workers node failure."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.boss_node', return_value=initial_state.copy()):
            with patch('deep_research_engine.engine.workers_node') as mock_workers:
                mock_workers.return_value = {**initial_state, "fatal_error": "Workers failed"}

                with pytest.raises(RuntimeError, match="Workers node failed"):
                    await _run_graph_sync(initial_state, config)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_handles_critic_loop(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test sync handles critic retry loop."""
        config = WorkflowConfig(**sample_config_dict)
        config.phases = {"critic": {"config": {"max_retries": 3}}}

        call_count = 0

        async def mock_critic(state):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {**state, "critique": {"recommendation": "revise"}, "iteration_count": call_count}
            else:
                return {**state, "critique": {"recommendation": "pass"}}

        with patch('deep_research_engine.engine.boss_node', return_value=initial_state.copy()):
            with patch('deep_research_engine.engine.workers_node', return_value=initial_state.copy()):
                with patch('deep_research_engine.engine.critic_node', side_effect=mock_critic):
                    with patch('deep_research_engine.engine.synthesis_node', return_value=initial_state.copy()):
                        await _run_graph_sync(initial_state, config)

                        # Critic should be called 3 times
                        assert call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_handles_synthesis_failure(self, initial_state: EngineState, sample_config_dict: Dict[str, Any]):
        """Test sync handles synthesis node failure."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.boss_node', return_value=initial_state.copy()):
            with patch('deep_research_engine.engine.workers_node', return_value=initial_state.copy()):
                with patch('deep_research_engine.engine.critic_node', return_value=initial_state.copy()):
                    with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                        mock_synthesis.return_value = {**initial_state, "fatal_error": "Synthesis failed"}

                        with pytest.raises(RuntimeError, match="Synthesis node failed"):
                            await _run_graph_sync(initial_state, config)


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
            {"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]},
            {"worker_results": [{}], "current_phase": "critic", "phase_history": ["boss", "workers", "critic"]},
            {"critique": {"passed": True}, "current_phase": "synthesis", "phase_history": ["boss", "workers", "critic", "synthesis"]},
            {"synthesis_output": "Final report", "current_phase": "complete"},
        ]

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.boss_node', return_value=state_sequence[0]):
                with patch('deep_research_engine.engine.workers_node', return_value=state_sequence[1]):
                    with patch('deep_research_engine.engine.critic_node', return_value=state_sequence[2]):
                        with patch('deep_research_engine.engine.synthesis_node', return_value=state_sequence[3]):
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

            with patch('deep_research_engine.engine.boss_node', return_value={"plan": ["task"], "current_phase": "workers", "phase_history": ["boss", "workers"]}):
                await run_workflow(
                    input_data="Test",
                    config=config,
                    storage_backends=["memory", "postgres"],
                    stream=False
                ).__anext__()

                # Should initialize specified backends
                assert mock_storage.initialize.call_count >= 1
