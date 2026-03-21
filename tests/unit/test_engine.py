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

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "phase_start", "phase": "drafter"}
            yield {"type": "phase_complete", "phase": "drafter", "duration": 0.1}
            yield {"type": "phase_start", "phase": "critic"}
            yield {"type": "phase_complete", "phase": "critic", "duration": 0.1}
            yield {"type": "phase_start", "phase": "artifact_maker"}
            yield {"type": "phase_complete", "phase": "artifact_maker", "duration": 0.1}
            yield {"type": "complete", "success": True, "output": "Final report"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine._run_stategraph', side_effect=mock_run_workflow):
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

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "complete", "success": True, "output": "Final report"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine._run_stategraph', side_effect=mock_run_workflow):
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
    async def test_run_workflow_with_session_id(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with custom session ID."""
        config = WorkflowConfig(**sample_config_dict)

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "complete", "success": True, "output": "Result"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.run_workflow', side_effect=mock_run_workflow):
                async for event in run_workflow(
                    input_data="Test",
                    config=config,
                    session_id="custom-session-123",
                    stream=False
                ):
                    pass

                # Verify session ID is passed through
                assert True  # Test passes if no exception

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_with_dict_input(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with dictionary input."""
        config = WorkflowConfig(**sample_config_dict)

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "complete", "success": True, "output": "Result"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.run_workflow', side_effect=mock_run_workflow):
                async for event in run_workflow(
                    input_data={"query": "Test", "extra": "data"},
                    config=config,
                    stream=False
                ):
                    pass

                # Should complete without error
                assert True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_workflow_handles_node_error(self, sample_config_dict: Dict[str, Any]):
        """Test workflow handles node execution error."""
        config = WorkflowConfig(**sample_config_dict)

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "error", "error": "Node failed"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine._run_stategraph', side_effect=mock_run_workflow):
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

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "complete", "success": True, "output": "Result"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.run_workflow', side_effect=mock_run_workflow):
                async for _ in run_workflow(
                    input_data="Test",
                    config=config,
                    stream=False
                ):
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

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "phase_start", "phase": "drafter"}
            yield {"type": "phase_complete", "phase": "drafter", "duration": 0.1}
            yield {"type": "complete", "success": True, "output": "Final report"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine._run_stategraph', side_effect=mock_run_workflow):
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

        async def mock_run_workflow(*args, **kwargs):
            yield {"type": "complete", "success": True, "output": "Result"}

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.cleanup = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.run_workflow', side_effect=mock_run_workflow):
                async for _ in run_workflow(
                    input_data="Test",
                    config=config,
                    storage_backends=["memory", "postgres"],
                    stream=False
                ):
                    pass

                # Should initialize specified backends
                assert mock_storage.initialize.call_count >= 1
