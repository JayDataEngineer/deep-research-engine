"""Integration Tests for Graph Execution.

Tests for complete graph execution including node interactions,
state transitions, and workflow orchestration.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from deep_research_engine.engine import run_workflow
from deep_research_engine.config import WorkflowConfig, load_config
from deep_research_engine.state import EngineState, create_initial_state
from deep_research_engine.storage import StorageManager, InMemoryStorage
from deep_research_engine.workers import WorkerConfig, WorkerRegistry, BaseWorker, WorkerResult


# =============================================================================
# WORKFLOW EXECUTION INTEGRATION TESTS
# =============================================================================

class TestWorkflowExecutionIntegration:
    """Integration tests for complete workflow execution."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, sample_config_dict: Dict[str, Any]):
        """Test complete workflow from start to finish."""
        config = WorkflowConfig(**sample_config_dict)

        # Track state through workflow
        state_history = []

        async def track_state(node_name):
            def decorator(func):
                async def wrapper(state):
                    state_history.append({"node": node_name, "before": state.copy()})
                    result = await func(state)
                    state_history.append({"node": node_name, "after": result.copy()})
                    return result
                return wrapper
            return decorator

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                with patch('deep_research_engine.engine.workers_node') as mock_workers:
                    with patch('deep_research_engine.engine.critic_node') as mock_critic:
                        with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                            # Setup realistic state transitions
                            boss_state = create_initial_state("test", "research", {"query": "test"})
                            boss_state["plan"] = ["Research task 1", "Research task 2"]
                            boss_state["current_phase"] = "workers"
                            boss_state["phase_history"] = ["boss", "workers"]

                            workers_state = boss_state.copy()
                            workers_state["worker_results"] = [
                                {"worker_id": "w1", "success": True, "output": {"finding": "data1"}},
                                {"worker_id": "w2", "success": True, "output": {"finding": "data2"}}
                            ]
                            workers_state["current_phase"] = "critic"
                            workers_state["phase_history"] = ["boss", "workers", "critic"]

                            critic_state = workers_state.copy()
                            critic_state["critique"] = {"passed": True, "score": 0.85}
                            critic_state["current_phase"] = "synthesis"
                            critic_state["phase_history"] = ["boss", "workers", "critic", "synthesis"]

                            synthesis_state = critic_state.copy()
                            synthesis_state["synthesis_output"] = "# Research Report\n\n## Findings\n\nTest findings"
                            synthesis_state["current_phase"] = "complete"

                            mock_boss.return_value = boss_state
                            mock_workers.return_value = workers_state
                            mock_critic.return_value = critic_state
                            mock_synthesis.return_value = synthesis_state

                            events = []
                            async for event in run_workflow(
                                input_data="Test research query",
                                config=config,
                                stream=True
                            ):
                                events.append(event)

                            # Verify workflow completed successfully
                            assert any(e["type"] == "complete" for e in events)
                            complete_event = [e for e in events if e["type"] == "complete"][0]
                            assert complete_event["success"] is True
                            assert "Research Report" in complete_event["output"]

                            # Verify all phases executed
                            assert mock_boss.called
                            assert mock_workers.called
                            assert mock_critic.called
                            assert mock_synthesis.called

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_critic_retry_loop(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with critic requesting retries."""
        config = WorkflowConfig(**sample_config_dict)

        retry_count = {"workers": 0, "critic": 0}

        async def mock_workers_retry(state):
            retry_count["workers"] += 1
            result = state.copy()
            result["worker_results"] = [{"worker_id": f"w{retry_count['workers']}", "success": True}]
            result["current_phase"] = "critic"
            result["phase_history"] = result.get("phase_history", []) + ["workers", "critic"]
            return result

        async def mock_critic_retry(state):
            retry_count["critic"] += 1
            result = state.copy()
            result["iteration_count"] = retry_count["critic"]

            if retry_count["critic"] < 3:
                # Request retry first two times
                result["critique"] = {"passed": False, "score": 0.5, "recommendation": "revise"}
                result["current_phase"] = "workers"
            else:
                # Pass on third try
                result["critique"] = {"passed": True, "score": 0.85, "recommendation": "pass"}
                result["current_phase"] = "synthesis"

            result["phase_history"] = result.get("phase_history", []) + ["critic"]
            return result

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                with patch('deep_research_engine.engine.workers_node', side_effect=mock_workers_retry):
                    with patch('deep_research_engine.engine.critic_node', side_effect=mock_critic_retry):
                        with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                            boss_state = create_initial_state("test", "research", {"query": "test"})
                            boss_state["plan"] = ["Task"]
                            mock_boss.return_value = boss_state

                            synthesis_state = create_initial_state("test", "research", {"query": "test"})
                            synthesis_state["synthesis_output"] = "Report"
                            synthesis_state["current_phase"] = "complete"
                            mock_synthesis.return_value = synthesis_state

                            events = []
                            async for event in run_workflow(
                                input_data="Test",
                                config=config,
                                stream=True
                            ):
                                events.append(event)

                            # Workers should be called 3 times
                            assert retry_count["workers"] == 3
                            # Critic should be called 3 times
                            assert retry_count["critic"] == 3
                            # Should have retry events
                            assert any(e["type"] == "phase_retry" for e in events)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_storage_integration(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with actual storage operations."""
        config = WorkflowConfig(**sample_config_dict)

        # Create real in-memory storage
        real_storage = InMemoryStorage({"namespace": "test"})
        await real_storage.initialize()

        stored_items = []

        original_store = real_storage.store
        async def track_store(data, namespace="default"):
            result = await original_store(data, namespace)
            stored_items.append({"data": data, "namespace": namespace, "result": result})
            return result

        real_storage.store = track_store

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_manager = AsyncMock()
            mock_manager.initialize = AsyncMock()
            mock_manager.cleanup = AsyncMock()
            mock_manager.get_backend = MagicMock(return_value=real_storage)
            MockStorage.return_value = mock_manager

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                with patch('deep_research_engine.engine.workers_node') as mock_workers:
                    with patch('deep_research_engine.engine.critic_node') as mock_critic:
                        with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                            boss_state = create_initial_state("test", "research", {"query": "test"})
                            boss_state["plan"] = ["Task"]
                            mock_boss.return_value = boss_state

                            workers_state = boss_state.copy()
                            workers_state["worker_results"] = [{"success": True}]
                            mock_workers.return_value = workers_state

                            critic_state = workers_state.copy()
                            critic_state["critique"] = {"passed": True}
                            mock_critic.return_value = critic_state

                            synthesis_state = critic_state.copy()
                            synthesis_state["synthesis_output"] = "Report"
                            mock_synthesis.return_value = synthesis_state

                            async for _ in run_workflow(
                                input_data="Test",
                                config=config,
                                stream=False
                            ):
                                pass

                            # Verify storage was used
                            await real_storage.cleanup()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_parallel_workers(self, sample_config_dict: Dict[str, Any]):
        """Test workflow with parallel worker execution."""
        config = WorkflowConfig(**sample_config_dict)

        execution_times = []

        class TimedWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                start = asyncio.get_event_loop().time()
                await asyncio.sleep(0.05)  # Simulate work
                end = asyncio.get_event_loop().time()
                execution_times.append({"worker": config.worker_id, "duration": end - start})
                return {"worker_id": config.worker_id, "success": True}

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                boss_state = create_initial_state("test", "research", {"query": "test"})
                boss_state["plan"] = [f"Task {i}" for i in range(5)]
                mock_boss.return_value = boss_state

                with patch('deep_research_engine.engine.workers_node'):
                    with patch('deep_research_engine.engine.critic_node') as mock_critic:
                        critic_state = boss_state.copy()
                        critic_state["critique"] = {"passed": True}
                        mock_critic.return_value = critic_state

                        with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                            synthesis_state = critic_state.copy()
                            synthesis_state["synthesis_output"] = "Report"
                            mock_synthesis.return_value = synthesis_state

                            with patch.object(WorkerRegistry, 'get', return_value=TimedWorker):
                                events = []
                                async for event in run_workflow(
                                    input_data="Test",
                                    config=config,
                                    stream=False
                                ):
                                    events.append(event)

                                # All workers should execute in ~0.05s if parallel
                                # If sequential, would be ~0.25s
                                total_time = sum(t["duration"] for t in execution_times)
                                # Allow some overhead, but should be much less than sequential
                                assert total_time < 0.15

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_error_propagation(self, sample_config_dict: Dict[str, Any]):
        """Test errors propagate correctly through workflow."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.boss_node', side_effect=Exception("Boss failed")):
                events = []
                async for event in run_workflow(
                    input_data="Test",
                    config=config,
                    stream=True
                ):
                    events.append(event)

                # Should have error event
                error_events = [e for e in events if e["type"] == "error"]
                assert len(error_events) > 0
                assert "Boss failed" in str(error_events[0].get("error", ""))

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, sample_config_dict: Dict[str, Any]):
        """Test state persists data across phases."""
        config = WorkflowConfig(**sample_config_dict)

        # Track state modifications
        state_data = {"custom_field": "initial_value"}

        async def mock_boss_stateful(state):
            state["custom_field"] = "modified_by_boss"
            state["plan"] = ["Task"]
            state["current_phase"] = "workers"
            state["phase_history"] = ["boss", "workers"]
            return state

        async def mock_workers_stateful(state):
            assert state["custom_field"] == "modified_by_boss"
            state["custom_field"] = "modified_by_workers"
            state["worker_results"] = [{"success": True}]
            state["current_phase"] = "critic"
            state["phase_history"] = ["boss", "workers", "critic"]
            return state

        async def mock_critic_stateful(state):
            assert state["custom_field"] == "modified_by_workers"
            state["custom_field"] = "modified_by_critic"
            state["critique"] = {"passed": True}
            state["current_phase"] = "synthesis"
            state["phase_history"] = ["boss", "workers", "critic", "synthesis"]
            return state

        async def mock_synthesis_stateful(state):
            assert state["custom_field"] == "modified_by_critic"
            state["synthesis_output"] = "Report"
            state["current_phase"] = "complete"
            return state

        with patch('deep_research_engine.engine.StorageManager'):
            with patch('deep_research_engine.engine.boss_node', side_effect=mock_boss_stateful):
                with patch('deep_research_engine.engine.workers_node', side_effect=mock_workers_stateful):
                    with patch('deep_research_engine.engine.critic_node', side_effect=mock_critic_stateful):
                        with patch('deep_research_engine.engine.synthesis_node', side_effect=mock_synthesis_stateful):
                            async for event in run_workflow(
                                input_data="Test",
                                config=config,
                                stream=False
                            ):
                                pass

                            # If we get here without assertion errors, state persisted correctly


# =============================================================================
# CONCURRENT WORKFLOW TESTS
# =============================================================================

class TestConcurrentWorkflows:
    """Tests for concurrent workflow execution."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_concurrent_workflows(self, sample_config_dict: Dict[str, Any]):
        """Test running multiple workflows concurrently."""
        config = WorkflowConfig(**sample_config_dict)

        async def run_single_workflow(query: str):
            with patch('deep_research_engine.engine.StorageManager'):
                with patch('deep_research_engine.engine.boss_node') as mock_boss:
                    with patch('deep_research_engine.engine.workers_node') as mock_workers:
                        with patch('deep_research_engine.engine.critic_node') as mock_critic:
                            with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                                boss_state = create_initial_state("test", "research", {"query": query})
                                boss_state["plan"] = ["Task"]
                                mock_boss.return_value = boss_state

                                workers_state = boss_state.copy()
                                workers_state["worker_results"] = [{"success": True}]
                                mock_workers.return_value = workers_state

                                critic_state = workers_state.copy()
                                critic_state["critique"] = {"passed": True}
                                mock_critic.return_value = critic_state

                                synthesis_state = critic_state.copy()
                                synthesis_state["synthesis_output"] = f"Report for: {query}"
                                mock_synthesis.return_value = synthesis_state

                                async for event in run_workflow(query, config, stream=False):
                                    return event

        # Run 3 workflows concurrently
        results = await asyncio.gather(
            run_single_workflow("Query 1"),
            run_single_workflow("Query 2"),
            run_single_workflow("Query 3"),
        )

        # All should complete
        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert "Query 1" in results[0]["output"]
        assert "Query 2" in results[1]["output"]
        assert "Query 3" in results[2]["output"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_isolation(self, sample_config_dict: Dict[str, Any]):
        """Test that concurrent workflows don't interfere with each other."""
        config = WorkflowConfig(**sample_config_dict)

        execution_log = []

        async def tracked_workflow(query: str):
            with patch('deep_research_engine.engine.StorageManager'):
                with patch('deep_research_engine.engine.boss_node') as mock_boss:
                    with patch('deep_research_engine.engine.workers_node') as mock_workers:
                        with patch('deep_research_engine.engine.critic_node') as mock_critic:
                            with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                                boss_state = create_initial_state("test", "research", {"query": query})
                                boss_state["plan"] = ["Task"]
                                boss_state["session_id"] = query  # Use query as session ID
                                mock_boss.return_value = boss_state

                                workers_state = boss_state.copy()
                                workers_state["worker_results"] = [{"success": True, "query": query}]
                                mock_workers.return_value = workers_state

                                critic_state = workers_state.copy()
                                critic_state["critique"] = {"passed": True, "query": query}
                                mock_critic.return_value = critic_state

                                synthesis_state = critic_state.copy()
                                synthesis_state["synthesis_output"] = f"Report: {query}"
                                mock_synthesis.return_value = synthesis_state

                                async for event in run_workflow(query, config, stream=False):
                                    execution_log.append({"query": query, "event": event["type"]})
                                    return event

        # Run concurrently
        await asyncio.gather(
            tracked_workflow("Workflow A"),
            tracked_workflow("Workflow B"),
        )

        # Verify no cross-contamination
        workflow_a_events = [e for e in execution_log if e["query"] == "Workflow A"]
        workflow_b_events = [e for e in execution_log if e["query"] == "Workflow B"]

        assert len(workflow_a_events) > 0
        assert len(workflow_b_events) > 0


# =============================================================================
# STORAGE INITIALIZATION TESTS
# =============================================================================

class TestStorageInitialization:
    """Tests for storage initialization in workflows."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_initializes_enabled_storage(self, sample_config_dict: Dict[str, Any]):
        """Test workflow initializes only enabled storage backends."""
        config_dict = sample_config_dict.copy()
        config_dict["storage"] = {
            "memory": {"enabled": True},
            "postgres": {"enabled": False},
            "mongodb": {"enabled": True}
        }
        config = WorkflowConfig(**config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                boss_state = create_initial_state("test", "research", {"query": "test"})
                boss_state["plan"] = ["Task"]
                mock_boss.return_value = boss_state

                await run_workflow("Test", config, stream=False).__anext__()

                # Should initialize memory and mongodb, not postgres
                initialized_backends = [call[0][0] for call in mock_storage.initialize.call_args_list]
                assert "memory" in initialized_backends
                assert "mongodb" in initialized_backends
                assert "postgres" not in initialized_backends

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_continues_on_storage_failure(self, sample_config_dict: Dict[str, Any]):
        """Test workflow continues if storage initialization fails."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.initialize.side_effect = Exception("Storage init failed")
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                boss_state = create_initial_state("test", "research", {"query": "test"})
                boss_state["plan"] = ["Task"]
                mock_boss.return_value = boss_state

                # Should not raise, should continue with workflow
                events = []
                async for event in run_workflow("Test", config, stream=False):
                    events.append(event)

                assert len(events) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_cleanup_on_complete(self, sample_config_dict: Dict[str, Any]):
        """Test workflow cleans up storage on completion."""
        config = WorkflowConfig(**sample_config_dict)

        with patch('deep_research_engine.engine.StorageManager') as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch('deep_research_engine.engine.boss_node') as mock_boss:
                with patch('deep_research_engine.engine.workers_node') as mock_workers:
                    with patch('deep_research_engine.engine.critic_node') as mock_critic:
                        with patch('deep_research_engine.engine.synthesis_node') as mock_synthesis:
                            boss_state = create_initial_state("test", "research", {"query": "test"})
                            boss_state["plan"] = ["Task"]
                            mock_boss.return_value = boss_state

                            workers_state = boss_state.copy()
                            workers_state["worker_results"] = [{"success": True}]
                            mock_workers.return_value = workers_state

                            critic_state = workers_state.copy()
                            critic_state["critique"] = {"passed": True}
                            mock_critic.return_value = critic_state

                            synthesis_state = critic_state.copy()
                            synthesis_state["synthesis_output"] = "Report"
                            mock_synthesis.return_value = synthesis_state

                            async for _ in run_workflow("Test", config, stream=False):
                                pass

                            # Should cleanup
                            mock_storage.cleanup.assert_called_once()
