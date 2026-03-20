"""End-to-End Tests for Complete Workflows.

Tests for:
- Full workflow execution with mocked components
- Output generation and validation
- Real-world workflow scenarios
"""

import pytest
import asyncio
from typing import Dict, Any
from pathlib import Path

from deep_research_engine.config import load_config, WorkflowConfig
from deep_research_engine.state import EngineState, create_initial_state
from deep_research_engine.workers import WorkerConfig, WorkerRegistry
from deep_research_engine.workers.base import BaseWorker, WorkerResult


# =============================================================================
# WORKER RESULT E2E TESTS
# =============================================================================

class TestWorkerResultE2E:
    """End-to-end tests for worker results."""

    @pytest.mark.e2e
    def test_worker_result_structure(self, sample_worker_result: Dict[str, Any]):
        """Test worker result has correct structure."""
        required_fields = [
            "worker_id", "worker_type", "task",
            "success", "output", "output_type",
            "confidence", "sources", "source_count",
            "timing", "source_count"
        ]

        for field in required_fields:
            assert field in sample_worker_result

    @pytest.mark.e2e
    def test_worker_result_timing(self, sample_worker_result: Dict[str, Any]):
        """Test worker result timing structure."""
        timing = sample_worker_result["timing"]

        assert "start" in timing
        assert "end" in timing
        assert "duration" in timing
        assert isinstance(timing["duration"], (int, float))

    @pytest.mark.e2e
    def test_worker_result_sources(self, sample_worker_result: Dict[str, Any]):
        """Test worker result sources structure."""
        sources = sample_worker_result["sources"]
        source_count = sample_worker_result["source_count"]

        assert isinstance(sources, list)
        assert isinstance(source_count, int)
        assert source_count == len(sources)


# =============================================================================
# STATE MANAGEMENT E2E TESTS
# =============================================================================

class TestStateManagementE2E:
    """End-to-end tests for state management."""

    @pytest.mark.e2e
    def test_initial_state_creation(self):
        """Test initial state creation."""
        state = create_initial_state(
            workflow_name="test_workflow",
            workflow_type="research",
            input_data={"query": "test query"},
            session_id="test_session",
            run_id="test_run",
            max_iterations=3
        )

        assert state["workflow_name"] == "test_workflow"
        assert state["workflow_type"] == "research"
        assert state["session_id"] == "test_session"
        assert state["run_id"] == "test_run"
        assert state["max_iterations"] == 3

    @pytest.mark.e2e
    def test_state_has_required_fields(self, initial_state: EngineState):
        """Test state has all required fields."""
        required_fields = [
            "workflow_id", "workflow_name", "workflow_type",
            "current_phase", "input_data", "plan",
            "worker_results", "context",
            "session_id", "run_id",
        ]

        for field in required_fields:
            assert field in initial_state, f"Missing field: {field}"

    @pytest.mark.e2e
    def test_state_context_extensibility(self, initial_state: EngineState):
        """Test state context can be extended."""
        initial_state["context"]["custom_key"] = "custom_value"
        initial_state["context"]["nested"] = {"data": [1, 2, 3]}

        assert initial_state["context"]["custom_key"] == "custom_value"
        assert initial_state["context"]["nested"]["data"] == [1, 2, 3]


# =============================================================================
# WORKER REGISTRY E2E TESTS
# =============================================================================

class TestWorkerRegistryE2E:
    """End-to-end tests for worker registry."""

    @pytest.mark.e2e
    def test_worker_registry_list(self):
        """Test worker registry lists workers."""
        workers = WorkerRegistry.list_workers()
        assert isinstance(workers, list)

    @pytest.mark.e2e
    def test_worker_config_creation(self):
        """Test worker config creation."""
        config = WorkerConfig(
            worker_id="test_worker",
            worker_type="web_researcher",
            task="Research AI trends"
        )

        assert config.worker_id == "test_worker"
        assert config.worker_type == "web_researcher"
        assert config.task == "Research AI trends"
        assert config.tools == []
        assert config.max_retries == 3


# =============================================================================
# CONFIG WORKFLOW E2E TESTS
# =============================================================================

class TestConfigWorkflowE2E:
    """End-to-end tests for config-based workflows."""

    @pytest.mark.e2e
    def test_rag_config_phases(self):
        """Test RAG config has expected phases."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        phase_names = [p["name"] for p in config.phases]
        assert len(phase_names) >= 3

    @pytest.mark.e2e
    def test_config_storage_enabled(self):
        """Test config has storage enabled."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        # Memory storage should be enabled
        assert "memory" in config.storage
        assert config.storage["memory"]["enabled"] is True


# =============================================================================
# OUTPUT VALIDATION E2E TESTS
# =============================================================================

class TestOutputValidationE2E:
    """End-to-end tests for output validation."""

    @pytest.mark.e2e
    def test_state_serialization(self, initial_state: EngineState):
        """Test state can be serialized."""
        import json

        # Convert to dict and serialize
        state_dict = dict(initial_state)
        json_str = json.dumps(state_dict, default=str)

        # Should serialize without error
        assert len(json_str) > 0

        # Should deserialize back
        restored = json.loads(json_str)
        assert restored["workflow_name"] == initial_state["workflow_name"]

    @pytest.mark.e2e
    def test_worker_result_serialization(self, sample_worker_result: Dict[str, Any]):
        """Test worker result can be serialized."""
        import json

        json_str = json.dumps(sample_worker_result, default=str)
        assert len(json_str) > 0

        restored = json.loads(json_str)
        assert restored["worker_id"] == sample_worker_result["worker_id"]


# =============================================================================
# REAL-WORLD SCENARIO E2E TESTS
# =============================================================================

class TestRealWorldScenariosE2E:
    """End-to-end tests for real-world scenarios."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_state_creation(self):
        """Test multiple states can be created concurrently."""
        async def create_state(name: str) -> EngineState:
            return create_initial_state(
                workflow_name=name,
                workflow_type="research",
                input_data={"query": f"query for {name}"},
                session_id=f"session_{name}",
                run_id=f"run_{name}",
            )

        # Create multiple states concurrently
        tasks = [
            asyncio.create_task(create_state(f"workflow_{i}"))
            for i in range(5)
        ]
        states = await asyncio.gather(*tasks)

        # All states should be created
        assert len(states) == 5
        assert all(s["workflow_name"].startswith("workflow_") for s in states)

    @pytest.mark.e2e
    def test_config_load_and_validate(self):
        """Test config can be loaded and validated."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))

        # Should be valid WorkflowConfig
        assert isinstance(config, WorkflowConfig)

        # Should have required attributes
        assert config.workflow is not None
        assert config.input is not None
        assert config.phases is not None


# =============================================================================
# PERFORMANCE E2E TESTS
# =============================================================================

class TestPerformanceE2E:
    """Performance tests for end-to-end scenarios."""

    @pytest.mark.e2e
    def test_state_creation_performance(self):
        """Test state creation is fast."""
        import time

        start = time.time()
        for i in range(100):
            state = create_initial_state(
                workflow_name=f"test_{i}",
                workflow_type="research",
                input_data={"query": f"query {i}"},
                session_id=f"session_{i}",
                run_id=f"run_{i}",
            )
        duration = time.time() - start

        # Should create 100 states in under 1 second
        assert duration < 1.0

    @pytest.mark.e2e
    def test_config_load_performance(self):
        """Test config loading is fast."""
        import time

        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"

        start = time.time()
        for _ in range(10):
            config = load_config(str(config_path))
            assert config is not None
        duration = time.time() - start

        # Should load 10 times in under 1 second
        assert duration < 1.0
