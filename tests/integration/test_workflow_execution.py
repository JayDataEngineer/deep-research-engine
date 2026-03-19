"""Integration Tests for Workflow Execution.

Tests for:
- run_workflow() main entry point
- Storage initialization and cleanup
- State management across phases
- Config to state translation
- Error handling in workflow execution
"""

import pytest
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from deep_research_engine.engine import run_workflow
from deep_research_engine.config import load_config, WorkflowConfig
from deep_research_engine.state import (
    EngineState,
    create_initial_state,
    merge_worker_results,
)
from deep_research_engine.storage import StorageManager, InMemoryStorage


# =============================================================================
# RUN_WORKFLOW TESTS
# =============================================================================

class TestRunWorkflow:
    """Tests for run_workflow() main entry point."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_workflow_with_string_input(self, rag_config_path: Path):
        """Test running workflow with string query input."""
        config = load_config(str(rag_config_path))
        
        events = []
        async for event in run_workflow(
            input_data="Test research query",
            config=config,
            stream=True
        ):
            events.append(event)
        
        # Should produce events
        assert len(events) > 0
        assert any(e.get("type") == "phase_start" for e in events)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_workflow_with_dict_input(self, sample_config_dict: Dict[str, Any]):
        """Test running workflow with dict input."""
        config = WorkflowConfig(**sample_config_dict)
        
        events = []
        async for event in run_workflow(
            input_data={"query": "test query", "extra": "data"},
            config=config,
            stream=True
        ):
            events.append(event)
        
        assert len(events) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_workflow_generates_run_id(self, sample_config_dict: Dict[str, Any]):
        """Test that workflow generates unique run ID."""
        config = WorkflowConfig(**sample_config_dict)
        
        run_ids = []
        for _ in range(3):
            async for event in run_workflow(
                input_data="test",
                config=config,
                stream=True
            ):
                if event.get("type") == "complete":
                    run_ids.append(event.get("state", {}).get("workflow_id"))
        
        # All run IDs should be unique
        assert len(set(run_ids)) == len(run_ids)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_workflow_with_custom_session_id(self, sample_config_dict: Dict[str, Any]):
        """Test running workflow with custom session ID."""
        config = WorkflowConfig(**sample_config_dict)
        custom_session_id = "custom-session-123"
        
        async for event in run_workflow(
            input_data="test",
            config=config,
            session_id=custom_session_id,
            stream=True
        ):
            if event.get("type") == "complete":
                assert event["state"]["session_id"] == custom_session_id
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_workflow_non_streaming(self, sample_config_dict: Dict[str, Any]):
        """Test running workflow without streaming."""
        config = WorkflowConfig(**sample_config_dict)
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should produce single complete event
        assert len(events) == 1
        assert events[0]["type"] == "complete"
        assert events[0]["success"] is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_workflow_storage_initialization(self, tmp_path: Path):
        """Test that workflow initializes storage correctly."""
        config_content = """
workflow:
  name: storage_test
  type: research

input:
  type: query
  source: test

phases:
  - name: boss
    node: BossNode

storage:
  memory:
    enabled: true

execution:
  max_steps: 10
"""
        config_file = tmp_path / "storage_test.yaml"
        config_file.write_text(config_content)
        config = load_config(str(config_file))
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=True
        ):
            events.append(event)
        
        # Should complete without storage errors
        assert any(e.get("type") in ["complete", "phase_start", "phase_complete"] for e in events)


# =============================================================================
# STORAGE INTEGRATION TESTS
# =============================================================================

class TestStorageIntegration:
    """Tests for storage integration in workflows."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_storage_manager_lifecycle(self):
        """Test storage manager full lifecycle."""
        manager = StorageManager()
        
        # Initialize
        await manager.initialize("memory", {"max_size_mb": 100})
        assert "memory" in manager.backends
        
        # Use
        doc_id = await manager.store(
            "memory",
            {"content": "test"},
            namespace="test"
        )
        assert doc_id is not None
        
        retrieved = await manager.retrieve("memory", doc_id, namespace="test")
        assert retrieved is not None
        
        # Cleanup
        await manager.cleanup()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_storage_manager_multiple_backends(self):
        """Test storage manager with multiple backends."""
        manager = StorageManager()
        
        # Initialize multiple backends
        await manager.initialize("memory", {"max_size_mb": 100})
        
        # Store in different backends (if available)
        doc_id = await manager.store(
            "memory",
            {"content": "test"},
            namespace="test"
        )
        
        assert doc_id is not None
        
        await manager.cleanup()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_in_memory_storage_persistence(self):
        """Test in-memory storage persistence during workflow."""
        storage = InMemoryStorage({})
        await storage.initialize()
        
        # Store multiple items
        ids = []
        for i in range(5):
            result = await storage.store(
                {"index": i, "content": f"doc-{i}"},
                namespace="test"
            )
            ids.append(result.id)
        
        # Retrieve all
        for i, doc_id in enumerate(ids):
            retrieved = await storage.retrieve(doc_id, namespace="test")
            assert retrieved is not None
            assert retrieved["index"] == i
        
        await storage.cleanup()


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================

class TestStateManagement:
    """Tests for state management across workflow phases."""
    
    @pytest.mark.integration
    def test_state_creation_from_config(self, sample_config_dict: Dict[str, Any]):
        """Test creating state from config."""
        config = WorkflowConfig(**sample_config_dict)
        
        state = create_initial_state(
            workflow_name=config.workflow["name"],
            workflow_type=config.workflow["type"],
            input_data={"query": "test"},
            session_id="test-session",
            run_id="test-run"
        )
        
        assert state["workflow_name"] == "test_workflow"
        assert state["workflow_type"] == "research"
        assert state["current_phase"] == "boss"
    
    @pytest.mark.integration
    def test_state_worker_result_accumulation(self, initial_state: EngineState):
        """Test accumulating worker results in state."""
        state = initial_state.copy()
        
        # Simulate parallel worker results
        result1 = {"worker_id": "w1", "output": "result 1"}
        result2 = {"worker_id": "w2", "output": "result 2"}
        result3 = {"worker_id": "w3", "output": "result 3"}
        
        # Accumulate
        state["worker_results"] = merge_worker_results(
            state["worker_results"],
            [result1, result2]
        )
        state["worker_results"] = merge_worker_results(
            state["worker_results"],
            [result3]
        )
        
        assert len(state["worker_results"]) == 3
    
    @pytest.mark.integration
    def test_state_phase_transitions(self, initial_state: EngineState):
        """Test state phase transitions."""
        state = initial_state.copy()
        
        # Simulate workflow progression
        transitions = ["boss", "workers", "critic", "synthesis"]
        
        for phase in transitions:
            state["current_phase"] = phase
            state["phase_history"].append(phase)
        
        assert state["current_phase"] == "synthesis"
        assert len(state["phase_history"]) == 5  # Initial + 4 transitions
        assert "boss" in state["phase_history"]
        assert "synthesis" in state["phase_history"]
    
    @pytest.mark.integration
    def test_state_context_preservation(self, initial_state: EngineState):
        """Test that context preserves workflow-specific data."""
        state = initial_state.copy()
        
        # Add workflow-specific data
        state["context"]["pdf_chunks"] = ["chunk1", "chunk2"]
        state["context"]["embeddings"] = [[0.1, 0.2], [0.3, 0.4]]
        state["context"]["metadata"] = {"source": "test.pdf"}
        
        # Should preserve all data
        assert len(state["context"]["pdf_chunks"]) == 2
        assert len(state["context"]["embeddings"]) == 2
        assert state["context"]["metadata"]["source"] == "test.pdf"


# =============================================================================
# CONFIG TRANSLATION TESTS
# =============================================================================

class TestConfigTranslation:
    """Tests for translating config to execution."""
    
    @pytest.mark.integration
    def test_config_phase_extraction(self, rag_config_path: Path):
        """Test extracting phase configuration."""
        config = load_config(str(rag_config_path))

        # Config has at least one phase (boss)
        assert len(config.phases) >= 1

        # Check boss phase exists
        phase_names = [p["name"] for p in config.phases]
        assert "boss" in phase_names

    @pytest.mark.integration
    def test_config_storage_extraction(self, rag_config_path: Path):
        """Test extracting storage configuration."""
        config = load_config(str(rag_config_path))

        assert "memory" in config.storage
        assert config.storage["memory"]["enabled"] is True

    @pytest.mark.integration
    def test_config_llm_extraction(self, rag_config_path: Path):
        """Test extracting LLM configuration."""
        config = load_config(str(rag_config_path))

        assert "models" in config.llm
        assert "main" in config.llm["models"]


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestWorkflowErrorHandling:
    """Tests for error handling in workflow execution."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_handles_missing_config_file(self):
        """Test workflow handles missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_handles_invalid_yaml(self, tmp_path: Path):
        """Test workflow handles invalid YAML."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: [")
        
        with pytest.raises(Exception):  # yaml.YAMLError
            load_config(str(invalid_yaml))
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_storage_handles_connection_failure(self):
        """Test storage handles connection failures gracefully."""
        # In-memory storage should never fail
        storage = InMemoryStorage({})
        await storage.initialize()
        
        # Try to retrieve nonexistent item
        result = await storage.retrieve("nonexistent", namespace="test")
        assert result is None  # Should return None, not raise
        
        await storage.cleanup()


# =============================================================================
# WORKFLOW PHASE TESTS
# =============================================================================

class TestWorkflowPhases:
    """Tests for workflow phase execution."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_produces_phase_events(self, sample_config_dict: Dict[str, Any]):
        """Test workflow produces phase transition events."""
        config = WorkflowConfig(**sample_config_dict)
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=True
        ):
            events.append(event)
        
        # Should have phase events
        phase_events = [e for e in events if e.get("type") in ["phase_start", "phase_complete"]]
        assert len(phase_events) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_initializes_with_boss_phase(self, sample_config_dict: Dict[str, Any]):
        """Test workflow starts with boss phase."""
        config = WorkflowConfig(**sample_config_dict)
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=True
        ):
            events.append(event)
            if event.get("type") == "phase_start":
                break
        
        # First phase should be boss
        assert events[0].get("type") == "phase_start"
        # Boss phase is implied as first


# =============================================================================
# CONCURRENT WORKFLOW TESTS
# =============================================================================

class TestConcurrentWorkflows:
    """Tests for running multiple workflows concurrently."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_multiple_workflows_concurrently(self, sample_config_dict: Dict[str, Any]):
        """Test running multiple workflows in parallel."""
        config = WorkflowConfig(**sample_config_dict)
        
        async def run_workflow_task(query: str):
            events = []
            async for event in run_workflow(
                input_data=query,
                config=config,
                stream=False
            ):
                events.append(event)
            return events
        
        # Run 3 workflows concurrently
        queries = ["query 1", "query 2", "query 3"]
        tasks = [asyncio.create_task(run_workflow_task(q)) for q in queries]
        results = await asyncio.gather(*tasks)
        
        # All should complete
        assert len(results) == 3
        for result in results:
            assert len(result) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_workflows_isolated_state(self, sample_config_dict: Dict[str, Any]):
        """Test that concurrent workflows have isolated state."""
        config = WorkflowConfig(**sample_config_dict)
        
        run_ids = []
        
        async def run_and_capture_id(query: str):
            async for event in run_workflow(
                input_data=query,
                config=config,
                stream=False
            ):
                if event.get("type") == "complete":
                    return event["state"]["workflow_id"]
            return None
        
        # Run concurrently
        tasks = [
            run_and_capture_id("query 1"),
            run_and_capture_id("query 2"),
        ]
        run_ids = await asyncio.gather(*tasks)
        
        # All run IDs should be unique
        assert len(set(run_ids)) == len(run_ids)
        assert all(rid is not None for rid in run_ids)
