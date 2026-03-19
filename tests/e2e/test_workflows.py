"""End-to-End Tests for Complete Workflows.

Tests for:
- Full RAG workflow execution
- PDF summarizer workflow
- Config file-based workflows
- Output generation and validation
- Real-world workflow scenarios
"""

import pytest
import asyncio
from typing import Dict, Any, List
from pathlib import Path
import json

from deep_research_engine.engine import run_workflow
from deep_research_engine.config import load_config, WorkflowConfig
from deep_research_engine.state import EngineState


# =============================================================================
# RAG WORKFLOW E2E TESTS
# =============================================================================

class TestRAGWorkflowE2E:
    """End-to-end tests for RAG workflow."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_workflow_basic_execution(self):
        """Test basic RAG workflow execution."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))
        
        events = []
        async for event in run_workflow(
            input_data="What are the latest developments in AI?",
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should complete
        assert len(events) > 0
        assert events[-1].get("type") == "complete"
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_workflow_streaming_events(self):
        """Test RAG workflow streaming events."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))
        
        events = []
        async for event in run_workflow(
            input_data="Test query",
            config=config,
            stream=True
        ):
            events.append(event)
        
        # Should have multiple events
        assert len(events) > 0
        
        # Check event types
        event_types = [e.get("type") for e in events]
        assert any("phase" in t for t in event_types)
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_workflow_storage_enabled(self):
        """Test RAG workflow with storage enabled."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))
        
        # Memory storage should be enabled
        assert config.storage["memory"]["enabled"] is True
        
        events = []
        async for event in run_workflow(
            input_data="Test with storage",
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should complete successfully
        assert events[-1].get("success") is True
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_rag_workflow_custom_session_id(self):
        """Test RAG workflow with custom session ID."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))
        
        custom_session_id = "custom-rag-session-123"
        
        events = []
        async for event in run_workflow(
            input_data="Test query",
            config=config,
            session_id=custom_session_id,
            stream=False
        ):
            events.append(event)
        
        final_state = events[-1].get("state", {})
        assert final_state.get("session_id") == custom_session_id


# =============================================================================
# PDF SUMMARIZER WORKFLOW E2E TESTS
# =============================================================================

class TestPDFSummarizerWorkflowE2E:
    """End-to-end tests for PDF summarizer workflow."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pdf_summarizer_config_loads(self):
        """Test PDF summarizer config loads correctly."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "pdf_summarizer.yaml"
        config = load_config(str(config_path))
        
        assert config.workflow["name"] == "pdf_summarizer"
        assert config.workflow["type"] == "ingestion"
        assert len(config.phases) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_pdf_summarizer_workflow_execution(self, tmp_path: Path):
        """Test PDF summarizer workflow execution."""
        # Create a test PDF config
        config_content = """
workflow:
  name: pdf_summarizer_test
  type: ingestion
  version: 1.0.0

input:
  type: pdf
  path: "/tmp/test.pdf"

phases:
  - name: boss
    node: BossNode
    config:
      strategy: chunk_based

storage:
  memory:
    enabled: true

execution:
  max_steps: 50
"""
        config_file = tmp_path / "pdf_test.yaml"
        config_file.write_text(config_content)
        config = load_config(str(config_file))
        
        events = []
        async for event in run_workflow(
            input_data={"path": "/tmp/test.pdf"},
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should complete
        assert len(events) > 0


# =============================================================================
# SKILL GENERATOR WORKFLOW E2E TESTS
# =============================================================================

class TestSkillGeneratorWorkflowE2E:
    """End-to-end tests for skill generator workflow."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_skill_generator_config_loads(self):
        """Test skill generator config loads correctly."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "skill_generator.yaml"
        config = load_config(str(config_path))
        
        assert config.workflow["name"] == "skill_generator"
        assert config.workflow["type"] == "generation"
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_skill_generator_workflow_execution(self):
        """Test skill generator workflow execution."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "skill_generator.yaml"
        config = load_config(str(config_path))
        
        events = []
        async for event in run_workflow(
            input_data="https://example.com/docs",
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should complete
        assert len(events) > 0


# =============================================================================
# TELEGRAM INGEST WORKFLOW E2E TESTS
# =============================================================================

class TestTelegramIngestWorkflowE2E:
    """End-to-end tests for telegram ingest workflow."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_telegram_ingest_config_loads(self):
        """Test telegram ingest config loads correctly."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "telegram_ingest.yaml"
        config = load_config(str(config_path))
        
        assert config.workflow["name"] == "telegram_ingest"
        assert config.workflow["type"] == "ingestion"
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_telegram_ingest_workflow_execution(self):
        """Test telegram ingest workflow execution."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "telegram_ingest.yaml"
        config = load_config(str(config_path))
        
        events = []
        async for event in run_workflow(
            input_data={"chat_id": "test_chat", "export_path": "/tmp/telegram_export"},
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should complete
        assert len(events) > 0


# =============================================================================
# OUTPUT VALIDATION TESTS
# =============================================================================

class TestOutputValidationE2E:
    """Tests for validating workflow outputs."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_output_structure(self, sample_config_dict: Dict[str, Any]):
        """Test workflow output has correct structure."""
        config = WorkflowConfig(**sample_config_dict)
        
        events = []
        async for event in run_workflow(
            input_data="test query",
            config=config,
            stream=False
        ):
            events.append(event)
        
        final_event = events[-1]
        
        # Check output structure
        assert "type" in final_event
        assert final_event["type"] == "complete"
        assert "success" in final_event
        assert "state" in final_event
        
        state = final_event["state"]
        assert "workflow_id" in state
        assert "workflow_name" in state
        assert "synthesis_output" in state
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_state_completeness(self, sample_config_dict: Dict[str, Any]):
        """Test workflow state has all required fields."""
        config = WorkflowConfig(**sample_config_dict)
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        state = events[-1]["state"]
        
        # Required fields
        required_fields = [
            "workflow_id",
            "workflow_name",
            "workflow_type",
            "current_phase",
            "input_data",
            "worker_results",
            "context",
            "start_time",
            "session_id",
            "run_id",
        ]
        
        for field in required_fields:
            assert field in state, f"Missing field: {field}"
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_timing_information(self, sample_config_dict: Dict[str, Any]):
        """Test workflow includes timing information."""
        config = WorkflowConfig(**sample_config_dict)
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        state = events[-1]["state"]
        
        # Should have timing info
        assert "start_time" in state
        assert state["start_time"] is not None


# =============================================================================
# REAL-WORLD SCENARIO TESTS
# =============================================================================

class TestRealWorldScenariosE2E:
    """Tests for real-world workflow scenarios."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_step_research_workflow(self):
        """Test multi-step research workflow."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "rag.yaml"
        config = load_config(str(config_path))
        
        # Complex query
        query = """
        Research the following topics:
        1. Latest developments in large language models
        2. Vector database technologies for RAG
        3. Agent-based AI systems
        """
        
        events = []
        async for event in run_workflow(
            input_data=query,
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should handle complex query
        assert events[-1].get("success") is True
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_with_env_var_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test workflow with environment variable configuration."""
        monkeypatch.setenv("MODEL_NAME", "test-model")
        monkeypatch.setenv("STORAGE_ENABLED", "true")
        
        config_content = """
workflow:
  name: env_test
  type: research

input:
  type: query
  source: "${MODEL_NAME-test}"

phases:
  - name: boss
    node: BossNode

storage:
  memory:
    enabled: true

execution:
  max_steps: 20
"""
        config_file = tmp_path / "env_test.yaml"
        config_file.write_text(config_content)
        config = load_config(str(config_file))
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        assert events[-1].get("success") is True
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, sample_config_dict: Dict[str, Any]):
        """Test workflow error recovery."""
        config = WorkflowConfig(**sample_config_dict)
        
        # Workflow should handle errors gracefully
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should complete (even if with error state)
        assert len(events) > 0
        final = events[-1]
        assert "type" in final


# =============================================================================
# PERFORMANCE AND SCALABILITY TESTS
# =============================================================================

class TestPerformanceE2E:
    """Performance and scalability tests."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_response_time(self, sample_config_dict: Dict[str, Any]):
        """Test workflow completes within reasonable time."""
        config = WorkflowConfig(**sample_config_dict)
        
        import time
        start = time.time()
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        duration = time.time() - start
        
        # Should complete within 10 seconds (scaffold implementation)
        assert duration < 10.0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, sample_config_dict: Dict[str, Any]):
        """Test multiple workflows running concurrently."""
        config = WorkflowConfig(**sample_config_dict)

        async def execute_workflow(query: str):
            events = []
            async for event in run_workflow(
                input_data=query,
                config=config,
                stream=False
            ):
                events.append(event)
            return events

        # Run 5 workflows concurrently
        queries = [f"query {i}" for i in range(5)]
        tasks = [asyncio.create_task(execute_workflow(q)) for q in queries]
        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 5
        assert all(len(r) > 0 for r in results)
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_workflow_memory_usage(self, sample_config_dict: Dict[str, Any]):
        """Test workflow memory usage with large context."""
        config = WorkflowConfig(**sample_config_dict)
        
        # Large context
        large_context = {
            "documents": [f"doc_{i}" for i in range(100)],
            "metadata": {"size": "large"}
        }
        
        events = []
        async for event in run_workflow(
            input_data={"query": "test", "context": large_context},
            config=config,
            stream=False
        ):
            events.append(event)
        
        # Should handle large context
        assert events[-1].get("success") is True


# =============================================================================
# CONFIGURATION VARIATION TESTS
# =============================================================================

class TestConfigVariationsE2E:
    """Tests for different configuration variations."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_minimal_config_workflow(self, tmp_path: Path):
        """Test workflow with minimal configuration."""
        config_content = """
workflow:
  name: minimal
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
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(config_content)
        config = load_config(str(config_file))
        
        events = []
        async for event in run_workflow(
            input_data="test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        assert len(events) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_maximal_config_workflow(self, tmp_path: Path):
        """Test workflow with full configuration."""
        config_content = """
workflow:
  name: maximal
  type: research
  version: 1.0.0
  description: Full configuration test

input:
  type: query
  source: comprehensive test

phases:
  - name: boss
    node: BossNode
    config:
      strategy: complexity_based
      workers:
        min: 1
        max: 5
  - name: workers
    node: WorkerNode
    config:
      worker_type: web_researcher
      parallel: true
  - name: critic
    node: CriticNode
    config:
      evaluate: worker_outputs
  - name: synthesis
    node: SynthesisNode
    config:
      output:
        format: markdown

storage:
  memory:
    enabled: true
    max_size_mb: 100

llm:
  models:
    main: test-model
    tool: test-tool-model

tools:
  enabled:
    - web_search

execution:
  max_steps: 100
  timeout: 600
  logging: info

output:
  destination:
    type: memory
  format: markdown
"""
        config_file = tmp_path / "maximal.yaml"
        config_file.write_text(config_content)
        config = load_config(str(config_file))
        
        events = []
        async for event in run_workflow(
            input_data="comprehensive test",
            config=config,
            stream=False
        ):
            events.append(event)
        
        assert events[-1].get("success") is True
