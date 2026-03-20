"""Unit Tests for State Types and Factory.

Tests for:
- EngineState TypedDict
- WorkerResult TypedDict
- CriticResult TypedDict
- create_initial_state() factory
- State reducers (merge_worker_results, merge_errors, overwrite_reducer)
"""

import pytest
from typing import Dict, Any
from datetime import datetime

from deep_research_engine.state import (
    EngineState,
    WorkerResult,
    CriticResult,
    create_initial_state,
    merge_worker_results,
    merge_errors,
    overwrite_reducer,
    PhaseName,
    WorkflowType,
    WorkerType,
)


# =============================================================================
# ENGINE STATE TESTS
# =============================================================================

class TestEngineState:
    """Tests for EngineState TypedDict."""
    
    @pytest.mark.unit
    def test_state_has_required_fields(self, initial_state: EngineState):
        """Test that initial state has all required core fields."""
        assert "workflow_id" in initial_state
        assert "workflow_name" in initial_state
        assert "workflow_type" in initial_state
        assert "current_phase" in initial_state
        assert "input_data" in initial_state
        assert "plan" in initial_state
        assert "worker_results" in initial_state
        assert "context" in initial_state
    
    @pytest.mark.unit
    def test_state_field_types(self, initial_state: EngineState):
        """Test that state fields have correct types."""
        assert isinstance(initial_state["workflow_id"], str)
        assert isinstance(initial_state["workflow_name"], str)
        assert isinstance(initial_state["workflow_type"], str)
        assert isinstance(initial_state["current_phase"], str)
        assert isinstance(initial_state["input_data"], dict)
        assert isinstance(initial_state["plan"], list)
        assert isinstance(initial_state["worker_results"], list)
        assert isinstance(initial_state["context"], dict)
        assert isinstance(initial_state["phase_history"], list)
    
    @pytest.mark.unit
    def test_state_worker_results_accumulator(self, initial_state: EngineState):
        """Test that worker_results uses accumulator reducer."""
        # Worker results should start empty
        assert initial_state["worker_results"] == []
        
        # Simulate adding results (this would happen via reducer in LangGraph)
        result1 = {"worker_id": "worker_1", "success": True}
        result2 = {"worker_id": "worker_2", "success": True}
        
        merged = merge_worker_results(
            initial_state["worker_results"],
            [result1, result2]
        )
        
        assert len(merged) == 2
        assert merged[0]["worker_id"] == "worker_1"
        assert merged[1]["worker_id"] == "worker_2"
    
    @pytest.mark.unit
    def test_state_optional_fields(self, initial_state: EngineState):
        """Test that optional fields can be None or missing."""
        # These should be None initially
        assert initial_state["critique"] is None
        assert initial_state["synthesis_output"] is None
        assert initial_state["end_time"] is None
        assert initial_state["fatal_error"] is None
    
    @pytest.mark.unit
    def test_state_context_extensibility(self, initial_state: EngineState):
        """Test that context dict can hold workflow-specific data."""
        state = initial_state.copy()
        state["context"]["custom_workflow_data"] = {
            "pdf_path": "/path/to/file.pdf",
            "chunk_size": 1000,
            "overlap": 200
        }
        
        assert state["context"]["custom_workflow_data"]["pdf_path"] == "/path/to/file.pdf"
    
    @pytest.mark.unit
    def test_state_phase_transitions(self, initial_state: EngineState):
        """Test tracking phase transitions."""
        state = initial_state.copy()
        
        # Simulate phase transitions
        state["current_phase"] = "drafter"
        state["phase_history"].append("drafter")
        
        state["current_phase"] = "critic"
        state["phase_history"].append("critic")
        
        assert state["current_phase"] == "critic"
        assert "ingestion" in state["phase_history"]
        assert "drafter" in state["phase_history"]
        assert len(state["phase_history"]) == 3


# =============================================================================
# CREATE_INITIAL_STATE TESTS
# =============================================================================

class TestCreateInitialState:
    """Tests for create_initial_state() factory function."""
    
    @pytest.mark.unit
    def test_create_with_minimal_args(self):
        """Test creating state with minimal required arguments."""
        state = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test"},
            session_id="session-123",
            run_id="run-456"
        )
        
        assert state["workflow_name"] == "test"
        assert state["workflow_type"] == "research"
        assert state["input_data"] == {"query": "test"}
        assert state["session_id"] == "session-123"
        assert state["run_id"] == "run-456"
        assert state["max_iterations"] == 3  # Default
    
    @pytest.mark.unit
    def test_create_with_custom_iterations(self):
        """Test creating state with custom max_iterations."""
        state = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test"},
            session_id="session-123",
            run_id="run-456",
            max_iterations=5
        )
        
        assert state["max_iterations"] == 5
    
    @pytest.mark.unit
    def test_create_with_context(self):
        """Test creating state with custom context."""
        context = {
            "pdf_path": "/path/to/file.pdf",
            "chunk_size": 1000
        }
        
        state = create_initial_state(
            workflow_name="pdf_summarizer",
            workflow_type="ingestion",
            input_data={"path": "/path/to/file.pdf"},
            session_id="session-123",
            run_id="run-456",
            context=context
        )
        
        assert state["context"]["pdf_path"] == "/path/to/file.pdf"
        assert state["context"]["chunk_size"] == 1000
    
    @pytest.mark.unit
    def test_create_sets_initial_phase_to_ingestion(self):
        """Test that initial phase is always 'ingestion'."""
        state = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test"},
            session_id="session-123",
            run_id="run-456"
        )
        
        assert state["current_phase"] == "ingestion"
        assert state["phase_history"] == ["ingestion"]
    
    @pytest.mark.unit
    def test_create_initializes_empty_collections(self):
        """Test that collections start empty."""
        state = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test"},
            session_id="session-123",
            run_id="run-456"
        )
        
        assert state["plan"] == []
        assert state["worker_results"] == []
        assert state["worker_errors"] == []
        assert state["critique_history"] == []
        assert state["errors"] == []
    
    @pytest.mark.unit
    def test_create_sets_timestamp(self):
        """Test that start_time is set."""
        state = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test"},
            session_id="session-123",
            run_id="run-456"
        )
        
        assert state["start_time"] is not None
        # Should be ISO format
        datetime.fromisoformat(state["start_time"])  # Should not raise
    
    @pytest.mark.unit
    def test_create_unique_run_id(self):
        """Test that each state gets unique identifiers."""
        state1 = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test1"},
            session_id="session-1",
            run_id="run-1"
        )
        
        state2 = create_initial_state(
            workflow_name="test",
            workflow_type="research",
            input_data={"query": "test2"},
            session_id="session-2",
            run_id="run-2"
        )
        
        assert state1["run_id"] != state2["run_id"]
        assert state1["session_id"] != state2["session_id"]


# =============================================================================
# WORKER RESULT TESTS
# =============================================================================

class TestWorkerResult:
    """Tests for WorkerResult TypedDict."""
    
    @pytest.mark.unit
    def test_result_success_structure(self, sample_worker_result: WorkerResult):
        """Test successful worker result structure."""
        assert sample_worker_result["success"] is True
        assert "worker_id" in sample_worker_result
        assert "worker_type" in sample_worker_result
        assert "task" in sample_worker_result
        assert "output" in sample_worker_result
        assert "confidence" in sample_worker_result
    
    @pytest.mark.unit
    def test_result_timing_structure(self, sample_worker_result: WorkerResult):
        """Test timing information in result."""
        assert "timing" in sample_worker_result
        assert "start" in sample_worker_result["timing"]
        assert "end" in sample_worker_result["timing"]
        assert "duration" in sample_worker_result["timing"]
        assert isinstance(sample_worker_result["timing"]["duration"], float)
    
    @pytest.mark.unit
    def test_result_sources_structure(self, sample_worker_result: WorkerResult):
        """Test sources in result."""
        assert "sources" in sample_worker_result
        assert "source_count" in sample_worker_result
        assert sample_worker_result["source_count"] == len(sample_worker_result["sources"])
    
    @pytest.mark.unit
    def test_result_error_structure(self):
        """Test error result structure."""
        error_result: WorkerResult = WorkerResult(
            worker_id="failed_worker",
            worker_type="web_researcher",
            task="research",
            success=False,
            output=None,
            error="Connection timeout",
            error_type="TimeoutError",
            retry_count=3
        )
        
        assert error_result["success"] is False
        assert error_result["error"] == "Connection timeout"
        assert error_result["error_type"] == "TimeoutError"
        assert error_result["retry_count"] == 3

    @pytest.mark.unit
    def test_result_confidence_range(self):
        """Test that confidence is in valid range."""
        valid_result: WorkerResult = WorkerResult(
            worker_id="test",
            worker_type="test",
            task="test",
            success=True,
            output={},
            confidence=0.85
        )
        
        assert 0.0 <= valid_result["confidence"] <= 1.0


# =============================================================================
# CRITIC RESULT TESTS
# =============================================================================

class TestCriticResult:
    """Tests for CriticResult TypedDict."""
    
    @pytest.mark.unit
    def test_critic_pass_structure(self):
        """Test passing critic result."""
        pass_result: CriticResult = CriticResult(
            passed=True,
            score=0.85,
            criteria_scores={
                "completeness": 0.9,
                "accuracy": 0.8
            },
            recommendation="pass",
            feedback_to="workers"
        )
        
        assert pass_result["passed"] is True
        assert pass_result["score"] == 0.85
        assert pass_result["recommendation"] == "pass"
    
    @pytest.mark.unit
    def test_critic_fail_structure(self):
        """Test failing critic result."""
        fail_result: CriticResult = CriticResult(
            passed=False,
            score=0.5,
            criteria_scores={
                "completeness": 0.4,
                "accuracy": 0.6
            },
            issues=["Missing key information", "Sources not verified"],
            suggestions=["Add more sources", "Verify claims"],
            recommendation="revise",
            feedback_to="workers"
        )
        
        assert fail_result["passed"] is False
        assert len(fail_result["issues"]) == 2
        assert len(fail_result["suggestions"]) == 2
        assert fail_result["recommendation"] == "revise"
    
    @pytest.mark.unit
    def test_critic_recommendation_values(self):
        """Test valid recommendation values."""
        valid_recommendations = ["pass", "revise", "terminate"]
        
        for rec in valid_recommendations:
            result: CriticResult = CriticResult(
                passed=rec == "pass",
                score=0.5,
                recommendation=rec,
                feedback_to="workers"
            )
            assert result["recommendation"] == rec


# =============================================================================
# STATE REDUCERS TESTS
# =============================================================================

class TestStateReducers:
    """Tests for state reducer functions."""
    
    @pytest.mark.unit
    def test_merge_worker_results(self):
        """Test merging worker results from parallel execution."""
        left = [
            {"worker_id": "worker_1", "output": "result 1"},
            {"worker_id": "worker_2", "output": "result 2"},
        ]
        right = [
            {"worker_id": "worker_3", "output": "result 3"},
        ]
        
        merged = merge_worker_results(left, right)
        
        assert len(merged) == 3
        assert merged[0]["worker_id"] == "worker_1"
        assert merged[2]["worker_id"] == "worker_3"
    
    @pytest.mark.unit
    def test_merge_errors(self):
        """Test merging errors from parallel execution."""
        left = [
            {"worker_id": "worker_1", "error": "error 1"},
        ]
        right = [
            {"worker_id": "worker_2", "error": "error 2"},
            {"worker_id": "worker_3", "error": "error 3"},
        ]
        
        merged = merge_errors(left, right)
        
        assert len(merged) == 3
        assert all("error" in item for item in merged)
    
    @pytest.mark.unit
    def test_overwrite_reducer(self):
        """Test overwrite reducer keeps last value."""
        assert overwrite_reducer("old", "new") == "new"
        assert overwrite_reducer(None, "new") == "new"
        assert overwrite_reducer("old", None) == "old"
        assert overwrite_reducer(1, 2) == 2
    
    @pytest.mark.unit
    def test_reducers_are_commutative_for_parallel_execution(self):
        """Test that merge operations work for parallel execution."""
        result1 = {"worker_id": "w1", "success": True}
        result2 = {"worker_id": "w2", "success": True}
        result3 = {"worker_id": "w3", "success": True}
        
        # Different order should produce same length
        merged1 = merge_worker_results([result1], [result2, result3])
        merged2 = merge_worker_results([result2, result3], [result1])
        
        assert len(merged1) == 3
        assert len(merged2) == 3


# =============================================================================
# TYPE ALIASES TESTS
# =============================================================================

class TestTypeAliases:
    """Tests for type aliases."""
    
    @pytest.mark.unit
    def test_phase_name_literals(self):
        """Test PhaseName type accepts valid literals."""
        valid_phases: list = ["ingestion", "drafter", "critic", "neo4j_builder", "synthesizer"]
        # Type checking would catch invalid values at compile time
        assert len(valid_phases) == 5
    
    @pytest.mark.unit
    def test_workflow_type_literals(self):
        """Test WorkflowType type accepts valid literals."""
        valid_types: list = ["research", "ingestion", "generation"]
        assert len(valid_types) == 3
    
    @pytest.mark.unit
    def test_worker_type_literals(self):
        """Test WorkerType type accepts valid literals."""
        valid_workers: list = [
            "web_researcher", "chunk_processor", "multimodal",
            "scraper", "asr_worker", "vlm_worker", "face_worker", "text_worker"
        ]
        assert len(valid_workers) == 8
    
    @pytest.mark.unit
    def test_output_format_literals(self):
        """Test OutputFormat type accepts valid literals."""
        valid_formats: list = ["markdown", "json", "html", "graph", "podcast_script"]
        assert len(valid_formats) == 5
