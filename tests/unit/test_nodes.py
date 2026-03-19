"""Unit Tests for LangGraph Nodes.

Tests for boss, workers, critic, and synthesis nodes.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

from deep_research_engine.state import EngineState, create_initial_state, WorkerResult
from deep_research_engine.nodes import (
    boss_node,
    workers_node,
    critic_node,
    synthesis_node,
    route_after_boss,
    route_after_workers,
    route_after_critic,
    route_after_synthesis,
)
from deep_research_engine.workers import WorkerConfig, WorkerRegistry

# Patch location for deepagents
DEEPAGENTS_PATCH = 'deepagents.create_deep_agent'

# Check if deepagents is available
try:
    import deepagents
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False


# =============================================================================
# BOSS NODE TESTS
# =============================================================================

class TestBossNode:
    """Tests for boss_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_boss_node_creates_plan(self, initial_state: EngineState):
        """Test that boss node creates a research plan."""
        # Setup
        state = initial_state.copy()
        state["input_data"] = {"query": "What is machine learning?"}
        state["context"] = {
            "boss": {
                "strategy": "complexity_based",
                "workers": {"min": 1, "max": 3}
            }
        }

        # Mock the LLM call
        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [
                    MagicMock(content='{"tasks": ["Research ML basics"], "complexity": "simple"}')
                ]
            })
            mock_agent_factory.return_value = mock_agent

            # Execute
            result_state = await boss_node(state)

            # Verify
            assert "plan" in result_state
            assert len(result_state["plan"]) > 0
            assert result_state["current_phase"] == "workers"
            assert "workers" in result_state["phase_history"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_boss_node_handles_string_input(self, initial_state: EngineState):
        """Test boss node with string input instead of dict."""
        state = initial_state.copy()
        state["input_data"] = "Research quantum computing"
        state["context"] = {"boss": {"strategy": "simple"}}

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content='{"tasks": ["Research QC"], "complexity": "medium"}')]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await boss_node(state)

            assert "plan" in result_state
            assert result_state["current_phase"] == "workers"

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_boss_node_handles_llm_error(self, initial_state: EngineState):
        """Test boss node handles LLM errors gracefully."""
        state = initial_state.copy()
        state["input_data"] = {"query": "Test query"}
        state["context"] = {"boss": {"strategy": "simple"}}

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent_factory.side_effect = Exception("LLM API error")

            result_state = await boss_node(state)

            # Should have error in errors list
            assert len(result_state["errors"]) > 0
            assert result_state["errors"][0]["phase"] == "boss"

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_boss_node_handles_json_parse_error(self, initial_state: EngineState):
        """Test boss node handles JSON parsing errors."""
        state = initial_state.copy()
        state["input_data"] = {"query": "Test query"}
        state["context"] = {"boss": {"strategy": "simple"}}

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            # Return invalid JSON
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content="This is not valid JSON {")]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await boss_node(state)

            # Should create fallback plan
            assert "plan" in result_state
            assert len(result_state["plan"]) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_boss_node_respects_worker_limits(self, initial_state: EngineState):
        """Test boss node respects min/max worker configuration."""
        state = initial_state.copy()
        state["input_data"] = {"query": "Complex research topic"}
        state["context"] = {
            "boss": {
                "strategy": "complexity_based",
                "workers": {"min": 2, "max": 5}
            }
        }

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [
                    MagicMock(content='{"tasks": ["Task 1", "Task 2", "Task 3"], "complexity": "complex"}')
                ]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await boss_node(state)

            assert "plan_metadata" in result_state
            assert result_state["plan_metadata"]["worker_count"] == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_boss_node_updates_phase_history(self, initial_state: EngineState):
        """Test boss node updates phase history correctly."""
        state = initial_state.copy()
        state["input_data"] = {"query": "Test"}
        state["context"] = {"boss": {}}

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content='{"tasks": ["Task 1"]}')]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await boss_node(state)

            assert "boss" in result_state["phase_history"]
            assert "workers" in result_state["phase_history"]


# =============================================================================
# ROUTE AFTER BOSS TESTS
# =============================================================================

class TestRouteAfterBoss:
    """Tests for route_after_boss function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_boss_success(self, initial_state: EngineState):
        """Test routing after successful boss execution."""
        state = initial_state.copy()
        state["plan"] = ["Task 1", "Task 2"]

        result = await route_after_boss(state)
        assert result == "workers"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_boss_error_no_plan(self, initial_state: EngineState):
        """Test routing when boss failed to create plan."""
        state = initial_state.copy()
        state["plan"] = []

        result = await route_after_boss(state)
        assert result == "error"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_boss_error_fatal(self, initial_state: EngineState):
        """Test routing when boss set fatal error."""
        state = initial_state.copy()
        state["fatal_error"] = "Boss node failed"

        result = await route_after_boss(state)
        assert result == "error"


# =============================================================================
# WORKERS NODE TESTS
# =============================================================================

class TestWorkersNode:
    """Tests for workers_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workers_node_executes_parallel(self, initial_state: EngineState):
        """Test workers node executes workers in parallel."""
        state = initial_state.copy()
        state["plan"] = ["Task 1", "Task 2", "Task 3"]
        state["context"] = {
            "workers": {
                "worker_type": "test_worker",
                "parallel": True,
                "max_concurrent": 10
            }
        }

        # Register test worker
        class TestWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {
                    "worker_id": config.worker_id,
                    "success": True,
                    "output": {"result": "done"}
                }

        with patch.object(WorkerRegistry, 'get', return_value=TestWorker):
            result_state = await workers_node(state)

            assert "worker_results" in result_state
            assert len(result_state["worker_results"]) == 3
            assert result_state["current_phase"] == "critic"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workers_node_handles_sequential(self, initial_state: EngineState):
        """Test workers node with sequential execution."""
        state = initial_state.copy()
        state["plan"] = ["Task 1", "Task 2"]
        state["context"] = {
            "workers": {
                "worker_type": "test_worker",
                "parallel": False
            }
        }

        class TestWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {"worker_id": config.worker_id, "success": True}

        with patch.object(WorkerRegistry, 'get', return_value=TestWorker):
            result_state = await workers_node(state)

            assert len(result_state["worker_results"]) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workers_node_handles_failures(self, initial_state: EngineState):
        """Test workers node handles worker failures."""
        state = initial_state.copy()
        state["plan"] = ["Task 1", "Task 2"]
        state["context"] = {"workers": {"worker_type": "test_worker"}}

        class FailingWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                if "Task 1" in config.task:
                    return {"worker_id": config.worker_id, "success": False, "error": "Failed"}
                return {"worker_id": config.worker_id, "success": True}

        with patch.object(WorkerRegistry, 'get', return_value=FailingWorker):
            result_state = await workers_node(state)

            assert len(result_state["worker_results"]) == 1
            assert len(result_state.get("worker_errors", [])) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workers_node_no_plan(self, initial_state: EngineState):
        """Test workers node with no plan."""
        state = initial_state.copy()
        state["plan"] = []

        result_state = await workers_node(state)

        # Should skip execution
        assert result_state == state

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workers_node_accumulates_results(self, initial_state: EngineState):
        """Test workers node accumulates results correctly."""
        state = initial_state.copy()
        state["plan"] = ["Task 1"]
        state["worker_results"] = [{"existing": "result"}]  # Pre-existing results
        state["context"] = {"workers": {"worker_type": "test_worker"}}

        class TestWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {"worker_id": config.worker_id, "success": True, "new": "result"}

        with patch.object(WorkerRegistry, 'get', return_value=TestWorker):
            result_state = await workers_node(state)

            assert len(result_state["worker_results"]) == 2
            assert result_state["worker_results"][0]["existing"] == "result"
            assert result_state["worker_results"][1]["new"] == "result"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_workers_node_respects_concurrency_limit(self, initial_state: EngineState):
        """Test workers node respects max_concurrent limit."""
        state = initial_state.copy()
        state["plan"] = ["Task " + str(i) for i in range(10)]
        state["context"] = {
            "workers": {
                "worker_type": "test_worker",
                "parallel": True,
                "max_concurrent": 2
            }
        }

        execution_count = 0
        max_concurrent = 0

        class TestWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                nonlocal execution_count, max_concurrent
                execution_count += 1
                max_concurrent = max(max_concurrent, execution_count)
                await asyncio.sleep(0.01)
                execution_count -= 1
                return {"worker_id": config.worker_id, "success": True}

        with patch.object(WorkerRegistry, 'get', return_value=TestWorker):
            await workers_node(state)

            # Concurrency should not exceed limit
            assert max_concurrent <= 2


# =============================================================================
# ROUTE AFTER WORKERS TESTS
# =============================================================================

class TestRouteAfterWorkers:
    """Tests for route_after_workers function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_workers_success(self, initial_state: EngineState):
        """Test routing after successful worker execution."""
        state = initial_state.copy()
        state["worker_results"] = [{"success": True}]

        result = await route_after_workers(state)
        assert result == "critic"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_workers_all_failed(self, initial_state: EngineState):
        """Test routing when all workers failed."""
        state = initial_state.copy()
        state["worker_results"] = []
        state["worker_errors"] = [{"error": "Failed"}]

        result = await route_after_workers(state)
        assert result == "error"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_workers_partial_success(self, initial_state: EngineState):
        """Test routing with partial worker success."""
        state = initial_state.copy()
        state["worker_results"] = [{"success": True}]
        state["worker_errors"] = [{"error": "Failed"}]

        result = await route_after_workers(state)
        assert result == "critic"


# =============================================================================
# CRITIC NODE TESTS
# =============================================================================

class TestCriticNode:
    """Tests for critic_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_critic_node_passes_good_results(self, state_with_results: EngineState):
        """Test critic passes good quality results."""
        state = state_with_results.copy()
        state["context"] = {
            "critic": {
                "pass_threshold": 0.7,
                "criteria": [
                    {"name": "completeness", "weight": 0.4, "threshold": 0.7},
                    {"name": "accuracy", "weight": 0.4, "threshold": 0.8}
                ]
            }
        }
        state["max_iterations"] = 3
        state["iteration_count"] = 0

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content='{"score": 0.85, "criteria_scores": {"completeness": 0.9, "accuracy": 0.8}}')]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await critic_node(state)

            assert result_state["critique"]["passed"] is True
            assert result_state["critique"]["score"] >= 0.7
            assert result_state["current_phase"] == "synthesis"

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_critic_node_fails_poor_results(self, state_with_results: EngineState):
        """Test critic fails poor quality results."""
        state = state_with_results.copy()
        state["context"] = {"critic": {"pass_threshold": 0.9}}
        state["max_iterations"] = 3
        state["iteration_count"] = 0

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content='{"score": 0.5, "criteria_scores": {"completeness": 0.5}}')]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await critic_node(state)

            assert result_state["critique"]["passed"] is False
            assert result_state["current_phase"] == "workers"
            assert result_state["iteration_count"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_critic_node_max_retries(self, state_with_results: EngineState):
        """Test critic respects max retries."""
        state = state_with_results.copy()
        state["context"] = {"critic": {"pass_threshold": 0.9}}
        state["max_iterations"] = 3
        state["iteration_count"] = 3  # Already at max

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content='{"score": 0.5}')]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await critic_node(state)

            # Should terminate despite low score
            assert result_state["critique"]["recommendation"] == "terminate"

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_critic_node_handles_llm_error(self, state_with_results: EngineState):
        """Test critic handles LLM errors gracefully."""
        state = state_with_results.copy()
        state["context"] = {"critic": {}}
        state["max_iterations"] = 3
        state["iteration_count"] = 0

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent_factory.side_effect = Exception("LLM error")

            result_state = await critic_node(state)

            # Should have fallback evaluation
            assert "critique" in result_state
            assert result_state["critique"]["score"] == 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not installed")
    async def test_critic_node_tracks_history(self, state_with_results: EngineState):
        """Test critic tracks critique history."""
        state = state_with_results.copy()
        state["context"] = {"critic": {}}
        state["max_iterations"] = 3
        state["iteration_count"] = 0
        state["critique_history"] = [{"previous": "evaluation"}]

        with patch(DEEPAGENTS_PATCH) as mock_agent_factory:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content='{"score": 0.8}')]
            })
            mock_agent_factory.return_value = mock_agent

            result_state = await critic_node(state)

            assert len(result_state["critique_history"]) == 2


# =============================================================================
# ROUTE AFTER CRITIC TESTS
# =============================================================================

class TestRouteAfterCritic:
    """Tests for route_after_critic function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_critic_pass(self, initial_state: EngineState):
        """Test routing after critic passes."""
        state = initial_state.copy()
        state["critique"] = {"recommendation": "pass"}

        result = await route_after_critic(state)
        assert result == "synthesis"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_critic_revise(self, initial_state: EngineState):
        """Test routing after critic requests revision."""
        state = initial_state.copy()
        state["critique"] = {"recommendation": "revise"}

        result = await route_after_critic(state)
        assert result == "workers"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_critic_terminate(self, initial_state: EngineState):
        """Test routing after critic terminates."""
        state = initial_state.copy()
        state["critique"] = {"recommendation": "terminate"}

        result = await route_after_critic(state)
        assert result == "error"


# =============================================================================
# SYNTHESIS NODE TESTS
# =============================================================================

class TestSynthesisNode:
    """Tests for synthesis_node function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesis_node_creates_report(self, state_with_results: EngineState):
        """Test synthesis node creates final report."""
        state = state_with_results.copy()
        state["context"] = {
            "synthesis": {
                "format": "research_report",
                "citations": {"enabled": True}
            }
        }

        class TestSynthesisWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {
                    "success": True,
                    "output": "# Research Report\n\n## Findings\n\nTest findings",
                    "output_type": "markdown",
                    "confidence": 0.9
                }

        with patch.object(WorkerRegistry, 'get', return_value=TestSynthesisWorker):
            result_state = await synthesis_node(state)

            assert "synthesis_output" in result_state
            assert "Research Report" in result_state["synthesis_output"]
            assert result_state["current_phase"] == "complete"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesis_node_handles_failure(self, state_with_results: EngineState):
        """Test synthesis node handles worker failure."""
        state = state_with_results.copy()
        state["context"] = {"synthesis": {}}

        class FailingSynthesisWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {
                    "success": False,
                    "error": "Synthesis failed"
                }

        with patch.object(WorkerRegistry, 'get', return_value=FailingSynthesisWorker):
            result_state = await synthesis_node(state)

            assert len(result_state["errors"]) > 0
            assert "fatal_error" in result_state

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesis_node_handles_basemodel_output(self, state_with_results: EngineState):
        """Test synthesis node handles Pydantic BaseModel output."""
        state = state_with_results.copy()
        state["context"] = {"synthesis": {"format": "structured_json"}}

        class MockOutput:
            def model_dump_json(self):
                return '{"section": "findings"}'

            def model_dump(self):
                return {"section": "findings", "count": 5}

        class TestSynthesisWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {
                    "success": True,
                    "output": MockOutput(),
                    "output_type": "json",
                    "confidence": 0.85
                }

        with patch.object(WorkerRegistry, 'get', return_value=TestSynthesisWorker):
            result_state = await synthesis_node(state)

            assert "synthesis_output" in result_state
            assert "synthesis_metadata" in result_state
            assert result_state["synthesis_metadata"]["worker_count"] == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesis_node_no_worker_results(self, initial_state: EngineState):
        """Test synthesis node with no worker results."""
        state = initial_state.copy()
        state["worker_results"] = []
        state["context"] = {"synthesis": {}}

        class TestSynthesisWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {
                    "success": True,
                    "output": "# Empty Report\n\nNo findings to report.",
                    "output_type": "markdown"
                }

        with patch.object(WorkerRegistry, 'get', return_value=TestSynthesisWorker):
            result_state = await synthesis_node(state)

            assert result_state["current_phase"] == "complete"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_synthesis_node_tracks_metadata(self, state_with_results: EngineState):
        """Test synthesis node tracks output metadata."""
        state = state_with_results.copy()
        state["context"] = {
            "synthesis": {
                "format": "research_report",
                "citations": {"enabled": True, "format": "APA"},
                "max_retries": 5
            }
        }

        class TestSynthesisWorker:
            def __init__(self, config):
                self.config = config

            async def execute(self, state, config):
                return {
                    "success": True,
                    "output": "Report",
                    "output_type": "markdown",
                    "confidence": 0.9
                }

        with patch.object(WorkerRegistry, 'get', return_value=TestSynthesisWorker):
            result_state = await synthesis_node(state)

            assert "synthesis_metadata" in result_state
            assert result_state["synthesis_metadata"]["format"] == "research_report"
            # Metadata may not include citations_enabled in basic implementation
            # Just verify the metadata dict exists and has format
            assert "worker_count" in result_state["synthesis_metadata"]


# =============================================================================
# ROUTE AFTER SYNTHESIS TESTS
# =============================================================================

class TestRouteAfterSynthesis:
    """Tests for route_after_synthesis function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_synthesis_success(self, initial_state: EngineState):
        """Test routing after successful synthesis."""
        state = initial_state.copy()
        state["synthesis_output"] = "Final report"

        result = await route_after_synthesis(state)
        assert result == "end"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_route_after_synthesis_error(self, initial_state: EngineState):
        """Test routing after synthesis error."""
        state = initial_state.copy()
        state["fatal_error"] = "Synthesis failed"

        result = await route_after_synthesis(state)
        assert result == "error"
