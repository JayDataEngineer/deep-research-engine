"""Tests for Critic Node logic."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from deep_research_engine.nodes.critic import critic_node, CriticEvaluation
from deep_research_engine.state import EngineState

@pytest.mark.unit
@pytest.mark.asyncio
async def test_critic_node_integration_mocked(state_with_results: EngineState):
    """Test critic_node with mocked LLM and Prompt."""
    state = state_with_results.copy()
    # Ensure there's a draft in worker_results
    state["worker_results"][-1]["output"] = "This is a high quality research draft."
    state["context"] = {
        "config": {
            "critic": {
                "enabled": True,
                "pass_threshold": 0.8,
                "max_retries": 3
            }
        }
    }
    state["iteration_count"] = 0

    # Mock PromptWrapper
    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = "Compiled prompt"

    # Mock LLM and structured output
    mock_evaluation = CriticEvaluation(
        criteria_scores={"completeness": 0.9, "accuracy": 0.9},
        overall_score=0.9,
        issues=[],
        suggestions=[]
    )

    mock_structured_llm = AsyncMock()
    mock_structured_llm.ainvoke.return_value = mock_evaluation

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("deep_research_engine.nodes.critic.get_prompt", AsyncMock(return_value=mock_prompt)), \
         patch("deep_research_engine.nodes.critic.get_chat_llm", AsyncMock(return_value=mock_llm)):

        command = await critic_node(state)

        # Verify
        assert command.update["critic_passed"] is True
        assert command.goto == "neo4j_builder"
        assert command.update["critique"]["score"] == 0.9
        assert command.update["iteration_count"] == 1
        assert len(command.update["critique_history"]) == 1

@pytest.mark.unit
@pytest.mark.asyncio
async def test_critic_node_fail_and_retry(state_with_results: EngineState):
    """Test critic_node fails and routes back to drafter."""
    state = state_with_results.copy()
    state["worker_results"][-1]["output"] = "Poor draft."
    state["context"] = {
        "config": {
            "critic": {
                "enabled": True,
                "pass_threshold": 0.8,
                "max_retries": 3
            }
        }
    }
    state["iteration_count"] = 0

    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = "Compiled prompt"

    mock_evaluation = CriticEvaluation(
        criteria_scores={"completeness": 0.5, "accuracy": 0.5},
        overall_score=0.5,
        issues=["Too short"],
        suggestions=["Add more detail"]
    )

    mock_structured_llm = AsyncMock()
    mock_structured_llm.ainvoke.return_value = mock_evaluation

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("deep_research_engine.nodes.critic.get_prompt", AsyncMock(return_value=mock_prompt)), \
         patch("deep_research_engine.nodes.critic.get_chat_llm", AsyncMock(return_value=mock_llm)):

        command = await critic_node(state)

        # Verify
        assert command.update["critic_passed"] is False
        assert command.goto == "drafter"
        assert command.update["critique"]["score"] == 0.5
        assert "Too short" in command.update["critique"]["issues"]

@pytest.mark.unit
@pytest.mark.asyncio
async def test_critic_node_max_retries(state_with_results: EngineState):
    """Test critic_node passes after max retries even if score is low."""
    state = state_with_results.copy()
    state["worker_results"][-1]["output"] = "Still poor draft."
    state["context"] = {
        "config": {
            "critic": {
                "enabled": True,
                "pass_threshold": 0.8,
                "max_retries": 3
            }
        }
    }
    state["iteration_count"] = 3 # Already at max retries

    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = "Compiled prompt"

    mock_evaluation = CriticEvaluation(
        criteria_scores={"completeness": 0.5, "accuracy": 0.5},
        overall_score=0.5,
        issues=["Still too short"],
        suggestions=[]
    )

    mock_structured_llm = AsyncMock()
    mock_structured_llm.ainvoke.return_value = mock_evaluation

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("deep_research_engine.nodes.critic.get_prompt", AsyncMock(return_value=mock_prompt)), \
         patch("deep_research_engine.nodes.critic.get_chat_llm", AsyncMock(return_value=mock_llm)):

        command = await critic_node(state)

        # Verify
        assert command.update["critic_passed"] is True # Forced pass due to max retries
        assert command.goto == "neo4j_builder"
        assert command.update["iteration_count"] == 4
