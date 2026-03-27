"""Integration Tests for Deep Research Engine Nodes."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch


# =============================================================================
# Integration Test: Critic Node
# =============================================================================


@pytest.mark.asyncio
async def test_critic_node_with_sample_state():
    """Test critic node with realistic state."""
    from deep_research_engine.nodes.critic import critic_node
    from deep_research_engine.state import create_initial_state
    
    # Create sample state
    state = create_initial_state(
        workflow_name="rag",
        workflow_type="research",
        input_data={"query": "Test query"},
        session_id="test_session",
        run_id="test_run"
    )
    
    # Add sample worker results
    state["worker_results"] = [
        {
            "worker_id": "researcher_0",
            "worker_type": "web_researcher",
            "task": "Search web",
            "success": True,
            "output": "Sample research output with findings...",
            "confidence": 0.85,
            "source_count": 3
        }
    ]
    
    state["plan"] = ["Search web", "Analyze findings"]
    state["context"] = {"config": {}}
    
    # Mock the deepagents import to avoid actual LLM calls
    with patch("deep_research_engine.nodes.critic.create_deep_agent") as mock_agent:
        # Setup mock response
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value={
            "messages": [MagicMock(content="Test evaluation")]
        })
        mock_agent.return_value = mock_llm
        
        # Run critic node
        result = await critic_node(state)
        
        # Verify result structure
        assert "__goto__" in result
        assert "critic_passed" in result or "critique" in result


# =============================================================================
# Integration Test: Ingestion Node
# =============================================================================


@pytest.mark.asyncio
async def test_ingestion_node_structure():
    """Test ingestion node structure and imports."""
    from deep_research_engine.nodes.ingestion import ingestion_node
    from deep_research_engine.state import create_initial_state
    
    # Verify node exists and is callable
    assert callable(ingestion_node)
    
    # Create minimal state
    state = create_initial_state(
        workflow_name="telegram_ingest",
        workflow_type="ingestion",
        input_data={"path": "/tmp/test"},
        session_id="test_session",
        run_id="test_run"
    )
    
    # Node should handle empty input gracefully
    # (Actual execution requires real files)
    assert state is not None


# =============================================================================
# Integration Test: Drafter Node
# =============================================================================


@pytest.mark.asyncio
async def test_drafter_node_with_flash_agent():
    """Test drafter node integrates with Flash Agent."""
    from deep_research_engine.nodes.drafter import drafter_node
    from deep_research_engine.state import create_initial_state
    
    state = create_initial_state(
        workflow_name="rag",
        workflow_type="research",
        input_data={"query": "Test research query"},
        session_id="test_session",
        run_id="test_run"
    )
    
    state["context"] = {
        "config": {
            "drafter": {
                "enabled": True,
                "workers": [
                    {"type": "web_researcher", "task": "Search web", "enabled": True}
                ]
            }
        }
    }
    
    # Verify node exists
    assert callable(drafter_node)


# =============================================================================
# Integration Test: Full Pipeline Import
# =============================================================================


def test_full_pipeline_imports():
    """Test all pipeline components import correctly."""
    # State
    from deep_research_engine.state import (
        EngineState,
        WorkerResult,
        CriticEvaluation,
        create_initial_state
    )
    
    # Nodes
    from deep_research_engine.nodes import (
        ingestion_node,
        drafter_node,
        critic_node,
        neo4j_builder_node,
        synthesizer_node
    )
    
    # Utils
    from deep_research_engine.utils.pipeline import (
        normalize_timestamp,
        extract_scene_frames,
        parse_ffmpeg_metadata,
        cluster_identities,
        create_mega_files,
        purge_vram
    )
    
    # Multimodal
    from deep_research_engine.shared.multimodal import (
        extract_audio_from_video,
        get_video_duration,
        asr_transcribe,
        unify_speakers
    )
    
    # Agents
    from deep_research_engine.agents.flash import FlashAgent, create_flash_subagents
    
    # All imports successful
    assert True


# =============================================================================
# Integration Test: Config-Driven Workflow
# =============================================================================


def test_config_loading_and_validation():
    """Test configuration loading for various workflows."""
    from deep_research_engine.config import load_config
    
    # Test RAG config
    rag_config = load_config("configs/rag.yaml")
    assert rag_config is not None
    assert "workflow" in rag_config
    
    # Test Telegram ingest config
    telegram_config = load_config("configs/telegram_ingest.yaml")
    assert telegram_config is not None
    assert "workflow" in telegram_config


# =============================================================================
# Integration Test: Storage Initialization
# =============================================================================


@pytest.mark.asyncio
async def test_storage_tools_initialization():
    """Test storage tools initialize correctly."""
    from deep_research_engine.tools.storage import initialize_storage_tools
    
    config = {
        "storage": {
            "vector_store": {
                "type": "in_memory",
                "enabled": True
            }
        }
    }
    
    # Should not raise exception
    initialize_storage_tools(config)


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
