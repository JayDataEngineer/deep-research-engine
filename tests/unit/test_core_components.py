"""Unit Tests for Deep Research Engine Core Components."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test: Critic Node
# =============================================================================


class TestCriticNode:
    """Tests for the critic node evaluation logic."""

    @pytest.mark.asyncio
    async def test_critic_evaluation_schema(self):
        """Test CriticEvaluation schema creation."""
        from deep_research_engine.state import CriticEvaluation
        
        evaluation = CriticEvaluation(
            passed=True,
            score=0.85,
            completeness=0.9,
            accuracy=0.8,
            citation_quality=0.85,
            coherence=0.85,
            issues=["Minor formatting issue"],
            suggestions=["Add more citations"],
            recommendation="pass",
            feedback_to="artifact_maker"
        )
        
        assert evaluation.passed is True
        assert evaluation.score == 0.85
        assert len(evaluation.issues) == 1
        assert evaluation.recommendation == "pass"

    @pytest.mark.asyncio
    async def test_critic_node_import(self):
        """Test critic node imports correctly."""
        from deep_research_engine.nodes.critic import critic_node
        assert critic_node is not None


# =============================================================================
# Test: Video Audio Extraction
# =============================================================================


class TestVideoAudioExtraction:
    """Tests for video audio extraction functions."""

    @pytest.mark.asyncio
    async def test_extract_audio_from_video_import(self):
        """Test audio extraction function imports."""
        from deep_research_engine.shared.multimodal import (
            extract_audio_from_video,
            get_video_duration
        )
        assert extract_audio_from_video is not None
        assert get_video_duration is not None

    @pytest.mark.asyncio
    async def test_get_video_duration_nonexistent(self):
        """Test duration extraction for nonexistent file."""
        from deep_research_engine.shared.multimodal import get_video_duration
        
        duration = await get_video_duration(Path("/nonexistent/video.mp4"))
        assert duration == 0.0


# =============================================================================
# Test: FFmpeg Metadata Parsing
# =============================================================================


class TestFFmpegMetadata:
    """Tests for FFmpeg metadata parsing."""

    @pytest.mark.asyncio
    async def test_parse_ffmpeg_metadata_import(self):
        """Test metadata parsing function imports."""
        from deep_research_engine.utils.pipeline import (
            parse_ffmpeg_metadata,
            extract_scene_frames
        )
        assert parse_ffmpeg_metadata is not None
        assert extract_scene_frames is not None

    @pytest.mark.asyncio
    async def test_parse_ffmpeg_metadata_nonexistent(self):
        """Test parsing nonexistent metadata file."""
        from deep_research_engine.utils.pipeline import parse_ffmpeg_metadata
        
        frames = await parse_ffmpeg_metadata(Path("/nonexistent/time.txt"))
        assert len(frames) == 0


# =============================================================================
# Test: State Types
# =============================================================================


class TestStateTypes:
    """Tests for state type definitions."""

    def test_engine_state_import(self):
        """Test EngineState imports."""
        from deep_research_engine.state import EngineState
        assert EngineState is not None

    def test_worker_result_import(self):
        """Test WorkerResult imports."""
        from deep_research_engine.state import WorkerResult
        assert WorkerResult is not None

    def test_critic_evaluation_import(self):
        """Test CriticEvaluation imports."""
        from deep_research_engine.state import CriticEvaluation
        assert CriticEvaluation is not None

    def test_create_initial_state(self):
        """Test state factory function."""
        from deep_research_engine.state import create_initial_state
        
        state = create_initial_state(
            workflow_name="test_workflow",
            workflow_type="research",
            input_data={"query": "test query"},
            session_id="test_session",
            run_id="test_run"
        )
        
        assert state["workflow_name"] == "test_workflow"
        assert state["workflow_type"] == "research"
        assert state["input_data"]["query"] == "test query"
        assert state["critic_passed"] is False


# =============================================================================
# Test: Flash Agent
# =============================================================================


class TestFlashAgent:
    """Tests for Flash Agent implementation."""

    def test_flash_agent_import(self):
        """Test FlashAgent imports."""
        from deep_research_engine.agents.flash import FlashAgent
        assert FlashAgent is not None

    def test_flash_agent_creation(self):
        """Test FlashAgent instantiation."""
        from deep_research_engine.agents.flash import FlashAgent
        
        agent = FlashAgent(
            name="test_agent",
            task="Test research task",
            model="openai:gpt-4o-mini",
            max_iterations=2
        )
        
        assert agent.name == "test_agent"
        assert agent.task == "Test research task"
        assert agent.max_iterations == 2

    def test_create_flash_subagents(self):
        """Test flash subagent creation."""
        from deep_research_engine.agents.flash import create_flash_subagents
        
        configs = [
            {"type": "web_researcher", "task": "Search web", "enabled": True},
            {"type": "chunk_processor", "task": "Process PDFs", "enabled": True}
        ]
        
        import asyncio
        subagents = asyncio.run(create_flash_subagents(configs))
        
        assert len(subagents) == 2
        assert subagents[0]["name"] == "web_researcher_specialist"
        assert subagents[1]["name"] == "chunk_processor_specialist"


# =============================================================================
# Test: Pipeline Utilities
# =============================================================================


class TestPipelineUtils:
    """Tests for pipeline utility functions."""

    def test_purge_vram_import(self):
        """Test purge_vram imports."""
        from deep_research_engine.utils.pipeline import purge_vram
        assert purge_vram is not None

    def test_cluster_identities_empty(self):
        """Test clustering with empty embeddings."""
        import numpy as np
        from deep_research_engine.utils.pipeline import cluster_identities
        
        labels = cluster_identities(np.array([]))
        assert len(labels) == 0


# =============================================================================
# Test: Ingestion Node
# =============================================================================


class TestIngestionNode:
    """Tests for ingestion node."""

    @pytest.mark.asyncio
    async def test_ingestion_node_import(self):
        """Test ingestion node imports."""
        from deep_research_engine.nodes.ingestion import ingestion_node
        assert ingestion_node is not None


# =============================================================================
# Test: Config Loading
# =============================================================================


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_config_function_exists(self):
        """Test load_config function exists."""
        from deep_research_engine.config import load_config
        assert load_config is not None


# =============================================================================
# Test: Worker Registry
# =============================================================================


class TestWorkerRegistry:
    """Tests for worker registry."""

    def test_worker_registry_import(self):
        """Test WorkerRegistry imports."""
        from deep_research_engine.workers.base import WorkerRegistry
        assert WorkerRegistry is not None


# =============================================================================
# Integration Test Helpers
# =============================================================================


@pytest.fixture
def sample_worker_results():
    """Sample worker results for testing."""
    return [
        {
            "worker_id": "researcher_0",
            "worker_type": "web_researcher",
            "task": "Search for information",
            "success": True,
            "output": "Research findings...",
            "confidence": 0.85,
            "source_count": 5
        }
    ]


@pytest.fixture
def sample_plan():
    """Sample research plan."""
    return [
        "Search web for primary sources",
        "Analyze findings",
        "Synthesize report"
    ]


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
