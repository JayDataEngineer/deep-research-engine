"""Unit Tests for Deep Research Engine Core Components.

This module tests the core functionality added in the March 27, 2026 implementation:
- Critic evaluation with structured output
- Video audio extraction
- FFmpeg metadata parsing
- Flash Agent orchestration
- State management
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test: Critic Evaluation Schema
# =============================================================================


class TestCriticEvaluation:
    """Tests for CriticEvaluation Pydantic model."""

    def test_critic_evaluation_creation(self):
        """Test creating CriticEvaluation with valid data."""
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
        assert evaluation.feedback_to == "artifact_maker"

    def test_critic_evaluation_default_values(self):
        """Test CriticEvaluation default field values."""
        from deep_research_engine.state import CriticEvaluation
        
        evaluation = CriticEvaluation(
            passed=False,
            score=0.5,
            completeness=0.5,
            accuracy=0.5,
            citation_quality=0.5,
            coherence=0.5,
            recommendation="revise",
            feedback_to="drafter"
        )
        
        assert evaluation.issues == []
        assert evaluation.suggestions == []

    def test_critic_evaluation_score_validation(self):
        """Test score field validation (0-1 range)."""
        from deep_research_engine.state import CriticEvaluation
        from pydantic import ValidationError
        
        # Valid scores
        evaluation = CriticEvaluation(
            passed=True, score=1.0, completeness=1.0, accuracy=1.0,
            citation_quality=1.0, coherence=1.0,
            recommendation="pass", feedback_to="artifact_maker"
        )
        assert evaluation.score == 1.0
        
        # Invalid score should raise validation error
        with pytest.raises(ValidationError):
            CriticEvaluation(
                passed=True, score=1.5, completeness=0.5, accuracy=0.5,
                citation_quality=0.5, coherence=0.5,
                recommendation="pass", feedback_to="artifact_maker"
            )

    def test_critic_evaluation_model_dump(self):
        """Test serialization to dict."""
        from deep_research_engine.state import CriticEvaluation
        
        evaluation = CriticEvaluation(
            passed=True,
            score=0.85,
            completeness=0.9,
            accuracy=0.8,
            citation_quality=0.85,
            coherence=0.85,
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            recommendation="pass",
            feedback_to="artifact_maker"
        )
        
        dump = evaluation.model_dump()
        assert dump["passed"] is True
        assert dump["score"] == 0.85
        assert "Issue 1" in dump["issues"]


# =============================================================================
# Test: Video Audio Extraction
# =============================================================================


class TestVideoAudioExtraction:
    """Tests for video audio extraction functions."""

    def test_imports(self):
        """Test audio extraction functions import correctly."""
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

    @pytest.mark.asyncio
    async def test_extract_audio_from_video_nonexistent(self):
        """Test audio extraction for nonexistent file raises error."""
        from deep_research_engine.shared.multimodal import extract_audio_from_video
        
        with pytest.raises(FileNotFoundError):
            await extract_audio_from_video(Path("/nonexistent/video.mp4"))


# =============================================================================
# Test: FFmpeg Metadata Parsing
# =============================================================================


class TestFFmpegMetadata:
    """Tests for FFmpeg metadata parsing."""

    def test_imports(self):
        """Test metadata parsing functions import correctly."""
        from deep_research_engine.utils.pipeline import (
            parse_ffmpeg_metadata,
            extract_scene_frames
        )
        assert parse_ffmpeg_metadata is not None
        assert extract_scene_frames is not None

    @pytest.mark.asyncio
    async def test_parse_ffmpeg_metadata_nonexistent(self, tmp_path):
        """Test parsing nonexistent metadata file."""
        from deep_research_engine.utils.pipeline import parse_ffmpeg_metadata
        
        frames = await parse_ffmpeg_metadata(tmp_path / "nonexistent.txt")
        assert len(frames) == 0

    @pytest.mark.asyncio
    async def test_parse_ffmpeg_metadata_new_format(self, tmp_path):
        """Test parsing new FFmpeg metadata format."""
        from deep_research_engine.utils.pipeline import parse_ffmpeg_metadata
        
        # Create test metadata file with new format
        metadata_file = tmp_path / "time.txt"
        metadata_file.write_text(
            "[Parsed_metadata_1 @ 0x7f8b8c0d4e00] pts:108859 pts_time:1.20954\n"
            "lavfi.scene_score=0.436456\n"
            "[Parsed_metadata_1 @ 0x7f8b8c0d4e00] pts:217718 pts_time:2.41908\n"
            "lavfi.scene_score=0.523412\n"
        )
        
        frames = await parse_ffmpeg_metadata(metadata_file)
        
        assert len(frames) == 2
        assert frames[0]["pts_time"] == 1.20954
        assert frames[0]["scene_score"] == 0.436456
        assert frames[0]["timestamp_ms"] == 1209
        assert frames[0]["timestamp_formatted"] == "00:00:01.209"

    @pytest.mark.asyncio
    async def test_parse_ffmpeg_metadata_old_format(self, tmp_path):
        """Test parsing old FFmpeg metadata format."""
        from deep_research_engine.utils.pipeline import parse_ffmpeg_metadata
        
        # Create test metadata file with old format
        metadata_file = tmp_path / "time.txt"
        metadata_file.write_text(
            "frame=    1 pts=      1 time=00:00:01.23\n"
            "frame=    2 pts=     45 time=00:00:04.50\n"
        )
        
        frames = await parse_ffmpeg_metadata(metadata_file)
        
        assert len(frames) == 2
        assert frames[0]["frame_num"] == 1
        assert frames[0]["timestamp_ms"] == 1230
        assert frames[1]["frame_num"] == 2
        assert frames[1]["timestamp_ms"] == 4500


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
        assert "context" in state


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

    def test_flash_agent_system_prompt(self):
        """Test FlashAgent system prompt generation."""
        from deep_research_engine.agents.flash import FlashAgent
        
        agent = FlashAgent(
            name="research_specialist",
            task="Research quantum computing"
        )
        
        prompt = agent._build_system_prompt()
        
        assert "research" in prompt.lower()
        assert "PLAN" in prompt
        assert "DELEGATE" in prompt
        assert "SYNTHESIZE" in prompt

    @pytest.mark.asyncio
    async def test_create_flash_subagents(self):
        """Test flash subagent creation."""
        from deep_research_engine.agents.flash import create_flash_subagents

        configs = [
            {"type": "web_researcher", "task": "Search web", "enabled": True},
            {"type": "content_classifier", "task": "Classify content", "enabled": True}
        ]

        subagents = await create_flash_subagents(configs)

        assert len(subagents) == 2
        assert subagents[0]["name"] == "web_researcher_specialist"
        assert subagents[1]["name"] == "content_classifier_specialist"


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
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
