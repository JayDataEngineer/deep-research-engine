"""Pytest Configuration and Shared Fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


# =============================================================================
# Test Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


# =============================================================================
# Shared Fixtures
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
            "source_count": 5,
            "timing": {
                "start": 1234567890.0,
                "end": 1234567895.0,
                "duration": 5.0
            }
        },
        {
            "worker_id": "researcher_1",
            "worker_type": "chunk_processor",
            "task": "Process PDF chunks",
            "success": True,
            "output": "PDF analysis results...",
            "confidence": 0.92,
            "source_count": 3,
            "timing": {
                "start": 1234567890.0,
                "end": 1234567893.0,
                "duration": 3.0
            }
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


@pytest.fixture
def sample_state(sample_worker_results, sample_plan):
    """Sample engine state for testing."""
    from deep_research_engine.state import create_initial_state
    
    state = create_initial_state(
        workflow_name="test_workflow",
        workflow_type="research",
        input_data={"query": "Test query"},
        session_id="test_session",
        run_id="test_run"
    )
    
    state["worker_results"] = sample_worker_results
    state["plan"] = sample_plan
    state["context"] = {"config": {}}
    
    return state


@pytest.fixture
def sample_critic_evaluation():
    """Sample critic evaluation for testing."""
    from deep_research_engine.state import CriticEvaluation
    
    return CriticEvaluation(
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


@pytest.fixture
def mock_deep_agent():
    """Mock deep agent for testing."""
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Mock response")]
    })
    return mock_agent


@pytest.fixture
def mock_structured_output():
    """Mock structured output for testing."""
    def _create_mock(evaluation_data=None):
        if evaluation_data is None:
            evaluation_data = {
                "passed": True,
                "score": 0.85,
                "issues": [],
                "suggestions": [],
                "recommendation": "pass"
            }
        
        mock_evaluation = MagicMock()
        for key, value in evaluation_data.items():
            setattr(mock_evaluation, key, value)
        mock_evaluation.model_dump.return_value = evaluation_data
        
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_evaluation)
        
        return mock_llm
    
    return _create_mock


@pytest.fixture
def tmp_config_file(tmp_path):
    """Create temporary config file for testing."""
    def _create_config(content: str, name: str = "test_config.yaml"):
        config_file = tmp_path / name
        config_file.write_text(content)
        return config_file
    
    return _create_config


@pytest.fixture
def sample_video_file(tmp_path):
    """Create sample video file for testing."""
    # Note: This creates an empty file, not a real video
    # Real video tests would need actual video files
    video_file = tmp_path / "test_video.mp4"
    video_file.touch()
    return video_file


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create sample audio file for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.touch()
    return audio_file


@pytest.fixture
def sample_image_file(tmp_path):
    """Create sample image file for testing."""
    image_file = tmp_path / "test_image.jpg"
    image_file.touch()
    return image_file


@pytest.fixture
def ffmpeg_metadata_file(tmp_path):
    """Create FFmpeg metadata file for testing."""
    def _create_metadata(format_type="new"):
        metadata_file = tmp_path / "time.txt"
        
        if format_type == "new":
            content = (
                "[Parsed_metadata_1 @ 0x7f8b8c0d4e00] pts:108859 pts_time:1.20954\n"
                "lavfi.scene_score=0.436456\n"
                "[Parsed_metadata_1 @ 0x7f8b8c0d4e00] pts:217718 pts_time:2.41908\n"
                "lavfi.scene_score=0.523412\n"
            )
        else:  # old format
            content = (
                "frame=    1 pts=      1 time=00:00:01.23\n"
                "frame=    2 pts=     45 time=00:00:04.50\n"
            )
        
        metadata_file.write_text(content)
        return metadata_file
    
    return _create_metadata


# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_worker_result(
    worker_type="web_researcher",
    success=True,
    output="Mock output",
    confidence=0.85
):
    """Helper to create mock worker result."""
    return {
        "worker_id": f"{worker_type}_0",
        "worker_type": worker_type,
        "task": "Mock task",
        "success": success,
        "output": output,
        "confidence": confidence,
        "source_count": 3
    }


def create_mock_state_with_results(worker_results):
    """Helper to create mock state with worker results."""
    from deep_research_engine.state import create_initial_state
    
    state = create_initial_state(
        workflow_name="test",
        workflow_type="research",
        input_data={"query": "Test"},
        session_id="test",
        run_id="test"
    )
    
    state["worker_results"] = worker_results
    state["plan"] = ["Test plan"]
    state["context"] = {"config": {}}
    
    return state
