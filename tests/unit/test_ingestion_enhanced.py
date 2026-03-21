"""Tests for enhanced ingestion node."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from deep_research_engine.nodes.ingestion import ingestion_node
from deep_research_engine.state import EngineState

@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingestion_node_generic_files(initial_state, tmp_path):
    """Test ingestion node with various file types."""
    state = initial_state.copy()

    # Setup test files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "test.pdf").write_text("dummy pdf content")
    (input_dir / "test.jpg").write_text("dummy image content")
    (input_dir / "test.mp3").write_text("dummy audio content")

    state["input_data"] = {"path": str(input_dir)}
    state["session_id"] = "test_session"
    state["context"] = {
        "config": {
            "phases": {
                "ingestion": {
                    "enabled": True
                }
            }
        }
    }

    # Mock multimodal utils
    with patch("deep_research_engine.nodes.ingestion.vlm_analyze", return_value="Image description") as mock_vlm, \
         patch("deep_research_engine.nodes.ingestion.face_recognize", return_value=[{"name": "Alice"}]) as mock_face, \
         patch("deep_research_engine.nodes.ingestion.asr_transcribe", return_value="Audio transcript") as mock_asr, \
         patch("deep_research_engine.nodes.ingestion.extract_entities", return_value={"people": ["Alice"], "topics": ["test"]}) as mock_entities:

        # We also need to mock pymupdf4llm if it's used
        with patch("pymupdf4llm.to_markdown", return_value="# PDF Content") as mock_pdf:
            command = await ingestion_node(state)

            assert command.goto == "drafter"
            assert "ingested_items" in command.update
            # Should have 3 files + big dump
            ingested_items = command.update["ingested_items"]
            # Filter for our specific md files
            md_files = [f for f in ingested_items if f.endswith(".md")]
            assert len(md_files) >= 3

            # Check context updates
            assert "ingested_text_files" in command.update["context"]
            assert "big_text_dump" in command.update["context"]

            # Verify the big dump exists
            dump_path = Path(command.update["context"]["big_text_dump"])
            assert dump_path.exists()
            content = dump_path.read_text()
            assert "test_pdf.md" in content
            assert "test_image.md" in content
            assert "test_audio.md" in content

@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingestion_node_single_file(initial_state, tmp_path):
    """Test ingestion node with a single file."""
    state = initial_state.copy()

    test_pdf = tmp_path / "single.pdf"
    test_pdf.write_text("dummy pdf content")

    state["input_data"] = {"path": str(test_pdf)}
    state["session_id"] = "single_file_session"
    state["context"] = {"config": {"phases": {"ingestion": {"enabled": True}}}}

    with patch("pymupdf4llm.to_markdown", return_value="# Single PDF Content") as mock_pdf:
        command = await ingestion_node(state)

        assert command.goto == "drafter"
        assert len(command.update["ingested_items"]) == 1
        assert "single_pdf.md" in command.update["ingested_items"][0]
