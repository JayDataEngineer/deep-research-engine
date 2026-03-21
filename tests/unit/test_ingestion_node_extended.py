import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from deep_research_engine.nodes.ingestion import ingestion_node, _process_pdf, _process_image, _process_audio
from deep_research_engine.state import EngineState

@pytest.fixture
def mock_state(tmp_path):
    return {
        "session_id": "test_session",
        "input_data": {"path": str(tmp_path)},
        "context": {
            "config": {
                "ingestion": {
                    "enabled": True
                }
            }
        },
        "phase_history": []
    }

@pytest.mark.asyncio
async def test_process_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page text content"
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_open.return_value = mock_doc

        result = await _process_pdf(pdf_path, output_dir)

        assert result is not None
        assert result.exists()
        assert "Page text content" in result.read_text()
        mock_open.assert_called_once_with(pdf_path)

@pytest.mark.asyncio
async def test_process_image(tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake image data")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("deep_research_engine.nodes.ingestion.vlm_analyze", new_callable=AsyncMock) as mock_vlm, \
         patch("deep_research_engine.nodes.ingestion.face_recognize", new_callable=AsyncMock) as mock_face:

        mock_vlm.return_value = "A beautiful test image"
        mock_face.return_value = [{"name": "Alice", "confidence": 0.95}]

        result = await _process_image(image_path, output_dir)

        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "A beautiful test image" in content
        assert "Alice" in content
        assert "0.95" in content

@pytest.mark.asyncio
async def test_process_audio(tmp_path):
    audio_path = tmp_path / "test.mp3"
    audio_path.write_bytes(b"fake audio data")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("deep_research_engine.nodes.ingestion.asr_transcribe", new_callable=AsyncMock) as mock_asr:
        mock_asr.return_value = "This is a test audio transcript."

        result = await _process_audio(audio_path, output_dir)

        assert result is not None
        assert result.exists()
        assert "This is a test audio transcript." in result.read_text()

@pytest.mark.asyncio
async def test_ingestion_node_general_files(mock_state, tmp_path):
    # Setup files
    (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4 test")
    (tmp_path / "img1.jpg").write_bytes(b"fake image")
    (tmp_path / "snd1.mp3").write_bytes(b"fake audio")

    # Nested file
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "doc2.pdf").write_bytes(b"%PDF-1.4 nested test")

    with patch("fitz.open") as mock_fitz_open, \
         patch("deep_research_engine.nodes.ingestion.vlm_analyze", new_callable=AsyncMock) as mock_vlm, \
         patch("deep_research_engine.nodes.ingestion.face_recognize", new_callable=AsyncMock) as mock_face, \
         patch("deep_research_engine.nodes.ingestion.asr_transcribe", new_callable=AsyncMock) as mock_asr:

        # Mocking PDF
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "PDF text"
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc

        # Mocking VLM/Face
        mock_vlm.return_value = "Image description"
        mock_face.return_value = []

        # Mocking ASR
        mock_asr.return_value = "Audio transcript"

        command = await ingestion_node(mock_state)

        assert command.goto == "drafter"
        update = command.update
        assert "ingested_items" in update
        assert len(update["ingested_items"]) == 4  # 2 PDFs, 1 Image, 1 Audio

        context = update["context"]
        assert "ingested_text_files" in context
        assert len(context["ingested_text_files"]) == 4
        assert "big_text_dump" in context

        big_dump = Path(context["big_text_dump"])
        assert big_dump.exists()
        dump_content = big_dump.read_text()
        assert "PDF text" in dump_content
        assert "Image description" in dump_content
        assert "Audio transcript" in dump_content
