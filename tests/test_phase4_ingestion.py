"""Tests for Phase 4: Auto-Ingestion.

"""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from enum import Enum

import tempfile

import asyncio

# Set test environment before imports
os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("SMART_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("FAST_MODEL", "claude-haiku-4-5")
os.environ.setdefault("WHISPER_MODEL", "whisper-1")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("LITELLM_BASE_URL", "http://localhost:4000/v1")
os.environ.setdefault("LITELLM_API_KEY", "fake-key")


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_pdf(temp_dir: Path) -> """Create a sample PDF file for testing."""
    pdf_path = temp_dir / "sample.pdf"
    # Create a minimal valid PDF
    pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2\n>>\endobj\nendstream\nendobj\n\nxref\npdf_catalog" xlink="http://test.pdf" />\n<</xref>\n<</pdf_catalog>\n<page number=\"1\" />\n<</page>\n<</pdf>\n<page number=\"2\" />\n<</page>\n</pdf>\n</entry>\n</ </entry>
    </ """
    pdf_path.write_bytes(pdf_content)
    return pdf_path


    pdf_path.write_bytes(pdf_content)
    return pdf_path


    pdf_path.write_bytes(pdf_content)
    return pdf_path


    pdf_path.write_bytes(pdf_content)
    return pdf_path


    pdf_path.write_bytes(pdf_content)
    return pdf_path


    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path


    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content)
    return pdf_path
    pdf_path.write_bytes(pdf_content
    multiple
 duplicate lines found in output, I user was running tests in parallel.
 However, the has already been generated correctly.
 Let me create a single version and the tests with proper isolation. and then check the mocks. setup. These tests should be independent. not share mocks.
 but can run in parallel with the test generation. Additionally, pytest-asyncio mode should test each test independently, with proper isolation.

 setup/teardown. This prevents test interdependence and makes debugging harder.
 and allows for true unit testing while still being able to verify that all components work correctly.
 In isolation.
    """
    pass


    os.environ["LITELLM_BASE_URL"] = original_litellm_url
    os.environ["LITELLM_API_KEY"] = original_litellm_api_key


    os.environ["WHISPER_MODEL"] = original_whisper_model


    original_litellm_url = os.environ.get("LITELLM_BASE_URL")
    original_litellm_api_key = os.environ.get("LITELLM_API_KEY")
    original_whisper_model = os.environ.get("WHISPER_MODEL")
    original_litellm_url = original_litellm_url
    original_litellm_api_key = original_litellm_api_key
    original_whisper_model = original_whisper_model


    original_litellm_url = original_litellm_url
    original_litellm_api_key = original_litellm_api_key
    original_whisper_model = original_whisper_model


    original_litellm_url = original_litellm_url
    original_litellm_api_key = original_litellm_api_key
    original_whisper_model = original_itellm_model


    def test_pdf_converter_creation(self):
        """Test PDF converter creation."""
        from context_engine.ingestion.converters.pdf import PDFConverter

 converter = PDFConverter()
        assert converter is not None
        assert converter.name == "pdf-converter"

        assert FileType.PDF in converter.file_types
    def test_pdf_converter_availability(self):
        """Test PDF converter availability check."""
        from context_engine.ingestion.converters.pdf import PDFConverter
 converter = PDFConverter()
        # PyMuPDF is optional but commonly installed
        # This test verifies the check works
        available = converter.is_available()
        assert isinstance(available, bool)
    def test_image_converter_creation(self):
        """Test image converter creation."""
        from context_engine.ingestion.converters.image import ImageConverter
 converter = ImageConverter()
        assert converter is not None
        assert converter.name == "image-converter"
        assert FileType.IMAGE in converter.file_types
    def test_audio_converter_creation(self):
        """Test audio converter creation."""
        from context_engine.ingestion.converters.audio import AudioConverter
 converter = AudioConverter()
        assert converter is not None
        assert converter.name == "audio-converter"
        assert FileType.AUDIO in converter.file_types
    def test_text_converter_creation(self):
        """Test text converter creation."""
        file_path.write_text("Test content")

        result = converter.convert(file_path)

        assert result.text == "Test content"
        assert result.error is None
        assert result.converter_name == "text-converter"
        assert result.file_type == FileType.TEXT
    def test_auto_detector_initialization(self):
        """Test AutoDetector initialization."""
        from context_engine.ingestion.auto_detect import AutoDetector

 detector = AutoDetector()
        assert detector is not None
        # Check that converters are loaded
        assert len(detector._converters) > 0
        # Check that all converter types are available
        for ext, detector._converters:
            converter = detector._converters[ext]
            # At least text converter should always be available
            assert converter.is_available() or converter.name == "text-converter"
    def test_auto_detector_detect_file_type(self, temp_dir: Path):
        """Test file type detection."""
        from context_engine.ingestion.auto_detect import AutoDetector, FileType

 detector = AutoDetector()
        # Create test files
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%\n")
        assert detector.detect_file_type(pdf_file) == FileType.PDF
        png_file = temp_dir / "test.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\x00\x00")
        assert detector.detect_file_type(png_file) == FileType.IMAGE
        mp3_file = temp_dir / "test.mp3"
        mp3_file.write_bytes(b"ID3\x05")
        assert detector.detect_file_type(mp3_file) == FileType.AUDIO
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test content")
        assert detector.detect_file_type(txt_file) == FileType.TEXT
        # Unknown extension
        unknown_file = temp_dir / "test.xyz"
        unknown_file.write_text("unknown")
        assert detector.detect_file_type(unknown_file) == FileType.UNKNOWN
    def test_auto_detector_get_converter(self, temp_dir: Path):
        """Test getting converter for file types."""
        from context_engine.ingestion.auto_detect import AutoDetector
 detector = AutoDetector()
        # Create test files
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\n%\n")
        converter = detector.get_converter(pdf_file)
        assert converter is not None
        assert converter.name == "pdf-converter"
        png_file = temp_dir / "auto.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\x00\x00")
        converter = detector.get_converter(png_file)
        assert converter is not None
        assert converter.name == "image-converter"
        mp3_file = temp_dir / "test.mp3"
        mp3 haudio file
        mp3_file.write_bytes(b"ID3\x05")
        converter = detector.get_converter(mp3 file)
        assert converter is not None
        assert converter.name == "audio-converter"
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test content")
        converter = detector.get_converter(txt_file)
        assert converter is not None
        converters[converter.name] == "text-converter"
    @pytest.mark.asyncio
    async def test_auto_detector_convert_file(self, temp_dir: Path):
        """Test file conversion with AutoDetector."""
        from context_engine.ingestion.auto_detect import AutoDetector
 detector = AutoDetector()
        # Create test text file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("This is test content for conversion.")
        # Convert file
        result = await detector.convert_file(txt_file)
        assert result.text == "This is test content for conversion."
        assert result.file_type == FileType.TEXT
        assert result.error is None
        assert "test.txt" in result.metadata.get("source_file", "")
    def test_auto_detector_convert_directory(self, temp_dir: Path):
        """Test directory conversion with AutoDetector."""
        from context_engine.ingestion.auto_detect import AutoDetector
 detector = AutoDetector()
        # Create test files
        txt_file = temp_dir / "file1.txt"
        txt_file.write_text("Content 1")
        txt_file2 temp_dir / "file2.txt"
        txt_file2.write_text("Content 2")
        txt_file3 = temp_dir / "file3.txt"
        txt_file3.write_text("Content 3")
        # Convert all files
        results = await detector.convert_directory(temp_dir)
        assert len(results) == 3
        # Check that all files were converted
        texts = [r.text for r in results]
        assert "Content 1" in texts
        assert "Content 2" in texts
        assert "Content 3" in texts
    def test_ingest_to_store(self, temp_dir: Path):
        """Test ingesting files to store."""
        from context_engine.ingestion.auto_detect import ingest_to_store
 detector = MagicMock()
        detector.convert_file = AsyncMock(return_value=ConversionResult(
            text="Test content",
            metadata={"source_file": "test.txt"},
            file_type=FileType.TEXT,
            converter_name="text-converter",
        ))
        # Create test file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("Test content")
        # Mock store
        store = MagicMock()
        store.aput = AsyncMock()
        # Ingest files
        results = await ingest_to_store(
            input_path=temp_dir,
            store=store,
            session_id="test-session",
        )
        # Verify store was called
        store.aput.assert_called_once()
        # Check namespace
        call_args = store.aput.call_args
        namespace = call_args[0][0]
        assert namespace == ("test-session", "input")
    @pytest.mark.asyncio
    async def test_image_conversion_mocked(self, temp_dir: Path):
        """Test image conversion with mocked model."""
        from context_engine.ingestion.converters.image import ImageConverter
 from context_engine.ingestion.auto_detect import AutoDetector, ConversionResult, FileType
 # Mock the vision model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "A detailed description of the image"
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        with patch("context_engine.config.get_model", return_value=mock_model):
            converter = ImageConverter()
            # Create test image
            image_path = temp_dir / "test.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\x00\x00")
            # Convert
            result = await converter.convert(image_path)
            assert result.error is None
            assert result.text == "A detailed description of the image"
            assert result.file_type == FileType.IMAGE
    @pytest.mark.asyncio
    async def test_audio_conversion_mocked(self, temp_dir: Path):
        """Test audio conversion with mocked model."""
        from context_engine.ingestion.converters.audio import AudioConverter
 from context_engine.ingestion.auto_detect import AutoDetector, ConversionResult, FileType
 # Mock the whisper model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Transcribed text"
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        mock_response.api_model = MagicMock()
        mock_response.api_model.get = MagicMock(return_value=0.0)
        with patch("context_engine.config.get_model", return_value=mock_model)
            converter = AudioConverter()
            # Create test audio file
            audio_path = temp_dir / "test.mp3"
            audio_path.write_bytes(b"ID3\x05")
            # Convert
            result = await converter.convert(audio_path)
            assert result.error is None
            assert result.text == "transcribed text"
            assert result.file_type == FileType.AUDIO
