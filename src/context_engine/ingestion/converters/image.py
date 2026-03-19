"""Image converter using LiteLLM Vision model.

Converts images to text descriptions.
"""

import time
import base64
from pathlib import Path
from typing import Any

from dataclasses import dataclass
from loguru import logger

from context_engine.config import get_model
from context_engine.ingestion.converters.base import (
    BaseConverter,
    FileType,
    ConversionResult,
)


class ImageConverter(BaseConverter):
    """Convert images to text using Vision model."""

    @property
    def file_types(self) -> list[FileType]:
        return [FileType.IMAGE]

    @property
    def name(self) -> str:
        return "image-converter"

    def __init__(self) -> None:
        """Initialize image converter."""
        self._pil_available = False
        try:
            from PIL import Image as PILImage
            self._pil_available = True
            logger.info("Pillow available for image conversion")
        except ImportError:
            logger.warning("Pillow not installed - image conversion unavailable")

            self._pil_available = False

        logger.info(f"Image converter initialized (PIL available: {self._pil_available})")

    def is_available(self) -> bool:
        """Check if PIL is installed."""
        return self._pil_available

    def convert(self, file_path: Path) -> ConversionResult:
        """Convert image to text using Vision model.

        Args:
            file_path: Path to image file

        Returns:
            ConversionResult with extracted text
        """
        start_time = time.time()

        try:
            # Determine image format
            ext = file_path.suffix.lower()
            image_format = ext.lstrip(".")

            # Read and encode image
            with open(file_path, "rb") as f:
                image_data = f.read()

            if not image_data:
                return ConversionResult(
                    text="",
                    metadata={},
                    source_file=str(file_path),
                    file_type=FileType.IMAGE,
                    converter_name=self.name,
                    error=f"Failed to read image file: {file_path}",
                )

            # Convert to text via Vision model
            model = get_model("fast")  # Haiku for vision
            if model is None:
                return ConversionResult(
                    text="",
                    metadata={},
                    source_file=str(file_path),
                    file_type=FileType.IMAGE,
                    converter_name=self.name,
                    error="Vision model not available",
                )

            prompt = """Describe this image in detail. Focus on:
- Main objects and their purpose
- Colors, lighting, atmosphere
- Any text or readable content
- Overall mood and feeling of the image
- Notable features or patterns

Keep the description factual and objective.
            """

            # Use multimodal input for vision model
            response = model.invoke(
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{image_data}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ]
            )

            elapsed = time.time() - start_time
            text = response.content
            char_count = len(text)
            metadata = {
                "source_file": str(file_path),
                "image_format": image_format,
                "file_size_bytes": file_path.stat().st_size,
                "conversion_method": "vision",
            }

            return ConversionResult(
                text=text,
                metadata=metadata,
                source_file=str(file_path),
                file_type=FileType.IMAGE,
                converter_name=self.name,
                conversion_time=elapsed,
                char_count=char_count,
                tokens_used=response.usage_metadata.get("total_tokens", 0),
                model_name=model.model,
            )

        except Exception as e:
            logger.error(f"Error converting image {file_path}: {e}")
            return ConversionResult(
                text="",
                metadata={},
                source_file=str(file_path),
                file_type=FileType.IMAGE,
                converter_name=self.name,
                error=str(e),
            )
