"""Character Card Parser.

Parse character cards from various formats:
- JSON
- YAML
- PNG (SillyTavern format with embedded data)
"""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    from PIL import Image
    from PIL import PngImagePlugin
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from .models import CharacterCard, CharacterBook, CharacterBookEntry

logger = logging.getLogger(__name__)


def parse_character_card(
    source: Union[str, Path, bytes],
    format: str = "auto"
) -> Optional[CharacterCard]:
    """Parse a character card from file or bytes.

    Args:
        source: File path (str/Path) or raw bytes
        format: File format ("json", "yaml", "png", or "auto" to detect)

    Returns:
        CharacterCard or None if parsing fails
    """
    # Handle bytes input
    if isinstance(source, bytes):
        return _parse_from_bytes(source, format)

    # Handle file path
    filepath = Path(source)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    # Auto-detect format from extension
    if format == "auto":
        suffix = filepath.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in [".yaml", ".yml"]:
            format = "yaml"
        elif suffix == ".png":
            format = "png"
        else:
            logger.error(f"Unknown file format: {suffix}")
            return None

    # Parse based on format
    if format == "json":
        return _parse_json(filepath)
    elif format == "yaml":
        return _parse_yaml(filepath)
    elif format == "png":
        return _parse_png(filepath)
    else:
        logger.error(f"Unsupported format: {format}")
        return None


def _parse_from_bytes(data: bytes, format: str) -> Optional[CharacterCard]:
    """Parse character card from bytes."""
    if format == "png" and _HAS_PIL:
        try:
            img = Image.open(BytesIO(data))
            return _extract_png_data(img)
        except Exception as e:
            logger.error(f"Failed to parse PNG from bytes: {e}")
            return None
    else:
        # Try to decode as text
        try:
            text = data.decode("utf-8")
            data_dict = json.loads(text)
            return CharacterCard(**data_dict)
        except Exception as e:
            logger.error(f"Failed to parse bytes: {e}")
            return None


def _parse_json(filepath: Path) -> Optional[CharacterCard]:
    """Parse JSON character card."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CharacterCard(**data)
    except Exception as e:
        logger.error(f"Failed to parse JSON {filepath}: {e}")
        return None


def _parse_yaml(filepath: Path) -> Optional[CharacterCard]:
    """Parse YAML character card."""
    if not _HAS_YAML:
        logger.error("PyYAML not installed")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return CharacterCard(**data)
    except Exception as e:
        logger.error(f"Failed to parse YAML {filepath}: {e}")
        return None


def _parse_png(filepath: Path) -> Optional[CharacterCard]:
    """Parse PNG character card with embedded data (SillyTavern format)."""
    if not _HAS_PIL:
        logger.error("PIL not installed")
        return None

    try:
        img = Image.open(filepath)
        return _extract_png_data(img)
    except Exception as e:
        logger.error(f"Failed to parse PNG {filepath}: {e}")
        return None


def _extract_png_data(img: Image.Image) -> Optional[CharacterCard]:
    """Extract character card data from PNG image.

    SillyTavern embeds character data in PNG tEXt chunks
    prefixed with "chara0", "chara1", etc.
    """
    char_data = ""

    # Read all chara chunks
    i = 0
    while True:
        key = f"chara{i}"
        if key in img.text:
            char_data += img.text[key]
            i += 1
        else:
            break

    if not char_data:
        logger.error("No character data found in PNG")
        return None

    # Decode base64
    try:
        decoded = base64.b64decode(char_data).decode("utf-8")
        data = json.loads(decoded)
        return CharacterCard(**data)
    except Exception as e:
        logger.error(f"Failed to decode PNG character data: {e}")
        return None


def parse_character_book(
    source: Union[str, Path, dict],
    format: str = "auto"
) -> Optional[CharacterBook]:
    """Parse a character book (lorebook).

    Args:
        source: File path, dict, or JSON string
        format: File format ("json", "yaml", or "auto")

    Returns:
        CharacterBook or None
    """
    # Handle dict input
    if isinstance(source, dict):
        return _parse_book_dict(source)

    # Handle JSON string
    if isinstance(source, str) and source.strip().startswith("{"):
        try:
            data = json.loads(source)
            return _parse_book_dict(data)
        except Exception as e:
            logger.error(f"Failed to parse character book JSON string: {e}")
            return None

    # Handle file path
    filepath = Path(source)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    # Auto-detect format
    if format == "auto":
        suffix = filepath.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in [".yaml", ".yml"]:
            format = "yaml"
        else:
            format = "json"  # Default to JSON

    # Parse
    if format == "json":
        return _parse_book_json(filepath)
    elif format == "yaml":
        return _parse_book_yaml(filepath)
    else:
        logger.error(f"Unsupported format: {format}")
        return None


def _parse_book_dict(data: dict) -> Optional[CharacterBook]:
    """Parse character book from dict."""
    try:
        # Convert entries to CharacterBookEntry
        entries = []
        for entry_data in data.get("entries", []):
            entries.append(CharacterBookEntry(**entry_data))

        data["entries"] = entries
        return CharacterBook(**data)
    except Exception as e:
        logger.error(f"Failed to parse character book dict: {e}")
        return None


def _parse_book_json(filepath: Path) -> Optional[CharacterBook]:
    """Parse JSON character book."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _parse_book_dict(data)
    except Exception as e:
        logger.error(f"Failed to parse book JSON {filepath}: {e}")
        return None


def _parse_book_yaml(filepath: Path) -> Optional[CharacterBook]:
    """Parse YAML character book."""
    if not _HAS_YAML:
        logger.error("PyYAML not installed")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return _parse_book_dict(data)
    except Exception as e:
        logger.error(f"Failed to parse book YAML {filepath}: {e}")
        return None


def extract_character_from_png(
    filepath: Union[str, Path]
) -> Optional[Tuple[CharacterCard, bytes]]:
    """Extract character card and image from PNG file.

    Args:
        filepath: Path to PNG file

    Returns:
        Tuple of (CharacterCard, image_bytes) or None
    """
    if not _HAS_PIL:
        logger.error("PIL not installed")
        return None

    try:
        img = Image.open(filepath)

        # Extract character data
        char = _extract_png_data(img)
        if not char:
            return None

        # Get image bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return char, img_bytes.read()
    except Exception as e:
        logger.error(f"Failed to extract from PNG {filepath}: {e}")
        return None
