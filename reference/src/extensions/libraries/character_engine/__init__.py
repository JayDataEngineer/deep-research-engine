"""Character Engine - Character Card and Book Management.

This library provides:
- Pydantic models for character cards and character books (lorebooks)
- Parsers for JSON, YAML, and PNG (SillyTavern) formats
- CRUD operations for character storage
- API routes for character management

Example:
    from src.extensions.libraries.character_engine import (
        parse_character_card,
        CharacterStorage,
    )

    # Parse a card
    card = parse_character_card("path/to/card.json")

    # Use storage
    storage = CharacterStorage()
    storage.save_card(card, format="json")
"""

from .models import CharacterCard, CharacterBook, CharacterBookEntry
from .parser import (
    parse_character_card,
    parse_character_book,
    extract_character_from_png,
)
from .crud import CharacterStorage, get_character_storage
from . import routes

__all__ = [
    # Models
    "CharacterCard",
    "CharacterBook",
    "CharacterBookEntry",
    # Parser
    "parse_character_card",
    "parse_character_book",
    "extract_character_from_png",
    # CRUD
    "CharacterStorage",
    "get_character_storage",
    # Routes
    "routes",
    "router",
]

# Export router for easy access
router = routes.router
