"""Regex Engine Manager

Manages regex pattern functionality.
Refactored to use BaseManager and StorageManager.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.extensions.libraries.base.manager_base import BaseItem, BaseManager
from src.utils.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class RegexPattern(BaseItem):
    """A regex pattern definition."""

    id: str
    name: str
    pattern: str
    description: str = ""
    flags: int = 0  # re flags (re.I, re.M, etc.)
    replacement: str = ""  # Optional replacement string for substitution
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "pattern": self.pattern,
            "description": self.description,
            "flags": self.flags,
            "replacement": self.replacement,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegexPattern":
        return cls(**data)


class RegexEngineManager(BaseManager):
    """Manages regex pattern functionality."""

    # Default patterns
    DEFAULT_PATTERNS = {
        "url": RegexPattern(
            id="url",
            name="url",
            pattern=r'https?://[^\s<>"]+',
            description="Match URLs",
        ),
        "email": RegexPattern(
            id="email",
            name="email",
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            description="Match email addresses",
        ),
        "markdown_links": RegexPattern(
            id="markdown_links",
            name="markdown_links",
            pattern=r'\[([^\]]+)\]\(([^)]+)\)',
            description="Match markdown links",
        ),
    }

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the regex engine manager.

        Args:
            base_dir: Base directory for storage (auto-created if None)
        """
        super().__init__(base_dir=base_dir, feature_name="regex_patterns", storage_file="patterns.json")

        # Load defaults if empty
        if not self._items:
            self._load_defaults()

    def _load_defaults(self):
        """Load default patterns."""
        for pattern in self.DEFAULT_PATTERNS.values():
            if pattern.id not in self._items:
                self._items[pattern.id] = pattern
        self._save()

    def _get_item_class(self):
        """Get the RegexPattern item class."""
        return RegexPattern

    # ========================================================================
    # Pattern-specific methods
    # ========================================================================

    def get_pattern(self, name: str) -> Optional[RegexPattern]:
        """Get a pattern by name.

        Args:
            name: Pattern name

        Returns:
            RegexPattern or None
        """
        return self._items.get(name)

    def add_pattern(self, name: str, pattern: str, description: str = "", flags: int = 0, replacement: str = "") -> bool:
        """Add a new pattern.

        Args:
            name: Pattern name
            pattern: Regex pattern
            description: Optional description
            flags: Optional regex flags
            replacement: Optional replacement string

        Returns:
            True if added
        """
        regex_pattern = RegexPattern(
            id=StorageManager.generate_id_from_name(name),
            name=name,
            pattern=pattern,
            description=description,
            flags=flags,
            replacement=replacement,
        )
        return self.add_item(regex_pattern)

    def list_patterns(self, include_disabled: bool = False) -> List[RegexPattern]:
        """List all patterns.

        Args:
            include_disabled: Include disabled patterns

        Returns:
            List of patterns
        """
        if include_disabled:
            return list(self._items.values())
        return [p for p in self._items.values() if p.enabled]

    def apply_pattern(self, pattern_name: str, text: str, replacement: Optional[str] = None) -> Optional[str]:
        """Apply a pattern to text.

        Args:
            pattern_name: Pattern name
            text: Text to apply pattern to
            replacement: Optional replacement string (overrides pattern's replacement)

        Returns:
            Result string or None
        """
        pattern = self.get_pattern(pattern_name)
        if not pattern or not pattern.enabled:
            return None

        try:
            compiled = re.compile(pattern.pattern, pattern.flags)

            if replacement is not None:
                # Use provided replacement
                return compiled.sub(replacement, text)
            elif pattern.replacement:
                # Use pattern's replacement
                return compiled.sub(pattern.replacement, text)
            else:
                # Just return matches
                matches = compiled.findall(text)
                return f"Pattern '{pattern_name}' found {len(matches)} matches: {matches}"

        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern_name}': {e}")
            return f"Invalid regex pattern: {e}"

    def test_pattern(self, pattern_name: str, text: str) -> List[str]:
        """Test a pattern against text.

        Args:
            pattern_name: Pattern name
            text: Text to test

        Returns:
            List of matches
        """
        pattern = self.get_pattern(pattern_name)
        if not pattern or not pattern.enabled:
            return []

        try:
            compiled = re.compile(pattern.pattern, pattern.flags)
            return compiled.findall(text)
        except re.error:
            return []

    def search_patterns(self, query: str) -> List[RegexPattern]:
        """Search patterns by name or description.

        Args:
            query: Search query

        Returns:
            Matching patterns
        """
        query = query.lower()
        return [
            p for p in self._items.values()
            if query in p.name.lower() or query in p.description.lower()
        ]