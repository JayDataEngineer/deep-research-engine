"""Lorebook Activation Result.

Models for storing and rendering activation results.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict


class ActivationResult:
    """Result of lorebook activation.

    Contains activated entries, token usage, and rendering methods.
    """

    def __init__(self):
        self.activated_entries: Dict[int, Tuple] = {}  # entry_id -> (entry, content)
        self.activated_by_position: Dict[str, List[int]] = defaultdict(list)
        self.total_tokens_used: int = 0
        self.skipped_for_budget: List[int] = []
        self.matched_keywords: Dict[int, Set[str]] = defaultdict(set)

    def add_entry(
        self,
        entry,
        content: str,
        position: str,
        matched_keywords: Set[str] = None
    ):
        """Add an activated entry.

        Args:
            entry: The lorebook entry
            content: Rendered content
            position: Injection position
            matched_keywords: Set of keywords that matched
        """
        entry_id = getattr(entry, 'id', id(entry))
        self.activated_entries[entry_id] = (entry, content)
        self.activated_by_position[position].append(entry_id)
        self.total_tokens_used += len(content) // 4

        if matched_keywords:
            self.matched_keywords[entry_id].update(matched_keywords)

    def get_entry(self, entry_id: int) -> Optional[Tuple]:
        """Get an activated entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Tuple of (entry, content) or None
        """
        return self.activated_entries.get(entry_id)

    def is_empty(self) -> bool:
        """Check if any entries were activated."""
        return len(self.activated_entries) == 0

    def count(self) -> int:
        """Get number of activated entries."""
        return len(self.activated_entries)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response.

        Returns:
            Dict with activation info
        """
        entries_dict = {}
        for entry_id, (entry, content) in self.activated_entries.items():
            entries_dict[entry_id] = {
                "id": entry_id,
                "name": getattr(entry, 'name', None),
                "content": content,
                "position": getattr(entry, 'position', 'after_char'),
                "priority": getattr(entry, 'priority', getattr(entry, 'insertion_order', 0)),
                "matched_keywords": list(self.matched_keywords.get(entry_id, set())),
            }

        return {
            "activated_count": len(self.activated_entries),
            "entries": entries_dict,
            "by_position": dict(self.activated_by_position),
            "total_tokens": self.total_tokens_used,
            "skipped_for_budget": self.skipped_for_budget,
        }

    def render_context(self) -> Tuple[str, str]:
        """Render activated entries into context strings.

        Returns:
            Tuple of (before_content, after_content)
        """
        before_parts = []
        after_parts = []

        for entry_id, (entry, content) in self.activated_entries.items():
            position = getattr(entry, 'position', 'after_char')
            name = getattr(entry, 'name', f"Entry {entry_id}")

            if position in ("before_char", "EMTop", "ANTop"):
                before_parts.append(f"[{name}]\n{content}")
            else:
                after_parts.append(f"[{name}]\n{content}")

        before_content = "\n\n".join(before_parts)
        after_content = "\n\n".join(after_parts)

        return before_content, after_content

    def render_combined(self) -> str:
        """Render all entries into a single context string.

        Returns:
            Combined context string
        """
        before, after = self.render_context()
        parts = []
        if before:
            parts.append(before)
        if after:
            parts.append(after)
        return "\n\n---\n\n".join(parts)
