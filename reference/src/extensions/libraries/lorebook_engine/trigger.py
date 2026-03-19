"""Lorebook Trigger Engine - Keyword detection and activation.

This module handles:
- Keyword matching (primary, secondary, regex)
- Case-sensitive and case-insensitive matching
- AND/OR logic for keywords
- Regex pattern matching
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class TriggerMatch:
    """Result of a trigger match."""

    def __init__(
        self,
        entry_id: int,
        matched: bool,
        matched_keywords: Set[str] = None,
        score: float = 0.0
    ):
        self.entry_id = entry_id
        self.matched = matched
        self.matched_keywords = matched_keywords or set()
        self.score = score


class KeywordMatcher:
    """Handles keyword matching logic for lorebook entries."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for case-insensitive matching."""
        return text.lower().strip()

    @staticmethod
    def matches_pattern(
        text: str,
        pattern: str,
        case_sensitive: bool = False,
        use_regex: bool = False,
    ) -> bool:
        """Check if text matches a pattern.

        Args:
            text: Text to search in
            pattern: Pattern to match
            case_sensitive: Whether to respect case
            use_regex: Whether to use regex matching

        Returns:
            True if pattern matches
        """
        search_text = text if case_sensitive else KeywordMatcher.normalize_text(text)
        search_pattern = pattern if case_sensitive else KeywordMatcher.normalize_text(pattern)

        if use_regex:
            try:
                return bool(re.search(search_pattern, search_text))
            except re.error:
                # Invalid regex, fall back to literal
                pass

        return search_pattern in search_text

    @staticmethod
    def entry_matches_text(
        entry,
        text: str
    ) -> Tuple[bool, Set[str]]:
        """Check if an entry matches the given text.

        Args:
            entry: Lorebook entry with keys, secondary_keys, etc.
            text: Text to check against

        Returns:
            Tuple of (matches, matched_keywords_set)
        """
        # Check if entry is enabled
        if hasattr(entry, 'enabled') and not entry.enabled:
            return False, set()

        # Constant entries always match
        if hasattr(entry, 'constant') and entry.constant:
            return True, set()

        # Get keywords - handle different field names
        primary_keys = getattr(entry, 'keys', []) or []
        secondary_keys = getattr(entry, 'secondary_keys', []) or []

        # Handle alternate field names (some systems use 'keywords' vs 'keys')
        if not primary_keys and hasattr(entry, 'keywords'):
            primary_keys = entry.keywords

        if not secondary_keys and hasattr(entry, 'keysecondary'):
            secondary_keys = entry.keysecondary

        # Normalize search text once
        case_sensitive = getattr(entry, 'case_sensitive', False)
        use_regex = getattr(entry, 'use_regex', False)
        search_text = text if case_sensitive else KeywordMatcher.normalize_text(text)

        # Check probability (0-100 scale)
        probability = 100
        if hasattr(entry, 'probability'):
            probability = entry.probability
        elif hasattr(entry, 'extensions') and isinstance(entry.extensions, dict):
            probability = entry.extensions.get('probability', 100)

        if probability < 100:
            import random
            if random.randint(0, 100) > probability:
                return False, set()

        matched_keywords = set()

        # Get selective mode (AND vs OR logic)
        selective = getattr(entry, 'selective', False)

        if selective:
            # AND logic: ALL primary keys must match
            # AND if secondary_keys exist, ALL must also match
            all_primary_match = True
            for keyword in primary_keys:
                if KeywordMatcher.matches_pattern(search_text, keyword, case_sensitive, use_regex):
                    matched_keywords.add(keyword)
                else:
                    all_primary_match = False
                    break

            if all_primary_match and secondary_keys:
                for keyword in secondary_keys:
                    if KeywordMatcher.matches_pattern(search_text, keyword, case_sensitive, use_regex):
                        matched_keywords.add(keyword)
                    else:
                        return False, set()

            return all_primary_match and bool(primary_keys), matched_keywords
        else:
            # OR logic: ANY keyword can match
            for keyword in primary_keys:
                if KeywordMatcher.matches_pattern(search_text, keyword, case_sensitive, use_regex):
                    matched_keywords.add(keyword)
                    return True, matched_keywords

            # Check secondary keywords (OR with primary)
            for keyword in secondary_keys:
                if KeywordMatcher.matches_pattern(search_text, keyword, case_sensitive, use_regex):
                    matched_keywords.add(keyword)
                    return True, matched_keywords

            return False, set()


class LorebookTrigger:
    """Main trigger engine for lorebooks.

    Scans conversation text and activates matching entries.
    """

    def __init__(
        self,
        default_scan_depth: int = 10,
        default_token_budget: int = 2000,
    ):
        """Initialize the trigger engine.

        Args:
            default_scan_depth: Default messages to scan back
            default_token_budget: Default token budget
        """
        self.default_scan_depth = default_scan_depth
        self.default_token_budget = default_token_budget

    def trigger(
        self,
        entries: List,
        conversation_text: str,
        conversation_messages: List[Dict] = None,
        scan_depth: int = None,
        token_budget: int = None,
    ) -> 'ActivationResult':
        """Trigger entries based on conversation.

        Args:
            entries: List of lorebook entries to check
            conversation_text: Full conversation text (pre-built)
            conversation_messages: Optional raw messages for auto-building text
            scan_depth: How many messages back to scan
            token_budget: Maximum tokens for activated entries

        Returns:
            ActivationResult with activated entries
        """
        from .result import ActivationResult

        result = ActivationResult()

        scan_depth = scan_depth or self.default_scan_depth
        budget = token_budget or self.default_token_budget

        # Build conversation text if not provided
        if conversation_messages and not conversation_text:
            conversation_text = self._build_conversation_text(
                conversation_messages, scan_depth
            )

        # Sort entries by priority (if available)
        sorted_entries = self._sort_entries_by_priority(entries)

        # Trigger entries
        used_tokens = 0
        for entry in sorted_entries:
            matches, matched_kw = KeywordMatcher.entry_matches_text(entry, conversation_text)

            if matches:
                # Estimate tokens
                content = getattr(entry, 'content', '')
                entry_tokens = len(content) // 4

                # Check budget
                if used_tokens + entry_tokens > budget:
                    result.skipped_for_budget.append(getattr(entry, 'id', id(entry)))
                    continue

                # Determine position
                position = getattr(entry, 'position', 'after_char')
                if position == "at_depth":
                    position = "after_char"

                result.add_entry(entry, content, position, matched_kw)
                used_tokens += entry_tokens

                if used_tokens >= budget:
                    break

        return result

    def _sort_entries_by_priority(self, entries: List) -> List:
        """Sort entries by priority.

        Higher priority = earlier activation.
        Uses 'priority' or 'insertion_order' field.
        """
        def get_priority(entry):
            # Try different priority field names
            if hasattr(entry, 'priority'):
                return entry.priority
            if hasattr(entry, 'insertion_order'):
                return entry.insertion_order
            return 0

        return sorted(entries, key=lambda e: (get_priority(e), getattr(e, 'id', 0)), reverse=True)

    def _build_conversation_text(
        self,
        messages: List[Dict],
        scan_depth: int,
    ) -> str:
        """Build searchable text from conversation messages.

        Args:
            messages: List of message dicts
            scan_depth: Number of recent messages to include

        Returns:
            Concatenated text
        """
        # Get last N messages
        recent = messages[-scan_depth:] if scan_depth < len(messages) else messages

        parts = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle multimodal content
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                content = str(content)

            parts.append(f"{role}: {content}")

        return "\n\n".join(parts)
