"""CitationResolver: Convert filesystem path citations to numbered format.

Handles the citation system for the virtual filesystem:
- Extracts [/research/tier_*/source_id.md] citations from text
- Assigns unique numbers to each cited source
- Converts to [1], [2], etc. format
- Builds formatted citations section

Example:
    >>> resolver = CitationResolver(filesystem_backend)
    >>> text = "See [/research/tier_a/abc123.md] for details."
    >>> processed, citations = resolver.convert_paths_to_numbered(text)
    >>> processed
    'See [1] for details.'
    >>> citations[0].number
    1
    >>> citations[0].title
    'Source Title'
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CitationReference:
    """A citation reference found in text."""
    path: str                      # "/research/tier_a/abc123.md"
    line_number: Optional[int] = None  # Line number where cited
    context: str = ""              # Surrounding text context


@dataclass
class Citation:
    """A fully resolved citation with metadata."""
    number: int                    # Citation number [1], [2], etc.
    path: str                      # Original filesystem path
    title: str                     # Source title
    url: str                       # Source URL
    excerpt: str = ""               # Relevant excerpt
    tier: str = ""                  # Source tier (A/B/C)
    confidence: float = 0.8         # Confidence score
    metadata: Dict[str, Any] = field(default_factory=dict)


class CitationResolver:
    """Resolve and convert filesystem path citations to numbered format.

    Path Citation Format:
        [/research/tier_a/abc123.md]

    Numbered Format:
        [1], [2], [3], etc.

    Final Citations Section:
        ## References

        1. Title of Source - URL
        2. Another Source - URL
    """

    # Pattern to match path citations: [/research/...]
    PATH_CITATION_PATTERN = r'\[/research/tier_[abc]/[a-zA-Z0-9_-]+\.md\]'

    # Default configuration values (can be overridden by yaml_config)
    DEFAULT_EXCERPT_READ_LINES = 20
    DEFAULT_EXCERPT_MAX_LENGTH = 200
    DEFAULT_PATH_PREVIEW_LENGTH = 60
    DEFAULT_ID_PREVIEW_LENGTH = 8

    def __init__(self, filesystem_backend, run_logger=None, yaml_config: Optional[Dict[str, Any]] = None):
        """Initialize CitationResolver.

        Args:
            filesystem_backend: ResearchSourcesBackend instance
            run_logger: Optional RunLogger for logging
            yaml_config: Optional YAML config dict with citation settings
        """
        self.filesystem = filesystem_backend
        self.run_logger = run_logger

        # Get citation config from YAML or use defaults
        citation_config = yaml_config.get("citations", {}) if yaml_config else {}
        self._excerpt_read_lines = citation_config.get("excerpt_read_lines", self.DEFAULT_EXCERPT_READ_LINES)
        self._excerpt_max_length = citation_config.get("excerpt_max_length", self.DEFAULT_EXCERPT_MAX_LENGTH)
        self._path_preview_length = citation_config.get("path_preview_length", self.DEFAULT_PATH_PREVIEW_LENGTH)
        self._id_preview_length = citation_config.get("id_preview_length", self.DEFAULT_ID_PREVIEW_LENGTH)

        self.citation_map: Dict[str, int] = {}  # path → number
        self.citations: List[Citation] = []
        self.next_number = 1

    # ========================================================================
    # Extraction
    # ========================================================================

    def extract_path_citations(self, text: str) -> List[CitationReference]:
        """Extract all [/research/...] citations from text.

        Args:
            text: Text to search

        Returns:
            List of CitationReference objects

        Example:
            >>> text = "See [/research/tier_a/abc123.md] and [/research/tier_b/def456.md]"
            >>> refs = resolver.extract_path_citations(text)
            >>> len(refs)
            2
            >>> refs[0].path
            '[/research/tier_a/abc123.md]'
        """
        references = []

        for match in re.finditer(self.PATH_CITATION_PATTERN, text):
            path = match.group(0)

            # Find line number
            line_num = text[:match.start()].count("\n") + 1

            # Extract context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            references.append(CitationReference(
                path=path,
                line_number=line_num,
                context=context.strip()
            ))

        return references

    def get_unique_paths(self, text: str) -> List[str]:
        """Get unique cited paths from text.

        Args:
            text: Text to search

        Returns:
            List of unique paths in order of first appearance
        """
        seen = set()
        unique_paths = []

        for match in re.finditer(self.PATH_CITATION_PATTERN, text):
            path = match.group(0)
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)

        return unique_paths

    # ========================================================================
    # Metadata Resolution
    # ========================================================================

    def get_source_metadata(self, path: str) -> Dict[str, Any]:
        """Get source metadata from filesystem backend.

        Args:
            path: Virtual filesystem path

        Returns:
            Dictionary with source metadata

        Example:
            >>> metadata = resolver.get_source_metadata("/research/tier_a/abc123.md")
            >>> metadata['title']
            'Article Title'
            >>> metadata['url']
            'https://example.com/article'
        """
        if hasattr(self.filesystem, 'get_source_metadata'):
            # ResearchSourcesBackend has this method
            return self.filesystem.get_source_metadata(path)

        # Generic fallback
        tier, source_id = self.filesystem._parse_path(path)

        if not tier or not source_id:
            return {"path": path, "error": "invalid_path"}

        # Try to read the source and extract metadata
        try:
            content = self.filesystem.read(path, limit=50)
            lines = content.split("\n")

            # Extract title from first line (remove line number)
            title = ""
            for line in lines:
                if line.strip() and not line.strip().startswith("|"):
                    title = line.strip().lstrip("#").strip()
                    break

            return {
                "source_id": source_id,
                "path": path,
                "tier": tier,
                "title": title,
                "url": "",
                "excerpt": title
            }
        except Exception as e:
            if self.run_logger:
                self.run_logger._write_to_file(f"[CITATION] Error getting metadata for {path}: {e}")

            return {
                "path": path,
                "error": str(e)
            }

    # ========================================================================
    # Number Assignment
    # ========================================================================

    def assign_number(self, path: str) -> int:
        """Assign citation number to a path (idempotent).

        Args:
            path: Filesystem path

        Returns:
            Assigned citation number
        """
        if path not in self.citation_map:
            self.citation_map[path] = self.next_number
            self.next_number += 1

        return self.citation_map[path]

    def get_number(self, path: str) -> Optional[int]:
        """Get existing citation number for a path.

        Args:
            path: Filesystem path

        Returns:
            Citation number or None if not assigned
        """
        return self.citation_map.get(path)

    # ========================================================================
    # Conversion
    # ========================================================================

    def convert_paths_to_numbered(
        self,
        text: str
    ) -> Tuple[str, List[Citation]]:
        """Convert all [/research/...] paths to [1], [2], etc.

        Args:
            text: Text containing path citations

        Returns:
            Tuple of (processed_text, citations_list)

        Example:
            >>> text = "See [/research/tier_a/abc123.md] and [/research/tier_a/def456.md]."
            >>> processed, citations = resolver.convert_paths_to_numbered(text)
            >>> processed
            'See [1] and [2].'
            >>> len(citations)
            2
        """
        # Get unique cited paths
        unique_paths = self.get_unique_paths(text)

        if self.run_logger:
            self.run_logger._write_to_file(
                f"[CITATIONS] Converting {len(unique_paths)} unique path citations "
                f"to numbered format"
            )

        # Assign numbers and build citation objects
        citations = []

        for path in unique_paths:
            # Assign number
            number = self.assign_number(path)

            # Get metadata
            metadata = self.get_source_metadata(path)

            # Read source to get excerpt
            excerpt = ""
            if "error" not in metadata:
                try:
                    content = self.filesystem.read(path, limit=self._excerpt_read_lines)
                    # Get first non-empty line
                    for line in content.split("\n"):
                        clean_line = line.strip()
                        if clean_line and not clean_line.startswith("#"):
                            excerpt = clean_line[:self._excerpt_max_length]
                            break
                except (FileNotFoundError, PermissionError, OSError, IOError, UnicodeDecodeError):
                    excerpt = metadata.get("excerpt", "")[:self._excerpt_max_length]

            citations.append(Citation(
                number=number,
                path=path,
                title=metadata.get("title", Path(path).stem),
                url=metadata.get("url", ""),
                excerpt=excerpt,
                tier=metadata.get("tier", ""),
                confidence=metadata.get("relevance_score", 0.8),
                metadata=metadata
            ))

        # Replace paths with numbers in text
        def replace_path(match):
            path = match.group(0)
            number = self.citation_map.get(path)
            if number:
                return f"[{number}]"
            return path

        processed_text = re.sub(self.PATH_CITATION_PATTERN, replace_path, text)

        return processed_text, citations

    # ========================================================================
    # Formatting
    # ========================================================================

    def format_citations_section(
        self,
        citations: Optional[List[Citation]] = None,
        style: str = "markdown"
    ) -> str:
        """Format citations list for final output.

        Args:
            citations: Optional list of Citation objects (uses internal list if None)
            style: Format style ("markdown", "academic", "html", "json")

        Returns:
            Formatted citations section

        Styles:
            markdown: Markdown with tier badges
            academic: Standard academic format
            html: HTML list with links
            json: JSON array
        """
        if citations is None:
            citations = self.citations

        if not citations:
            return ""

        if style == "markdown":
            return self._format_markdown(citations)
        elif style == "academic":
            return self._format_academic(citations)
        elif style == "html":
            return self._format_html(citations)
        elif style == "json":
            return self._format_json(citations)
        else:
            return self._format_markdown(citations)

    def _format_markdown(self, citations: List[Citation]) -> str:
        """Format citations as Markdown."""
        lines = ["\n## Sources\n"]

        for cit in sorted(citations, key=lambda c: c.number):
            tier_badge = ""
            if cit.tier:
                emoji_map = {"A": "🟢", "B": "🟡", "C": "⚪"}
                tier_badge = f" {emoji_map.get(cit.tier, '')} **[Tier {cit.tier}]**"

            lines.append(
                f"- [{cit.number}] {cit.title}{tier_badge}\n"
                f"  **URL:** {cit.url}\n"
                f"  **Path:** `{cit.path}`"
            )

        return "\n".join(lines)

    def _format_academic(self, citations: List[Citation]) -> str:
        """Format citations in academic style."""
        lines = ["\n## References\n"]

        for cit in sorted(citations, key=lambda c: c.number):
            lines.append(f"{cit.number}. {cit.title}. {cit.url}")

        return "\n".join(lines)

    def _format_html(self, citations: List[Citation]) -> str:
        """Format citations as HTML."""
        lines = ['<h2>Sources</h2>\n<ol>']

        for cit in sorted(citations, key=lambda c: c.number):
            tier_badge = f" <em>(Tier {cit.tier})</em>" if cit.tier else ""
            lines.append(
                f'<li><a href="{cit.url}">{cit.title}</a>{tier_badge}</li>'
            )

        lines.append("</ol>")
        return "\n".join(lines)

    def _format_json(self, citations: List[Citation]) -> str:
        """Format citations as JSON."""
        data = [
            {
                "number": cit.number,
                "path": cit.path,
                "title": cit.title,
                "url": cit.url,
                "tier": cit.tier,
                "excerpt": cit.excerpt
            }
            for cit in sorted(citations, key=lambda c: c.number)
        ]

        return f"\n```json\n{json.dumps(data, indent=2)}\n```\n"

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get citation statistics.

        Returns:
            Dictionary with citation stats
        """
        if not self.citations:
            return {
                "total_citations": 0,
                "unique_sources": 0,
                "by_tier": {}
            }

        by_tier = {"A": 0, "B": 0, "C": 0}

        for cit in self.citations:
            if cit.tier:
                by_tier[cit.tier] = by_tier.get(cit.tier, 0) + 1

        return {
            "total_citations": len(self.citations),
            "unique_sources": len(self.citation_map),
            "by_tier": by_tier,
            "next_number": self.next_number
        }

    # ========================================================================
    # Reset
    # ========================================================================

    def reset(self) -> None:
        """Reset resolver state (for new iteration)."""
        self.citation_map.clear()
        self.citations.clear()
        self.next_number = 1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_citations_from_text(text: str) -> List[str]:
    """Extract all path citations from text (convenience function).

    Args:
        text: Text to search

    Returns:
        List of citation paths

    Example:
        >>> text = "See [/research/tier_a/abc123.md]"
        >>> extract_citations_from_text(text)
        ['[/research/tier_a/abc123.md]']
    """
    pattern = r'\[/research/tier_[abc]/[a-zA-Z0-9_-]+\.md\]'
    return re.findall(pattern, text)


def count_citations(text: str) -> int:
    """Count total citations in text (including duplicates).

    Args:
        text: Text to count citations in

    Returns:
        Number of citations found

    Example:
        >>> text = "[/research/a.md] and [/research/a.md] again"
        >>> count_citations(text)
        2
    """
    pattern = r'\[/research/tier_[abc]/[a-zA-Z0-9_-]+\.md\]'
    return len(re.findall(pattern, text))
